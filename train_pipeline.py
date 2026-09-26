import os
import sys
import time
import joblib
import polars as pl
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import preprocess_dataframe
from src.blocking import run_blocking_pipeline
from src.features import extract_pairwise_features
from src.model import add_labels_to_features, train_lgbm_model, predict_matches

def run_training_pipeline(sample_size: int = 100000):
    """
    Train and evaluate Entity Resolution model pipeline.
    sample_size: Number of S1 records to sample for fast training/validation iteration.
    """
    t0 = time.time()
    print(f"=== Starting Training Pipeline (S1 Sample Size: {sample_size:,}) ===")

    # 1. Load Data
    s1_all = pl.read_csv('dataset/train/train_source1.tsv', separator='\t')
    if sample_size and sample_size < len(s1_all):
        s1 = s1_all.head(sample_size)
    else:
        s1 = s1_all

    s1_ids_set = set(s1['entity_id'])
    gt = pl.read_csv('dataset/train/train_ground_truth.tsv', separator='\t')
    gt_sample = gt.filter(pl.col('source1_entity_id').is_in(s1_ids_set))

    # Identify matching S2/S3 entity IDs
    matched_ids_set = set()
    for row in gt_sample.iter_rows(named=True):
        if row['matched_entity_ids']:
            matched_ids_set.update(row['matched_entity_ids'].split(','))

    print(f"Loaded {len(s1):,} S1 records and {len(matched_ids_set):,} relevant matching S2/S3 records.")

    # Load S2 and S3 (including all matches + background noise)
    s2 = pl.read_csv('dataset/train/train_source2.tsv', separator='\t')
    s3 = pl.read_csv('dataset/train/train_source3.tsv', separator='\t')

    # 2. Preprocess records
    s1_proc = preprocess_dataframe(s1)
    s23_proc = preprocess_dataframe(pl.concat([s2, s3]))

    # 3. Candidate Blocking
    candidates = run_blocking_pipeline(s1, s2, s3, top_k=25)

    # Calculate Blocking Recall Ceiling
    gt_pairs = set()
    for row in gt_sample.iter_rows(named=True):
        if row['matched_entity_ids']:
            for m_id in row['matched_entity_ids'].split(','):
                gt_pairs.add((row['source1_entity_id'], m_id))

    cand_pairs = set(zip(candidates['source1_entity_id'], candidates['candidate_entity_id']))
    recalled_pairs = gt_pairs.intersection(cand_pairs)
    blocking_recall = len(recalled_pairs) / max(len(gt_pairs), 1)

    print(f"=== Blocking Evaluation ===")
    print(f"Total True Positive Pairs: {len(gt_pairs):,}")
    print(f"Recalled Pairs in Blocking: {len(recalled_pairs):,}")
    print(f"Blocking Recall Ceiling: {blocking_recall:.4%}")

    # 4. Feature Extraction
    features_df = extract_pairwise_features(candidates, s1_proc, s23_proc)

    # 5. Add Labels
    labeled_df = add_labels_to_features(features_df, gt_sample)

    # 6. Train-Validation Split by S1 Entity ID
    unique_s1 = s1['entity_id'].to_list()
    train_ids, val_ids = train_test_split(unique_s1, test_size=0.2, random_state=42)

    train_df = labeled_df.filter(pl.col('source1_entity_id').is_in(set(train_ids)))
    val_df = labeled_df.filter(pl.col('source1_entity_id').is_in(set(val_ids)))

    # 7. Model Training
    model = train_lgbm_model(train_df, val_df)

    # Save model
    joblib.dump(model, 'model.pkl')
    print("Saved model to 'model.pkl'")

    # 8. Offline Evaluation on Validation Set
    val_preds = predict_matches(model, val_df, val_ids, threshold=0.45)
    
    val_gt_dict = {}
    for row in gt_sample.filter(pl.col('source1_entity_id').is_in(set(val_ids))).iter_rows(named=True):
        val_gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()

    tp, fp, fn = 0, 0, 0
    for row in val_preds.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        pred_set = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
        actual_set = val_gt_dict.get(s1_id, set())

        tp += len(pred_set.intersection(actual_set))
        fp += len(pred_set - actual_set)
        fn += len(actual_set - pred_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    print("=== Validation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Training completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    run_training_pipeline(sample_size=50000)
