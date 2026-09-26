import os
import sys
import time
import joblib
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import preprocess_dataframe
from src.blocking import run_blocking_pipeline
from src.features import extract_pairwise_features
from src.model import predict_matches
from src.postprocess import (
    export_candidate_pairs_tsv,
    export_matching_results_tsv,
    verify_submission_files
)

def run_prediction_pipeline(
    cand_out_path: str = "output/candidate_pairs.tsv",
    match_out_path: str = "output/matching_results.tsv",
    threshold: float = 0.45
):
    """
    Full inference pipeline for competition test dataset.
    Generates output/candidate_pairs.tsv and output/matching_results.tsv.
    """
    t0 = time.time()
    print("=== Starting Test Prediction Pipeline ===")

    # 1. Load Test Datasets
    print("Loading test dataset files...")
    test_s1 = pl.read_csv('dataset/test/test_source1.tsv', separator='\t')
    test_s2 = pl.read_csv('dataset/test/test_source2.tsv', separator='\t')
    test_s3 = pl.read_csv('dataset/test/test_source3.tsv', separator='\t')

    s1_ids = test_s1['entity_id'].to_list()
    print(f"Loaded Test Records - S1: {len(test_s1):,}, S2: {len(test_s2):,}, S3: {len(test_s3):,}")

    # 2. Preprocess Test Records
    print("Preprocessing test records...")
    s1_proc = preprocess_dataframe(test_s1)
    s23_proc = preprocess_dataframe(pl.concat([test_s2, test_s3]))

    # 3. Candidate Blocking
    print("Running Candidate Blocking on Test Data...")
    candidates = run_blocking_pipeline(test_s1, test_s2, test_s3, top_k=25)

    # 4. Export candidate_pairs.tsv
    export_candidate_pairs_tsv(candidates, s1_ids, output_path=cand_out_path)

    # 5. Extract Pairwise Features
    print("Extracting Pairwise Features for Candidate Pairs...")
    features_df = extract_pairwise_features(candidates, s1_proc, s23_proc)

    # 6. Load Trained Model & Predict Matches
    print("Loading trained model ('model.pkl')...")
    model = joblib.load('model.pkl')

    print(f"Predicting entity matches with threshold tau = {threshold}...")
    predictions = predict_matches(model, features_df, s1_ids, threshold=threshold)

    # 7. Export matching_results.tsv
    export_matching_results_tsv(predictions, output_path=match_out_path)

    # 8. Rigorous Submission Integrity Verification
    verification_success = verify_submission_files(s1_ids, cand_path=cand_out_path, match_path=match_out_path)
    
    if verification_success:
        print(f"🎉 PREDICTION PIPELINE SUCCESSFUL in {time.time() - t0:.2f}s!")
    else:
        print("❌ PREDICTION PIPELINE VERIFICATION FAILED!")

if __name__ == '__main__':
    run_prediction_pipeline()
