import time
import numpy as np
import polars as pl
import lightgbm as lgb
from typing import List, Dict, Tuple, Set

FEATURE_COLS = [
    'blocking_score',
    'name_ratio',
    'name_token_sort',
    'name_token_set',
    'name_partial',
    'addr_ratio',
    'addr_token_sort',
    'addr_token_set',
    'num_match_ratio',
    'is_source2',
    'addr_is_empty',
]

def add_labels_to_features(
    features_df: pl.DataFrame,
    gt_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Construct ground truth lookup set and assign binary target label (1 if true match, 0 otherwise).
    """
    gt_pairs = set()
    gt_valid = gt_df.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != ''))
    
    for row in gt_valid.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids'].split(',')
        for m_id in matches:
            gt_pairs.add((s1_id, m_id))

    s1_ids = features_df['source1_entity_id'].to_list()
    cand_ids = features_df['candidate_entity_id'].to_list()

    labels = np.array([
        1.0 if (s1_ids[i], cand_ids[i]) in gt_pairs else 0.0
        for i in range(len(features_df))
    ], dtype=np.float32)

    return features_df.with_columns(pl.Series('target', labels))

def train_lgbm_model(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame | None = None
) -> lgb.LGBMClassifier:
    """Train LightGBM binary classifier on candidate features."""
    t0 = time.time()
    X_train = train_df.select(FEATURE_COLS).to_numpy()
    y_train = train_df['target'].to_numpy()

    print(f"Training LightGBM on {len(X_train):,} samples (Positives: {int(y_train.sum()):,}, Negatives: {int(len(y_train) - y_train.sum()):,})...")

    # Pos weight to balance hard negatives
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)

    clf = lgb.LGBMClassifier(
        n_estimators=600,
        max_depth=7,
        num_leaves=63,
        learning_rate=0.05,
        scale_pos_weight=min(pos_weight, 5.0), # cap weight to avoid high false positives
        n_jobs=16,
        random_state=42
    )

    if val_df is not None:
        X_val = val_df.select(FEATURE_COLS).to_numpy()
        y_val = val_df['target'].to_numpy()
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
    else:
        clf.fit(X_train, y_train)

    print(f"LGBM model trained in {time.time() - t0:.2f}s")
    return clf

def predict_matches(
    model: lgb.LGBMClassifier,
    features_df: pl.DataFrame,
    s1_all_ids: List[str],
    threshold: float = 0.45
) -> pl.DataFrame:
    """
    Predict entity matches for all S1 entities given features and threshold.
    Returns DataFrame matching exact competition format: [source1_entity_id, matched_entity_ids]
    """
    if len(features_df) > 0:
        X_feat = features_df.select(FEATURE_COLS).to_numpy()
        raw_probs = model.predict_proba(X_feat)
        probs = np.asarray(raw_probs)[:, 1]
        
        pred_df = features_df.with_columns(pl.Series('prob', probs))
        
        # Filter candidate pairs exceeding decision threshold
        filtered = pred_df.filter(pl.col('prob') >= threshold)
        
        # Group matched candidate entity IDs per source1_entity_id
        grouped = filtered.group_by('source1_entity_id').agg(
            pl.col('candidate_entity_id').implode().map_elements(lambda l: ','.join(l), return_dtype=pl.String).alias('matched_entity_ids')
        )
    else:
        grouped = pl.DataFrame(schema={'source1_entity_id': pl.String, 'matched_entity_ids': pl.String})

    # Right join to ensure EVERY single S1 entity is present in exact original order
    s1_base = pl.DataFrame({'source1_entity_id': s1_all_ids})
    result = s1_base.join(grouped, on='source1_entity_id', how='left').with_columns(
        pl.col('matched_entity_ids').fill_null('')
    )

    return result
