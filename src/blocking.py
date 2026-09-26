import time
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict
# pyrefly: ignore [missing-import]
from src.preprocessing import preprocess_dataframe

def generate_candidates_for_country(
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    top_k: int = 25,
    batch_size: int = 2500
) -> List[Tuple[str, List[str], List[float]]]:
    """
    Generate candidate pairs for S1 records against S2/S3 records within a country partition.
    Returns list of tuples: (s1_id, list_of_cand_ids, list_of_scores)
    """
    if len(s1_df) == 0 or len(s23_df) == 0:
        return []

    s1_text = (s1_df['clean_name'] + ' ' + s1_df['clean_address']).to_list()
    s23_text = (s23_df['clean_name'] + ' ' + s23_df['clean_address']).to_list()
    
    s1_ids = s1_df['entity_id'].to_list()
    s23_ids = s23_df['entity_id'].to_list()

    # Fit TF-IDF Vectorizer
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, sublinear_tf=True)
    all_text = s1_text + s23_text
    vec.fit(all_text)

    X_s1: csr_matrix = vec.transform(s1_text).tocsr()
    X_s23: csr_matrix = vec.transform(s23_text).tocsr()

    results = []
    
    # Process S1 in memory-efficient batches
    num_s1 = X_s1.shape[0]
    num_s23 = X_s23.shape[0]
    eff_k = min(top_k, num_s23)

    for i in range(0, num_s1, batch_size):
        sub_X1 = X_s1[i : i + batch_size]
        scores = (sub_X1 @ X_s23.transpose()).toarray() # shape: (batch_size, num_s23)

        # Get top_k indices for each row in batch
        top_k_indices = np.argpartition(scores, -eff_k, axis=1)[:, -eff_k:]

        for r_idx in range(sub_X1.shape[0]):
            s1_id = s1_ids[i + r_idx]
            row_scores = scores[r_idx]
            cand_indices = top_k_indices[r_idx]

            # Sort candidate indices by score descending
            sorted_cand_indices = cand_indices[np.argsort(-row_scores[cand_indices])]
            
            # Keep non-zero score candidates
            valid_indices = [c for c in sorted_cand_indices if row_scores[c] > 0.01]
            
            cand_ids = [s23_ids[c] for c in valid_indices]
            cand_scores = [float(row_scores[c]) for c in valid_indices]

            results.append((s1_id, cand_ids, cand_scores))

    return results

def run_blocking_pipeline(
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    top_k: int = 25
) -> pl.DataFrame:
    """
    Run country-partitioned blocking across S1 and combined (S2 + S3).
    Returns DataFrame with columns: [source1_entity_id, candidate_entity_id, blocking_score]
    """
    t0 = time.time()
    
    # Preprocess inputs
    print("Preprocessing S1...")
    s1_proc = preprocess_dataframe(s1_df)
    print("Preprocessing S2...")
    s2_proc = preprocess_dataframe(s2_df)
    print("Preprocessing S3...")
    s3_proc = preprocess_dataframe(s3_df)

    s23_proc = pl.concat([s2_proc, s3_proc])

    countries = s1_proc['country'].unique().to_list()
    print(f"Found {len(countries)} country partition(s): {countries}")

    all_pairs_s1 = []
    all_pairs_cand = []
    all_scores = []

    for country in countries:
        c_s1 = s1_proc.filter(pl.col('country') == country)
        c_s23 = s23_proc.filter(pl.col('country') == country)
        
        print(f"--- Blocking for country: '{country}' (S1: {len(c_s1):,}, S23: {len(c_s23):,}) ---")
        
        c_results = generate_candidates_for_country(c_s1, c_s23, top_k=top_k)
        
        for s1_id, cand_ids, scores in c_results:
            for c_id, sc in zip(cand_ids, scores):
                all_pairs_s1.append(s1_id)
                all_pairs_cand.append(c_id)
                all_scores.append(sc)

    cand_df = pl.DataFrame({
        'source1_entity_id': all_pairs_s1,
        'candidate_entity_id': all_pairs_cand,
        'blocking_score': all_scores
    })

    print(f"Blocking complete! Generated {len(cand_df):,} candidate pairs in {time.time() - t0:.2f}s")
    return cand_df
