import time
import numpy as np
import polars as pl
from rapidfuzz import fuzz

def extract_pairwise_features(
    candidates_df: pl.DataFrame,
    s1_proc: pl.DataFrame,
    s23_proc: pl.DataFrame
) -> pl.DataFrame:
    """
    Given candidate pairs [source1_entity_id, candidate_entity_id, blocking_score],
    joins original preprocessed record features and computes pairwise similarity metrics.
    """
    t0 = time.time()

    # Rename columns for clarity before joining
    s1_sub = s1_proc.select([
        pl.col('entity_id').alias('source1_entity_id'),
        pl.col('clean_name').alias('s1_name'),
        pl.col('clean_address').alias('s1_address'),
        pl.col('sorted_name').alias('s1_sorted_name'),
        pl.col('address_nums').alias('s1_addr_nums'),
    ])

    s23_sub = s23_proc.select([
        pl.col('entity_id').alias('candidate_entity_id'),
        pl.col('clean_name').alias('cand_name'),
        pl.col('clean_address').alias('cand_address'),
        pl.col('sorted_name').alias('cand_sorted_name'),
        pl.col('address_nums').alias('cand_addr_nums'),
    ])

    # Join preprocessed attributes to candidate pairs
    merged = candidates_df.join(s1_sub, on='source1_entity_id', how='left') \
                          .join(s23_sub, on='candidate_entity_id', how='left')

    s1_names = merged['s1_name'].fill_null('').to_list()
    cand_names = merged['cand_name'].fill_null('').to_list()

    s1_addrs = merged['s1_address'].fill_null('').to_list()
    cand_addrs = merged['cand_address'].fill_null('').to_list()

    s1_nums_list = merged['s1_addr_nums'].fill_null('').to_list()
    cand_nums_list = merged['cand_addr_nums'].fill_null('').to_list()

    cand_ids = merged['candidate_entity_id'].to_list()
    n_pairs = len(merged)

    # Pre-allocate feature arrays
    name_ratio = np.zeros(n_pairs, dtype=np.float32)
    name_token_sort = np.zeros(n_pairs, dtype=np.float32)
    name_token_set = np.zeros(n_pairs, dtype=np.float32)
    name_partial = np.zeros(n_pairs, dtype=np.float32)

    addr_ratio = np.zeros(n_pairs, dtype=np.float32)
    addr_token_sort = np.zeros(n_pairs, dtype=np.float32)
    addr_token_set = np.zeros(n_pairs, dtype=np.float32)

    num_match_ratio = np.zeros(n_pairs, dtype=np.float32)
    is_source2 = np.zeros(n_pairs, dtype=np.float32)
    addr_is_empty = np.zeros(n_pairs, dtype=np.float32)

    # Compute rapidfuzz metrics loop
    for i in range(n_pairs):
        n1, n2 = s1_names[i], cand_names[i]
        a1, a2 = s1_addrs[i], cand_addrs[i]

        name_ratio[i] = fuzz.ratio(n1, n2) / 100.0
        name_token_sort[i] = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_token_set[i] = fuzz.token_set_ratio(n1, n2) / 100.0
        name_partial[i] = fuzz.partial_ratio(n1, n2) / 100.0

        if a1 and a2:
            addr_ratio[i] = fuzz.ratio(a1, a2) / 100.0
            addr_token_sort[i] = fuzz.token_sort_ratio(a1, a2) / 100.0
            addr_token_set[i] = fuzz.token_set_ratio(a1, a2) / 100.0
        else:
            addr_is_empty[i] = 1.0

        # Number overlap ratio
        num1_set = set(s1_nums_list[i].split()) if s1_nums_list[i] else set()
        num2_set = set(cand_nums_list[i].split()) if cand_nums_list[i] else set()
        if num1_set and num2_set:
            num_match_ratio[i] = len(num1_set.intersection(num2_set)) / max(len(num1_set), len(num2_set))

        is_source2[i] = 1.0 if cand_ids[i].startswith('S2-') else 0.0

    features_df = merged.with_columns([
        pl.Series('name_ratio', name_ratio),
        pl.Series('name_token_sort', name_token_sort),
        pl.Series('name_token_set', name_token_set),
        pl.Series('name_partial', name_partial),
        pl.Series('addr_ratio', addr_ratio),
        pl.Series('addr_token_sort', addr_token_sort),
        pl.Series('addr_token_set', addr_token_set),
        pl.Series('num_match_ratio', num_match_ratio),
        pl.Series('is_source2', is_source2),
        pl.Series('addr_is_empty', addr_is_empty),
    ])

    print(f"Extracted features for {n_pairs:,} candidate pairs in {time.time() - t0:.2f}s")
    return features_df
