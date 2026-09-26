import os
import polars as pl

def export_candidate_pairs_tsv(
    candidates_df: pl.DataFrame,
    s1_all_ids: list[str],
    output_path: str = "output/candidate_pairs.tsv"
):
    """
    Format and export candidate pairs produced by blocking into candidate_pairs.tsv.
    Columns: source1_entity_id, candidate_entity_ids (comma separated)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if len(candidates_df) > 0:
        grouped = candidates_df.group_by('source1_entity_id').agg(
            pl.col('candidate_entity_id').unique().implode().map_elements(lambda l: ','.join(l), return_dtype=pl.String).alias('candidate_entity_ids')
        )
    else:
        grouped = pl.DataFrame(schema={'source1_entity_id': pl.String, 'candidate_entity_ids': pl.String})

    s1_base = pl.DataFrame({'source1_entity_id': s1_all_ids})
    final_cands = s1_base.join(grouped, on='source1_entity_id', how='left').with_columns(
        pl.col('candidate_entity_ids').fill_null('')
    )

    final_cands.write_csv(output_path, separator='\t', quote_style='never')
    print(f"Exported candidate_pairs.tsv: {len(final_cands):,} rows to {output_path}")

def export_matching_results_tsv(
    results_df: pl.DataFrame,
    output_path: str = "output/matching_results.tsv"
):
    """
    Export final predicted matches to matching_results.tsv.
    Columns: source1_entity_id, matched_entity_ids (comma separated)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.write_csv(output_path, separator='\t', quote_style='never')
    print(f"Exported matching_results.tsv: {len(results_df):,} rows to {output_path}")

def verify_submission_files(
    s1_test_ids: list[str],
    cand_path: str = "output/candidate_pairs.tsv",
    match_path: str = "output/matching_results.tsv"
) -> bool:
    """
    Rigorously verify compliance of submission files with challenge rules.
    """
    print("=== Running Submission Integrity Verification ===")
    
    # 1. Check existence
    if not os.path.exists(cand_path) or not os.path.exists(match_path):
        print(f"ERROR: Submission file missing! cand_path: {os.path.exists(cand_path)}, match_path: {os.path.exists(match_path)}")
        return False

    cands = pl.read_csv(cand_path, separator='\t', quote_char=None)
    matches = pl.read_csv(match_path, separator='\t', quote_char=None)

    # 2. Check row counts
    expected_count = len(s1_test_ids)
    if len(matches) != expected_count:
        print(f"ERROR: matching_results.tsv row count mismatch! Expected {expected_count}, got {len(matches)}")
        return False
    if len(cands) != expected_count:
        print(f"ERROR: candidate_pairs.tsv row count mismatch! Expected {expected_count}, got {len(cands)}")
        return False

    # 3. Check exact columns
    if matches.columns != ['source1_entity_id', 'matched_entity_ids']:
        print(f"ERROR: Incorrect columns in matching_results.tsv: {matches.columns}")
        return False
    if cands.columns != ['source1_entity_id', 'candidate_entity_ids']:
        print(f"ERROR: Incorrect columns in candidate_pairs.tsv: {cands.columns}")
        return False

    # 4. Check candidate containment: every matched ID must be in candidates
    cand_dict = {}
    for row in cands.iter_rows(named=True):
        cand_dict[row['source1_entity_id']] = set(row['candidate_entity_ids'].split(',')) if row['candidate_entity_ids'] else set()

    for row in matches.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        m_set = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
        c_set = cand_dict.get(s1_id, set())

        if not m_set.issubset(c_set):
            diff = m_set - c_set
            print(f"ERROR: Matched ID(s) {diff} for entity {s1_id} not present in candidate_pairs.tsv!")
            return False

    print("SUCCESS: All verification checks passed 100%! Ready for submission.")
    return True
