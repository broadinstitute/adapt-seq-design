#!/bin/python
#
# Combine CCF005 and CCF024 data (the .curated.tsv files).
# The only reason that this is needed -- as opposed to just concatenating
# the TSV files -- is that there is a small number of shared guide-target
# pairs between the two experiments (e.g., wildtype target, which has 0
# mismatches against guide), and this data should be combined to one row.


from collections import defaultdict

import numpy as np


def read_curated_data(in_fn):
    """Read .curated.tsv file.

    Args:
        in_fn: path to file

    Returns:
        list of column names, dict {column name: index}, list of rows
    """
    cols = []
    col_idx = {}
    rows = []
    with open(in_fn) as f:
        for i, line in enumerate(f):
            ls = line.rstrip().split('\t')

            if i == 0:
                # Header
                cols = ls
                col_idx = {cn: i for i, cn in enumerate(cols)}
                continue

            rows += [ls]
    return cols, col_idx, rows


def merge_rows(rows, cols, col_idx):
    """Merge rows with the same guide-target pair.

    Args:
        rows: list of rows across multiple datasets
        cols: list of column names
        col_idx: dict {column name: index of column}

    Returns:
        rows with unique guide-target pairs for each row,
        which pool measurements across the input rows
    """
    # Make a dict mapping: {guide-target pair info: row}
    rows_pooled = defaultdict(list)
    for row in rows:
        guide_seq = row[col_idx['guide_seq']]
        target_at_guide = row[col_idx['target_at_guide']]
        target_before = row[col_idx['target_before']]
        target_after = row[col_idx['target_after']]
        row_id = (guide_seq, target_at_guide, target_before, target_after)
        rows_pooled[row_id].append(row)

    # Merge measurements with the same guide-target pair
    rows_merged = []
    for row_id, rows_with_id in rows_pooled.items():
        measurements_str = []
        for row in rows_with_id:
            measurements_str += row[col_idx['out_logk_measurements']].split(',')
        measurements_float = [float(x) for x in measurements_str]
        median = np.median(measurements_float)
        stdev = np.std(measurements_float)
        count = len(measurements_float)
        merged_measurements_str = ','.join(measurements_str)

        # Find which columns have data to merge, and which should be
        # shared across rows
        cols_to_merge = set(['out_logk_median', 'out_logk_stdev', 'out_logk_replicate_count', 'out_logk_measurements'])
        cols_shared = set(cols) - cols_to_merge

        row_merged = []
        for col in cols:
            if col in cols_shared:
                # Find the shared value across all rows with this guide-target pair
                vals = [row[col_idx[col]] for row in rows_with_id]
                assert len(set(vals)) == 1  # check that there is 1 shared value
                val = list(set(vals))[0]
            else:
                if col == 'out_logk_median':
                    val = median
                elif col == 'out_logk_stdev':
                    val = stdev
                elif col == 'out_logk_replicate_count':
                    val = count
                elif col == 'out_logk_measurements':
                    val = merged_measurements_str
                else:
                    raise Exception("Unknown column '%s' to merge" % col)
            row_merged += [str(val)]

        rows_merged += [row_merged]

    return rows_merged


def write_merged_rows(cols, rows, out_fn):
    """Write a TSV file output, after reformatting.

    Args:
        cols: list of column names
        rows: list of rows, merged
        out_fn: path to output TSV file
    """
    with open(out_fn, 'w') as fw:
        def write_list(l):
            fw.write('\t'.join([str(x) for x in l]) + '\n')
        # Write header
        write_list(cols)
        # Write each row
        for row in rows:
            write_list(row)


def main():
    # Paths to input/output files
    IN_CCF005 = "CCF005_pairs_annotated.curated.tsv"
    IN_CCF024 = "CCF024_pairs_annotated.curated.tsv"
    OUT = "CCF_merged_pairs_annotated.curated.tsv"

    ccf005_cols, ccf005_col_idx, ccf005_rows = read_curated_data(IN_CCF005)
    ccf024_cols, ccf024_col_idx, ccf024_rows = read_curated_data(IN_CCF024)
        
    # Check that column names and indices are the same
    assert ccf005_cols == ccf024_cols
    assert ccf005_col_idx == ccf024_col_idx
    cols = ccf005_cols
    col_idx = ccf005_col_idx

    # Concatenate rows
    rows = ccf005_rows + ccf024_rows

    # Merge rows
    rows_merged = merge_rows(rows, cols, col_idx)

    # Write merged TSV
    write_merged_rows(cols, rows_merged, OUT)


if __name__ == "__main__":
    main()
