#!/bin/python

# Resample replicates (droplets measurements) from CCF data to pick the
# sampe number of replicates per guide-target pair.
#
# Each guide-target pair has some number of technical replicates; the
# median is 14. These are each droplets in the CARMEN experiment.
# We'll sample a particular number for each guide-target pair, randomly
# with replacement; if the number of guide-target pairs is N and we
# sample S, we'll end up with N*S data points (or 'samples') in our
# dataset.
# Note that this is preferable to just using all technical replicates/
# measurements in the dataset, because the number of replicates varies
# per guide-target pair and this ensures that every guide-target pair
# is represented with the same number of samples in the dataset.


import gzip

import numpy as np

import merge_ccf_data


def make_row_per_measurement(cols, col_idx, rows,
        num_replicates_to_sample=10):
    """Resample measurements, making one row per measurement.

    Args:
        cols: list of column names
        col_idx: dict {column name: index of column}
        rows: list of rows across multiple datasets
        num_replicates_to_sample: number of measurements to sample,
            with replacement, for each guide-target row

    Returns:
        cols and rows, with one measurement per row; cols is
        slightly different (for 'out_*' values) than the
        input cols
    """
    cols_input = [c for c in cols if not c.startswith('out_')]
    new_cols = cols_input + ['out_logk_measurement']

    new_rows = []
    num_with_sufficient_replicates = 0
    for row in rows:
        # Start the new rows with input (non-output) values
        new_row_start = []
        for c in cols_input:
            new_row_start += [row[col_idx[c]]]

        # Get measurements for this guide-target pair
        measurements_str = row[col_idx['out_logk_measurements']].split(',')
        measurements = [float(x) for x in measurements_str]
        measurements_sampled = np.random.choice(measurements,
                size=num_replicates_to_sample)

        if len(measurements) >= num_replicates_to_sample:
            num_with_sufficient_replicates += 1
        
        for m in measurements_sampled:
            new_row = new_row_start + [m]
            new_rows += [new_row]

    print(("Number of input rows (guide-target pairs) with >= %d "
        "measurements is %d of %d") % (num_replicates_to_sample,
            num_with_sufficient_replicates, len(rows)))
    return new_cols, new_rows


def write_rows(cols, rows, out_fn):
    """Write a .tsv.gz file output.

    Args:
        cols: list of column names
        rows: list of rows, merged
        out_fn: path to output TSV file
    """
    with gzip.open(out_fn, 'wt') as fw:
        def write_list(l):
            fw.write('\t'.join([str(x) for x in l]) + '\n')
        # Write header
        write_list(cols)
        # Write each row
        for row in rows:
            write_list(row)


def main():
    # Set seed
    np.random.seed(1)

    # Paths to input/output files
    IN_CCF = "CCF-curated/CCF_merged_pairs_annotated.curated.tsv"
    OUT = "CCF-curated/CCF_merged_pairs_annotated.curated.resampled.tsv.gz"

    # Read
    cols, col_idx, rows = merge_ccf_data.read_curated_data(IN_CCF)

    # Transform to have 1 row per measurement
    new_cols, new_rows = make_row_per_measurement(cols, col_idx, rows)

    # Write
    write_rows(new_cols, new_rows, OUT)


if __name__ == "__main__":
    main()
