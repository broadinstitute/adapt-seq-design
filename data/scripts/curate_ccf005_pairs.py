#!/bin/python
#
# This data is from Nick Haradhvala's synthetic library of Cas13 crRNAs
# tested using CARMEN/Cas13.
# This is for the fisrt experiment we did (Spring, 2019): CCF005.


# Amount of target sequence context to extract for each guide
CONTEXT_NT = 20


from collections import defaultdict
import csv
import gzip
import math
import statistics


def read_input(in_fn):
    """Read annotated (summarized) input csv file.

    In this file, every line represents a guide-target pair.

    Args:
        in_fn: path to input file

    Returns:
        list of dicts where each element corresponds to a row
    """
    col_names = {}
    lines = []
    with open(in_fn) as f:
        for i, line in enumerate(f):
            # Split the line and skip the first column (row number)
            ls = line.rstrip().split(',')
            ls = ls[1:]

            if i == 0:
                # Read header
                for j, col_name in enumerate(ls):
                    col_names[j] = col_name
            else:
                # Read line
                row = {}
                for j, val in enumerate(ls):
                    row[col_names[j]] = val
                lines += [row]
    return lines


def read_droplet_input(in_droplets):
    """Read input csv file of droplets.

    In this file, every line represents a droplet. There may be multiple
    droplets for each guide-target pair, so a pair can be represented by many
    lines.

    This file is messy -- e.g., newline characters within quotes -- so let's
    use the csv module here to read.

    Args:
        in_droplets: path to input file

    Returns:
        list of dicts where each element corresponds to a droplet
    """
    # Only keep a subset of the columns
    cols_to_keep = ['Target', 'crRNA', 'k']

    col_name_idx = {}
    lines = []
    with gzip.open(in_droplets, 'rt') as f:
        reader = csv.reader(f)
        col_names = next(reader, None)
        col_name_idx = {k: i for i, k in enumerate(col_names)}
        for i, ls in enumerate(reader):
            row = {}
            for col in cols_to_keep:
                row[col] = ls[col_name_idx[col]]
            lines += [row]
    return lines


def filter_controls(rows):
    """Remove crRNA controls from rows.

    This leaves in target controls.

    Returns:
        rows with only experiments
    """
    rows_filtered = []
    for row in rows:
        if row['guide_type'] == 'exp':
            # Check this row
            assert 'control' not in row['crRNA']
            rows_filtered += [row]
        else:
            # Check this is a control
            assert row['guide_type'] == 'neg' or row['guide_type'] == 'pos'
            assert 'control' in row['crRNA']
    return rows_filtered


def filter_inactive_guides(rows):
    """Filter two inactive guides.

    For some reason, two guides were completely inactive -- probably a
    technical issue. Filter these out.

    Returns:
        rows with two inactive guides filtered
    """
    inactive_guides = ['block18_guide0', 'block7_guide13']

    rows_filtered = []
    for row in rows:
        if row['crRNA'] in inactive_guides:
            # Verify this is inactive
            assert float(row['median']) < -2.5
        else:
            # Keep it
            rows_filtered += [row]
    return rows_filtered


def hamming_dist(a, b):
    """Compute Hamming distance between two strings.
    """
    assert len(a) == len(b)
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def reverse_complement(x):
    """Construct reverse complement of string.
    """
    rc = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    x = x.upper()
    return ''.join(rc[b] for b in x[::-1])


def reformat_row(row):
    """Verify and summarize contents of a row.

    Args:
        row: dict representing a row (guide-target pair)

    Returns:
        row with new columns, removed columns, and renamed columns
    """
    # Check that guide_target is the reverse complement of spacer_seq
    spacer_rc = reverse_complement(row['spacer_seq'].replace('u', 't'))
    assert spacer_rc == row['guide_target']
    guide_seq = row['guide_target']
    guide_pos = int(row['pos'])
    full_target_seq = row['target_seq']

    # Check that the Hamming distance to the target is reasonable
    target_at_guide = full_target_seq[guide_pos:(guide_pos + len(guide_seq))]
    hd = hamming_dist(guide_seq, target_at_guide)
    if row['target_type'] == 'exp':
        # 1 mismatch (if mismatch is within guide)
        if int(float(row['mismatch_position'])) < 28:
            assert hd == 1
        else:
            assert hd == 0
    elif row['target_type'] == 'pos':
        # matching
        assert hd == 0
    elif row['target_type'] == 'neg':
        # not matching
        assert hd > 1

    # Extract target sequence before and after guide (context)
    target_before = full_target_seq[(guide_pos - CONTEXT_NT):guide_pos]
    target_after = full_target_seq[(guide_pos + len(guide_seq)):(guide_pos + len(guide_seq) + CONTEXT_NT)]
    assert (target_before + target_at_guide + target_after) in full_target_seq

    # Add 'N' before or after target context if there are no bases there
    if len(target_before) < CONTEXT_NT:
        missing_bases = CONTEXT_NT - len(target_before)
        target_before = 'N'*missing_bases + target_before
    if len(target_after) < CONTEXT_NT:
        missing_bases = CONTEXT_NT - len(target_after)
        target_after = 'N'*missing_bases + target_after

    # Check the PFS
    if row['PFS'] != '':
        assert row['PFS'] == target_after[:2]

    # Extract the block
    block = int(float(row['block']))
    assert block == float(row['block'])

    # Remake row
    row_new = {}
    row_new['crrna'] = row['crRNA']
    row_new['target'] = row['Target']
    row_new['guide_seq'] = guide_seq
    row_new['guide_pos_nt'] = guide_pos
    row_new['target_at_guide'] = target_at_guide
    row_new['target_before'] = target_before
    row_new['target_after'] = target_after
    row_new['crrna_block'] = block
    row_new['type'] = row['target_type']
    row_new['guide_target_hamming_dist'] = hd
    row_new['out_logk_median'] = float(row['median'])
    row_new['out_logk_stdev'] = float(row['std']) if row['count'] != '1' else 0
    row_new['out_logk_replicate_count'] = int(row['count'])

    return row_new


def add_replicate_measurements(rows, droplets):
    """Add a column giving replicate information to each row.

    Each technical replicate measurement is a droplet. For each guide-target
    pair, there are 1 or more replicate measurements.

    Args:
        rows: list of dicts, where each element represents a guide-target pair
        droplets: list of dicts, where each element represents a droplet

    Returns:
        rows with an added column 'out_logk_measurements', as given by the
        individual droplets
    """
    # Construct a mapping {(target, crRNA): [replicate measurements]}
    measurements = defaultdict(list)
    
    for droplet in droplets:
        # Note that, in droplets, 'k' is really log(k)
        target = droplet['Target']
        crrna = droplet['crRNA']
        logk = float(droplet['k'])
        measurements[(target, crrna)].append(logk)

    rows_new = []
    for row in rows:
        # Fetch and sort the list of measurements for this guide-target pair
        m = measurements[(row['target'], row['crrna'])]
        m = sorted(m)

        # Check that the summary statistics agree with the measurements
        assert len(m) >= 1
        assert row['out_logk_replicate_count'] == len(m)
        assert math.isclose(row['out_logk_median'], statistics.median(m),
                rel_tol=1e-5)
        if len(m) == 1:
            assert row['out_logk_stdev'] == 0
        else:
            assert math.isclose(row['out_logk_stdev'], statistics.stdev(m),
                    rel_tol=1e-5)

        # Comma-separate the measurements
        m_str = ','.join(str(v) for v in m)

        row['out_logk_measurements'] = m_str
        rows_new += [row]

    return rows_new


def write_output(rows, out_fn):
    """Write a TSV file output, after reformatting.
    """
    cols = ['guide_seq', 'guide_pos_nt', 'target_at_guide', 'target_before',
            'target_after', 'crrna_block', 'type', 'guide_target_hamming_dist',
            'out_logk_median', 'out_logk_stdev', 'out_logk_replicate_count',
            'out_logk_measurements']
    with open(out_fn, 'w') as fw:
        def write_list(l):
            fw.write('\t'.join([str(x) for x in l]) + '\n')
        write_list(cols)
        for row in rows:
            row_list = [row[c] for c in cols]
            write_list(row_list)


def main():
    # Paths to input/output files
    IN = "CCF005_pairs_annotated.csv"
    IN_DROPLETS = "CCF005_pairs_droplets.filtered.csv.gz"
    OUT = "CCF005_pairs_annotated.curated.tsv"

    rows = read_input(IN)
    rows = filter_controls(rows)
    rows = filter_inactive_guides(rows)

    # Reformat rows and check a few things
    new_rows = []
    for row in rows:
        row_new = reformat_row(row)
        new_rows += [row_new]
    rows = new_rows

    # Add droplet-level (replicate) measurements
    droplets = read_droplet_input(IN_DROPLETS)
    rows = add_replicate_measurements(rows, droplets)

    write_output(rows, OUT)


if __name__ == "__main__":
    main()
