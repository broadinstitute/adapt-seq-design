#!/bin/python
#
# This data is from Nick Haradhvala's synthetic library of Cas13 crRNAs
# tested using CARMEN/Cas13.


# Paths to input/output files
IN = "CCF005_pairs_annotated.csv"
OUT = "CCF005_pairs_annotated.curated.tsv"

# Amount of target sequence context to extract for each guide
CONTEXT_NT = 20


def read_input():
    """Read input csv file.

    Returns:
        list of dicts where each element corresponds to a row
    """
    col_names = {}
    lines = []
    with open(IN) as f:
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
    row_new['out_replicate_count'] = int(row['count'])

    return row_new


def write_output(rows):
    """Write a TSV file output, after reformatting.
    """
    cols = ['guide_seq', 'guide_pos_nt', 'target_at_guide', 'target_before',
            'target_after', 'crrna_block', 'type', 'guide_target_hamming_dist',
            'out_logk_median', 'out_logk_stdev', 'out_replicate_count']
    with open(OUT, 'w') as fw:
        def write_list(l):
            fw.write('\t'.join([str(x) for x in l]) + '\n')
        write_list(cols)
        for row in rows:
            row_list = [row[c] for c in cols]
            write_list(row_list)


def main():
    rows = read_input()
    rows = filter_controls(rows)

    new_rows = []
    for row in rows:
        row_new = reformat_row(row)
        new_rows += [row_new]

    write_output(new_rows)


if __name__ == "__main__":
    main()
