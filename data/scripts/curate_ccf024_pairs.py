#!/bin/python
#
# This data is from Nick Haradhvala's synthetic library of Cas13 crRNAs
# tested using CARMEN/Cas13.
# This is for the second experiment we did (Fall, 2019): CCF024.


# Amount of target sequence context to extract for each guide
CONTEXT_NT = 20


from collections import defaultdict
import csv
import gzip
import math
import statistics

# Because this is highly similar to parsing CCF005, use some of those
# functions when possible
import curate_ccf005_pairs
read_input = curate_ccf005_pairs.read_input
read_droplet_input = curate_ccf005_pairs.read_droplet_input
filter_controls = curate_ccf005_pairs.filter_controls
filter_inactive_guides = curate_ccf005_pairs.filter_inactive_guides
hamming_dist = curate_ccf005_pairs.hamming_dist
reverse_complement = curate_ccf005_pairs.reverse_complement
add_replicate_measurements = curate_ccf005_pairs.add_replicate_measurements
write_output = curate_ccf005_pairs.write_output


def read_targets(in_fn):
    """Read target sequences.

    Args:
        in_fn: path to tsv with target sequences

    Returns:
        dict {target name: sequence}
    """
    target_seqs = {}
    with open(in_fn) as f:
        for i, line in enumerate(f):
            ls = line.rstrip().split('\t')
            name, seq = ls
            
            if i == 0:
                # Check header
                assert name == "name"
                assert seq == "seq"
                continue

            target_seqs[name] = seq

    return target_seqs


def filter_blank_targets(rows):
    """Filter data points against blank targets.

    Returns:
        rows with blank targets removed
    """
    rows_filtered = []
    for row in rows:
        if row['Target'] == 'blank':
            # Verify this is inactive
            assert float(row['median']) == -4
        else:
            # Keep it
            rows_filtered += [row]
    return rows_filtered


def add_crrna_info(rows, in_fn):
    """Read crRNA info -- this info was taken from CCF005 -- and add it to
    rows.

    Args:
        rows: list of dicts representing each input row
        in_fn: path to tsv with info

    Returns:
        rows with added columns
    """
    # Read file with info on crRNAs
    col_names = {}
    crrna_info = {}
    with open(in_fn) as f:
        for i, line in enumerate(f):
            ls = line.rstrip().split('\t')

            if i == 0:
                # Read header
                for j, col_name in enumerate(ls):
                    col_names[j] = col_name
            else:
                # Read line
                row = {}
                for j, val in enumerate(ls):
                    row[col_names[j]] = val
                name = row['crRNA']
                del row['crRNA']
                crrna_info[name] = row

    # Add info to rows
    for row in rows:
        crrna = row['crRNA']
        for k, v in crrna_info[crrna].items():
            row[k] = v
    return rows


def reformat_row(row, target_seqs):
    """Verify and summarize contents of a row.

    Args:
        row: dict representing a row (guide-target pair)
        target_seqs: dict {target name: target sequence}

    Returns:
        row with new columns, removed columns, and renamed columns
    """
    # Read the guide sequence and target sequence (just where the guide is)
    guide_seq = row['guide_seq']
    target_at_guide = row['target_seq']

    # Check that guide_seq is the reverse complement of spacer_seq
    # for this crRNA
    crrna_spacer_seq = row['spacer_seq']
    spacer_rc = reverse_complement(crrna_spacer_seq.replace('u', 't')).upper()
    assert spacer_rc == guide_seq

    # Get the full target sequence
    full_target_seq = target_seqs[row['Target']]

    # 'target_type' is not in this csv; set it based on target name
    if 'Target_WT' in row['Target']:
        target_type = 'pos'
    elif 'Target_neg' in row['Target']:
        target_type = 'neg'
    elif 'Target_MMM' in row['Target']:
        target_type = 'exp'
    else:
        raise Exception("Unknown target type")

    # Find guide_pos by searching for target_at_guide in full_target_seq
    # (note that in CCF005, pos is 0-based, so it's good to be
    # 0-based here too)
    assert full_target_seq.count(target_at_guide) == 1
    guide_pos = full_target_seq.index(target_at_guide)
    assert guide_pos >= 0

    # Check that the Hamming distance to the target is reasonable
    # and matches the given 'n_mismatch' value
    hd = hamming_dist(guide_seq, target_at_guide)
    assert hd == int(row['n_mismatch'])
    if target_type == 'exp':
        # 1 mismatch (if mismatch is within guide)
        # 0 to ~14 mismatches
        assert 0 <= hd <= 14
    elif target_type == 'pos':
        # matching
        assert hd == 0
    elif target_type == 'neg':
        # not matching
        assert hd > 14

    # Verify guide_pos with crRNA info file
    assert guide_pos == int(row['pos'])

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

    # Pull out the PFS
    pfs = target_after[:2]
    assert len(pfs) == 2

    # Get the block
    block = int(float(row['block']))

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
    row_new['type'] = target_type
    row_new['guide_target_hamming_dist'] = hd
    row_new['out_logk_median'] = float(row['median'])
    row_new['out_logk_stdev'] = float(row['std']) if row['count'] != '1' else 0
    row_new['out_logk_replicate_count'] = int(row['count'])

    return row_new


def main():
    # Paths to input/output files
    IN = "CCF024_pairs_annotated.csv"
    IN_DROPLETS = "CCF024_pairs_droplets.filtered.csv.gz"
    IN_TARGETS = "CCF024_targets.tsv"
    IN_CRRNA_INFO = "CCF024_crrnas.tsv"
    OUT = "CCF-curated/CCF024_pairs_annotated.curated.tsv"

    rows = read_input(IN)

    # This file does not have as much info for each crRNA as CCF005;
    # add columns
    rows = add_crrna_info(rows, IN_CRRNA_INFO)

    rows = filter_controls(rows)
    rows = filter_inactive_guides(rows)
    rows = filter_blank_targets(rows)

    targets = read_targets(IN_TARGETS)

    # Reformat rows and check a few things
    new_rows = []
    for row in rows:
        row_new = reformat_row(row, targets)
        new_rows += [row_new]
    rows = new_rows

    # Add droplet-level (replicate) measurements
    droplets = read_droplet_input(IN_DROPLETS)
    rows = add_replicate_measurements(rows, droplets)

    write_output(rows, OUT)


if __name__ == "__main__":
    main()
