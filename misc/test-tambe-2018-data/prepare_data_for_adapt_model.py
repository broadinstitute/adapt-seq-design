"""Prepare data from Tambe et al. 2018 for input to ADAPT's predictive model.

Notably, that paper uses 20-nt spacers, whereas ADAPT's model requires 28-nt
spacer (technically, protospacers) as input. This script pads the guide
on the 3' end of the spacer (5' end of the protospacer) to reach 28 nt.
Likewise, the flanking sequence in the target that the paper uses is too
short, so this script pads the target sequences on the ends.
"""

import argparse
import gzip
import random

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>'


# Set lengths of input sequences for ADAPT's model
ADAPT_MODEL_GUIDE_LEN = 28
ADAPT_MODEL_TARGET_FLANKING5_LEN = 10
ADAPT_MODEL_TARGET_FLANKING3_LEN = 10

# Use the same padding sequences for all guide-target pairs
pad_seqs = None

# Tambe et al. crRNA sequences -- note that these are the
# reverse complement of the spacer sequence (i.e., same
# sequence as the RNA target)
TAMBE_CRRNA_GUIDE_X = 'CGCCUGAACCACCAGGCUAU'
TAMBE_CRRNA_GUIDE_Y = 'CCGCACUAUCGGAAGUUCAC'


rc_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
def reverse_complement(seq):
    """Take reverse complement.
    """
    return ''.join([rc_map.get(b, b) for b in seq[::-1]])


def make_pad_seqs(guide_pad_len, target_flanking5_pad_len,
        target_flanking3_pad_len):
    """Make padding sequences.

    This ensures that each guide-target pair has the same padding
    sequence.
    """
    def rand_seq(seq_len):
        return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(seq_len))

    global pad_seqs
    pad_seqs = {}
    pad_seqs['guide'] = rand_seq(guide_pad_len)
    pad_seqs['target_flanking5'] = rand_seq(target_flanking5_pad_len)
    pad_seqs['target_flanking3'] = rand_seq(target_flanking3_pad_len)


def prepare_for_adapt_input(guide, target, guide_pos):
    """Prepare guide and target from Tambe et al. for input to ADAPT's model.

    Note that guide should be the protospacer (i.e., (near-)identical to the
    target), so we should not have to take its reverse complement (which we
    would have to do if it were the spacer).

    Args:
        guide: protospacer sequence
        target: target sequence
        guide_pos: start position (0-based) of the guide in the target

    Returns:
        tuple (guide, target) for ADAPT's input
    """
    # Assume that guide and target are shorter than what ADAPT requires
    assert len(guide) < ADAPT_MODEL_GUIDE_LEN
    assert len(target) < ADAPT_MODEL_TARGET_FLANKING5_LEN + ADAPT_MODEL_GUIDE_LEN + ADAPT_MODEL_TARGET_FLANKING3_LEN

    # Make target uppercase (because of the 'gg' on the 5' end)
    target = target.upper()

    # Replace 'U' with 'T'
    guide = guide.replace('U', 'T')
    target = target.replace('U', 'T')

    # Pull out the parts of the target sequence
    target_flanking5 = target[:guide_pos]
    target_at_guide = target[guide_pos:(guide_pos + len(guide))]
    target_flanking3 = target[(guide_pos + len(guide)):]
    assert len(target_flanking5) + len(target_at_guide) + len(target_flanking3) == len(target)

    # If target_flanking5 is too long, trim it
    if len(target_flanking5) > ADAPT_MODEL_TARGET_FLANKING5_LEN:
        trim_nt = len(target_flanking5) - ADAPT_MODEL_TARGET_FLANKING5_LEN
        target_flanking5 = target_flanking5[trim_nt:]

    # Determine how much to pad each sequence
    guide_pad_len = ADAPT_MODEL_GUIDE_LEN - len(guide)
    target_flanking5_pad_len = ADAPT_MODEL_TARGET_FLANKING5_LEN - len(target_flanking5)
    target_flanking3_pad_len = ADAPT_MODEL_TARGET_FLANKING3_LEN - len(target_flanking3)
    if target_flanking5_pad_len < 0:
        target_flanking5_pad_len = 0
    if target_flanking3_pad_len < 0:
        target_flanking3_pad_len = 0

    if pad_seqs is None:
        # Make padding sequences
        make_pad_seqs(guide_pad_len, target_flanking5_pad_len, target_flanking3_pad_len)

    # Check lengths of padding sequences are as needed, in case they were previously
    # made from a different guide-target pair
    assert len(pad_seqs['guide']) == guide_pad_len
    assert len(pad_seqs['target_flanking5']) == target_flanking5_pad_len
    assert len(pad_seqs['target_flanking3']) == target_flanking3_pad_len

    # Pad nucleotides on the 3' end of the spacer, which is the 5' end of
    # the protospacer
    guide_padded = pad_seqs['guide'] + guide

    # Pad nucleotides to the target ends
    # Also, since we're adding to the guide, add
    # the same padding sequence to the target where the guide will bind
    target_padded = (pad_seqs['target_flanking5'] + target_flanking5 +
        pad_seqs['guide'] + target_at_guide + target_flanking3 + pad_seqs['target_flanking3'])

    return (guide_padded, target_padded)


def prepare_cleavage_rates(args):
    with open(args.output_data, 'w') as fout:
        def write_row(row):
            fout.write('\t'.join(str(x) for x in row) + '\n')

        # Write output header
        out_header = ['target_with_context', 'guide', 'number_of_mismatches',
                'tambe_value']
        write_row(out_header)

        with open(args.input_data) as fin:
            expected_in_header = ['spacer_rc', 'target',
                    'mean_normalized_cleavage_rate_relative_to_no_mismatch',
                    'number_of_mismatches']
            for i, line in enumerate(fin):
                line = line.rstrip()
                if i == 0:
                    # Check that the header is as expected
                    assert line.split('\t') == expected_in_header
                    continue
                if len(line) == 0:
                    continue

                # Read input row
                guide, target, tambe_val, num_mismatches = line.split('\t')
                if tambe_val == "ND":
                    # Replace 'ND' with 0 (the lowest value possible in their dataset)
                    tambe_val = 0
                else:
                    tambe_val = float(tambe_val)
                num_mismatches = int(num_mismatches)

                # Prepare guide-target sequences for ADAPT's input
                guide_prep, target_prep = prepare_for_adapt_input(guide,
                        target, args.guide_pos)

                write_row([target_prep, guide_prep, num_mismatches, tambe_val])


def prepare_binding(args):
    # Sequence used on the 5' end of the target's reverse complement (so 3' end
    # of what we will label the target)
    # These sequences come from a Word doc sent by Mitch
    target_rc_5_X = 'GCA'
    target_rc_3_X = 'CATGCTTAAGATCGGA'
    target_rc_5_Y = 'CTG'
    target_rc_3_Y = 'TTGAATCCAGATCGGA'

    target_5_X = reverse_complement(target_rc_3_X)
    target_3_X = reverse_complement(target_rc_5_X)
    target_5_Y = reverse_complement(target_rc_3_Y)
    target_3_Y = reverse_complement(target_rc_5_Y)

    with gzip.open(args.output_data, 'wt') as fout:
        def write_row(row):
            fout.write('\t'.join(str(x) for x in row) + '\n')

        # Write output header
        out_header = ['target_with_context', 'guide', 'number_of_mismatches',
                'tambe_value']
        write_row(out_header)

        with gzip.open(args.input_data, 'rt') as fin:
            expected_in_header = ['crrna', 'orig_crrna_name',
                    'target_seq_rc', 'mismatch_pos', 'num_mismatches',
                    'fc', 'fc_normalized', 'fc_normalized_without_rescale']
            for i, line in enumerate(fin):
                line = line.rstrip()
                if i == 0:
                    # Check that the header is as expected
                    assert line.split('\t') == expected_in_header
                    continue
                if len(line) == 0:
                    continue

                # Read input row
                ls = line.split('\t')
                crrna = ls[0]
                target_rc = ls[2]
                num_mismatches = int(ls[4])
                fc = ls[5]
                fc_normalized = ls[6]
                fc_normalized_without_rescale = ls[7]
                if fc == 'nan':
                    # Skip this data point; it is likely that the control
                    # was too low to measure (in the denominator of the
                    # fold-change)
                    continue

                fc = float(fc)
                fc_normalized = float(fc_normalized)
                fc_normalized_without_rescale = float(fc_normalized_without_rescale)

                if crrna == 'RNA_X':
                    guide = TAMBE_CRRNA_GUIDE_X
                elif crrna == 'RNA_Y':
                    guide = TAMBE_CRRNA_GUIDE_Y

                # Replace 'U' with 'T'
                guide = guide.replace('U', 'T')
                target_rc = target_rc.replace('U', 'T')

                # Take reverse complement of target
                target = reverse_complement(target_rc)

                # Add on additional flanking sequence that was used in the
                # library but not in the analysis in Tambe et al
                if crrna == 'RNA_X':
                    target = target_5_X + target + target_3_X
                elif crrna == 'RNA_Y':
                    target = target_5_Y + target + target_3_Y

                tambe_val = fc_normalized

                # Prepare guide-target sequences for ADAPT's input
                guide_prep, target_prep = prepare_for_adapt_input(guide,
                        target, args.guide_pos)

                write_row([target_prep, guide_prep, num_mismatches, tambe_val])

def main(args):
    # Set the seed
    random.seed(args.seed)

    if args.data_type == 'cleavage-rates':
        prepare_cleavage_rates(args)
    elif args.data_type == 'binding':
        prepare_binding(args)
    else:
        raise Exception("Unknown data type '%s'" % args.data_type)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_type',
            choices=['cleavage-rates', 'binding'],
            help=("Type of data to prepare: 'cleavage-rates' or 'binding'"))
    parser.add_argument('input_data',
            help=("Path to TSV of data from Tambe et al. 2018"))
    parser.add_argument('guide_pos', type=int,
            help=("Start position (0-based) of the guide in the target sequence"))
    parser.add_argument('output_data',
            help=("Path to output TSV, prepared for ADAPT's model"))
    parser.add_argument('--seed', type=int,
            default=1,
            help=("Random seed to use when determining nucleotides"))

    args = parser.parse_args()

    main(args)

