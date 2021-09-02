"""Prepare data from Abudayyeh et al. 2017 for input to ADAPT's predictive model.

When mismatches are present (because the allele is not given), this uses a
random allele for the mismatch. It also hard-codes the corresponding target
sequence for each guide, found via BLAST.
"""

import argparse
import random

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>'


# Target sequence for each guide, including 10 nt context around the
# guide 
GLUC_1_TARGET = 'GAAATCAAAATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCT' # uses MF882921.1 for context
CXCR4_TARGET = 'CCTGGTATTGTCATCCTGTCCTGCTATTGCATTATCATCTCCAAGCTG' # uses NM_001008540.2 for context
GLUC_2_TARGET = 'AAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGT' # uses MF882921.1 for context


rc_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
def reverse_complement(seq):
    """Take reverse complement.
    """
    return ''.join([rc_map.get(b, b) for b in seq[::-1]])


def prepare_knockdown(args):
    with open(args.output_data, 'w') as fout:
        def write_row(row):
            fout.write('\t'.join(str(x) for x in row) + '\n')

        # Write output header
        out_header = ['target_with_context', 'guide', 'number_of_mismatches',
                'abudayyeh_value']
        write_row(out_header)

        with open(args.input_data) as fin:
            expected_in_header = ['guide_name', 'spacer_seq',
                    'spacer_mismatch_pos', 'number_of_mismatches',
                    'mean_normalized_knockdown_level']

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
                guide_name = ls[0]
                spacer_seq = ls[1]
                if ls[2] == 'NA':
                    spacer_mismatch_pos = None
                else:
                    spacer_mismatch_pos = int(ls[2])
                num_mismatches = int(ls[3])
                knockdown = float(ls[4])

                if spacer_mismatch_pos is None:
                    assert num_mismatches == 0
                if num_mismatches == 0:
                    assert spacer_mismatch_pos is None

                # Replace 'U' with 'T' in the spacer
                spacer_seq = spacer_seq.replace('U', 'T')

                # Determine the target sequence
                if guide_name == 'Gluc_1':
                    target = GLUC_1_TARGET
                elif guide_name == 'CXCR4':
                    target = CXCR4_TARGET
                elif guide_name == 'Gluc_2':
                    target = GLUC_2_TARGET
                else:
                    raise Exception(("Unknown guide '%s'") % guide_name)

                # Insert mismatches with random alleles
                if num_mismatches == 0:
                    spacer_mismatched = spacer_seq
                else:
                    if num_mismatches == 1:
                        spacer_mismatch_pos = [spacer_mismatch_pos]
                    elif num_mismatches == 2:
                        spacer_mismatch_pos = [spacer_mismatch_pos, spacer_mismatch_pos + 1]
                    else:
                        raise Exception("Number of mismatches must be 0, 1, or 2")
                    spacer_mismatched = list(spacer_seq)
                    for pos in spacer_mismatch_pos:
                        pos0 = pos - 1
                        curr_allele = spacer_mismatched[pos0]
                        # Sort the list so that, even with a seed, the
                        # random choices are the same (the hashing in the
                        # set could vary the order of choices, unsorted)
                        choices = sorted(list({'A','C','G','T'} -
                            {curr_allele}))
                        mismatch_allele = random.choice(choices)
                        spacer_mismatched[pos0] = mismatch_allele
                    spacer_mismatched = ''.join(spacer_mismatched)

                # Make the guide be the reverse complement of the
                # mismatched spacer
                guide = reverse_complement(spacer_mismatched)

                # For the wildtype, check that guide is in the target
                # And check that it is not when mismatched
                if num_mismatches == 0:
                    assert guide in target
                else:
                    assert guide not in target

                write_row([target, guide, num_mismatches, knockdown])


def main(args):
    # Set the seed
    random.seed(args.seed)

    prepare_knockdown(args)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data',
            help=("Path to TSV of data from Tambe et al. 2018"))
    parser.add_argument('output_data',
            help=("Path to output TSV, prepared for ADAPT's model"))
    parser.add_argument('--seed', type=int,
            default=1,
            help=("Random seed to use when determining nucleotides"))

    args = parser.parse_args()

    main(args)
