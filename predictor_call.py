"""Call the predictor module to make predictions on data.

This is meant for data outside the training/test data.

Currently this is oriented toward predictions for Cas13 regresion
on active data points.
"""

import argparse

import predictor

__author__ = 'Hayden Metsky <hayden@mit.edu>'


def read_inputs(fn):
    """Read guides and targets to test.

    Args:
        fn: path to TSV file; col 1 is target with context, col 2 is
            guide

    Returns:
         list of tuples (target with context, guide)
    """
    inputs = []
    with open(fn) as f:
        for line in f:
            line = line.rstrip()
            ls = line.split('\t')
            assert len(ls) == 2
            target_with_context, guide = ls

            if len(target_with_context) < len(guide):
                raise Exception(("Target with context is shorter than "
                    "guide in input"))

            inputs += [(target_with_context, guide)]
    return inputs


def write_preds(inputs, preds, fn):
    """Write input together with predictions.

    Args:
        inputs: output of read_inputs()
        preds: list of predicted values, one for each input
        fn: path to output TSV file
    """
    with open(fn, 'w') as fw:
        for (target_with_context, guide), pred in zip(inputs, preds):
            row = [target_with_context, guide, pred]
            fw.write('\t'.join(str(x) for x in row) + '\n')


def main(args):
    # Load model and inputs
    model = predictor.load_model_for_cas13_regression_on_active(
            args.model_path)
    inputs = read_inputs(args.input_seqs)

    # Check context_nt matches input data
    for target_with_context, guide in inputs:
        expected_target_len = len(guide) + 2*model.context_nt
        if expected_target_len != len(target_with_context):
            raise Exception(("Amount of context in target does not match "
                "expected amount from model (%d)") % model.context_nt)

    # Make predictions
    preds = predictor.pred_from_nt(model, inputs)
    write_preds(inputs, preds, args.output)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',
            help=("Path from which to load parameters and model weights for "
                  "model"))
    parser.add_argument('input_seqs',
            help=("Path to TSV file containing input sequences. Each row "
                  "corresponds to a guide-target pair to test. First col "
                  "gives the target with context and second col gives the "
                  "guide (guide should match the center)"))
    parser.add_argument('output',
            help=("Path to output TSV; format is the input TSV, with an "
                  "added column giving predicted value"))

    args = parser.parse_args()

    main(args)
