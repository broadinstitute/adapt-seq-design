"""Compute regression statistics without measurement error.

The regression outputs include measurement error, which will pull
down correlation statistics.
"""

import argparse
from collections import defaultdict
import gzip

import numpy as np
import scipy.stats


def parse_args():
    """Parse arguments.

    Returns:
        argument namespace
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('regression_results_tsv',
            help=("Path to .tsv.gz file with regression results"))
    args = parser.parse_args()

    return args


def read_results(fn):
    """Read file of results.

    Args:
        fn: path to file with regression results (.tsv.gz)

    Returns:
        list of dict
    """
    dp = []
    with gzip.open(fn, 'rt') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            ls = line.split('\t')
            if i == 0:
                # Header
                header = ls
                continue
            row = {}
            for j, v in enumerate(ls):
                k = header[j]
                if k in ('true_activity', 'predicted_activity', 'crrna_pos'):
                    v = float(v)
                row[k] = v
            dp += [row]
    return dp


def points_with_error(dp):
    """Pull out all data points.

    Args:
        dp: list of dict, each giving information for a row

    Returns:
        tuple (list of true values, list of predicted values)
    """
    true = []
    pred = []
    for p in dp:
        true += [p['true_activity']]
        pred += [p['predicted_activity']]
    return (true, pred)


def points_without_error(dp):
    """Take summary statistic of true values -- i.e., remove error.

    Args:
        dp: list of dict, each giving information for a row

    Returns:
        tuple (list of true values, list of predicted values)
    """
    # Group points by (target, guide) pair
    same = defaultdict(list)
    for p in dp:
        same[(p['target'], p['guide'])].append(p)

    # Check that predicted value is the same in each group
    # (it should be because the input is the same, but allow
    # some numerical tolerance)
    for _, ps in same.items():
        pred_values = [p['predicted_activity'] for p in ps]
        assert np.allclose(pred_values, [pred_values[0]]*len(pred_values))

    # Collapse groups
    true = []
    pred = []
    for _, ps in same.items():
        pred_values = [p['predicted_activity'] for p in ps]
        pred_value = np.mean(pred_values)

        true_values = [p['true_activity'] for p in ps]
        true_value = np.mean(true_values)
        
        true += [true_value]
        pred += [pred_value]

    return (true, pred)


def print_stats(true, pred):
    """Print regression statistics.

    Args:
        true: list of true activity values
        pred: list of predicted activity values
    """
    rho = scipy.stats.spearmanr(true, pred)
    print(rho)

    r = scipy.stats.pearsonr(true, pred)
    print(r)


if __name__ == '__main__':
    args = parse_args()
    dp = read_results(args.regression_results_tsv)

    p_with_error_true, p_with_error_pred = points_with_error(dp)
    print_stats(p_with_error_true, p_with_error_pred)

    p_without_error_true, p_without_error_pred = points_without_error(dp)
    print_stats(p_without_error_true, p_without_error_pred)
