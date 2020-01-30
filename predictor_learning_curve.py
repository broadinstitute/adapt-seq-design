"""Compute and plot learning curves of the predictor.
"""

import argparse
import gzip
import math
import os
import pickle
import random

import parse_data
import predictor
import predictor_hyperparam_search

import numpy as np


__author__ = 'Hayden Metsky <hayden@mit.edu>'


def construct_model(params, regression, x):
    """Construct model according to hyperparameter search.

    Note that this reads hyperparameters to construct a model, but does not
    load weights.

    Args:
        params: hyperparameters for model
        regression: if True, perform regression; if False, classification
        x: train data (only needed for shape)

    Returns:..
        cnn.CasCNNWithParallelFilters object
    """
    # Construct the model
    model = predictor.construct_model(params, x.shape,
            regression=regression)

    return model


def compute_learning_curve(x, y, regression, context_nt, num_splits=5,
        num_inner_splits=5, num_sizes=10):
    """Compute a learning curve.

    This splits the data num_splits ways. In each split S, there is a training
    set S_T and a validation (test) set S_V. S_T gets further split so that
    a fraction of it can be used for early stopping during training. Let
    S_T' be what remains of it. Then, this trains the model using
    (n / num_sizes)*|S_T'| training data points for n in [1,...,num_sizes].
    For each number of training data points, it computes a training loss
    and validation loss.

    Note that at the largest sampling, not all training data points will
    be used: some will be reserved for validation/testing, and others
    will be reserved for early stopping during training.

    Args:
        x: input data
        y: labels/output
        regression: if True, perform regression; if False, classification
        context_nt: number of nt on each side of guide to use for context
        num_splits: number of splits of the data to make (i.e., k in k-fold);
            at each size, this yields num_splits different train/validation
            metrics based on the different splits of the data
        num_inner_splits: number of splits to perform for hyperparameter
            search on each outer fold and sampling size
        num_sizes: number of different sizes of the training data to compute
            a train/validation metrics for

    Returns:
        dict
            {fold index:
                {sampling approach ('sample_all' indicating
        sampling across all data points, or 'sample_crrnas' indicating
        sampling from the crRNAs):
                    {sample size:
                        {'train': training metrics,
                         'val': validation metrics}
                    }
                }
            }
    """
    def sample_over_all_data(num, xx, yy):
        # Produce a subsample, of size num, across all data points
        # in xx and yy
        assert len(xx) == len(yy)
        assert num <= len(xx)
        idx = random.sample(list(range(len(xx))), num)
        return xx[idx], yy[idx]

    def sample_over_crrnas(num, xx, yy):
        # Produce a subsample, of num crRNAs, across the different
        # crRNAs in xx and yy; for each crRNA, include all data points
        # This should produce the same number of data points as
        # subsample_over_all_data(num*(number of targets per crRNA), xx, yy)
        # but with fewer crRNAs; the output of subsample_over_all_data() may
        # encompass all crRNAs even with a small fraction of all data points
        # being sampled
        assert len(xx) == len(yy)
        # Determine the crRNA positions of each data point in xx; this
        # effectively serves as an identifier for the crRNA
        xx_with_pos = [(xi, parse_data._split_parser.pos_for_input(xi))
                for xi in xx]
        all_pos = set([pos for xi, pos in xx_with_pos])
        assert num <= len(all_pos)
        crrna_pos_sampled = set(random.sample(list(all_pos), num))
        idx = [i for i, (xi, xi_pos) in enumerate(xx_with_pos)
                if xi_pos in crrna_pos_sampled]
        return xx[idx], yy[idx]

    # Determine the sizes to sample for each sampling approach
    # This should be the minimum number, for each sampling fraction, taken
    # across the folds (the numbers to sample should be very similar
    # across the folds, but might be slightly different; this helps ensure
    # that we sample the same set of sizes for each fold)
    sample_sizes_all = [None]*num_sizes
    sample_sizes_crrnas = [None]*num_sizes
    split_iter = data_parser.split(x, y, num_splits=num_splits,
            stratify_by_pos=True)
    for x_train, y_train, _, _ in split_iter:
        # Reserve some of x_train for early stopping
        train_split_iter = data_parser.split(x_train, y_train, num_splits=4,
                stratify_by_pos=True)
        x_train_for_train, y_train_for_train, _, _ = next(train_split_iter)

        # Use num_sizes different subsamplings of x_train_for_train, in
        # increasing size
        for i in range(num_sizes):
            n = i + 1
            frac_to_sample = (float(n) / num_sizes)

            # Number to sample, sampling from all data points
            num_points_to_sample = round(frac_to_sample * len(x_train_for_train))
            if sample_sizes_all[i] is None or num_points_to_sample < sample_sizes_all[i]:
                sample_sizes_all[i] = num_points_to_sample

            # Number to sample, sampling from crRNAs
            crrna_pos = set([parse_data._split_parser.pos_for_input(xi) for xi in
                x_train_for_train])
            num_crrnas_to_sample = round(frac_to_sample * len(crrna_pos))
            if sample_sizes_crrnas[i] is None or num_crrnas_to_sample < sample_sizes_crrnas[i]:
                sample_sizes_crrnas[i] = num_crrnas_to_sample

    # Construct the learning curve
    learning_curve = {}
    fold = 0
    split_iter = data_parser.split(x, y, num_splits=num_splits,
            stratify_by_pos=True)
    for x_train, y_train, x_validate, y_validate in split_iter:
        print('STARTING SPLIT {} of {}'.format(fold+1, num_splits))
        print('  Training up to n={}, validating on n={}'.format(len(x_train),
            len(x_validate)))

        learning_curve[fold] = {'sample_all': {}, 'sample_crrnas': {}}

        # Use x_validate, y_validate for validation of the model on
        # this fold

        # Use num_sizes different subsamplings of x_train_for_train, in
        # increasing size
        for size_i in range(num_sizes):
            print('Training on sampling {} of {} of the training set'.format(
                size_i+1, num_sizes))

            if regression:
                # Print what the MSE would be if only predicting the mean of
                # the training data
                print('  MSE on train data if predicting mean of train data:',
                        np.mean(np.square(np.mean(y_train) - y_train.ravel())))
                print('  MSE on validation data if predicting mean of train data:',
                        np.mean(np.square(np.mean(y_train) - y_validate.ravel())))

            # Sample from all data points, train, and then run
            # predictor.test() on the validation data for this fold to
            # get the validation results
            size = sample_sizes_all[size_i]
            x_train_sample, y_train_sample = sample_over_all_data(
                    size, x_train_for_train, y_train_for_train)
            # Perform hyperparameter search; do this because hyperparameters
            # may vary based on sampling size
            hyperparams, _, _ = predictor_hyperparam_search.search_for_hyperparams(
                    x_train_sample, y_train_sample, 'random', regression,
                    context_nt, num_splits=num_inner_splits,
                    num_random_samples=100)
            # Split the training data ({x,y}_train_sample) *again* to get a
            # separate validation set ({x,y}_train_sample_for_es)) to
            # pass to predictor.train_and_validate(), which it will use for early
            # stopping
            # This way, we do not use the same validation set both for early
            # stopping and for measuring the performance of the model (otherwise
            # our 'training' of the model, which includes deciding when to
            # stop, would have seen and factored in the data used for evaluating
            # it on the fold)
            # Only use one of the splits, with 3/4 being used to train the model
            # and 1/4 for early stopping
            train_split_iter = data_parser.split(x_train_sample, y_train_sample,
                    num_splits=4, stratify_by_pos=True)
            x_train_sample_for_train, y_train_sample_for_train, x_train_sample_for_es, y_train_sample_for_es = next(train_split_iter)
            # Construct model
            model = construct_model(hyperparams, regression,
                    x_train_sample_for_train)
            train_results, _ = predictor.train_and_validate(
                    model, x_train_sample_for_train, y_train_sample_for_train,
                    x_train_sample_for_es, y_train_sample_for_es,
                    hyperparams['max_num_epochs'], data_parser)
            val_results = predictor.test(model, x_validate, y_validate,
                    data_parser)
            learning_curve[fold]['sample_all'][size] = {'train': train_results,
                    'val': val_results}

            # Sample from the crRNAs, train, and then run
            # predictor.test() on the validation data for this fold to
            # get the validation results
            # Above, size is the total number of data points; for this, it is
            # the number of crRNAs
            size = sample_sizes_crrnas[size_i]
            x_train_sample, y_train_sample = sample_over_crrnas(
                    size, x_train_for_train, y_train_for_train)
            # Perform hyperparameter search; do this because hyperparameters
            # may vary based on sampling size
            hyperparams, _, _ = predictor_hyperparam_search.search_for_hyperparams(
                    x_train_sample, y_train_sample, 'random', regression,
                    context_nt, num_splits=num_inner_splits,
                    num_random_samples=100)
            # Split the training data ({x,y}_train_sample), as explained above
            train_split_iter = data_parser.split(x_train_sample, y_train_sample,
                    num_splits=4, stratify_by_pos=True)
            x_train_sample_for_train, y_train_sample_for_train, x_train_sample_for_es, y_train_sample_for_es = next(train_split_iter)
            # Construct model
            model = construct_model(hyperparams, regression,
                    x_train_sample_for_train)
            train_results, _ = predictor.train_and_validate(
                    model, x_train_sample_for_train, y_train_sample_for_train,
                    x_train_sample_for_es, y_train_sample_for_es,
                    hyperparams['max_num_epochs'], data_parser)
            val_results = predictor.test(model, x_validate, y_validate,
                    data_parser)
            learning_curve[fold]['sample_crrnas'][size] = {'train': train_results,
                    'val': val_results}

        print(('FINISHED SPLIT {} of {}').format(fold+1, num_splits))
        fold += 1

    return learning_curve


def main(args):
    # Seed the predictor
    predictor.set_seed(args.seed)

    # Make the data parser be module wide
    global data_parser

    # Read data
    # Do not have a validation set; read the test set if desired, but
    # don't use it for anything
    split_frac = (1.0 - args.test_split_frac, 0, args.test_split_frac)
    data_parser = predictor.read_data(args, split_frac)
    x, y = data_parser.train_set()
    x_test, y_test = data_parser.test_set()
    if args.dataset == 'cas13':
        if args.cas13_classify:
            regression = False
        else:
            regression = True

    # Compute the learning curve
    learning_curve = compute_learning_curve(x, y,
            regression, args.context_nt,
            num_splits=args.num_splits, num_sizes=args.num_sizes)

    # Write the learning curve to a TSV file
    with gzip.open(args.write_tsv, 'wt') as fw:
        def write_row(row):
            fw.write('\t'.join(str(c) for c in row) + '\n')

        # Write header
        header = ['fold', 'sampling_approach', 'size', 'dataset', 'metric', 'value']
        write_row(header)

        for fold in learning_curve.keys():
            for sampling_approach in learning_curve[fold].keys():
                for size in learning_curve[fold][sampling_approach].keys():
                    for dataset in learning_curve[fold][sampling_approach][size].keys():
                        for metric in learning_curve[fold][sampling_approach][size][dataset].keys():
                            value = learning_curve[fold][sampling_approach][size][dataset][metric]
                            write_row([fold, sampling_approach, size,
                                dataset, metric, value])


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-tsv',
            required=True,
            help=("Path to .tsv.gz file to which to write learning curve "
                  "results"))
    parser.add_argument('--dataset',
            choices=['cas13'],
            default='cas13',
            help=("Dataset to use."))
    parser.add_argument('--cas13-subset',
            choices=['exp', 'pos', 'neg', 'exp-and-pos'],
            help=("Use a subset of the Cas13 data. See parse_data module "
                  "for descriptions of the subsets. To use all data, do not "
                  "set."))
    parser.add_argument('--cas13-classify',
            action='store_true',
            help=("If set, only classify Cas13 activity into inactive/active"))
    parser.add_argument('--cas13-regress-on-all',
            action='store_true',
            help=("If set, perform regression for Cas13 data on all data "
                  "(this can be reduced using --cas13-subset)"))
    parser.add_argument('--cas13-regress-only-on-active',
            action='store_true',
            help=("If set, perform regression for Cas13 data only on the "
                  "active class"))
    parser.add_argument('--cas13-normalize-crrna-activity',
            action='store_true',
            help=("If set, normalize the activity of each crRNA (guide) "
                  "across its targets to have mean 0 and stdev 1; this means "
                  "prediction is performed based on target differences (e.g., "
                  "mismatches) rather than inherent sequence of the crRNA"))
    parser.add_argument('--context-nt',
            type=int,
            default=10,
            help=("nt of target sequence context to include alongside each "
                  "guide"))
    parser.add_argument('--num-splits',
            type=int,
            default=5,
            help=("Number of splits of the data to make (i.e., k in k-fold); "
                  "at each size, this yields num_splits different train/"
                  "validation metrics based on different splits of the data"))
    parser.add_argument('--num-sizes',
            type=int,
            default=10,
            help=("Number of different sizes of the training data to compute "
                  "train/validation metrics for"))
    parser.add_argument('--test-split-frac',
            type=float,
            default=0,
            help=("Fraction of the dataset to leave out (completely ignore) "
                  "from this analysis"))
    parser.add_argument('--seed',
            type=int,
            default=1,
            help=("Random seed"))

    args = parser.parse_args()

    # Print the arguments provided
    print(args)

    main(args)
