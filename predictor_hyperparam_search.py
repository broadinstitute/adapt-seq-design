"""Search over hyperparameters for the predictor.

The grid search here is similar to scikit-learn's GridSearchCV, and the
random search is similar to its RandomizedSearchCV.
"""

import argparse
import itertools
import random

import parse_data
import predictor

import numpy as np


__author__ = 'Hayden Metsky <hayden@mit.edu>'


def cross_validate(params, x, y, num_splits):
    """Perform k-fold cross-validation.

    This uses parse_data.split() to split data, which uses stratified
    k-fold cross validation.

    Args:
        params: dict of hyperparameters
        x: input data
        y: labels

    Returns:
        list of validation losses, one for each fold
    """
    val_losses = []
    i = 0
    split_iter = parse_data.split(x, y, num_splits=num_splits,
            stratify_by_pos=True)
    for x_train, y_train, x_validate, y_validate in split_iter:
        print('STARTING FOLD {} of {}'.format(i+1, num_splits))

        # Start a new model for this fold
        model = predictor.construct_model(params, x_train.shape)

        # Train and validate this fold
        results = predictor.train_and_validate(model, x_train, y_train,
                x_validate, y_validate, params['max_num_epochs'])
        val_loss, _ = results
        val_losses += [val_loss]

        print(('FINISHED FOLD {} of {}; current mean validation loss is '
            '{}').format(i+1, num_splits, np.mean(val_losses)))
        i += 1
    return val_losses


def hyperparam_grid():
    """Construct grid of hyperparameters.

    Iterates:
        params where each is a dict of possible parameter values (i.e., a
        point in the grid)
    """
    # Map each parameter to a list of possible values
    grid = {
            'conv_num_filters': [5, 15, 25],
            'conv_filter_width': [[1], [2], [1, 2], [1, 2, 3, 4]],
            'pool_window_width': [1, 2, 4],
            'fully_connected_dim': [[20], [40],
                [20, 20], [20, 40], [40, 40],
                [20, 20, 20], [20, 20, 40], [20, 40, 40], [40, 40, 40]],
            'pool_strategy': ['max', 'avg', 'max-and-avg'],
            'locally_connected_width': [None, [1], [2], [1, 2], [1, 2, 3, 4]],
            'locally_connected_dim': [1, 2, 4],
            'dropout_rate': [0, 0.25, 0.50],
            'l2_factor': [0, 0.0001, 0.001, 0.01, 0.1],
            'max_num_epochs': [1000]
    }
    keys_ordered = sorted(grid.keys())
    grid_iter = itertools.product(*(grid[k] for k in keys_ordered))
    for point in grid_iter:
        params = {}
        for i, k in enumerate(keys_ordered):
            params[k] = point[i]
        yield params


def hyperparam_random_dist(num_samples=250):
    """Construct distribution of hyperparameters.

    Args:
        num_samples: number of random points to yield from the space

    Iterates:
        params where each is a dict of possible parameter values (i.e.,
        a point selected randomly from the space)
    """
    # Define distributions
    def constant(x):
        def d():
            return x
        return d
    def uniform_discrete(x):
        def d():
            return random.choice(x)
        return d
    def uniform_int(low=0, high=10):
        def d():
            return np.random.randint(low, high)
        return d
    def uniform_continuous(low=0.0, high=1.0):
        def d():
            return np.random.uniform(low, high)
        return d
    def uniform_nested_dist(num_low, num_high, d_inner):
        # d_inner is a distribution function
        # v is chosen uniformly in [num_low, num_high), and v
        # values are sampled from d_inner and returned
        def d():
            v = np.random.randint(num_low, num_high)
            return [d_inner() for _ in range(v)]
        return d
    def lognormal(mean=0.0, stdev=1.0):
        # mean/stdev describe the log of the returned variable
        def d():
            return np.random.lognormal(mean, stdev)
        return d

    # Map each parameter to a distribution of values
    space = {
             'conv_num_filters': uniform_int(5, 100),
             'conv_filter_width': uniform_discrete([
                 [1], [2], [3], [4],
                 [1, 2], [1, 2, 3], [1, 2, 4], [1, 2, 3, 4]]),
             'pool_window_width': uniform_int(1, 7),
             'fully_connected_dim': uniform_nested_dist(1, 4,
                 uniform_int(10, 50)),
             'pool_strategy': uniform_discrete(['max', 'avg', 'max-and-avg']),
             'locally_connected_width': uniform_discrete([None,
                 [1], [2], [1, 2]]),
             'locally_connected_dim': uniform_int(1, 5),
             'dropout_rate': uniform_continuous(0, 0.75),
             'l2_factor': lognormal(-9.0, 4.0),
             'max_num_epochs': constant(1000)
    }
    for i in range(num_samples):
        params = {}
        for k in space.keys():
            k_dist = space[k]
            params[k] = k_dist()
        yield params


def search_for_hyperparams(x, y, search_type, num_splits=5, loss_out=None):
    """Search for optimal hyperparameters.

    This uses hyperparam_grid() to find the grid of hyperparameters over
    which to search.

    On each point p in the grid, this runs k-fold cross-validation, computes
    the mean validation loss over the folds for p, and then returns the
    p with the smallest loss.

    Args:
        x: input data
        y: labels
        search_type: 'grid' or 'random' search
        num_splits: number of splits of the data to make (i.e., k in k-fold)
        loss_out: if set, an opened file object to write to write the mean
            validation loss for each choice of hyperparameters

    Returns:
        tuple (params, loss) where params is the optimal choice of
        hyperparameters and loss is the mean validation loss at that choice
    """
    if search_type == 'grid':
        params_iter = hyperparam_grid()
    elif search_type == 'random':
        params_iter = hyperparam_random_dist()
    else:
        raise Exception("Unknown search type '%s'" % search_type)

    best_params = None
    best_loss = None
    for params in params_iter:
        # Compute a mean validation loss at this choice of params
        val_losses = cross_validate(params, x, y, num_splits)
        mean_loss = np.mean(val_losses)
        if best_params is None or mean_loss < best_loss:
            # This is the current best choice of params
            best_params = params
            best_loss = mean_loss

        if loss_out is not None:
            loss_out.write(str(params) + '\t' + str(mean_loss) + '\n')
    return (best_params, best_loss)


def nested_cross_validate(x, y, search_type,
        num_outer_splits=5, num_inner_splits=5, loss_out=None):
    """Perform nested cross-validation to validate model and search.

    This is useful to verify the overall model and model fitting approach
    (i.e., the hyperparameter search). The inner cross-validation procedure
    searches for optimal hyperparameters. The outer cross-validation
    procedure sees how well this search generalizes, by performing and
    testing it on different folds of the data. If the selected hyperparameters
    are similar across the outer folds, that's good.

    This is not for selecting a final model.

    Args:
        x: input data
        y: labels
        search_type: 'grid' or 'random' search
        num_outer_splits: number of folds in the outer cross-validation
            procedure
        num_inner_splits: number of folds in the inner cross-validation
            procedures
        loss_out: if set, an opened file object to write to write the mean
            validation loss for each choice of hyperparameters (once per
            each outer fold)

    Returns:
        list x of tuples (params, loss) where params is an optimal choice
        of parameters (a dict) and loss is the loss for that choice on the
        outer validation data; each element of x corresponds to an outer fold
        of the data
    """
    optimal_choices = []
    i = 0
    outer_split_iter = parse_data.split(x, y, num_splits=num_outer_splits,
            stratify_by_pos=True)
    for x_train, y_train, x_validate, y_validate in outer_split_iter:
        print('STARTING OUTER FOLD {} of {}'.format(i+1, num_outer_splits))

        # Search for hyperparameters on this outer fold of the data

        # Start a new model for this fold
        best_params, _ = search_for_hyperparams(x_train, y_train,
                search_type, num_splits=num_inner_splits, loss_out=loss_out)

        # Compute a loss for this choice of parameters as follows:
        #   (1) train the model on all the training data (in the inner
        #       fold loop of search_for_hyperparams(), the model will have
        #       only been trained on a subset of the data (i.e., on k-1
        #       folds))
        #   (2) test the model on the outer validation data (x_validate,
        #       y_validate), effectively treating this outer validation data
        #       as a test set for best_params
        best_model = predictor.construct_model(best_params, x_train.shape)
        results = predictor.train_and_validate(best_model, x_train, y_train,
                x_validate, y_validate, best_params['max_num_epochs'])
        val_loss, _ = results
        optimal_choices += [(best_params, val_loss)]
        print(('FINISHED OUTER FOLD {} of {}; validation loss on this outer '
            'fold is {}').format(i+1, num_outer_splits, val_loss))
        i += 1
    return optimal_choices


def main(args):
    # Seed the predictor, numpy, and python's random module
    predictor.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Read the data
    train_split_frac = 1.0 - args.test_split_frac
    if args.simulate_cas13:
        parser_class = parse_data.Cas13SimulatedData
    else:
        parser_class = parse_data.Doench2016Cas9ActivityParser
    data_parser = parser_class(
            subset=args.subset,
            context_nt=args.context_nt,
            split=(train_split_frac, 0, args.test_split_frac),
            shuffle_seed=args.seed,
            stratify_by_pos=True)
    data_parser.read()
    parse_data._split_parser = data_parser

    x, y = data_parser.train_set()
    x_test, y_test = data_parser.test_set()

    # Print the size of each data set
    print('DATA SIZES - (Train + validation): {}, Test: {}'.format(
        len(x), len(x_test)))

    # Print the fraction of the training data points that are in each class
    classes = set(tuple(yv) for yv in y)
    for c in classes:
        num_c = sum(1 for yv in y if tuple(yv) == c)
        frac_c = float(num_c) / len(y)
        print('Fraction of train data in class {}: {}'.format(
            c, frac_c))

    if args.command == 'hyperparam-search':
        # Search for optimal hyperparameters
        if args.params_mean_val_loss_out_tsv:
            # Use buffering=1 for line buffering (write each line immediately)
            params_mean_val_loss_out_tsv_f = open(
                    args.params_mean_val_loss_out_tsv, 'w', buffering=1)
        else:
            params_mean_val_loss_out_tsv_f = None

        params, loss = search_for_hyperparams(x, y,
                args.search_type,
                num_splits=args.hyperparam_search_cross_val_num_splits,
                loss_out=params_mean_val_loss_out_tsv_f)

        if params_mean_val_loss_out_tsv_f is not None:
            params_mean_val_loss_out_tsv_f.close()

        print('***')
        print('Hyperparameter search results:')
        print('  Best parameters: {}'.format(params))
        print('  Mean validation loss of these parameters: {}'.format(loss))
        print('***')

        # Test a model with these parameters
        # We could retrain on all the train data (the above search will have
        # performed cross-validation and thus only trained on a subset of the
        # data (i.e., k-1 folds)). But we did not perform nested
        # cross-validation with different sized inputs, so we do not know
        # how well the hyperparameter search performs on different training
        # sizes. Similarly we do not know how many epochs to train for (the
        # above search uses early stopping, and the number of epochs that
        # goes for may be less than the number needed on a larger training
        # input size).
        # So we will split x,y into train/validate sets according to the same
        # size used for the search (i.e., according to the number of splits),
        # so we train on the same size used for the hyperparameter search and
        # the validation data can also be used for early stopping during
        # training. Here, do the split on shuffled data.
        print('Testing model with optimal parameters')
        model = predictor.construct_model(params, x.shape)
        split_iter = parse_data.split(x, y,
                args.hyperparam_search_cross_val_num_splits,
                stratify_by_pos=True)
        # Only take the first split of the generator as the train/validation
        # split
        x_train, y_train, x_validate, y_validate = next(split_iter)
        predictor.train_and_validate(model, x_train, y_train,
                x_validate, y_validate, params['max_num_epochs'])

        # Test the model
        print('***')
        # test() prints results
        predictor.test(model, x_test, y_test)
        print('***')
    elif args.command == 'nested-cross-val':
        print('Note: The test data is not used at all in this command.')

        # Perform nested cross-validation
        if args.params_mean_val_loss_out_tsv:
            # Use buffering=1 for line buffering (write each line immediately)
            params_mean_val_loss_out_tsv_f = open(
                    args.params_mean_val_loss_out_tsv, 'w', buffering=1)
        else:
            params_mean_val_loss_out_tsv_f = None

        fold_results = nested_cross_validate(x, y,
                args.search_type,
                num_outer_splits=args.nested_cross_val_outer_num_splits,
                num_inner_splits=args.hyperparam_search_cross_val_num_splits,
                loss_out=params_mean_val_loss_out_tsv_f)

        if params_mean_val_loss_out_tsv_f is not None:
            params_mean_val_loss_out_tsv_f.close()

        # Print results (parameter selection and loss) for each outer fold
        print('***')
        for fold in range(len(fold_results)):
            print('Outer fold {} results:'.format(fold+1))
            params, loss = fold_results[fold]
            print('  Parameters: {}'.format(params))
            print('  Loss of parameters on outer validation data: {}'.format(loss))
    else:
        raise Exception("Unknown command '%s'" % args.command)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate-cas13',
            action='store_true',
            help=("Instead of Cas9 data, use Cas13 data simulated from the "
                  "Cas9 data"))
    parser.add_argument('--subset',
          choices=['guide-mismatch-and-good-pam', 'guide-match'],
          help=("Use a subset of the data. See parse_data module for "
                "descriptions of the subsets. To use all data, do not set."))
    parser.add_argument('--context-nt',
            type=int,
            default=20,
            help=("nt of target sequence context to include alongside each "
                  "guide"))
    parser.add_argument('--test-split-frac',
            type=float,
            default=0.3,
            help=("Fraction of the dataset to use for testing the final "
                  "model"))
    parser.add_argument('--command',
            choices=['hyperparam-search', 'nested-cross-val'],
            default='hyperparam-search',
            help=("'hyperparam-search': search for hyperparameters with "
                  "cross-validation to generate a final model, and test "
                  "this final model on the test set; 'nested-cross-val': "
                  "perform nested cross-validation to evaluate the "
                  "model and hyperparameter search"))
    parser.add_argument('--hyperparam-search-cross-val-num-splits',
            type=int,
            default=5,
            help=("Number of data splits to use (i.e., k in k-fold) for "
                  "hyperparameter search"))
    parser.add_argument('--nested-cross-val-outer-num-splits',
            type=int,
            default=5,
            help=("Number of splits to use in the outer fold for nested "
                  "cross-validation (the inner fold uses "
                  "HYPERPARAM_SEARCH_CROSS_VAL_NUM_SPLITS splits)"))
    parser.add_argument('--search-type',
            choices=['grid', 'random'],
            default='random',
            help=("Type of hyperparameter search ('grid' or 'random')"))
    parser.add_argument('--params-mean-val-loss-out-tsv',
            help=("If set, path to out TSV at which to write the mean "
                  "validation loss (across folds) for each choice of "
                  "hyperparameters; with nested cross-validation, each "
                  "choice of hyperparameters is written for *each* outer "
                  "fold"))
    parser.add_argument('--seed',
            type=int,
            default=1,
            help=("Random seed"))
    args = parser.parse_args()

    main(args)

