"""Search over hyperparameters for the predictor.

The grid search here is similar to scikit-learn's GridSearchCV, and the
random search is similar to its RandomizedSearchCV.
"""

import argparse
from collections import defaultdict
import itertools
import os
import pickle
import random

import parse_data
import predictor

import numpy as np
import scipy.stats


__author__ = 'Hayden Metsky <hayden@mit.edu>'



_regression_losses = ['mse', '1_minus_rho']
_default_regression_loss = '1_minus_rho'
_classification_losses = ['bce', '1_minus_auc-roc']
_default_classification_loss = '1_minus_auc-roc'
def determine_val_loss(results):
    """Determine loss values on validation data.

    Note that using the direct loss (i.e., what is optimized during training)
    means also including regularization component of the loss; we probably
    do not want to include this as it could lead us to choose models that
    underfit (e.g., small weights) or overfit (e.g., too small of a
    l2-factor hyperparameter). This metric is in results['loss'], and
    this function leaves it out. For a very similar metric (i.e., just the
    error part of the loss term), we can use binary cross-entropy
    (results['bce']) for classification or mean squared error (results['mse'])
    for regression. Similarly, the model itself can directly affect the
    loss value in results['loss'], which we do not want (e.g., higher
    dimensions of fully connected layers lead to more weights, which can
    increase the regularization term of the loss; similarly, *not* having
    a locally connected layer may lead to more weights (as the dimension of
    the fully connected layers that follow would increase), which would
    increase the loss).

    Args:
        results: dict returned by predictor.train_and_validate()

    Returns:
        (default loss value to use for ranking, dict of loss values for
        different metrics)
    """
    if 'bce' in results:
        # Classification results
        assert 'auc-roc' in results
        assert 'mse' not in results
        assert 'r-spearman' not in results

        losses = {'bce': results['bce'],
                '1_minus_auc-roc': 1.0 - results['auc-roc']}
        default_loss = losses[_default_classification_loss]
    elif 'mse' in results:
        # Regression results
        assert 'r-spearman' in results
        assert 'bce' not in results
        assert 'auc-roc' not in results

        losses = {'mse': results['mse'],
                '1_minus_rho': 1.0 - results['r-spearman']}
        default_loss = losses[_default_regression_loss]
    else:
        raise Exception("Unknown whether classification or regression")

    return default_loss, losses


def cross_validate(params, x, y, num_splits, regression):
    """Perform k-fold cross-validation.

    This uses parse_data.split() to split data, which uses stratified
    k-fold cross validation.

    Args:
        params: dict of hyperparameters
        x: input data
        y: output (labels, if classification)
        num_splits: number of folds
        regression: if True, perform regression; if False, classification

    Returns:
        tuple ([default validation loss for each fold], dict {metric: [list of
        validation losses for metric, for each fold]})
    """
    val_losses_default = []
    val_losses_different_metrics = defaultdict(list)
    i = 0
    split_iter = parse_data.split(x, y, num_splits=num_splits,
            stratify_by_pos=True)
    for x_train, y_train, x_validate, y_validate in split_iter:
        print('STARTING FOLD {} of {}'.format(i+1, num_splits))
        print('  Training on n={}, validating on n={}'.format(len(x_train),
            len(x_validate)))

        # Use x_validate, y_validate for validation of the model on
        # this fold

        # Split the training data (x_train, y_train) *again* to get a
        # separate validation set (x_validate_for_es, y_validate_for_es) to
        # pass to predictor.train_and_validate(), which it will use for early
        # stopping
        # This way, we do not use the same validation set both for early
        # stopping and for measuring the performance of the model (otherwise
        # our 'training' of the model, which includes deciding when to
        # stop, would have seen and factored in the data used for evaluating
        # it on the fold)
        # Only use one of the splits, with 3/4 being used to train the model
        # and 1/4 for early stopping
        train_split_iter = parse_data.split(x_train, y_train, num_splits=4,
                stratify_by_pos=True)
        x_train_for_train, y_train_for_train, x_validate_for_es, y_validate_for_es = next(train_split_iter)

        # Start a new model for this fold
        model = predictor.construct_model(params, x_train_for_train.shape,
                regression)

        # Train this fold
        predictor.train_and_validate(model, x_train_for_train,
                y_train_for_train, x_validate_for_es, y_validate_for_es,
                params['max_num_epochs'])

        # Run predictor.test() on the validation data for this fold, to
        # get its validation results
        results = predictor.test(model, x_validate, y_validate)
        a, b = determine_val_loss(results)
        val_losses_default += [a]
        for k in b.keys():
            val_losses_different_metrics[k].append(b[k])

        print(('FINISHED FOLD {} of {}').format(i+1, num_splits))
        i += 1
    return val_losses_default, val_losses_different_metrics


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


def hyperparam_random_dist(num_samples):
    """Construct distribution of hyperparameters.

    Args:
        num_samples: number of random points to yield from the space
            (if None, use default)

    Iterates:
        params where each is a dict of possible parameter values (i.e.,
        a point selected randomly from the space)
    """
    if num_samples is None:
        # Set default number of samples
        num_samples = 1000

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
             'conv_num_filters': uniform_int(5, 25),
             'conv_filter_width': uniform_discrete([
                 [1], [2], [3], [4],
                 [1, 2], [1, 2, 3], [1, 2, 4], [1, 2, 3, 4]]),
             'pool_window_width': uniform_int(1, 6),
             'fully_connected_dim': uniform_nested_dist(1, 3,
                 uniform_int(10, 50)),
             'pool_strategy': uniform_discrete(['max', 'avg', 'max-and-avg']),
             'locally_connected_width': uniform_discrete([None,
                 [1], [2], [3], [1, 2], [1, 2, 3]]),
             'locally_connected_dim': uniform_int(1, 6),
             'dropout_rate': uniform_continuous(0, 0.5),
             'l2_factor': lognormal(-12.0, 5.0),
             'max_num_epochs': constant(1000)
    }
    for i in range(num_samples):
        params = {}
        for k in space.keys():
            k_dist = space[k]
            params[k] = k_dist()
        yield params


def search_for_hyperparams(x, y, search_type, regression,
        num_splits=5, loss_out=None, num_random_samples=None,
        max_sem=None):
    """Search for optimal hyperparameters.

    This uses hyperparam_grid() to find the grid of hyperparameters over
    which to search.

    On each point p in the grid, this runs k-fold cross-validation, computes
    the mean validation loss over the folds for p, and then returns the
    p with the smallest loss. As a single loss value, this uses the default
    loss defined above determine_val_loss().

    Args:
        x: input data
        y: labels
        search_type: 'grid' or 'random' search
        regression: if True, perform regression; if False, classification
        num_splits: number of splits of the data to make (i.e., k in k-fold)
        loss_out: if set, an opened file object to write to write the
            validation loss values for each choice of hyperparameters
        num_random_samples: number of random samples to use, if search_type
            is 'random' (if None, use default)
        max_sem: maximum standard error of the mean (SEM) on the loss to allow
            the hyperparameters with that loss to be chosen as the 'best'
            choice of hyperparameters (if None, no limit)

    Returns:
        tuple (params, loss, sem) where params is the optimal choice of
        hyperparameters and loss is the mean validation loss at that choice
        and sem of the standard error of the loss
    """
    if num_random_samples is not None and search_type != 'random':
        raise Exception("Cannot set num_random_samples without random search")

    if search_type == 'grid':
        params_iter = hyperparam_grid()
    elif search_type == 'random':
        params_iter = hyperparam_random_dist(num_random_samples)
    else:
        raise Exception("Unknown search type '%s'" % search_type)

    best_params = None
    best_loss = None
    best_loss_sem = None
    for params in params_iter:
        # Compute a mean validation loss at this choice of params
        val_losses_default, val_losses_different_metrics = cross_validate(
                params, x, y, num_splits, regression)

        mean_loss = np.mean(val_losses_default)
        sem_loss = scipy.stats.sem(val_losses_default)

        # Decide whether to update the current best choice of hyperparameters
        if mean_loss is np.nan:
            # mean_loss can be nan, e.g., for Spearman's rho if all predicted
            # values are the same
            # Do not update it in this case
            update_choice = False
        else:
            if max_sem is not None and sem_loss > max_sem:
                # The SEM of the mean loss is too high; do not allow this
                # choice to be the best choice
                update_choice = False
            else:
                if best_params is None:
                    update_choice = True
                else:
                    # Update the best choice iff mean_loss < best_loss
                    update_choice = (mean_loss < best_loss)

        if update_choice:
            # This is the current best choice of params
            best_params = params
            best_loss = mean_loss
            best_loss_sem = sem_loss

        if loss_out is not None:
            row = [params]
            metrics_ordered = _regression_losses if regression else _classification_losses
            for metric in metrics_ordered:
                losses = val_losses_different_metrics[metric]
                row += [np.mean(losses)]
                row += [scipy.stats.sem(losses)]
                row += [','.join(str(l) for l in losses)]
            loss_out.write('\t'.join(str(x) for x in row) + '\n')

    if best_params is None:
        raise Exception(("Could not find best choice of hyperparameters "
            "because either (1) all losses were nan; or (2) all standard "
            "errors of the mean loss for each choice were too high (i.e., "
            "no model is robust enough)"))

    return (best_params, best_loss, best_loss_sem)


def nested_cross_validate(x, y, search_type, regression,
        num_outer_splits=5, num_inner_splits=5, loss_out=None,
        num_random_samples=None,
        max_sem=None):
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
        regression: if True, perform regression; if False, classification
        num_outer_splits: number of folds in the outer cross-validation
            procedure
        num_inner_splits: number of folds in the inner cross-validation
            procedures
        loss_out: if set, an opened file object to write to write the mean
            validation loss for each choice of hyperparameters (once per
            each outer fold)
        num_random_samples: number of random samples to use, if search_type
            is 'random' (if None, use default)
        max_sem: maximum standard error of the mean (SEM) on the loss to allow
            the hyperparameters with that loss to be chosen as the 'best'
            choice of hyperparameters (if None, no limit)

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
        best_params, _, _ = search_for_hyperparams(x_train, y_train,
                search_type, num_splits=num_inner_splits, loss_out=loss_out,
                num_random_samples=num_random_samples,
                max_sem=max_sem)

        # Compute a loss for this choice of parameters as follows:
        #   (1) train the model on all the training data (in the inner
        #       fold loop of search_for_hyperparams(), the model will have
        #       only been trained on a subset of the data (i.e., on k-1
        #       folds))
        #   (2) test the model on the outer validation data (x_validate,
        #       y_validate), effectively treating this outer validation data
        #       as a test set for best_params
        best_model = predictor.construct_model(best_params, x_train.shape,
                regression)
        results = predictor.train_and_validate(best_model, x_train, y_train,
                x_validate, y_validate, best_params['max_num_epochs'])
        val_loss, val_loss_different_metrics = determine_val_loss(results)
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
    if args.dataset == 'cas9':
        parser_class = parse_data.Doench2016Cas9ActivityParser
        subset = args.cas9_subset
        regression = False
    elif args.dataset == 'simulated-cas13':
        parser_class = parse_data.Cas13SimulatedData
        subset = args.cas9_subset
        regression = False
    elif args.dataset == 'cas13':
        parser_class = parse_data.Cas13ActivityParser
        subset = args.cas13_subset
        if args.cas13_classify:
            regression = False
        else:
            regression = True
    data_parser = parser_class(
            subset=subset,
            context_nt=args.context_nt,
            split=(train_split_frac, 0, args.test_split_frac),
            shuffle_seed=args.seed,
            stratify_by_pos=True)
    if args.dataset == 'cas13':
        classify_activity = args.cas13_classify
        regress_on_all = args.cas13_regress_on_all
        regress_only_on_active = args.cas13_regress_only_on_active
        data_parser.set_activity_mode(
                classify_activity, regress_on_all, regress_only_on_active)
    data_parser.read()
    parse_data._split_parser = data_parser

    x, y = data_parser.train_set()
    x_test, y_test = data_parser.test_set()

    # Print the size of each data set
    print('DATA SIZES - (Train + validation): {}, Test: {}'.format(
        len(x), len(x_test)))

    if regression:
        # Print the mean outputs
        print('Mean train output: {}'.format(np.mean(y)))
    else:
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

            # Write header
            header = ['params']
            metrics = _regression_losses if regression else _classification_losses
            for metric in metrics:
                header += [metric + '_mean']
                header += [metric + '_sem']
                header += [metric + '_values']
            params_mean_val_loss_out_tsv_f.write('\t'.join(header) + '\n')
        else:
            params_mean_val_loss_out_tsv_f = None

        params, loss, loss_sem = search_for_hyperparams(x, y,
                args.search_type,
                regression,
                num_splits=args.hyperparam_search_cross_val_num_splits,
                loss_out=params_mean_val_loss_out_tsv_f,
                num_random_samples=args.num_random_samples,
                max_sem=args.max_sem)

        if params_mean_val_loss_out_tsv_f is not None:
            params_mean_val_loss_out_tsv_f.close()

        print('***')
        print('Hyperparameter search results:')
        print('  Best parameters: {}'.format(params))
        print('  Mean validation loss of these parameters: {}'.format(loss))
        print('  Std. err. of validation loss of these parameters: {}'.format(loss_sem))
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
        # So we will split x,y into train/validate sets and the validation data
        # can also be used for early stopping during training.
        print('Testing model with optimal parameters')
        model = predictor.construct_model(params, x.shape, regression)
        split_iter = parse_data.split(x, y,
                args.hyperparam_search_cross_val_num_splits,
                stratify_by_pos=True)
        # Only take the first split of the generator as the train/validation
        # split
        x_train, y_train, x_validate, y_validate = next(split_iter)
        predictor.train_and_validate(model, x_train, y_train,
                x_validate, y_validate, params['max_num_epochs'])

        # Save the model weights and best parameters to args.save_best_model
        if args.save_best_model:
            if not os.path.exists(args.save_best_model):
                os.makedirs(args.save_best_model)

            # Note that we can only save the model weights, not the model itself
            # (incl. architecture), because they are subclassed models and
            # therefore are described by code (Keras only allows saving
            # the full model if they are Sequential or Functional models)
            # See https://www.tensorflow.org/beta/guide/keras/saving_and_serializing
            # for details on saving subclassed models
            model.save_weights(
                    os.path.join(args.save_best_model,
                        'best_model.weights'),
                    save_format='tf')
            print('Saved best model weights to {}'.format(args.save_best_model))

            params['regression'] = regression
            save_best_model_params = os.path.join(args.save_best_model,
                    'best_model.params.pkl')
            with open(save_best_model_params, 'wb') as f:
                pickle.dump(params, f)
            print('Saved best model parameters to {}'.format(
                args.save_best_model))

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

            # Write header
            header = ['params']
            metrics = _regression_losses if regression else _classification_losses
            for metric in metrics:
                header += [metric + '_mean']
                header += [metric + '_sem']
                header += [metric + '_values']
            params_mean_val_loss_out_tsv_f.write('\t'.join(header) + '\n')
        else:
            params_mean_val_loss_out_tsv_f = None

        fold_results = nested_cross_validate(x, y,
                args.search_type,
                regression,
                num_outer_splits=args.nested_cross_val_outer_num_splits,
                num_inner_splits=args.hyperparam_search_cross_val_num_splits,
                loss_out=params_mean_val_loss_out_tsv_f,
                num_random_samples=args.num_random_samples,
                max_sem=args.max_sem)

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
    parser.add_argument('--dataset',
            choices=['cas9', 'simulated-cas13', 'cas13'],
            required=True,
            help=("Dataset to use. 'simulated-cas13' is simulated Cas13 data "
                  "from the Cas9 data"))
    parser.add_argument('--cas9-subset',
            choices=['guide-mismatch-and-good-pam', 'guide-match'],
            help=("Use a subset of the Cas9 data or simulated Cas13 data. See "
                "parse_data module for descriptions of the subsets. To use all "
                "data, do not set."))
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
            default=3,
            help=("Number of splits to use in the outer fold for nested "
                  "cross-validation (the inner fold uses "
                  "HYPERPARAM_SEARCH_CROSS_VAL_NUM_SPLITS splits)"))
    parser.add_argument('--search-type',
            choices=['grid', 'random'],
            default='random',
            help=("Type of hyperparameter search ('grid' or 'random')"))
    parser.add_argument('--num-random-samples',
            type=int,
            default=1000,
            help=("Number of random samples to use for random search; "
                  "only applicable when SEARCH_TYPE is 'random'"))
    parser.add_argument('--max-sem',
            type=float,
            help=("If set, a maximum standard error of the mean loss (across "
                  "folds) to permit a choice of hyperparameters to be chosen "
                  "as the best choice. 0.05 is reasonable for most "
                  "applications. If not set (default), there is no "
                  "restriction on this"))
    parser.add_argument('--params-mean-val-loss-out-tsv',
            help=("If set, path to out TSV at which to write the mean "
                  "validation losses (across folds) for each choice of "
                  "hyperparameters; with nested cross-validation, each "
                  "choice of hyperparameters is written for *each* outer "
                  "fold"))
    parser.add_argument('--save-best-model',
            help=("If set, path to directory in which to save best parameters "
                  "and the model weights; only applies to hyperparameter "
                  "search"))
    parser.add_argument('--seed',
            type=int,
            default=1,
            help=("Random seed"))
    args = parser.parse_args()

    # Print the arguments provided
    print(args)

    main(args)

