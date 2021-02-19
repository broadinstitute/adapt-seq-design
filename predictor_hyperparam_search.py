"""Search over hyperparameters for the predictor.

The grid search here is similar to scikit-learn's GridSearchCV, and the
random search is similar to its RandomizedSearchCV.
"""

import argparse
from collections import defaultdict
import hashlib
import itertools
import math
import os
import pickle
import random

import parse_data
import predictor

import numpy as np
import scipy.stats
import tensorflow as tf


__author__ = 'Hayden Metsky <hayden@mit.edu>'


_regression_losses = ['mse', '1_minus_r', '1_minus_rho']
_default_regression_loss = '1_minus_rho'
_classification_losses = ['bce', '1_minus_auc-roc', '1_minus_auc-pr']
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
        results: dict returned by predictor.train_with_keras() or
            predictor.test_with_keras()

    Returns:
        (default loss value to use for ranking, dict of loss values for
        different metrics)
    """
    if 'bce' in results:
        # Classification results
        assert 'auc-roc' in results
        assert 'auc-pr' in results
        assert 'mse' not in results
        assert 'r-pearson' not in results
        assert 'r-spearman' not in results

        losses = {'bce': results['bce'],
                  '1_minus_auc-roc': 1.0 - results['auc-roc'],
                  '1_minus_auc-pr': 1.0 - results['auc-pr']}
        default_loss = losses[_default_classification_loss]
    elif 'mse' in results:
        # Regression results
        assert 'r-pearson' in results
        assert 'r-spearman' in results
        assert 'bce' not in results
        assert 'auc-roc' not in results
        assert 'auc-pr' not in results

        losses = {'mse': results['mse'],
                  '1_minus_r': 1.0 - results['r-pearson'],
                  '1_minus_rho': 1.0 - results['r-spearman']}
        default_loss = losses[_default_regression_loss]
    else:
        raise Exception("Unknown whether classification or regression")

    return default_loss, losses


def cross_validate(params, x, y, num_splits, regression,
        callback=None, dp=None, stop_early_for_loss=None):
    """Perform k-fold cross-validation.

    This uses data_parser.split() to split data, which uses stratified
    k-fold cross validation.

    Args:
        params: dict of hyperparameters
        x: input data
        y: output (labels, if classification)
        num_splits: number of folds
        regression: if True, perform regression; if False, classification
        callback: if set, a function to have the test function call
        dp: if set, a data parser to use rather than the module-wide
            data_parser object
        stop_early_for_loss: if set, stop early if any outer fold gives
            a validation loss (default) that exceeds this value

    Returns:
        tuple ([default validation loss for each fold], dict {metric: [list of
        validation losses for metric, for each fold]})
    """
    print('Performing cross-validation with parameters: {}'.format(params))

    if dp is None:
        dp = data_parser

    val_losses_default = []
    val_losses_different_metrics = defaultdict(list)
    i = 0
    split_iter = dp.split(x, y, num_splits=num_splits,
            stratify_by_pos=True)
    for x_train, y_train, x_validate, y_validate in split_iter:
        print('STARTING FOLD {} of {}'.format(i+1, num_splits))
        print('  Training on n={}, validating on n={}'.format(len(x_train),
            len(x_validate)))

        # Use x_validate, y_validate for validation of the model on
        # this fold

        # Split the training data (x_train, y_train) *again* to get a
        # separate validation set (x_validate_for_es, y_validate_for_es) to
        # pass to predictor.train_with_keras(), which it will use for early
        # stopping
        # This way, we do not use the same validation set both for early
        # stopping and for measuring the performance of the model (otherwise
        # our 'training' of the model, which includes deciding when to
        # stop, would have seen and factored in the data used for evaluating
        # it on the fold)
        # Only use one of the splits, with 3/4 being used to train the model
        # and 1/4 for early stopping
        train_split_iter = dp.split(x_train, y_train, num_splits=4,
                stratify_by_pos=True)
        x_train_for_train, y_train_for_train, x_validate_for_es, y_validate_for_es = next(train_split_iter)

        # Start a new model for this fold
        model = predictor.construct_model(params, x_train_for_train.shape,
                regression, compile_for_keras=True, y_train=y_train_for_train)

        # Train this fold
        predictor.train_with_keras(model, x_train_for_train,
                y_train_for_train, x_validate_for_es, y_validate_for_es)

        # Run predictor.test_with_keras() on the validation data for this fold, to
        # get its validation results
        results = predictor.test_with_keras(model, x_validate, y_validate,
                dp, callback=callback)
        a, b = determine_val_loss(results)
        val_losses_default += [a]
        for k in b.keys():
            val_losses_different_metrics[k].append(b[k])

        if stop_early_for_loss is not None:
            if (a > stop_early_for_loss or np.isnan(a)) and i+1 < num_splits:
                print(('STOPPING EARLY'))
                break

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
            'conv_filter_width': [None, [1], [2], [1, 2], [1, 2, 3, 4]],
            'conv_num_filters': [5, 25, 100, 200],
            'pool_window_width': [1, 2, 4],
            'fully_connected_dim': [[20], [40],
                [20, 20], [20, 40], [40, 40],
                [20, 20, 20], [20, 20, 40], [20, 40, 40], [40, 40, 40]],
            'pool_strategy': ['max', 'avg', 'max-and-avg'],
            'locally_connected_width': [None, [1], [2], [1, 2], [1, 2, 3, 4]],
            'locally_connected_dim': [1, 2, 4],
            'skip_batch_norm': [False, True],
            'add_gc_content': [False],
            'activation_fn': ['relu', 'elu'],
            'dropout_rate': [0, 0.25, 0.50],
            'l2_factor': [0, 0.0001, 0.001, 0.01, 0.1],
            'sample_weight_scaling_factor': [0],
            'batch_size': [16, 32, 64, 256],
            'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],
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
    def loguniform(log_low=-5.0, log_high=0.0, base=10.0):
        # values base^x where x is uniform in [log_low, log_high)
        def d():
            return base**np.random.uniform(log_low, log_high)
        return d

    # Map each parameter to a distribution of values
    space = {
             'conv_filter_width': uniform_discrete([
                 None,
                 [1], [2], [3], [4],
                 [1, 2], [1, 2, 3], [1, 2, 3, 4]]),
             'conv_num_filters': uniform_int(10, 250),
             'pool_window_width': uniform_int(1, 4),
             'fully_connected_dim': uniform_nested_dist(1, 3,
                 uniform_int(25, 75)),
             'pool_strategy': uniform_discrete(['max', 'avg', 'max-and-avg']),
             'locally_connected_width': uniform_discrete([
                 None,
                 [1], [2], [1, 2]]),
             'locally_connected_dim': uniform_int(1, 5),
             'skip_batch_norm': uniform_discrete([False, True]),
             'add_gc_content': constant(False),
             'activation_fn': uniform_discrete(['relu', 'elu']),
             'dropout_rate': uniform_continuous(0, 0.5),
             'l2_factor': lognormal(-13.0, 4.0),
             'sample_weight_scaling_factor': constant(0),
             'batch_size': uniform_int(32, 256),
             'learning_rate': loguniform(-6.0, -1.0, 10.0),
             'regression_clip': uniform_discrete([False, True]),
             'regression_clip_alpha': loguniform(-6.0, -1.0, 10.0),
             'max_num_epochs': constant(1000)
    }
    for i in range(num_samples):
        params = {}
        for k in space.keys():
            k_dist = space[k]
            params[k] = k_dist()
        yield params


def search_for_hyperparams(x, y, search_type, regression, context_nt,
        num_splits=5, loss_out=None, models_out=None, num_random_samples=None,
        max_sem=None, dp=None, stop_early_for_loss=None):
    """Search for optimal hyperparameters.

    This uses hyperparam_grid() to find the grid of hyperparameters over
    which to search.

    On each point p in the grid, this runs k-fold cross-validation, computes
    the mean validation loss over the folds for p, and then returns the
    p with the smallest mean loss. As a single loss value, this uses the default
    loss defined above determine_val_loss().

    Args:
        x: input data
        y: labels
        search_type: 'grid' or 'random' search
        regression: if True, perform regression; if False, classification
        context_nt: number of nt on each side of guide to use for context
        num_splits: number of splits of the data to make (i.e., k in k-fold)
        loss_out: if set, an opened file object to write to write the
            validation loss values for each choice of hyperparameters
        models_out: if set, a path to a directory in which to save model
            parameters and trained weights for each choice of hyperparameters
            (each choice is placed in a separate subfolder in here)
        num_random_samples: number of random samples to use, if search_type
            is 'random' (if None, use default)
        max_sem: maximum standard error of the mean (SEM) on the loss to allow
            the hyperparameters with that loss to be chosen as the 'best'
            choice of hyperparameters (if None, no limit)
        dp: if set, a data parser to use rather than the module-wide
            data_parser object
        stop_early_for_loss: if set, stop early if any outer fold gives
            a validation loss (default) that exceeds this value

    Returns:
        tuple (params, loss, sem) where params is the optimal choice of
        hyperparameters and loss is the mean validation loss at that choice
        and sem of the standard error of the loss
    """
    if dp is None:
        dp = data_parser

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
        params['context_nt'] = context_nt

        params_id = params_hash(params)

        # Compute a mean validation loss at this choice of params
        val_losses_default, val_losses_different_metrics = cross_validate(
                params, x, y, num_splits, regression, dp=dp,
                stop_early_for_loss=stop_early_for_loss)

        if len(val_losses_default) < num_splits:
            # Cross-validation stopped early on an outer fold; skip this
            # choice of hyperparameters
            continue

        mean_loss = np.mean(val_losses_default)
        if len(val_losses_default) == 1:
            sem_loss = scipy.nan
        else:
            sem_loss = scipy.stats.sem(val_losses_default)

        if models_out is not None:
            # Train a model across all data, and save hyperparameters and
            # weights
            model_out_path = os.path.join(models_out,
                    'model-' + params_id)
            train_and_save_model(params, x, y, regression, context_nt,
                    num_splits, model_out_path)

        # Decide whether to update the current best choice of hyperparameters
        # It seems that np.mean(x) where x contains nan is *not* np.nan
        # but math.isnan(..) is True
        if mean_loss is np.nan or math.isnan(mean_loss):
            # mean_loss can be nan, e.g., for Spearman's rho if all predicted
            # values are the same for at least one of the inner folds
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
            # Write a row listing this model's hyperparameters and validation
            # results
            row = [params_id, params]
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


def nested_cross_validate(x, y, search_type, regression, context_nt,
        num_outer_splits=5, num_inner_splits=5, loss_out=None,
        num_random_samples=None,
        max_sem=None,
        outer_splits_to_run=None):
    """Perform nested cross-validation to validate model and search.

    This is useful to verify the overall model and model fitting approach
    (i.e., the hyperparameter search). The inner cross-validation procedure
    searches for optimal hyperparameters. The outer cross-validation
    procedure sees how well this search generalizes, by performing and
    testing it on different folds of the data. If the selected hyperparameters
    are similar across the outer folds, and results on the validation set
    are similar across the outer folds, that's good.

    This is not for selecting a final model.

    Args:
        x: input data
        y: labels
        search_type: 'grid' or 'random' search
        regression: if True, perform regression; if False, classification
        context_nt: number of nt to use on each side of guide for context
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
        outer_splits_to_run: if set, a list of outer splits to run (0-based);
            if not set, run all

    Returns:
        list x where x[i] is a tuple (params, loss, metrics) where params is an
        optimal choice of parameters (a dict), loss is the (default) loss for
        that choice on the outer validation data, and metrics gives loss values
        for different metrics on the outer validation data; each x[i]
        corresponds to an outer fold of the data. If outer_splits_to_run is
        set, then x[i] is None if i is not in outer_splits_to_run
    """
    optimal_choices = []
    i = 0
    outer_split_iter = dp.split(x, y, num_splits=num_outer_splits,
            stratify_by_pos=True)
    for x_train, y_train, x_validate, y_validate in outer_split_iter:
        print('STARTING OUTER FOLD {} of {}'.format(i+1, num_outer_splits))
        print('  MSE on train data if predicting mean of train data:',
                np.mean(np.square(np.mean(y_train) - np.array(y_train))))
        print('  MSE on validate data if predicting mean of train data:',
                np.mean(np.square(np.mean(y_train) - np.array(y_validate))))

        if outer_splits_to_run is not None:
            if i not in outer_splits_to_run:
                print('  Skipping this outer split')
                optimal_choices += [None]
                i += 1

                # Advance random number generator
                random.random()
                np.random.random()
                tf.random.uniform([1])

                continue

        # Search for hyperparameters on this outer fold of the data

        # Start a new model for this fold
        best_params, _, _ = search_for_hyperparams(x_train, y_train,
                search_type, regression, context_nt,
                num_splits=num_inner_splits, loss_out=loss_out,
                num_random_samples=num_random_samples,
                max_sem=max_sem)

        # Compute a loss for this choice of parameters as follows:
        #   (1) train the model on all the training data (in the inner
        #       fold loop of search_for_hyperparams(), the model will have
        #       only been trained on a subset of the data (i.e., on k-1
        #       folds)); on this training data, split it further into
        #       data actually used for training and 'validation' data only
        #       used for early stopping
        #   (2) test the model on the outer validation data (x_validate,
        #       y_validate), effectively treating this outer validation data
        #       as a test set for best_params
        best_model = predictor.construct_model(best_params, x_train.shape,
                regression, compile_for_keras=True, y_train=y_train)
        # Split x_train,y_train into train/validate sets and the validation
        # data is used only for early stopping during training
        train_split_iter = dp.split(x_train, y_train,
                num_inner_splits,
                stratify_by_pos=True)
        # Only take the first split of the generator as the train/validation
        # split
        x_train_for_train, y_train_for_train, x_train_for_es, y_train_for_es = next(train_split_iter)
        predictor.train_with_keras(best_model,
                x_train_for_train, y_train_for_train,
                x_train_for_es, y_train_for_es)
        # Test the model on the validation data
        val_results = predictor.test_with_keras(best_model,
                x_validate, y_validate, dp)
        val_loss, val_loss_different_metrics = determine_val_loss(val_results)
        optimal_choices += [(best_params, val_loss, val_loss_different_metrics)]

        # Print metrics on fold
        print('Results on fold {}'.format(i+1))
        print('  Metrics on validation data: {}'.format(val_results))

        print(('FINISHED OUTER FOLD {} of {}; validation loss on this outer '
            'fold is {}').format(i+1, num_outer_splits, val_loss))
        i += 1
    return optimal_choices


def params_hash(params, length=8):
    """Hash parameters.

    This is useful for identifying a model (e.g., naming a directory in which
    to save it).

    This is the final 'length' hex digits of the hash. Python's hash()
    function produces a hash with size dependent on the size of the input and
    is inconsistent (depends on seed); using the SHA-224 hash function should
    produce more uniform hash values.

    Args:
        params: dict of hyperparameters for a model
        length: number of hex digits in hash

    Returns:
        hash of params
    """
    params_str = str(sorted(list(params.items())))
    return hashlib.sha224(params_str.encode()).hexdigest()[-length:]


def train_and_save_model(params, x, y, regression, context_nt,
        hyperparam_search_cross_val_num_splits, out_path):
    """Train and save a model.

    This trains across all train/validate (non-test) data.

    Args:
        params: dict of hyperparameters for model
        x: input data
        y: labels
        regression: True if performing regression; False if classification
        context_nt: number of nt on each side of guide to use for context
        hyperparam_search_cross_val_num_splits: number of splits used
            for cross-validation during hyperparameter search
        out_path: directory in which to save model parameters and
            trained weights
    """
    print('Training model across all train+validate data')
    # We could retrain on all the train data (the hyperparameter search will have
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
    split_iter = data_parser.split(x, y,
            hyperparam_search_cross_val_num_splits,
            stratify_by_pos=True)
    # Only take the first split of the generator as the train/validation
    # split
    x_train, y_train, x_validate, y_validate = next(split_iter)
    model = predictor.construct_model(params, x_train.shape, regression,
            compile_for_keras=True, y_train=y_train)
    predictor.train_with_keras(model, x_train, y_train,
            x_validate, y_validate)

    # Save the model weights and best parameters to out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Note that we can only save the model weights, not the model itself
    # (incl. architecture), because they are subclassed models and
    # therefore are described by code (Keras only allows saving
    # the full model if they are Sequential or Functional models)
    # See https://www.tensorflow.org/beta/guide/keras/saving_and_serializing
    # for details on saving subclassed models
    model.save_weights(
            os.path.join(out_path,
                'model.weights'),
            save_format='tf')
    print('Saved best model weights to {}'.format(out_path))

    params['regression'] = regression
    params['context_nt'] = context_nt
    out_path_params = os.path.join(out_path,
            'model.params.pkl')
    with open(out_path_params, 'wb') as f:
        pickle.dump(params, f)
    print('Saved model parameters to {}'.format(out_path))



def main(args):
    # Seed the predictor, numpy, and python's random module
    predictor.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Make the data_parser be module wide
    global data_parser

    # Read the data
    train_split_frac = 1.0 - args.test_split_frac
    if args.dataset == 'cas13':
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
            stratify_by_pos=True,
            use_median_measurement=args.use_median_measurement)
    if args.dataset == 'cas13':
        classify_activity = args.cas13_classify
        regress_on_all = args.cas13_regress_on_all
        regress_only_on_active = args.cas13_regress_only_on_active
        data_parser.set_activity_mode(
                classify_activity, regress_on_all, regress_only_on_active)
        if args.cas13_normalize_crrna_activity:
            data_parser.set_normalize_crrna_activity()
        if args.cas13_use_difference_from_wildtype_activity:
            data_parser.set_use_difference_from_wildtype_activity()
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
        print('Variance of train output: {}'.format(np.var(y)))
    else:
        # Print the fraction of the training data points that are in each class
        classes = set(tuple(yv) for yv in y)
        for c in classes:
            num_c = sum(1 for yv in y if tuple(yv) == c)
            frac_c = float(num_c) / len(y)
            print('Fraction of train data in class {}: {}'.format(
                c, frac_c))

    if args.command == 'hyperparam-search':
        print('Note: The test data is not used at all in this command.')

        # Search for optimal hyperparameters
        if args.params_mean_val_loss_out_tsv:
            # Use buffering=1 for line buffering (write each line immediately)
            params_mean_val_loss_out_tsv_f = open(
                    args.params_mean_val_loss_out_tsv, 'w', buffering=1)

            # Write header
            header = ['params_id', 'params']
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
                args.context_nt,
                num_splits=args.hyperparam_search_cross_val_num_splits,
                loss_out=params_mean_val_loss_out_tsv_f,
                models_out=args.save_models,
                num_random_samples=args.num_random_samples,
                max_sem=args.max_sem,
                stop_early_for_loss=args.stop_early_for_loss)

        if params_mean_val_loss_out_tsv_f is not None:
            params_mean_val_loss_out_tsv_f.close()

        print('***')
        print('Hyperparameter search results:')
        print('  Note: Best is chosen according to criteria above')
        print('  Best parameters hash: {}'.format(params_hash(params)))
        print('  Best parameters: {}'.format(params))
        print('  Mean validation loss of these parameters: {}'.format(loss))
        print('  Std. err. of validation loss of these parameters: {}'.format(loss_sem))
        print('***')
    elif args.command == 'nested-cross-val':
        print('Note: The test data is not used at all in this command.')
        if args.test_split_frac > 0:
            print(('WARNING: Performing nested cross-validation but there is '
                   'unused test data; it may make sense to set '
                   '--test-split-frac to 0'))

        if args.save_models is not None:
            print(('WARNING: --save-models is set but models are not saved '
                   'during nested cross-validation'))

        # Perform nested cross-validation
        if args.params_mean_val_loss_out_tsv:
            # Use buffering=1 for line buffering (write each line immediately)
            params_mean_val_loss_out_tsv_f = open(
                    args.params_mean_val_loss_out_tsv, 'w', buffering=1)

            # Write header
            header = ['params_id', 'params']
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
                args.context_nt,
                num_outer_splits=args.nested_cross_val_outer_num_splits,
                num_inner_splits=args.hyperparam_search_cross_val_num_splits,
                loss_out=params_mean_val_loss_out_tsv_f,
                num_random_samples=args.num_random_samples,
                max_sem=args.max_sem,
                outer_splits_to_run=args.nested_cross_val_run_for)

        if params_mean_val_loss_out_tsv_f is not None:
            params_mean_val_loss_out_tsv_f.close()

        if args.nested_cross_val_out_tsv:
            # Write results for each fold
            with open(args.nested_cross_val_out_tsv, 'w') as fw:
                header = ['fold', 'best_params_id', 'best_params']
                header += metrics
                fw.write('\t'.join(header) + '\n')
                for fold in range(len(fold_results)):
                    if fold_results[fold] is None:
                        continue
                    fold_params, fold_loss, fold_metrics = fold_results[fold]
                    row = [fold, params_hash(fold_params), fold_params]
                    for metric in metrics:
                        row += [fold_metrics[metric]]
                    fw.write('\t'.join(str(r) for r in row) + '\n')

        # Print results (parameter selection and loss) for each outer fold
        print('***')
        for fold in range(len(fold_results)):
            print('Outer fold {} results:'.format(fold+1))
            if fold_results[fold] is None:
                continue
            fold_params, fold_loss, fold_metrics = fold_results[fold]
            print('  Parameters: {}'.format(fold_params))
            print('  Loss of parameters on outer validation data: {}'.format(fold_loss))
            print('  Metrics of parameters on outer validation data: {}'.format(fold_metrics))
        print('***')
    else:
        raise Exception("Unknown command '%s'" % args.command)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--cas13-use-difference-from-wildtype-activity',
            action='store_true',
            help=("If set, use the activity value of a guide g and target t "
                  "pair to be the difference between the measured activity of "
                  "g-t and the mean activity between g and all wildtype "
                  "(matching) targets of g; this means prediction is "
                  "performed based on targeted differences (e.g., mismatches) "
                  "rather than inherent sequence of the crRNA"))
    parser.add_argument('--use-median-measurement',
            action='store_true',
            help=("If set, use the median measurment across replicates "
                  "(instead, resample)"))
    parser.add_argument('--context-nt',
            type=int,
            default=10,
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
                  "as the best choice. 0.10 is reasonable for most "
                  "applications. If not set (default), there is no "
                  "restriction on this"))
    parser.add_argument('--stop-early-for-loss',
            type=float,
            help=("If set, stop a choice of hyperparameters early if any "
                  "outer fold gives a default loss value exceeding this "
                  "value"))
    parser.add_argument('--params-mean-val-loss-out-tsv',
            help=("If set, path to out TSV at which to write the mean "
                  "validation losses (across folds) for each choice of "
                  "hyperparameters; with nested cross-validation, each "
                  "choice of hyperparameters is written for *each* outer "
                  "fold"))
    parser.add_argument('--nested-cross-val-out-tsv',
            help=("If set, path to out TSV at which to write metrics on "
                  "the validation data for each outer fold of nested "
                  "cross-validation (one row per outer fold; each column "
                  "gives a metric)"))
    parser.add_argument('--nested-cross-val-run-for',
            nargs='+',
            type=int,
            help=("If set, only run the given outer splits (0-based). If "
                  "not set, run for all."))
    parser.add_argument('--save-models',
            help=("If set, path to directory in which to save parameters and "
                  "model weights for each model validated (each is in a "
                  "subfolder); only applies to hyperparameter search"))
    parser.add_argument('--seed',
            type=int,
            default=1,
            help=("Random seed"))
    args = parser.parse_args()

    # Print the arguments provided
    print(args)

    main(args)

