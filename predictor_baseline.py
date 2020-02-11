"""Baselines for predicting guide sequence activity.
"""

import argparse

import fnn
import parse_data
import predictor
import rnn

import numpy as np
import scipy
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils
import tensorflow as tf


def parse_args():
    """Parse arguments.

    Returns:
        argument namespace
    """
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
    parser.add_argument('--context-nt',
            type=int,
            default=10,
            help=("nt of target sequence context to include alongside each "
                  "guide"))
    parser.add_argument('--regression-scoring-method',
            choices=['mse', 'rho'],
            default='mse',
            help=("Method to use for scoring regression results; 'mse' for "
                  "mean squared error, 'rho' for Spearman rank correlation"))
    parser.add_argument('--test-split-frac',
            type=float,
            default=0.3,
            help=("Fraction of the dataset to use for testing the final "
                  "model"))
    parser.add_argument('--models-to-use',
            nargs='+',
            help=("List of model names to use. If not set, use all."))
    parser.add_argument('--nested-cross-val',
            action='store_true',
            help=("If set, perform nested cross-validation to evaluate "
                  "model selection, rather than just cross-validation to "
                  "select a single model"))
    parser.add_argument('--nested-cross-val-outer-num-splits',
            type=int,
            default=5,
            help=("Number of outer folds to use for nested cross-validation"))
    parser.add_argument('--nested-cross-val-out-tsv',
            help=("Path to output TSV at which to write metrics on the "
                  "validation data for each outer fold of nested "
                  "cross-validation (one row per outer fold; each column "
                  "gives a metric)"))
    parser.add_argument('--seed',
            type=int,
            default=1,
            help=("Random seed"))
    args = parser.parse_args()

    # Print the arguments provided
    print(args)

    return args


def set_seed(seed):
    """Set tensorflow and numpy seed.

    sklearn appears to use the numpy random number generator, so setting
    the numpy seed applies to that too.

    Args:
        seed: random seed
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


# TODO increase n_iter
def random_search_cv(model_name, model_obj, cv, scorer, n_iter=2):
    """Construct a RandomizedSearchCV object.

    Args:
        model_name: name of model
        model_obj: model object; either a scikit-learn estimator or one that
            follows its interface
        cv: cross-validation splitting iterator
        scorer: scoring function to use
        n_iter: number of samples for random search

    Returns:
        sklearn.model_selection.RandomizedSearchCV object
    """
    # In some cases (e.g., pulling from distributions whose log is uniform),
    # we'll want to just draw many samples and have RandomizedSearchCV draw
    # from these; specify how many samples we draw and provide as
    # representative of the space
    space_size = 1000

    # Set params
    if model_name == 'l1_lr':
        params = {
            'alpha': np.logspace(-8, 8, base=10.0, num=space_size)
        }
    elif model_name == 'l2_lr':
        params = {
            'alpha': np.logspace(-8, 8, base=10.0, num=space_size)
        }
    elif model_name == 'l1l2_lr':
        params = {
            'l1_ratio': 1.0 - np.logspace(-5, 0, base=2.0, num=space_size)[::-1] + 2.0**(-5),
            'alpha': np.logspace(-8, 8, base=10.0, num=space_size)
        }
    elif model_name == 'gbt':
        params = {
            'learning_rate': np.logspace(-2, 0, base=10.0, num=space_size),
            'n_estimators': np.logspace(0, 8, base=2, num=space_size).astype(int),
            'min_samples_split': np.logspace(1, 3, base=2, num=space_size).astype(int),
            'min_samples_leaf': np.logspace(0, 2, base=2, num=space_size).astype(int),
            'max_depth': np.logspace(1, 3, base=2, num=space_size).astype(int),
            'max_features': [None, 0.1, 'sqrt', 'log2']
        }
    elif model_name == 'rf':
        # For max_depth, set 'None' 1/2 of the time
        params = {
            'n_estimators': np.logspace(0, 8, base=2, num=space_size).astype(int),
            'min_samples_split': np.logspace(1, 3, base=2, num=space_size).astype(int),
            'min_samples_leaf': np.logspace(0, 2, base=2, num=space_size).astype(int),
            'max_depth': [None]*space_size + list(np.logspace(1, 4, base=2,
                num=space_size).astype(int)),
            'max_features': [None, 0.1, 'sqrt', 'log2']
        }
    elif model_name == 'mlp':
        # Constructing a space of layer_dims requires choosing the number of
        # layers and the dimensions of each; note that layer_dims does NOT
        # include the output layer of the MLP (which has dimension=1)
        layer_dims = []
        for i in range(space_size):
            num_layers = np.random.randint(1, 4)
            dims = [np.random.randint(4, 128) for _ in range(num_layers)]
            layer_dims += [dims]

        params = {
            'layer_dims': layer_dims,
            'dropout_rate': scipy.stats.uniform(0, 0.5),
            'activation_fn': ['relu', 'elu']
        }
    elif model_name == 'lstm':
        params = {
            'units': np.logspace(1, 8, base=2, num=space_size).astype(int),
            'bidirectional': [False, True],
            'embed_dim': [None]*4 + list(range(1, 9)),
            'dropout_rate': scipy.stats.uniform(0, 0.5)
        }
    else:
        raise Exception("Unknown model: '%s'" % model_name)

    rs_cv = sklearn.model_selection.RandomizedSearchCV(model_obj,
                params, cv=cv, refit=True,
                scoring=scorer, n_iter=n_iter)
    return rs_cv


def classify(x_train, y_train, x_test, y_test,
        parsers,
        num_inner_splits=2):
    """Perform classification.

    Test data is used for evaluating the model with the best choice of
    hyperparameters, after refitting across *all* the train data.

    Args:
        x_{train,test}: input data for train/test
        y_{train,test}: output labels for train/test
        parsers: parse_data parsers to use for splitting data
        num_inner_splits: number of splits for cross-validation

    Returns:
        dict {'l1_logistic_regression': metrics on test data for best choice of
        hyperparameters with L1 logistic regression, 'l2_logistic_regression':
        metrics on test data for best choice of hyperparameters with L2
        logistic regression}
    """
    # TODO implement other models, e.g., SVM
    # TODO as done in regress() below, optimize hyperparameters for a
    # custom scoring function (e.g., AUC)

    # Determine class weights
    y_train_labels = list(y_train)
    class_weight = sklearn.utils.class_weight.compute_class_weight(
            'balanced', sorted(np.unique(y_train_labels)), y_train_labels)
    class_weight = list(class_weight)
    class_weight = {i: weight for i, weight in enumerate(class_weight)}

    # Construct models for L1 and L2 logistic regression
    # Perform built-in cross-validation to determine regularization strength

    # Set space for determining regularization strength: 10^-5 to 10^5 on a
    # log scale
    Cs = np.logspace(-5, 5, num=100, base=10.0)

    # TODO when there is more negative data, increase num_splits to 3 or 5
    # Note that, currently, there is so little negative data that when
    # num_splits=3, one split will have no negative data points in the test
    # set so all labels are from the same class. sklearn's log_loss
    # function complains about this, saying that all y_true are the same
    # and therefore the 'labels' argument must be provided to log_loss. One
    # way around this might be to follow
    # https://github.com/scikit-learn/scikit-learn/issues/9144#issuecomment-411347697
    # and specify a function/callable for the 'scoring' argument of
    # LogisticRegressionCV, which would be the output of
    # sklearn.metrics.make_scorer and wraps around sklearn's log_loss
    # function, passing in the 'labels' argument to it (one issue would be
    # having to also determine a 'sample_weight' argument to give the log_loss
    # function).
    def cv(k='baselinefeats'):
        assert k in ['baselinefeats', 'onehot']
        return parsers[k].split(x_train[k], y_train[k], num_inner_splits,
                stratify_by_pos=True, yield_indices=True)

    # It helps to provide an explicit function/callable as the scorer;
    # when using 'neg_log_loss' for 'scoring', it crashes because the
    # log_loss function expects a 'labels' argument when y_true contains
    # only one label

    metrics_for_models = {}
    for penalty in ['l1', 'l2']:
        # Build and fit model
        scoring = 'neg_log_loss'
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        lr = sklearn.linear_model.LogisticRegressionCV(
                Cs=Cs, cv=cv(), penalty=penalty, class_weight=class_weight,
                solver=solver, scoring=scoring, refit=True)
        lr.fit(x_train['baselinefeats'], y_train['baselinefeats'])

        # Test model
        y_pred = lr.predict(x_test['baselinefeats'])

        # Compute metrics
        # Include average precision score (ap), which is similar to area under
        # precision-recall (pr_auc), but not interpolated; this implementation
        # might be more conservative
        score = lr.score(x_test['baselinefeats'], y_test['baselinefeats'])
        roc_fpr, roc_tpr, roc_thresholds = sklearn.metrics.roc_curve(
                y_test['baselinefeats'], y_pred, pos_label=1)
        pr_precision, pr_recall, pr_thresholds = sklearn.metrics.precision_recall_curve(
                y_test['baselinefeats'], y_pred, pos_label=1)
        roc_auc = sklearn.metrics.auc(roc_fpr, roc_tpr)
        pr_auc = sklearn.metrics.auc(pr_recall, pr_precision)
        ap = sklearn.metrics.average_precision_score(
                y_test['baselinefeats'], y_pred, pos_label=1)

        # Print metrics
        print('#'*20)
        print("Classification with {} logistic regression:".format(penalty))
        print("    best C_    = {}".format(lr.C_))
        print("    AUC-ROC    = {}".format(roc_auc))
        print("    AUC-PR     = {}".format(pr_auc))
        print("    Avg. prec. = {}".format(ap))
        print("    Score ({}) = {}".format(scoring, score))
        print('#'*20)

        metrics = {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ap': ap}
        metrics_for_models[penalty + '_logistic_regression'] = metrics

    return metrics_for_models


def regress(x_train, y_train, x_test, y_test,
        parsers,
        num_inner_splits=5,
        scoring_method='mse',
        models_to_use=None):
    """Perform regression.

    Test data is used for evaluating the model with the best choice of
    hyperparameters, after refitting across *all* the train data.

    Args:
        x_{train,validate,test}: input data for train/validate/test
        y_{train,validate,test}: output labels for train/validate/test
        num_inner_splits: number of splits for cross-validation
        parsers: parse_data parsers to use for splitting data
        scoring_method: method to use for scoring test results; 'mse' (mean
            squared error) or 'rho' (Spearman's rank correlation)
        models_to_use: list of models to test; if None, test all

    Returns:
        dict {model: {input feats: metrics on test data for best choice of
            hyperparameters for model}}
    """
    # Check models_to_use
    all_models = ['lr', 'l1_lr', 'l2_lr', 'l1l2_lr', 'gbt', 'rf', 'mlp', 'lstm']
    if models_to_use is None:
        models_to_use = all_models
    assert set(models_to_use).issubset(all_models)

    # Set the input feats to use for different models
    # Use the same choice for all models *except* lstm, which should be in a
    # series form where each time step corresponds to a position
    input_feats = {}
    for m in all_models:
        if m == 'lstm':
            input_feats[m] = ['onehot']
        else:
            input_feats[m] = ['onehot-flat', 'onehot-simple', 'handcrafted',
                    'combined']

    # With models, perform cross-validation to determine hyperparameters
    # Most of the built-in cross-validators find the best choice based on
    # R^2; some of them do not support a custom scoring function via a
    # `scoring=...` argument. So instead wrap the regression with a
    # GridSearchCV object, which does support a custom scoring metric. Use
    # spearman rank correlation coefficient for this.
    def cv(feats):
        return parsers[feats].split(x_train[feats], y_train[feats],
                num_inner_splits, stratify_by_pos=True, yield_indices=True)

    def rho_f(y, y_pred):
        rho, _ = scipy.stats.spearmanr(y, y_pred)
        return rho
    rho_scorer = sklearn.metrics.make_scorer(rho_f,
            greater_is_better=True)
    mse_scorer = sklearn.metrics.make_scorer(
            sklearn.metrics.mean_squared_error,
            greater_is_better=False)
    if scoring_method == 'mse':
        scorer = mse_scorer
    elif scoring_method == 'rho':
        scorer = rho_scorer
    else:
        raise ValueError("Unknown scoring method %s" % scoring_method)

    def fit_and_test_model(reg, model_desc, hyperparams, feats):
        """Fit and test model.

        Args:
            reg: built regression model
            model_desc: string describing model
            hyperparams: list [p] of hyperparameters where each p is a string
                and reg.p gives the value chosen by the hyperparameter search
            feats: input features type

        Returns:
            dict giving metrics for the best choice of model hyperparameters
        """
        # Fit model
        reg.fit(x_train[feats], y_train[feats])

        # Test model
        y_pred = reg.predict(x_test[feats])

        # Compute metrics
        mse = sklearn.metrics.mean_squared_error(y_test[feats], y_pred)
        mae = sklearn.metrics.mean_absolute_error(y_test[feats], y_pred)
        R2 = sklearn.metrics.r2_score(y_test[feats], y_pred)
        r, _ = scipy.stats.pearsonr(y_test[feats], y_pred)
        rho, _ = scipy.stats.spearmanr(y_test[feats], y_pred)

        # Note that R2 does not necessarily equal r^2 here. The value R2
        # is computed by definition of R^2 (1 minus (residual sum of
        # squares)/(total sum of squares)) from the true vs. predicted
        # values. This is why R2 can be negative: it can do an even worse
        # job with prediction than just predicting the mean. r is computed
        # by simple least squares regression between y_test and y_pred, and
        # finding the Pearson's r of this curve (since this is simple
        # linear regression, r^2 should be nonnegative). The value R2 measures
        # the goodness-of-fit of the specific linear correlation
        # y_pred = y_test, whereas r measures the correlation from the
        # regression (y_pred = m*y_test + b).

        # Print metrics
        print('#'*20)
        print("Regression with {}".format(model_desc))
        if type(hyperparams) is list:
            for p in hyperparams:
                print("    best {}    = {}".format(p, getattr(reg, p)))
        else:
            print("    best params = {}".format(hyperparams.best_params_))
        print("    MSE = {}".format(mse))
        print("    MAE = {}".format(mae))
        print("    R^2 = {}".format(R2))
        print("    r   = {}".format(r))
        print("    rho = {}".format(rho))
        print('#'*20)

        return {'mse': mse, 'mae': mae, 'r2': R2, 'r': r, 'rho': rho,
                '1_minus_rho': 1.0-rho}

    # Linear regression (no regularization)
    def lr(feats):
        reg = sklearn.linear_model.LinearRegression(copy_X=True)    # no CV because there are no hyperparameters
        return fit_and_test_model(reg, 'Linear regression',
                hyperparams=[], feats=feats)

    # Note:
    #  For below models, increasing `max_iter` or increasing `tol` can reduce
    #  the warning 'ConvergenceWarning: Objective did not converge.'

    # L1 linear regression
    def l1_lr(feats):
        reg = sklearn.linear_model.Lasso(max_iter=100000, tol=0.001, copy_X=True)
        reg_cv = random_search_cv('l1_lr', reg, cv(feats), scorer)
        return fit_and_test_model(reg_cv, 'L1 linear regression',
                hyperparams=reg_cv, feats=feats)

    # L2 linear regression
    def l2_lr(feats):
        reg = sklearn.linear_model.Ridge(max_iter=100000, tol=0.001, copy_X=True)
        reg_cv = random_search_cv('l2_lr', reg, cv(feats), scorer)
        return fit_and_test_model(reg_cv, 'L2 linear regression',
                hyperparams=reg_cv, feats=feats)

    # Elastic net (L1+L2 linear regression)
    # Recommendation for l1_ratio is to place more values close to 1 (lasso)
    # and fewer closer to 0 (ridge)
    # A note to explain some potential confusion in the choice of
    #  l1_ratio: Ridge might be better than Lasso according to rho, but
    #  l1_ratio could still be chosen to be high (close to Lasso)
    #  especially if Lasso/Ridge are close; in part, this could be because
    #  fit_and_test_model() prints values on a hold-out set, but chooses
    #  hyperparameters on splits of the train set
    def l1l2_lr(feats):
        reg = sklearn.linear_model.ElasticNet(max_iter=100000, tol=0.001, copy_X=True)
        reg_cv = random_search_cv('l1l2_lr', reg, cv(feats), scorer)
        return fit_and_test_model(reg_cv, 'L1+L2 linear regression',
                hyperparams=reg_cv, feats=feats)

    # Gradient-boosted regression trees
    def gbt(feats):
        reg = sklearn.ensemble.GradientBoostingRegressor(loss='ls', tol=0.001)
        reg_cv = random_search_cv('gbt', reg, cv(feats), scorer)
        return fit_and_test_model(reg_cv, 'Gradient Boosting regression',
                hyperparams=reg_cv, feats=feats)

    # Random forest regression
    def rf(feats):
        reg = sklearn.ensemble.RandomForestRegressor(criterion='mse')
        reg_cv = random_search_cv('rf', reg, cv(feats), scorer)
        return fit_and_test_model(reg_cv, 'Random forest regression',
                hyperparams=reg_cv, feats=feats)

    # MLP
    def mlp(feats):
        reg = fnn.MultilayerPerceptron(parsers[feats].context_nt)
        reg_cv = random_search_cv('mlp', reg, cv(feats), scorer)
        return fit_and_test_model(reg_cv, 'Multilayer perceptron',
                hyperparams=reg_cv, feats=feats)

    # LSTM
    def lstm(feats):
        reg = rnn.LSTM(parsers[feats].context_nt)
        reg_cv = random_search_cv('lstm', reg, cv(feats), scorer)
        return fit_and_test_model(reg_cv, 'LSTM',
                hyperparams=reg_cv, feats=feats)

    metrics_for_models = {}
    for model_name in models_to_use:
        metrics_for_models[model_name] = {}
        for feats in input_feats[model_name]:
            print(("Running and evaluating model '%s' with input feature '%s'") %
                    (model_name, feats))
            model_fn = locals()[model_name]
            metrics_for_models[model_name][feats] = model_fn(feats)

    return metrics_for_models


def nested_cross_validate(x, y, num_outer_splits,
        regression, parsers, regression_scoring_method=None,
        models_to_use=None):
    """Perform nested cross-validation to validate model and search.

    Args:
        x: input data
        y: labels
        num_outer_splits: number of folds in the outer cross-validation
            procedure
        regression: True if performing regression; False if classification
        parsers: parse_data parsers to use for splitting data
        regression_scoring_method: if regression, method to use for 
            evaluating a model ('mse' or 'rho')
        models_to_use: list of models to test; if None, test all

    Returns:
        list x where each x[i] is an output of regress() or classify() on
        an outer fold
    """
    fold_results = []
    i = 0
    outer_split_iters = []
    outer_split_iters_feats = []
    for k in parsers.keys():
        outer_split_iters += [parsers[k].split(x[k], y[k],
            num_splits=num_outer_splits, stratify_by_pos=True)]
        outer_split_iters_feats += [k]
    for xy in zip(*outer_split_iters):
        print('STARTING OUTER FOLD {} of {}'.format(i+1, num_outer_splits))

        x_train = {}
        y_train = {}
        x_validate = {}
        y_validate = {}
        for k, xy_k in zip(outer_split_iters_feats, xy):
            x_train[k], y_train[k], x_validate[k], y_validate[k] = xy_k
            assert len(x_train[k]) == len(y_train[k])
            assert len(x_validate[k]) == len(y_validate[k])
            print(('Input feats {}: There are n={} train points and n={} '
                'validation points').format(
                k, len(x_train[k]), len(x_validate[k])))

        # Search for hyperparameters on this outer fold of the data
        if regression:
            metrics_for_models = regress(x_train, y_train,
                    x_validate, y_validate,
                    parsers,
                    scoring_method=regression_scoring_method,
                    models_to_use=models_to_use)
        else:
            metrics_for_models = classify(x_train, y_train,
                    x_validate, y_validate, parsers)
        fold_results += [metrics_for_models]

        # Print metrics on this fold
        print('Results on fold {}'.format(i+1))
        print('  Metrics on validation data')
        for model in metrics_for_models.keys():
            print('    Model: {}'.format(model))
            for feats in metrics_for_models[model].keys():
                print('        Input feats: {}'.format(feats))
                for metric in metrics_for_models[model][feats].keys():
                    print('          {} = {}'.format(metric,
                        metrics_for_models[model][feats][metric]))

        print(('FINISHED OUTER FOLD {} of {}').format(i+1, num_outer_splits))
        i += 1

    return fold_results


def main():
    # Read arguments
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Read data multiple times, each with different types of features
    x_train = {}
    y_train = {}
    x_validate = {}
    y_validate = {}
    x_test = {}
    y_test = {}
    parsers = {}
    split_frac = (1.0 - args.test_split_frac, 0.0, args.test_split_frac)
    for feats in ['onehot', 'onehot-flat', 'onehot-simple', 'handcrafted',
            'combined']:
        if feats == 'onehot':
            # For parse_data, treat this as not constructing features for
            # baseline
            make_feats_for_baseline = None
        else:
            make_feats_for_baseline = feats

        data_parser = predictor.read_data(args,
                split_frac=split_frac,
                make_feats_for_baseline=make_feats_for_baseline)
        x_train[feats], y_train[feats] = data_parser.train_set()
        x_validate[feats], y_validate[feats] = data_parser.validate_set()
        x_test[feats], y_test[feats] = data_parser.test_set()
        parsers[feats] = data_parser

        # Convert column to 1D array
        y_train[feats] = y_train[feats].ravel()
        y_validate[feats] = y_validate[feats].ravel()
        y_test[feats] = y_test[feats].ravel()

    # Determine, based on the dataset, whether to do regression or
    # classification
    if args.dataset == 'cas13':
        if args.cas13_classify:
            regression = False
        else:
            regression = True

    if args.nested_cross_val:
        # Perform nested cross-validation

        if args.test_split_frac > 0:
            print(('WARNING: Performing nested cross-validation but there is '
                   'unused test data; it may make sense to set '
                   '--test-split-frac to 0'))

        fold_results = nested_cross_validate(x_train, y_train,
                args.nested_cross_val_outer_num_splits,
                regression,
                parsers,
                regression_scoring_method=args.regression_scoring_method,
                models_to_use=args.models_to_use)

        if args.nested_cross_val_out_tsv:
            # Write fold results to a tsv file
            if regression:
                metrics = ['mse', '1_minus_rho']
            else:
                metrics = ['roc_auc', 'pr_auc']
            with open(args.nested_cross_val_out_tsv, 'w') as fw:
                header = ['fold', 'model'] + metrics
                fw.write('\t'.join(header) + '\n')
                for fold in range(len(fold_results)):
                    metrics_for_models = fold_results[fold]
                    for model in metrics_for_models.keys():
                        for feats in metrics_for_models[model].keys():
                            row = [fold, model, feats]
                            for metric in metrics:
                                row += [metrics_for_models[model][feats][metric]]
                            fw.write('\t'.join(str(r) for r in row) + '\n')
    else:
        # Simply perform a hyperparameter search for each model
        if regression:
            regress(x_train, y_train, x_test, y_test,
                    parsers,
                    scoring_method=args.regression_scoring_method,
                    models_to_use=args.models_to_use)
        else:
            classify(x_train, y_train, x_test, y_test, parsers)


if __name__ == "__main__":
    main()
