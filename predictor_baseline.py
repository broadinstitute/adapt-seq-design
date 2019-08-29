"""Baselines for predicting guide sequence activity.
"""

import argparse

import parse_data
import predictor

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


def classify(x_train, y_train, x_validate, y_validate, x_test, y_test):
    """Perform classification.

    Args:
        x_{train,validate,test}: input data for train/validate/test
        y_{train,validate,test}: output labels for train/validate/test
    """
    # TODO implement other models, e.g., SVM

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
    def cv(num_splits=2):
        return parse_data.split(x_train, y_train, num_splits,
                stratify_by_pos=True, yield_indices=True)

    # It helps to provide an explicit function/callable as the scorer;
    # when using 'neg_log_loss' for 'scoring', it crashes because the
    # log_loss function expects a 'labels' argument when y_true contains
    # only one label

    for penalty in ['l1', 'l2']:
        # Build and fit model
        scoring = 'neg_log_loss'
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        lr = sklearn.linear_model.LogisticRegressionCV(
                Cs=Cs, cv=cv(), penalty=penalty, class_weight=class_weight,
                solver=solver, scoring=scoring, refit=True)
        lr.fit(x_train, y_train)

        # Test model
        y_pred = lr.predict(x_test)

        # Compute metrics
        # Include average precision score (ap), which is similar to area under
        # precision-recall (pr_auc), but not interpolated; this implementation
        # might be more conservative
        score = lr.score(x_test, y_test)
        roc_fpr, roc_tpr, roc_thresholds = sklearn.metrics.roc_curve(
                y_test, y_pred, pos_label=1)
        pr_precision, pr_recall, pr_thresholds = sklearn.metrics.precision_recall_curve(
                y_test, y_pred, pos_label=1)
        roc_auc = sklearn.metrics.auc(roc_fpr, roc_tpr)
        pr_auc = sklearn.metrics.auc(pr_recall, pr_precision)
        ap = sklearn.metrics.average_precision_score(
                y_test, y_pred, pos_label=1)

        # Print metrics
        print('#'*20)
        print("Classification with {} logistic regression:".format(penalty))
        print("    best C_    = {}".format(lr.C_))
        print("    AUC-ROC    = {}".format(roc_auc))
        print("    AUC-PR     = {}".format(pr_auc))
        print("    Avg. prec. = {}".format(ap))
        print("    Score ({}) = {}".format(scoring, score))
        print('#'*20)


def regress(x_train, y_train, x_validate, y_validate, x_test, y_test):
    """Perform regression.

    Args:
        x_{train,validate,test}: input data for train/validate/test
        y_{train,validate,test}: output labels for train/validate/test
    """
    # With models, perform built-in cross-validation to determine
    # hyperparameters

    def cv(num_splits=5):
        return parse_data.split(x_train, y_train, num_splits,
                stratify_by_pos=True, yield_indices=True)

    def fit_and_test_model(reg, model_desc, hyperparams):
        """Fit and test model.

        Args:
            reg: built regression model
            model_desc: string describing model
            hyperparams: list [p] of hyperparameters where each p is a string
                and reg.p gives the value chosen by the hyperparameter search
        """
        # Fit model
        reg.fit(x_train, y_train)

        # Test model
        y_pred = reg.predict(x_test)

        # Compute metrics
        mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
        R2 = sklearn.metrics.r2_score(y_test, y_pred)
        r, _ = scipy.stats.pearsonr(y_test, y_pred)
        rho, _ = scipy.stats.spearmanr(y_test, y_pred)

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

    # Linear regression (no regularization)
    reg = sklearn.linear_model.LinearRegression(copy_X=True)    # no CV because there are no hyperparameters
    fit_and_test_model(reg, 'Linear regression', [])

    # L1 linear regression
    alphas = np.logspace(-5, 5, num=100, base=10.0)
    reg = sklearn.linear_model.LassoCV(alphas=alphas, cv=cv(),
            max_iter=10000, copy_X=True)
    fit_and_test_model(reg, 'L1 linear regression', hyperparams=['alpha_'])

    # L2 linear regression
    alphas = np.logspace(-5, 5, num=100, base=10.0)
    reg = sklearn.linear_model.RidgeCV(alphas=alphas, cv=cv(),
            scoring='neg_mean_squared_error')
    fit_and_test_model(reg, 'L2 linear regression', hyperparams=['alpha_'])

    # Elastic net (L1+L2 linear regression)
    # Recommendation for l1_ratio is to place more values close to 1 (lasso)
    # and fewer closer to 0 (ridge)
    # Note that this optimizes MSE; e.g., L1 (lasso) may do better than L2
    # (ridge) according to rho, but if ridge has lower MSE than lasso, then
    # this will select a small l1_ratio (closer to ridge); moreover, the
    # results printed by fit_and_test_model() are on the test data, but
    # hyperparameters are chosen on splits of the train data (this might
    # explain some confusion in why a certain l1_ratio is chosen)
    l1_ratios = 1.0 - np.logspace(-5, 0, num=10, base=2.0)[::-1] + 2.0**(-5)
    alphas = np.logspace(-5, 5, num=100, base=10.0)
    reg = sklearn.linear_model.ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas,
            cv=cv(), max_iter=1000, copy_X=True)
    fit_and_test_model(reg, 'L1+L2 linear regression',
            hyperparams=['l1_ratio_', 'alpha_'])

    # Gradient-boosted regression trees
    # TODO maybe set 'scoring=' for GridSearchCV; the default will use r^2
    #   but it seems the above functions are based on LinearModelCV, which uses
    #   MSE
    # TODO more generally, make sure CV scoring is consistent with what's
    #   done in predictor_hyperparam_search; perhaps change
    #   predictor_hyperparam_search to use the usual predictor loss function
    #   (binary cross entropy for classification and MSE for regression) rather
    #   than AUC/r-spearman; or change these to optimize rho, e.g., by
    #   using GridSearchCV with a custom scoring function
    params = {
            'learning_rate': np.logspace(-2, 0, num=5, base=10.0),
            'n_estimators': [2**k for k in range(0, 9)],
            'min_samples_split': [2**k for k in range(1, 5)],
            'min_samples_leaf': [2**k for k in range(0, 5)],
            'max_depth': [2**k for k in range(1, 5)],
            'max_features': [None, 0.1, 'sqrt', 'log2']
    }
    reg = sklearn.ensemble.GradientBoostingRegressor(loss='ls')
    reg_cv = sklearn.model_selection.GridSearchCV(reg,
            param_grid=params, cv=cv(), refit=True, verbose=2)
    fit_and_test_model(reg_cv, 'Gradient Boosting regression',
            hyperparams=reg_cv)


def main():
    # Read arguments and data
    args = parse_args()
    set_seed(args.seed)
    split_frac = (1.0 - args.test_split_frac, 0.0, args.test_split_frac)
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test), x_test_pos = predictor.read_data(args, split_frac=split_frac, make_feats_for_baseline=True)

    # Convert column to 1D array
    y_train = y_train.ravel()
    y_validate = y_validate.ravel()
    y_test = y_test.ravel()

    # Determine, based on the dataset, whether to do regression or
    # classification
    if args.dataset == 'cas13':
        if args.cas13_classify:
            regression = False
        else:
            regression = True
    elif args.dataset == 'cas9' or args.dataset == 'simulated-cas13':
        regression = False

    if regression:
        regress(x_train, y_train, x_validate, y_validate,
                x_test, y_test)
    else:
        classify(x_train, y_train, x_validate, y_validate,
                x_test, y_test)


if __name__ == "__main__":
    main()
