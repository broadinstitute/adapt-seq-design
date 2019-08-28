"""Baselines for predicting guide sequence activity.
"""

import argparse

import parse_data
import predictor

import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.metrics
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
    y_train_labels = [y_train[i][0] for i in range(len(y_train))]
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
    # and specifiy a function/callable for the 'scoring' argument of
    # LogisticRegressionCV, which would be the output of
    # sklearn.metrics.make_scorer and wraps around sklearn's log_loss
    # function, passing in the 'labels' argument to it (one issue would be
    # having to also determine a 'sample_weight' argument the log_loss
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
        print("    best C     = {}".format(lr.C_))
        print("    AUC-ROC    = {}".format(roc_auc))
        print("    AUC-PR     = {}".format(pr_auc))
        print("    Avg. prec. = {}".format(ap))
        print("    Score ({}) = {}".format(scoring, score))
        print('#'*20)


def main():
    # Read arguments and data
    args = parse_args()
    set_seed(args.seed)
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test), x_test_pos = predictor.read_data(args, make_feats_for_baseline=True)

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
        pass
    else:
        classify(x_train, y_train, x_validate, y_validate,
                x_test, y_test)


if __name__ == "__main__":
    main()
