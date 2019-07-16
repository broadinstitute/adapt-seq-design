"""Baseline for predicting guide sequence activity.

This uses logistic regression for classification and linear regression for
regression.
"""

import argparse

import parse_data
import predictor

import numpy as np
import sklearn
import tensorflow as tf


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
        choices=['exp', 'pos', 'neg'],
        help=("Use a subset of the Cas13 data. See parse_data module "
              "for descriptions of the subsets. To use all data, do not "
              "set."))
parser.add_argument('--cas13-classify',
        action='store_true',
        help=("If set, only classify Cas13 activity into inactive/active"))
parser.add_argument('--cas13-regress-only-on-active',
        action='store_true',
        help=("If set, perform regression for Cas13 data only on the "
              "active class"))
parser.add_argument('--context-nt',
        type=int,
        default=20,
        help=("nt of target sequence context to include alongside each "
              "guide"))
parser.add_argument('--l1-factor',
        type=float,
        default=0.0001,
        help=("L1 regularization factor"))
parser.add_argument('--seed',
        type=int,
        default=1,
        help=("Random seed"))
args = parser.parse_args()

# Print the arguments provided
print(args)


# Don't use a random seed for tensorflow
tf.random.set_seed(args.seed)


# Read data
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
        split=(0.6, 0.1, 0.3),
        shuffle_seed=args.seed,
        stratify_by_pos=True)
if args.dataset == 'cas13':
    classify_activity = args.cas13_classify
    regress_only_on_active = args.cas13_regress_only_on_active
    data_parser.set_activity_mode(
            classify_activity, regress_only_on_active)
    data_parser.set_make_feats_for_baseline()
data_parser.read()

x_train, y_train = data_parser.train_set()
x_validate, y_validate = data_parser.validate_set()
x_test, y_test = data_parser.test_set()

# Print the size of each data set
data_sizes = 'DATA SIZES - Train: {}, Validate: {}, Test: {}'
print(data_sizes.format(len(x_train), len(x_validate), len(x_test)))

if regression:
    # Print the mean outputs
    print('Mean train output: {}'.format(np.mean(y_train)))
else:
    # Print the fraction of the training data points that are in each class
    classes = set(tuple(y) for y in y_train)
    for c in classes:
        num_c = sum(1 for y in y_train if tuple(y) == c)
        frac_c = float(num_c) / len(y_train)
        frac_c_msg = 'Fraction of train data in class {}: {}'
        print(frac_c_msg.format(c, frac_c))
    for c in classes:
        num_c = sum(1 for y in y_validate if tuple(y) == c)
        frac_c = float(num_c) / len(y_validate)
        frac_c_msg = 'Fraction of validate data in class {}: {}'
        print(frac_c_msg.format(c, frac_c))
    for c in classes:
        num_c = sum(1 for y in y_test if tuple(y) == c)
        frac_c = float(num_c) / len(y_test)
        frac_c_msg = 'Fraction of test data in class {}: {}'
        print(frac_c_msg.format(c, frac_c))


# Construct a model for logistic regression; this can simply be a dense
# layer with one output dimension
# First, though, flatten the input while preseving the batch axis
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
regularizer = tf.keras.regularizers.l1(args.l1_factor)
if regression:
    activation_fn = 'linear'
else:
    activation_fn = 'sigmoid'
model.add(tf.keras.layers.Dense(
    1, activation=activation_fn, kernel_regularizer=regularizer))

# Define metrics for evaluation
if regression:
    mse_metric = tf.keras.metrics.MeanSquaredError('mse')
    mae_metric = tf.keras.metrics.MeanAbsoluteError('mae')
    mape_metric = tf.keras.metrics.MeanAbsolutePercentageError('mape')
    metrics = [mse_metric, mae_metric, mape_metric]
    loss_fn = 'mean_squared_error'
    class_weight = None
else:
    accuracy_metric = tf.keras.metrics.BinaryAccuracy('binary_accuracy')
    auc_roc_metric = tf.keras.metrics.AUC(
            num_thresholds=200, curve='ROC', name='auc_roc')
    auc_pr_metric = tf.keras.metrics.AUC(
            num_thresholds=200, curve='PR', name='auc_pr')
    metrics = [accuracy_metric, auc_roc_metric, auc_pr_metric]
    loss_fn = 'binary_crossentropy'

    # Determine class weights
    y_train_labels = [y_train[i][0] for i in range(len(y_train))]
    class_weight = sklearn.utils.class_weight.compute_class_weight(
            'balanced', sorted(np.unique(y_train_labels)), y_train_labels)
    class_weight = list(class_weight)
    class_weight = {i: weight for i, weight in enumerate(class_weight)}

# Compile and fit the model
model.compile(optimizer='Adam', loss=loss_fn,
        metrics=metrics)
model.fit(x_train, y_train, epochs=100, class_weight=class_weight,
    validation_data=(x_validate, y_validate))

# Evaluate model on test data
print('TEST:')
model.evaluate(x_test, y_test)

test_predictions = model.predict_on_batch(x_test)
r_pearson = predictor.pearson_corr(y_test, test_predictions)
r_spearman = predictor.spearman_corr(y_test, test_predictions)
print('Pearson corr: {}, Spearman corr: {}'.format(r_pearson, r_spearman))
