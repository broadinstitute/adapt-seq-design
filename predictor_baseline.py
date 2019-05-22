"""Logistic regression for a guide sequence, to serve as a baseline.

This is implemented for classifying Cas9 activity.
"""

import argparse

import parse_data

import numpy as np
import tensorflow as tf


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--subset',
        choices=['guide-mismatch-and-good-pam', 'guide-match'],
        help=("Use a subset of the data. See parse_data module for "
              "descriptions of the subsets. To use all data, do not set."))
parser.add_argument('--context-nt',
        type=int,
        default=20,
        help=("nt of target sequence context to include alongside each "
              "guide"))
parser.add_argument('--l1-factor',
        type=float,
        default=0.001,
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
data_parser = parse_data.Doench2016Cas9ActivityParser(
        subset=args.subset,
        context_nt=args.context_nt,
        split=(0.6, 0.2, 0.2),
        shuffle_seed=args.seed)
data_parser.read()

x_train, y_train = data_parser.train_set()
x_validate, y_validate = data_parser.validate_set()
x_test, y_test = data_parser.test_set()

# Print the size of each data set
data_sizes = 'DATA SIZES - Train: {}, Validate: {}, Test: {}'
print(data_sizes.format(len(x_train), len(x_validate), len(x_test)))

# Print the fraction of the training data points that are in each class
classes = set(tuple(y) for y in y_train)
for c in classes:
    num_c = sum(1 for y in y_train if tuple(y) == c)
    frac_c = float(num_c) / len(y_train)
    frac_c_msg = 'Fraction of train data in class {}: {}'
    print(frac_c_msg.format(c, frac_c))


# Construct a model for logistic regression; this can simply be a dense
# layer with one output dimension
# First, though, flatten the input while preseving the batch axis
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
regularizer = tf.keras.regularizers.l1(args.l1_factor)
model.add(tf.keras.layers.Dense(
    1, activation='sigmoid', kernel_regularizer=regularizer))

# Define metrics for evaluation
accuracy_metric = tf.keras.metrics.BinaryAccuracy('binary_accuracy')
auc_roc_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', name='auc_roc')
auc_pr_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='PR', name='auc_pr')

# Compile and fit the model
model.compile(optimizer='Adam', loss='binary_crossentropy',
        metrics=[accuracy_metric, auc_roc_metric, auc_pr_metric])
model.fit(x_train, y_train, epochs=50,
    validation_data=(x_validate, y_validate))

# Evaluate model on test data
print('TEST:')
model.evaluate(x_test, y_test)
