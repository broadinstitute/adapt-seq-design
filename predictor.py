"""Simple convnet (CNN) for a guide sequence.

This is implemented for classifying Cas9 activity.

This has one convolutional layer, a max pool layer, 2 fully connected hidden
layers, and fully connected output layer passed through sigmoid.
"""

import argparse

import parse_data

import numpy as np
import tensorflow as tf


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
parser.add_argument('--conv-num-filters',
        type=int,
        default=20,
        help=("Number of convolutional filters (i.e., output channels) "
              "in the first layer"))
parser.add_argument('--conv-filter-width',
        type=int,
        nargs='+',
        help=("Width of the convolutional filter (nt) (if "
              "--parallel-filter-model is set, then this is a collection "
              "of widths)"))
parser.add_argument('--pool-window-width',
        type=int,
        default=2,
        help=("Width of the pooling window"))
parser.add_argument('--fully-connected-dim',
        type=int,
        default=20,
        help=("Dimension of each fully connected layer (i.e., of its "
              "output space"))
parser.add_argument('--deeper-conv-model',
        action='store_true',
        help=("Use a model with 3 convolutional layers to reduce spatial "
              "dependence before the fully connected layers; note that "
              "this uses hard-coded hyperparameters that assume default "
              "values for the above arguments and context-nt==20"))
parser.add_argument('--parallel-filter-model',
        action='store_true',
        help=("Use a model with multiple filter widths in the first "
              "convolutional layer; the number of widths and value for "
              "each width is given by --conv-filter-width"))
parser.add_argument('--pool-strategy',
        choices=['max', 'avg', 'max-and-avg'],
        help=("For pooling, 'max' only does max pooling; 'avg' only does "
              "average pooling; 'max-and-avg' does both and concatenates. "
              "Only implemented for --parallel-filter-model."))
parser.add_argument('--dropout-rate',
        type=float,
        default=0.25,
        help=("Rate of dropout in between the 2 fully connected layers"))
parser.add_argument('--l2-factor',
        type=float,
        default=0,
        help=("L2 regularization factor. This is applied to weights "
              "(kernal_regularizer). Note that this does not regularize "
              "bias of activity."))
parser.add_argument('--epochs',
        type=int,
        default=50,
        help=("Number of training epochs"))
parser.add_argument('--seed',
        type=int,
        default=1,
        help=("Random seed"))
args = parser.parse_args()

# Check arguments and set defaults
if args.deeper_conv_model and args.parallel_filter_model:
    raise Exception("Cannot set two kinds of models simultaneously")

if not args.parallel_filter_model:
    if len(args.conv_filter_width) > 1:
        raise Exception(("Cannot set multiple widths (--conv-filter-width) without "
            "--parallel-filter-model"))
    args.conv_filter_width = args.conv_filter_width[0]
if args.conv_filter_width is None:
    if args.parallel_filter_model:
        args.conv_filter_width = [1, 2, 3, 4]   # default
    else:
        args.conv_filter_width = 2  # default

if args.pool_strategy is not None and not args.parallel_filter_model:
    raise Exception(("--pool-strategy is only implemented with "
        "--parallel-filter-model"))
if args.pool_strategy is None:
    args.pool_strategy = 'max'

# Print the arguments provided
print(args)


# Don't use a random seed for tensorflow
tf.random.set_seed(args.seed)

#####################################################################
# Read and batch input/output
#####################################################################
# Read data
if args.simulate_cas13:
    parser_class = parse_data.Cas13SimulatedData
else:
    parser_class = parse_data.Doench2016Cas9ActivityParser
data_parser = parser_class(
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

# Create datasets and batch data
batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)
validate_ds = tf.data.Dataset.from_tensor_slices(
        (x_validate, y_validate)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)
#####################################################################
#####################################################################


#####################################################################
# Construct a convolutional network model
#####################################################################

class Cas9CNN(tf.keras.Model):
    def __init__(self):
        super(Cas9CNN, self).__init__()

        # Construct the convolutional layer
        # Do not pad the input (`padding='valid'`) because all input
        # sequences should be the same length
        conv_layer_num_filters = args.conv_num_filters # ie, num of output channels
        conv_layer_filter_width = args.conv_filter_width
        self.conv = tf.keras.layers.Conv1D(
                conv_layer_num_filters,
                conv_layer_filter_width,
                strides=1,  # stride by 1
                padding='valid',
                activation='relu',
                name='conv')

        # Add a batch normalization layer
        # It should not matter whether this comes before or after the
        # pool layer, as long as it is after the conv layer
        # This is applied after the relu activation of the conv layer; the
        # original batch norm applies batch normalization before the
        # activation function, but more recent work seems to apply it
        # after activation
        self.batchnorm = tf.keras.layers.BatchNormalization()

        # Construct a pooling layer
        # Take the maximum over a window of width pool_window, for
        # each output channel of the conv layer (and, of course, for each batch)
        # Stride by max_pool_stride; note that if
        # max_pool_stride = pool_window, then the max pooling
        # windows are non-overlapping
        max_pool_window = args.pool_window_width
        max_pool_stride = int(max_pool_window / 2)
        self.maxpool = tf.keras.layers.MaxPooling1D(
                pool_size=max_pool_window,
                strides=max_pool_stride,
                name='maxpool')

        # Flatten the pooling output from above while preserving
        # the batch axis
        self.flatten = tf.keras.layers.Flatten()

        # Construct 2 fully connected hidden layers
        # Insert dropout between them for regularization
        # Set the dimension of each fully connected layer (i.e., dimension
        # of the output space) to fc_hidden_dim
        fc_hidden_dim = args.fully_connected_dim
        self.fc_1 = tf.keras.layers.Dense(
                fc_hidden_dim,
                activation='relu',
                name='fc_1')
        self.dropout = tf.keras.layers.Dropout(args.dropout_rate)
        self.fc_2 = tf.keras.layers.Dense(
                fc_hidden_dim,
                activation='relu',
                name='fc_2')

        # Construct the final layer (fully connected)
        # Set the dimension of this final fully connected layer to
        # fc_final_dim
        fc_final_dim = 1
        self.fc_final = tf.keras.layers.Dense(
                fc_final_dim,
                activation='sigmoid',
                name='fc_final')

        # Regularize weights on each layer
        l2_regularizer = tf.keras.regularizers.l2(args.l2_factor)
        for layer in self.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2_regularizer

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.batchnorm(x, training=training)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.dropout(x, training=training)
        x = self.fc_2(x)
        return self.fc_final(x)


class Cas9CNNWithDeeperConv(tf.keras.Model):
    def __init__(self):
        super(Cas9CNNWithDeeperConv, self).__init__()

        # Construct the convolutional layer
        conv1_layer_num_filters = args.conv_num_filters # ie, num of output channels
        conv1_layer_filter_width = args.conv_filter_width
        self.conv1 = tf.keras.layers.Conv1D(
                conv1_layer_num_filters,
                conv1_layer_filter_width,
                strides=1,  # stride by 1
                padding='valid',
                activation='relu',
                name='conv1')

        # Construct a pooling layer
        max_pool_window = args.pool_window_width
        max_pool_stride = int(max_pool_window / 2)
        self.maxpool = tf.keras.layers.MaxPooling1D(
                pool_size=max_pool_window,
                strides=max_pool_stride,
                name='maxpool1')

        # Add a batch normalization layer
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # Add another convolutional layer
        conv2_layer_num_filters = 20
        conv2_layer_filter_width = 6
        conv2_layer_filter_stride = 3
        self.conv2 = tf.keras.layers.Conv1D(
                conv2_layer_num_filters,
                conv2_layer_filter_width,
                strides=conv2_layer_filter_stride,
                padding='valid',
                activation='relu',
                name='conv2')

        # Construct another pooling layer
        self.maxpool2 = tf.keras.layers.MaxPooling1D(
                pool_size=max_pool_window,
                strides=max_pool_stride,
                name='maxpool2')

        # Add another batch normalization layer
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        # Add a final convolutional layer
        conv3_layer_num_filters = 20
        conv3_layer_filter_width = 4
        conv3_layer_filter_stride = 2
        self.conv3 = tf.keras.layers.Conv1D(
                conv3_layer_num_filters,
                conv3_layer_filter_width,
                strides=conv2_layer_filter_stride,
                padding='valid',
                activation='relu',
                name='conv3')

        # Flatten the pooling output from above while preserving
        # the batch axis
        self.flatten = tf.keras.layers.Flatten()

        # Construct 2 fully connected hidden layers
        # Insert dropout between them for regularization
        fc_hidden_dim = args.fully_connected_dim
        self.fc_1 = tf.keras.layers.Dense(
                fc_hidden_dim,
                activation='relu',
                name='fc_1')
        self.dropout = tf.keras.layers.Dropout(args.dropout_rate)
        self.fc_2 = tf.keras.layers.Dense(
                fc_hidden_dim,
                activation='relu',
                name='fc_2')

        # Construct the final layer (fully connected)
        fc_final_dim = 1
        self.fc_final = tf.keras.layers.Dense(
                fc_final_dim,
                activation='sigmoid',
                name='fc_final')

        # Regularize weights on each layer
        l2_regularizer = tf.keras.regularizers.l2(args.l2_factor)
        for layer in self.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2_regularizer

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x, training=training)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x, training=training)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.dropout(x, training=training)
        x = self.fc_2(x)
        return self.fc_final(x)


class Cas9CNNWithParallelFilters(tf.keras.Model):
    def __init__(self):
        super(Cas9CNNWithParallelFilters, self).__init__()

        # Construct groups, where each consists of a convolutional
        # layer with a particular width, a batch normalization layer, and
        # a pooling layer
        # Store these in separate lists, rather than as tuples in a single
        # list, so that they get stored in self.layers
        self.convs = []
        self.batchnorms = []
        self.pools = []
        self.pools_2 = []
        for filter_width in args.conv_filter_width:
            # Construct the convolutional layer
            conv_layer_num_filters = args.conv_num_filters # ie, num of output channels
            conv = tf.keras.layers.Conv1D(
                    conv_layer_num_filters,
                    filter_width,
                    strides=1,  # stride by 1
                    padding='valid',
                    activation='relu',
                    name='conv_w' + str(filter_width))
            # Note that the total number of filters in this layer will be
            # 4*conv_layer_num_filters since there are 4 groups

            # Add a batch normalization layer
            batchnorm = tf.keras.layers.BatchNormalization(
                    name='conv_w' + str(filter_width) + '_batchnorm')

            # Add a pooling layer
            pool_window_width = args.pool_window_width
            pool_stride = max(1, int(pool_window_width / 2))
            maxpool = tf.keras.layers.MaxPooling1D(
                    pool_size=pool_window_width,
                    strides=pool_stride,
                    name='conv_w' + str(filter_width) + '_maxpool')
            avgpool = tf.keras.layers.AveragePooling1D(
                    pool_size=pool_window_width,
                    strides=pool_stride,
                    name='conv_w' + str(filter_width) + '_avgpool')

            self.convs += [conv]
            self.batchnorms += [batchnorm]

            # If only using 1 pool, store this in self.pools
            # If using 2 pools, store one in self.pools and the other in
            # self.pools_2, and create self.pool_merge to concatenate the
            # outputs of these 2 pools
            if args.pool_strategy == 'max':
                self.pools += [maxpool]
                self.pools_2 += [None]
            elif args.pool_strategy == 'avg':
                self.pools += [avgpool]
                self.pools_2 += [None]
            elif args.pool_strategy == 'max-and-avg':
                self.pools += [maxpool]
                self.pools_2 += [avgpool]
                self.pool_merge = tf.keras.layers.Concatenate(axis=1)
            else:
                raise Exception(("Unknown --pool-strategy"))

        # Merge the outputs of the groups
        # The concatenation needs to happen along an axis, and all
        # inputs must have the same dimension along each axis except
        # for the concat axis
        # The axes are: (batch size, width, filters)
        # The concatenation can happen along either the width axis (1)
        # or the filters axis (2); it should not make a difference
        # Because the filters axis will all be the same dimension
        # (conv_layer_num_filters) but the width axis may be slightly
        # different (as each filter/kernel has a different width, so
        # the number that span the input may differ slightly), let's
        # concatenate along the width axis (axis=1)
        # Only create the merge layer if it is needed (i.e., there are
        # multiple filter widths)
        if len(args.conv_filter_width) > 1:
            self.merge = tf.keras.layers.Concatenate(axis=1)

        # Flatten the pooling output from above while preserving
        # the batch axis
        self.flatten = tf.keras.layers.Flatten()

        # Construct 2 fully connected hidden layers
        # Insert dropout between them for regularization
        fc_hidden_dim = args.fully_connected_dim
        self.fc_1 = tf.keras.layers.Dense(
                fc_hidden_dim,
                activation='relu',
                name='fc_1')
        self.dropout = tf.keras.layers.Dropout(args.dropout_rate)
        self.fc_2 = tf.keras.layers.Dense(
                fc_hidden_dim,
                activation='relu',
                name='fc_2')

        # Construct the final layer (fully connected)
        fc_final_dim = 1
        self.fc_final = tf.keras.layers.Dense(
                fc_final_dim,
                activation='sigmoid',
                name='fc_final')

        # Regularize weights on each layer
        l2_regularizer = tf.keras.regularizers.l2(args.l2_factor)
        for layer in self.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2_regularizer

    def call(self, x, training=False):
        group_outputs = []
        for conv, batchnorm, pool_1, pool_2 in zip(self.convs, self.batchnorms,
                self.pools, self.pools_2):
            group_x = conv(x)
            group_x = batchnorm(group_x, training=training)
            if pool_2 is None:
                group_x = pool_1(group_x)
            else:
                group_x_1 = pool_1(group_x)
                group_x_2 = pool_2(group_x)
                group_x = self.pool_merge([group_x_1, group_x_2])
            group_outputs += [group_x]
        if len(group_outputs) == 1:
            # Only 1 filter width; cannot merge across 1 input
            x = group_outputs[0]
        else:
            x = self.merge(group_outputs)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.dropout(x, training=training)
        x = self.fc_2(x)
        return self.fc_final(x)


if args.deeper_conv_model:
    model = Cas9CNNWithDeeperConv()
elif args.parallel_filter_model:
    model = Cas9CNNWithParallelFilters()
else:
    model = Cas9CNN()

# Print a model summary
model.build(x_train.shape)
print(model.summary())
#####################################################################
#####################################################################

#####################################################################
# Perform training and testing
#####################################################################
# Setup cross-entropy as the loss function
# This expects sigmoids (values in [0,1]) as the output; it will
# transform back to logits (not bounded betweem 0 and 1) before
# calling tf.nn.sigmoid_cross_entropy_with_logits
bce_per_sample = tf.keras.losses.BinaryCrossentropy()

# When outputting loss, take the mean across the samples from each batch
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
validate_loss_metric = tf.keras.metrics.Mean(name='validate_loss')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

# Also report on the accuracy and AUC for each epoch (each metric is updated
# with data from each batch, and computed using data from all batches)
train_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
train_auc_roc_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', name='train_auc_roc')
train_auc_pr_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='PR', name='train_auc_pr')
validate_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='validate_accuracy')
validate_auc_roc_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', name='validate_auc_roc')
validate_auc_pr_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='PR', name='validate_auc_pr')
test_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
test_auc_roc_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', name='test_auc_roc')
test_auc_pr_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='PR', name='test_auc_pr')

# Define an optimizer
optimizer = tf.keras.optimizers.Adam()

# Train the model using GradientTape; this is called on each batch
@tf.function
def train_step(seqs, labels):
    with tf.GradientTape() as tape:
        # Compute predictions and loss
        # Pass along `training=True` so that this can be given to
        # the batchnorm and dropout layers; an alternative to passing
        # it along would be to use `tf.keras.backend.set_learning_phase(1)`
        # to set the training phase
        predictions = model(seqs, training=True)
        prediction_loss = bce_per_sample(labels, predictions)
        # Add the regularization losses
        regularization_loss = tf.add_n(model.losses)
        loss = prediction_loss + regularization_loss
    # Compute gradients and opitmize parameters
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Record metrics
    train_loss_metric(loss)
    train_accuracy_metric(labels, predictions)
    train_auc_roc_metric(labels, predictions)
    train_auc_pr_metric(labels, predictions)

# Define functions for computing validation and test metrics; these are
# called on each batch
@tf.function
def validate_step(seqs, labels):
    # Compute predictions and loss
    predictions = model(seqs, training=False)
    prediction_loss = bce_per_sample(labels, predictions)
    regularization_loss = tf.add_n(model.losses)
    loss = prediction_loss + regularization_loss
    # Record metrics
    validate_loss_metric(loss)
    validate_accuracy_metric(labels, predictions)
    validate_auc_roc_metric(labels, predictions)
    validate_auc_pr_metric(labels, predictions)
@tf.function
def test_step(seqs, labels):
    # Compute predictions and loss
    predictions = model(seqs, training=False)
    prediction_loss = bce_per_sample(labels, predictions)
    regularization_loss = tf.add_n(model.losses)
    loss = prediction_loss + regularization_loss
    # Record metrics
    test_loss_metric(loss)
    test_accuracy_metric(labels, predictions)
    test_auc_roc_metric(labels, predictions)
    test_auc_pr_metric(labels, predictions)

# Train (and validate) for each epoch
for epoch in range(args.epochs):
    # Train on each batch
    for seqs, labels in train_ds:
        train_step(seqs, labels)

    # Validate on each batch
    # Note that we could run the validation data through the model all
    # at once (not batched), but batching may help with memory usage by
    # reducing how much data is run through the network at once
    for seqs, labels in validate_ds:
        validate_step(seqs, labels)

    # Log the metrics from this epoch
    print('EPOCH {}'.format(epoch+1))
    print('  Train metrics:')
    print('    Loss: {}'.format(train_loss_metric.result()))
    print('    Accuracy: {}'.format(train_accuracy_metric.result()))
    print('    AUC-ROC: {}'.format(train_auc_roc_metric.result()))
    print('    AUC-PR: {}'.format(train_auc_pr_metric.result()))
    print('  Validate metrics:')
    print('    Loss: {}'.format(validate_loss_metric.result()))
    print('    Accuracy: {}'.format(validate_accuracy_metric.result()))
    print('    AUC-ROC: {}'.format(validate_auc_roc_metric.result()))
    print('    AUC-PR: {}'.format(validate_auc_pr_metric.result()))

    # Reset metric states so they are not cumulative over epochs
    train_loss_metric.reset_states()
    train_accuracy_metric.reset_states()
    train_auc_roc_metric.reset_states()
    train_auc_pr_metric.reset_states()
    validate_loss_metric.reset_states()
    validate_accuracy_metric.reset_states()
    validate_auc_roc_metric.reset_states()
    validate_auc_pr_metric.reset_states()


# Test the model
for seqs, labels in test_ds:
    test_step(seqs, labels)
print('DONE')
print('  Test metrics:')
print('    Loss: {}'.format(test_loss_metric.result()))
print('    Accuracy: {}'.format(test_accuracy_metric.result()))
print('    AUC-ROC: {}'.format(test_auc_roc_metric.result()))
print('    AUC-PR: {}'.format(test_auc_pr_metric.result()))
test_loss_metric.reset_states()
test_accuracy_metric.reset_states()
test_auc_roc_metric.reset_states()
test_auc_pr_metric.reset_states()
#####################################################################
#####################################################################
