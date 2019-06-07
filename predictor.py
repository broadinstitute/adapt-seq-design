"""Simple convnet (CNN) for a guide sequence.

This is implemented for classifying Cas9 activity.
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
        default=[2],
        help=("Width of the convolutional filter (nt) (or multiple widths "
              "to perform parallel convolutions)"))
parser.add_argument('--pool-window-width',
        type=int,
        default=2,
        help=("Width of the pooling window"))
parser.add_argument('--fully-connected-dim',
        type=int,
        nargs='+',
        default=[20],
        help=("Dimension of each fully connected layer (i.e., of its "
              "output space); specify multiple dimensions for multiple "
              "fully connected layers"))
parser.add_argument('--pool-strategy',
        choices=['max', 'avg', 'max-and-avg'],
        default='max',
        help=("For pooling, 'max' only does max pooling; 'avg' only does "
              "average pooling; 'max-and-avg' does both and concatenates."))
parser.add_argument('--locally-connected-width',
        type=int,
        nargs='+',
        help=("If set, width (kernel size) of the locally connected layer. "
              "Use multiple widths to have parallel locally connected layers "
              "that get concatenated. If not set, do use have a locally "
              "connected layer."))
parser.add_argument('--locally-connected-dim',
        type=int,
        default=1,
        help=("Dimension of each locally connected layer (i.e., of its "
              "output space)"))
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
parser.add_argument('--max-num-epochs',
        type=int,
        default=1000,
        help=("Maximum number of training epochs (this employs early "
              "stopping)"))
parser.add_argument('--seed',
        type=int,
        default=1,
        help=("Random seed"))
args = parser.parse_args()

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
        self.lcs = []
        for filter_width in args.conv_filter_width:
            # Construct the convolutional layer
            # Do not pad the input (`padding='valid'`) because all input
            # sequences should be the same length
            conv_layer_num_filters = args.conv_num_filters # ie, num of output channels
            conv = tf.keras.layers.Conv1D(
                    conv_layer_num_filters,
                    filter_width,
                    strides=1,  # stride by 1
                    padding='valid',
                    activation='relu',
                    name='group_w' + str(filter_width) + '_conv')
            # Note that the total number of filters in this layer will be
            # len(conv_filter_width)*conv_layer_num_filters since there are
            # len(conv_filter_width) groups

            # Add a batch normalization layer
            # It should not matter whether this comes before or after the
            # pool layer, as long as it is after the conv layer
            # This is applied after the relu activation of the conv layer; the
            # original batch norm applies batch normalization before the
            # activation function, but more recent work seems to apply it
            # after activation
            batchnorm = tf.keras.layers.BatchNormalization(
                    name='group_w' + str(filter_width) + '_batchnorm')

            # Add a pooling layer
            # Pool over a window of width pool_window, for
            # each output channel of the conv layer (and, of course, for each batch)
            # Stride by pool_stride; note that if  pool_stride = pool_window,
            # then the pooling windows are non-overlapping
            pool_window_width = args.pool_window_width
            pool_stride = max(1, int(pool_window_width / 2))
            maxpool = tf.keras.layers.MaxPooling1D(
                    pool_size=pool_window_width,
                    strides=pool_stride,
                    name='group_w' + str(filter_width) + '_maxpool')
            avgpool = tf.keras.layers.AveragePooling1D(
                    pool_size=pool_window_width,
                    strides=pool_stride,
                    name='group_w' + str(filter_width) + '_avgpool')

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
            else:
                raise Exception(("Unknown --pool-strategy"))

            # Setup locally connected layers (if set)
            # Use one for each convolution filter grouping (if applied after
            # concatenating the groups, then the position-dependence may be
            # less meaningful because a single locally connected neuron may
            # be connected across two different groups)
            # This layer can be useful in this application because (unlike
            # convolution layers) it explicitly models the position-dependence --
            # i.e., weights can differ across a guide and across the sequence
            # context. Moreover, it can help collapse the different convolutional
            # filters down to a smaller number of values (dimensions) at
            # each position, effectively serving as a dimensionality reduction
            # before the fully connected layers.
            if args.locally_connected_width is not None:
                lcs_for_conv = []
                locally_connected_dim = args.locally_connected_dim
                for i, lc_width in enumerate(args.locally_connected_width):
                    # Stride by 1/2 the width
                    stride = max(1, int(lc_width / 2))
                    lc = tf.keras.layers.LocallyConnected1D(
                        locally_connected_dim,
                        lc_width,
                        strides=stride,
                        activation='relu',
                        name='group_w' + str(filter_width) + '_lc_w' + str(lc_width))
                    lcs_for_conv += [lc]
                self.lcs += [lcs_for_conv]
            else:
                self.lcs += [None]

        if args.pool_strategy == 'max-and-avg':
            # Setup layer to concatenate each max/avg pooling in each group
            self.pool_merge = tf.keras.layers.Concatenate(
                    axis=1,
                    name='merge_pool')

        if args.locally_connected_width is not None:
            if len(args.locally_connected_width) > 1:
                # Setup layer to concatenate the locally connected layers
                # in each group
                self.lc_merge = tf.keras.layers.Concatenate(
                        axis=1,
                        name='merge_lc')

        # Merge the outputs of the groups
        # The concatenation needs to happen along an axis, and all
        # inputs must have the same dimension along each axis except
        # for the concat axis
        # The axes are: (batch size, width, filters)
        # The concatenation can happen along either the width axis (1)
        # or the filters axis (2); it should not make a difference
        # Because the filters axis will all be the same dimension
        # (conv_layer_num_filters or, if there are locally connected layers,
        # then locally_connected_dim) but the width axis may be slightly
        # different (as each filter/kernel has a different width, so
        # the number that span the input may differ slightly), let's
        # concatenate along the width axis (axis=1)
        # Only create the merge layer if it is needed (i.e., there are
        # multiple filter widths)
        if len(args.conv_filter_width) > 1:
            self.merge = tf.keras.layers.Concatenate(
                    axis=1,
                    name='merge_groups')

        # Flatten the pooling output from above while preserving
        # the batch axis
        self.flatten = tf.keras.layers.Flatten()

        # Setup fully connected layers
        # Insert dropout before each of them for regularization
        # Set the dimension of each fully connected layer (i.e., dimension
        # of the output space) to args.fully_connected_dim[i]
        self.dropouts = []
        self.fcs = []
        for i, fc_hidden_dim in enumerate(args.fully_connected_dim):
            dropout = tf.keras.layers.Dropout(
                    args.dropout_rate,
                    name='dropout_' + str(i+1))
            fc = tf.keras.layers.Dense(
                    fc_hidden_dim,
                    activation='relu',
                    name='fc_' + str(i+1))
            self.dropouts += [dropout]
            self.fcs += [fc]

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
        # Run parallel convolution filters of different widths, each with
        # batch norm and pooling
        # If set, also add a locally connected layer(s) for each group
        group_outputs = []
        for conv, batchnorm, pool_1, pool_2, lcs in zip(self.convs, self.batchnorms,
                self.pools, self.pools_2, self.lcs):
            # Run the convolutional layer and batch norm on x, to
            # start this group
            group_x = conv(x)
            group_x = batchnorm(group_x, training=training)

            # Run the pooling layer on the current group output (group_x)
            if pool_2 is None:
                group_x = pool_1(group_x)
            else:
                group_x_1 = pool_1(group_x)
                group_x_2 = pool_2(group_x)
                group_x = self.pool_merge([group_x_1, group_x_2])

            if lcs is not None:
                # Run the locally connected layer (1 or more)
                if len(lcs) == 1:
                    # Only 1 locally connected layer
                    lc = lcs[0]
                    group_x = lc(group_x)
                else:
                    lc_outputs = []
                    for lc in lcs:
                        # Run the locally connected layer (lc) on the
                        # current output for this group (group_x)
                        lc_outputs += [lc(group_x)]
                    # Merge the outputs of the locally connected layers
                    group_x = self.lc_merge(lc_outputs)

            group_outputs += [group_x]

        # Merge the above groups
        if len(group_outputs) == 1:
            # Only 1 filter width; cannot merge across 1 input
            x = group_outputs[0]
        else:
            x = self.merge(group_outputs)
        x = self.flatten(x)

        # Run through fully connected layers
        for dropout, fc in zip(self.dropouts, self.fcs):
            x = dropout(x, training=training)
            x = fc(x)

        return self.fc_final(x)


model = Cas9CNNWithParallelFilters()

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

# Here we will effectively implement tf.keras.callbacks.EarlyStopping() to
# decide when to stop training; because we are not using model.fit(..) we
# cannot use this callback out-of-the-box
# Set the number of epochs that must pass with no improvement in the
# validation loss, after which we will stop training
STOPPING_PATIENCE = 5

# Train (and validate) for each epoch
best_val_loss = None
num_epochs_past_best_loss = 0
for epoch in range(args.max_num_epochs):
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

    val_loss = validate_loss_metric.result()

    # Reset metric states so they are not cumulative over epochs
    train_loss_metric.reset_states()
    train_accuracy_metric.reset_states()
    train_auc_roc_metric.reset_states()
    train_auc_pr_metric.reset_states()
    validate_loss_metric.reset_states()
    validate_accuracy_metric.reset_states()
    validate_auc_roc_metric.reset_states()
    validate_auc_pr_metric.reset_states()

    # Decide whether to stop at this epoch
    if best_val_loss is None or val_loss < best_val_loss:
        # Update the best validation loss
        best_val_loss = val_loss
        num_epochs_past_best_loss = 0
    else:
        # This loss is worse than one seen before
        num_epochs_past_best_loss += 1
    if num_epochs_past_best_loss >= STOPPING_PATIENCE:
        # Stop here
        print('  Stopping at EPOCH {}'.format(epoch+1))
        break


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
