"""CNN for predicting activity of a guide sequence: classification and
regression.
"""

import argparse

import parse_data

import numpy as np
import scipy
import sklearn
import tensorflow as tf


__author__ = 'Hayden Metsky <hayden@mit.edu>'


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
    parser.add_argument('--plot-roc-curve',
            help=("If set, path to PDF at which to save plot of ROC curve"))
    parser.add_argument('--plot-predictions',
            help=("If set, path to PDF at which to save plot of predictions "
                  "vs. true values"))
    args = parser.parse_args()

    # Print the arguments provided
    print(args)

    return args


def set_seed(seed):
    """Set tensorflow and numpy seed.

    Args:
        seed: random seed
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


def read_data(args):
    """Read input/output data.

    Args:
        args: argument namespace

    Returns:
        train, validate, test data where each is a tuple (x, y) and
        positions of each element in the test data
    """
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
    test_frac = 0.3
    train_frac = (1.0 - test_frac) * (2.0/3.0)
    validation_frac = (1.0 - test_frac) * (1.0/3.0)
    data_parser = parser_class(
            subset=subset,
            context_nt=args.context_nt,
            split=(train_frac, validation_frac, test_frac),
            shuffle_seed=args.seed,
            stratify_by_pos=True)
    if args.dataset == 'cas13':
        classify_activity = args.cas13_classify
        regress_on_all = args.cas13_regress_on_all
        regress_only_on_active = args.cas13_regress_only_on_active
        data_parser.set_activity_mode(
                classify_activity, regress_on_all, regress_only_on_active)
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
        if args.dataset == 'cas13' and args.cas13_classify:
            print('Note that inactive=1 and active=0')

    test_pos = [data_parser.pos_for_input(xi) for xi in x_test]

    return ((x_train, y_train),
            (x_validate, y_validate),
            (x_test, y_test),
            test_pos)


def make_dataset_and_batch(x, y, batch_size=32):
    """Make tensorflow dataset and batch.

    Args:
        x: input data
        y: outputs (labels if classification)
        batch_size: batch size

    Returns:
        batched tf.data.Dataset object
    """
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)


#####################################################################
# Construct a convolutional network model
#####################################################################
class CasCNNWithParallelFilters(tf.keras.Model):
    def __init__(self, params, regression):
        """
        Args:
            params: dict of hyperparameters
            regression: if True, perform regression; if False, classification
        """
        super(CasCNNWithParallelFilters, self).__init__()

        self.regression = regression

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
        for filter_width in params['conv_filter_width']:
            # Construct the convolutional layer
            # Do not pad the input (`padding='valid'`) because all input
            # sequences should be the same length
            conv_layer_num_filters = params['conv_num_filters'] # ie, num of output channels
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
            pool_window_width = params['pool_window_width']
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
            if params['pool_strategy'] == 'max':
                self.pools += [maxpool]
                self.pools_2 += [None]
            elif params['pool_strategy'] == 'avg':
                self.pools += [avgpool]
                self.pools_2 += [None]
            elif params['pool_strategy'] == 'max-and-avg':
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
            if params['locally_connected_width'] is not None:
                lcs_for_conv = []
                locally_connected_dim = params['locally_connected_dim']
                for i, lc_width in enumerate(params['locally_connected_width']):
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

        if params['pool_strategy'] == 'max-and-avg':
            # Setup layer to concatenate each max/avg pooling in each group
            self.pool_merge = tf.keras.layers.Concatenate(
                    axis=1,
                    name='merge_pool')

        if params['locally_connected_width'] is not None:
            if len(params['locally_connected_width']) > 1:
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
        if len(params['conv_filter_width']) > 1:
            self.merge = tf.keras.layers.Concatenate(
                    axis=1,
                    name='merge_groups')

        # Flatten the pooling output from above while preserving
        # the batch axis
        self.flatten = tf.keras.layers.Flatten()

        # Setup fully connected layers
        # Insert dropout before each of them for regularization
        # Set the dimension of each fully connected layer (i.e., dimension
        # of the output space) to params['fully_connected_dim'][i]
        self.dropouts = []
        self.fcs = []
        for i, fc_hidden_dim in enumerate(params['fully_connected_dim']):
            dropout = tf.keras.layers.Dropout(
                    params['dropout_rate'],
                    name='dropout_' + str(i+1))
            fc = tf.keras.layers.Dense(
                    fc_hidden_dim,
                    activation='relu',
                    name='fc_' + str(i+1))
            self.dropouts += [dropout]
            self.fcs += [fc]

        # Construct the final layer (fully connected)
        fc_final_dim = 1
        if regression:
            final_activation = 'linear'
        else:
            final_activation = 'sigmoid'
        self.fc_final = tf.keras.layers.Dense(
                fc_final_dim,
                activation=final_activation,
                name='fc_final')

        # Regularize weights on each layer
        l2_regularizer = tf.keras.regularizers.l2(params['l2_factor'])
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


def construct_model(params, shape, regression=False):
    """Construct model.

    Args:
        params: dict of hyperparameters
        shape: shape of input data; only used for printing model summary
        regression: if True, perform regression; if False, classification

    Returns:
        CasCNNWithParallelFilters object
    """
    model = CasCNNWithParallelFilters(params, regression)

    # Print a model summary
    model.build(shape)
    print(model.summary())

    return model


#####################################################################
# Perform training and testing
#####################################################################
# For classification, use cross-entropy as the loss function
# This expects sigmoids (values in [0,1]) as the output; it will
# transform back to logits (not bounded betweem 0 and 1) before
# calling tf.nn.sigmoid_cross_entropy_with_logits
bce_per_sample = tf.keras.losses.BinaryCrossentropy()

# For regression, use mean squared error as the loss function
mse_per_sample = tf.keras.losses.MeanSquaredError()

# When outputting loss, take the mean across the samples from each batch
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
validate_loss_metric = tf.keras.metrics.Mean(name='validate_loss')

# Define metrics for regression
# tf.keras.metrics does not have Pearson correlation or Spearman's correlation,
# so we have to define these; note that it becomes much easier to use these
# outside of the tf.function functions rather than inside of them (like the
# other metrics are used)
def pearson_corr(y_true, y_pred):
    r, _ = scipy.stats.pearsonr(y_true, y_pred)
    return r
def spearman_corr(y_true, y_pred):
    rho, _ = scipy.stats.spearmanr(y_true, y_pred)
    return rho
class Correlation:
    def __init__(self, corrtype, name='correlation'):
        assert corrtype in ('pearson_corr', 'spearman_corr')
        if corrtype == 'pearson_corr':
            self.corr_fn = pearson_corr
        if corrtype == 'spearman_corr':
            self.corr_fn = spearman_corr
        self.__name__ = name
        self.y_true = []
        self.y_pred = []
    def __call__(self, y_true, y_pred):
        # Convert from tensors to numpy arrays, then to regular Python lists
        y_true = list(tf.reshape(y_true, [-1]).numpy())
        y_pred = list(tf.reshape(y_pred, [-1]).numpy())
        self.y_true += y_true
        self.y_pred += y_pred
    def result(self):
        return self.corr_fn(self.y_true, self.y_pred)
    def reset_states(self):
        self.y_true = []
        self.y_pred = []
train_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
train_mape_metric = tf.keras.metrics.MeanAbsolutePercentageError(name='train_mape')
train_pearson_corr_metric = Correlation('pearson_corr', name='train_pearson_corr')
train_spearman_corr_metric = Correlation('spearman_corr', name='train_spearman_corr')
validate_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='validate_mae')
validate_mape_metric = tf.keras.metrics.MeanAbsolutePercentageError(name='validate_mape')
validate_pearson_corr_metric = Correlation('pearson_corr', name='validate_pearson_corr')
validate_spearman_corr_metric = Correlation('spearman_corr', name='validate_spearman_corr')
test_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='test_mae')
test_mape_metric = tf.keras.metrics.MeanAbsolutePercentageError(name='test_mape')
test_pearson_corr_metric = Correlation('pearson_corr', name='test_pearson_corr')
test_spearman_corr_metric = Correlation('spearman_corr', name='test_spearman_corr')

# Define metrics for classification
# Report on the accuracy and AUC for each epoch (each metric is updated
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


# Train the model using GradientTape; this is called on each batch
def train_step(model, seqs, outputs, optimizer, sample_weight=None):
    if model.regression:
        loss_fn = mse_per_sample
    else:
        loss_fn = bce_per_sample
    with tf.GradientTape() as tape:
        # Compute predictions and loss
        # Pass along `training=True` so that this can be given to
        # the batchnorm and dropout layers; an alternative to passing
        # it along would be to use `tf.keras.backend.set_learning_phase(1)`
        # to set the training phase
        predictions = model(seqs, training=True)
        prediction_loss = loss_fn(outputs, predictions,
                sample_weight=sample_weight)
        # Add the regularization losses
        regularization_loss = tf.add_n(model.losses)
        loss = prediction_loss + regularization_loss
    # Compute gradients and opitmize parameters
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Record metrics
    train_loss_metric(loss)
    if model.regression:
        train_mae_metric(outputs, predictions)
        train_mape_metric(outputs, predictions)
    else:
        train_accuracy_metric(outputs, predictions)
        train_auc_roc_metric(outputs, predictions)
        train_auc_pr_metric(outputs, predictions)

    return outputs, predictions

# Define functions for computing validation and test metrics; these are
# called on each batch
def validate_step(model, seqs, outputs, sample_weight=None):
    # Compute predictions and loss
    predictions = model(seqs, training=False)
    if model.regression:
        loss_fn = mse_per_sample
    else:
        loss_fn = bce_per_sample
    prediction_loss = loss_fn(outputs, predictions,
                sample_weight=sample_weight)
    regularization_loss = tf.add_n(model.losses)
    loss = prediction_loss + regularization_loss
    # Record metrics
    validate_loss_metric(loss)
    if model.regression:
        validate_mae_metric(outputs, predictions)
        validate_mape_metric(outputs, predictions)
    else:
        validate_accuracy_metric(outputs, predictions)
        validate_auc_roc_metric(outputs, predictions)
        validate_auc_pr_metric(outputs, predictions)
    return outputs, predictions

def test_step(model, seqs, outputs):
    # Compute predictions
    predictions = model(seqs, training=False)
    # Record metrics
    if model.regression:
        test_mae_metric(outputs, predictions)
        test_mape_metric(outputs, predictions)
    else:
        test_accuracy_metric(outputs, predictions)
        test_auc_roc_metric(outputs, predictions)
        test_auc_pr_metric(outputs, predictions)
    return outputs, predictions


# Here we will effectively implement tf.keras.callbacks.EarlyStopping() to
# decide when to stop training; because we are not using model.fit(..) we
# cannot use this callback out-of-the-box
# Set the number of epochs that must pass with no improvement in the
# validation loss, after which we will stop training
STOPPING_PATIENCE = 2

def train_and_validate(model, x_train, y_train, x_validate, y_validate,
        max_num_epochs):
    """Train the model and also validate on each epoch.

    Args:
        model: model object
        x_train, y_train: training input and outputs (labels, if
            classification)
        x_validate, y_validate: validation input and outputs (labels, if
            classification)
        max_num_epochs: maximum number of epochs to train for

    Returns:
        dict with validation metrics at the end (keys are 'loss'
        and ('auc-roc' or 'r-spearman'))
    """
    # Define an optimizer
    if model.regression:
        # When doing regression, sometimes the output would always be the
        # same value regardless of input; decreasing the learning rate fixed this
        optimizer = tf.keras.optimizers.Adam(lr=0.00001)
    else:
        optimizer = tf.keras.optimizers.Adam()

    # model may be new, but calling train_step on a new model will yield
    # an error; tf.function was designed such that a new one is needed
    # whenever there is a new model
    # (see
    # https://github.com/tensorflow/tensorflow/issues/27525#issuecomment-481025914)
    tf_train_step = tf.function(train_step)
    tf_validate_step = tf.function(validate_step)

    train_ds = make_dataset_and_batch(x_train, y_train)
    validate_ds = make_dataset_and_batch(x_validate, y_validate)

    # For classification, determine class weights
    if not model.regression:
        y_train_labels = [y_train[i][0] for i in range(len(y_train))]
        class_weights = sklearn.utils.class_weight.compute_class_weight(
                'balanced', sorted(np.unique(y_train_labels)), y_train_labels)
        class_weights = list(class_weights)
        print('Using class weights: {}'.format(class_weights))
    def determine_sample_weights(outputs):
        if not model.regression:
            labels = [int(o.numpy()[0]) for o in outputs]
            return [class_weights[label] for label in labels]
        else:
            return None

    best_val_loss = None
    num_epochs_past_best_loss = 0

    for epoch in range(max_num_epochs):
        # Train on each batch
        for seqs, outputs in train_ds:
            sample_weight = determine_sample_weights(outputs)
            y_true, y_pred = tf_train_step(model, seqs, outputs, optimizer,
                    sample_weight=sample_weight)
            if model.regression:
                train_pearson_corr_metric(y_true, y_pred)
                train_spearman_corr_metric(y_true, y_pred)

        # Validate on each batch
        # Note that we could run the validation data through the model all
        # at once (not batched), but batching may help with memory usage by
        # reducing how much data is run through the network at once
        for seqs, outputs in validate_ds:
            sample_weight = determine_sample_weights(outputs)
            y_true, y_pred = tf_validate_step(model, seqs, outputs,
                    sample_weight=sample_weight)
            if model.regression:
                validate_pearson_corr_metric(y_true, y_pred)
                validate_spearman_corr_metric(y_true, y_pred)

        # Log the metrics from this epoch
        print('EPOCH {}'.format(epoch+1))
        print('  Train metrics:')
        print('    Loss: {}'.format(train_loss_metric.result()))
        if model.regression:
            print('    MAE: {}'.format(train_mae_metric.result()))
            print('    MAPE: {}'.format(train_mape_metric.result()))
            print('    r-Pearson: {}'.format(train_pearson_corr_metric.result()))
            print('    r-Spearman: {}'.format(train_spearman_corr_metric.result()))
        else:
            print('    Accuracy: {}'.format(train_accuracy_metric.result()))
            print('    AUC-ROC: {}'.format(train_auc_roc_metric.result()))
            print('    AUC-PR: {}'.format(train_auc_pr_metric.result()))
        print('  Validate metrics:')
        print('    Loss: {}'.format(validate_loss_metric.result()))
        if model.regression:
            print('    MAE: {}'.format(validate_mae_metric.result()))
            print('    MAPE: {}'.format(validate_mape_metric.result()))
            print('    r-Pearson: {}'.format(validate_pearson_corr_metric.result()))
            print('    r-Spearman: {}'.format(validate_spearman_corr_metric.result()))
        else:
            print('    Accuracy: {}'.format(validate_accuracy_metric.result()))
            print('    AUC-ROC: {}'.format(validate_auc_roc_metric.result()))
            print('    AUC-PR: {}'.format(validate_auc_pr_metric.result()))

        val_loss = validate_loss_metric.result()
        if model.regression:
            val_spearman_corr = validate_spearman_corr_metric.result()
        else:
            val_auc_roc = validate_auc_roc_metric.result()

        # Reset metric states so they are not cumulative over epochs
        train_loss_metric.reset_states()
        train_mae_metric.reset_states()
        train_mape_metric.reset_states()
        train_pearson_corr_metric.reset_states()
        train_spearman_corr_metric.reset_states()
        train_accuracy_metric.reset_states()
        train_auc_roc_metric.reset_states()
        train_auc_pr_metric.reset_states()
        validate_mae_metric.reset_states()
        validate_mape_metric.reset_states()
        validate_pearson_corr_metric.reset_states()
        validate_spearman_corr_metric.reset_states()
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

    if model.regression:
        val_metrics = {'loss': val_loss, 'r-spearman': val_spearman_corr}
    else:
        val_metrics = {'loss': val_loss, 'auc-roc': val_auc_roc}
    return val_metrics


def test(model, x_test, y_test, plot_roc_curve=None, plot_predictions=None,
            x_test_pos=None):
    """Test a model.

    This prints metrics.

    Args:
        model: model object
        x_test, y_test: testing input and outputs (labels, if
            classification)
        plot_roc_curve: if set, path to PDF at which to save plot of ROC curve
        plot_predictions: if set, path to PDF at which to save plot of
            predictions vs. true values
        x_test_pos: if set, position of each element in x_test (used for
            plotting with plot_predictions)
    """
    tf_test_step = tf.function(test_step)

    test_ds = make_dataset_and_batch(x_test, y_test)

    all_outputs = []
    all_predictions = []
    for seqs, outputs in test_ds:
        y_true, y_pred = tf_test_step(model, seqs, outputs)
        if model.regression:
            test_pearson_corr_metric(y_true, y_pred)
            test_spearman_corr_metric(y_true, y_pred)
        all_outputs += list(tf.reshape(y_true, [-1]).numpy())
        all_predictions += list(tf.reshape(y_pred, [-1]).numpy())

    print('TEST DONE')
    print('  Test metrics:')
    if model.regression:
        print('    MAE: {}'.format(test_mae_metric.result()))
        print('    MAPE: {}'.format(test_mape_metric.result()))
        print('    r-Pearson: {}'.format(test_pearson_corr_metric.result()))
        print('    r-Spearman: {}'.format(test_spearman_corr_metric.result()))
    else:
        print('    Accuracy: {}'.format(test_accuracy_metric.result()))
        print('    AUC-ROC: {}'.format(test_auc_roc_metric.result()))
        print('    AUC-PR: {}'.format(test_auc_pr_metric.result()))
    test_mae_metric.reset_states()
    test_mape_metric.reset_states()
    test_pearson_corr_metric.reset_states()
    test_spearman_corr_metric.reset_states()
    test_accuracy_metric.reset_states()
    test_auc_roc_metric.reset_states()
    test_auc_pr_metric.reset_states()

    if plot_roc_curve:
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        fpr, tpr, thresholds = roc_curve(all_outputs, all_predictions)
        plt.figure(1)
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.show()
        plt.savefig(plot_roc_curve)
    if plot_predictions:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.scatter(all_outputs, all_predictions, c=x_test_pos)
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.title('True vs. predicted values')
        plt.show()
        plt.savefig(plot_predictions)


def main():
    # Read arguments and data
    args = parse_args()
    set_seed(args.seed)
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test), x_test_pos = read_data(args)

    # Determine, based on the dataset, whether to do regression or
    # classification
    if args.dataset == 'cas13':
        if args.cas13_classify:
            regression = False
        else:
            regression = True
    elif args.dataset == 'cas9' or args.dataset == 'simulated-cas13':
        regression = False

    if regression and args.plot_roc_curve:
        raise Exception(("Can only use --plot-roc-curve when doing "
            "classification"))
    if not regression and args.plot_predictions:
        raise Exception(("Can only use --plot-predictions when doing "
            "regression"))

    # Construct model
    params = vars(args)
    model = construct_model(params, x_train.shape, regression)

    # Train the model, with validation
    train_and_validate(model, x_train, y_train, x_validate, y_validate,
            args.max_num_epochs)

    # Test the model
    test(model, x_test, y_test, plot_roc_curve=args.plot_roc_curve,
            plot_predictions=args.plot_predictions,
            x_test_pos=x_test_pos)


if __name__ == "__main__":
    main()
