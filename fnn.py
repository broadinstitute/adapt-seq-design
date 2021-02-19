"""Feed forward neural networks for predicting activity of a guide sequence.

This is focused primarily on CNNs, but includes a simple multilayer perceptron
as well.

Can be used with classification and regression.
"""

import numpy as np
import tensorflow as tf


__author__ = 'Hayden Metsky <hayden@mit.edu>'


class CasCNNWithParallelFilters(tf.keras.Model):
    def __init__(self, params, regression):
        """
        Args:
            params: dict of hyperparameters
            regression: if True, perform regression; if False, classification
        """
        super(CasCNNWithParallelFilters, self).__init__()

        self.regression = regression
        self.add_gc_content = params['add_gc_content']
        self.context_nt = params['context_nt']

        # Note that this is only used for regression
        if params['sample_weight_scaling_factor'] < 0:
            raise ValueError(("Parameter 'sample_weight_scaling_factor' "
                "must be >=0"))
        self.sample_weight_scaling_factor = params['sample_weight_scaling_factor']

        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']

        if self.add_gc_content:
            # Construct a layer to extract the region (along the width axis)
            # of the guide sequence (and target)
            self.guide_slice = tf.keras.layers.Cropping1D((self.context_nt,
                self.context_nt))

        if params['conv_filter_width'] is None:
            # Use a layer of width 'None', to make it easier to construct
            # locally connected layers below
            conv_filter_widths = [None]
        else:
            conv_filter_widths = params['conv_filter_width']

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
        for filter_width in conv_filter_widths:
            if filter_width is not None:
                # Construct the convolutional layer
                # Do not pad the input (`padding='valid'`) because all input
                # sequences should be the same length
                conv_layer_num_filters = params['conv_num_filters'] # ie, num of output channels
                conv = tf.keras.layers.Conv1D(
                        conv_layer_num_filters,
                        filter_width,
                        strides=1,  # stride by 1
                        padding='valid',
                        activation=params['activation_fn'],
                        name='group_w' + str(filter_width) + '_conv')
                # Note that the total number of filters in this layer will be
                # len(conv_filter_width)*conv_layer_num_filters since there are
                # len(conv_filter_width) groups

                # Add a batch normalization layer
                # It should not matter whether this comes before or after the
                # pool layer, as long as it is after the conv layer
                # This is applied after the activation of the conv layer; the
                # original batch norm applies batch normalization before the
                # activation function, but more recent work seems to apply it
                # after activation
                # Only use if the parameter specifying to skip batch norm is
                # not set
                if params['skip_batch_norm'] is True:
                    batchnorm = None
                else:
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
            else:
                # No convolutional layer
                self.convs += [None]
                self.batchnorms += [None]
                self.pools += [None]
                self.pools_2 += [None]

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
                    if filter_width is not None:
                        name = 'group_w' + str(filter_width) + '_lc_w' + str(lc_width)
                    else:
                        name = 'lc_w' + str(lc_width)
                    # Stride by 1/2 the width
                    stride = max(1, int(lc_width / 2))
                    lc = tf.keras.layers.LocallyConnected1D(
                        locally_connected_dim,
                        lc_width,
                        strides=stride,
                        activation=params['activation_fn'],
                        name=name)
                    lcs_for_conv += [lc]
                self.lcs += [lcs_for_conv]
            else:
                self.lcs += [None]

        if conv_filter_widths != [None] and params['pool_strategy'] == 'max-and-avg':
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
        if len(conv_filter_widths) > 1:
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
                    activation=params['activation_fn'],
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

        if (regression and 'regression_clip' in params and
                params['regression_clip'] is True):
            # Clip output to be >= min_out
            min_out = -4
            def clip(x):
                return tf.keras.activations.relu(x - min_out) + min_out
            self.clip_output = clip
            alpha = params['regression_clip_alpha']
            def clip_leaky(x):
                return tf.keras.activations.relu(x - min_out,
                        alpha=alpha) + min_out
            self.clip_output_leaky = clip_leaky
        else:
            self.clip_output = None

        # Regularize weights on each layer
        l2_regularizer = tf.keras.regularizers.l2(params['l2_factor'])
        for layer in self.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2_regularizer

    def call(self, x, training=False):
        # Run parallel convolution filters of different widths, each with
        # batch norm and pooling
        # If set, also add a locally connected layer(s) for each group

        # If self.add_gc_content is True, then 'manually' add a feature to
        # the fully connected layer giving the GC content of the guide
        if self.add_gc_content:
            # Extract the region of the guide/target
            x_guide_region = self.guide_slice(x)
            # Pull out just the guide; in dimension 3, the first 4 bases
            # are target and the next 4 are guide
            x_guide = tf.slice(x_guide_region,
                    [0, 0, 4],  # begin at position 4 in dimension 3
                    [-1, -1, 4])    # take the 4 positions of the guide
            # Compute the number of bases for each base by summing the
            # one-hot encoding for the corresponding dimension
            base_count = tf.reduce_sum(x_guide, axis=1, keepdims=True)
            # Pull out just the C and G counts, and sum across these
            gc_count = base_count[:,:,1] + base_count[:,:,2]
            guide_len = tf.cast(tf.shape(x_guide)[1], tf.float32)
            gc_content = gc_count / guide_len

        group_outputs = []
        for conv, batchnorm, pool_1, pool_2, lcs in zip(self.convs, self.batchnorms,
                self.pools, self.pools_2, self.lcs):
            if conv is not None:
                # Run the convolutional layer and batch norm on x, to
                # start this group
                group_x = conv(x)
                if batchnorm is not None:
                    group_x = batchnorm(group_x, training=training)

                # Run the pooling layer on the current group output (group_x)
                if pool_2 is None:
                    group_x = pool_1(group_x)
                else:
                    group_x_1 = pool_1(group_x)
                    group_x_2 = pool_2(group_x)
                    group_x = self.pool_merge([group_x_1, group_x_2])
            else:
                # Skip the convolutional layer, as well as batch norm and
                # pooling
                group_x = x

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

        if self.add_gc_content:
            # Concatenate x and gc_content along the flattened dimension
            # (axis 0 is batch; use '-1' or '1' for the axis to concat along)
            x_with_gc = tf.concat([x, gc_content], -1)
            x = x_with_gc

        # Run through fully connected layers
        for dropout, fc in zip(self.dropouts, self.fcs):
            x = dropout(x, training=training)
            x = fc(x)

        x = dropout(x, training=training)
        x = self.fc_final(x)
        if self.clip_output is not None:
            if training:
                x = self.clip_output_leaky(x)
            else:
                x = self.clip_output(x)
        return x


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


class MultilayerPerceptron:
    """Multilayer perceptron (MLP) using Keras.

    This should be similar (if not identical) to the CasCNNWithParallelFilters,
    without convolutional or locally connected layers.
    """
    def __init__(self, context_nt, layer_dims=[64, 64],
            dropout_rate=0.5, activation_fn='relu', regression=True,
            class_weight=None, batch_size=32):
        """
        Args:
            context_nt: amount of context to use in target
            layer_dims: list of the dimensionality of each layer;
                len(layer_dims) specifies how many layers (this does NOT
                include the last layer)
            dropout_rate: dropout rate before each layer
            activation_fn: activation function to use for the hidden layers
                (everything but the final layer)
            regression: if True, perform regression; else, classification
            class_weight: class weight for training; only applicable for
                classification
            batch_size: batch size
        """
        self.context_nt = context_nt
        self.layer_dims = layer_dims
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.regression = regression
        self.class_weight=class_weight
        self.batch_size = batch_size

    # get_params() and set_params() are needed if we which to use this
    # class as a scikit-learn estimator
    # (see https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator)
    def get_params(self, deep=True):
        return {'context_nt': self.context_nt,
                'layer_dims': self.layer_dims,
                'dropout_rate': self.dropout_rate,
                'activation_fn': self.activation_fn,
                'regression': self.regression,
                'class_weight': self.class_weight,
                'batch_size': self.batch_size}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def setup(self, seq_len):
        """Setup the model.

        Args:
            seq_len: length of each sequence; only used to specify input shape
                to first layer
        """
        final_activation = 'linear' if self.regression else 'sigmoid'

        self.model = tf.keras.Sequential()

        # Flatten input
        self.model.add(tf.keras.layers.Flatten())

        # At each position of the input there are 8 bits for the one-hot
        # encoding (4 for the target, 4 for the guide); the total input size is
        # thus 8*seq_len
        input_dim = 8*seq_len

        # Add middle layers
        for i, dim in enumerate(self.layer_dims):
            self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
            if i == 0:
                self.model.add(tf.keras.layers.Dense(dim,
                    activation=self.activation_fn,
                    input_dim=input_dim))
            else:
                self.model.add(tf.keras.layers.Dense(dim,
                    activation=self.activation_fn))

        # Add a final layer
        self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
        self.model.add(tf.keras.layers.Dense(1, activation=final_activation))

        if self.regression:
            self.model.compile('adam', 'mse', metrics=[
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError()])
        else:
            self.model.compile('adam', 'binary_crossentropy', metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Accuracy()])

    def fit(self, x_train, y_train, max_num_epochs=1000):
        """Fit the model.

        Args:
            x_train/y_train: training data
            max_num_epochs: maximum number of epochs to run (early stopping
                should stop before this)
        """
        # Setup model; do this again in case parameters changed
        seq_len = len(x_train[0])
        self.setup(seq_len)

        # Setup early stopping
        # The validation data is only used for early stopping
        # Note that this uses a random train/val split to decide when to stop
        # early; this may not be ideal due to crRNA overlap between the
        # train/val sets (will likely stop too late and overfit)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                mode='min', patience=2)

        self.model.fit(x_train, y_train, validation_split=0.25,
                batch_size=self.batch_size,
                callbacks=[es], class_weight=self.class_weight,
                epochs=max_num_epochs,
                verbose=2)

    def predict(self, x_test):
        """Make predictions:

        Args:
            x_test: input data for predictions

        Returns:
            predictions
        """
        return self.model.predict(x_test).ravel()

