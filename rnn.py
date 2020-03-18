"""Classes and functions for working with RNNs.
"""

import parse_data

import numpy as np
import tensorflow as tf

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class LSTM:
    """Unidirectional and bidirectional LSTMs, which have been applied to
    protein sequence.

    This can optionally also use an embedding, but it doesn't make a lot of
    sense here: the vocabulary size is already very small.

    TODO: multiplicative LSTMs.
    """
    def __init__(self, context_nt, units=64, bidirectional=False,
            embed_dim=None, dropout_rate=0.5, regression=True,
            class_weight=None, batch_size=32):
        """
        Args:
            context_nt: amount of context to use in target
            units: dimensionality of LSTM output vector (and cell state vector)
            bidirectional: if True, use bidirectional LSTM
            embed_dim: if set, embed sequences with embedding layer and use
                this as the dimensionality; otherwise, use one-hot
                encoded sequence as input
            dropout_rate: dropout rate before final layer
            regression: if True, perform regression; else, classification
            class_weight: class weight for training; only application for
                classification
            batch_size: batch size
        """
        self.context_nt = context_nt
        self.units = units
        self.bidirectional = bidirectional
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.regression = regression
        self.class_weight=class_weight
        self.batch_size = batch_size

    # get_params() and set_params() are needed if we which to use this
    # class as a scikit-learn estimator
    # (see https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator)
    def get_params(self, deep=True):
        return {'context_nt': self.context_nt,
                'units': self.units,
                'bidirectional': self.bidirectional,
                'embed_dim': self.embed_dim,
                'dropout_rate': self.dropout_rate,
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
        if self.embed_dim is not None:
            # vocab size is 5 (A,C,G,T and no corresponding guide nt)
            # input shape is (seq_len, 2) -- seq_len nt long, and at each
            # position a value for the target and for the guide
            self.model.add(tf.keras.layers.Embedding(
                input_dim=5, output_dim=self.embed_dim,
                input_shape=(seq_len, 2)))
            # Merge the last two dimensions so that there's a single 1D
            # vector at each position of the guide-target (rather than a
            # vector [[embedded vector for target], [embedded vector for
            # guide]] at each position); i.e.,  change the shape from
            # [batch,length,2,embed_dim] -> [batch,length,2*embed_dim].
            # Note that, with tf.keras... we ignore the batch dimension
            # when reshaping
            self.model.add(tf.keras.layers.Reshape(
                (seq_len, 2*self.embed_dim)))
            # The input has seq_len timesteps and each step has
            # 2*self.embed_dim features
            lstm_input_shape = (seq_len, 2*self.embed_dim)
        else:
            # The input has seq_len timesteps and each step has 8 features (4
            # for the target and 4 for the guide)
            lstm_input_shape = (seq_len, 8)
        lstm = tf.keras.layers.LSTM(self.units, input_shape=lstm_input_shape)
        if self.bidirectional:
            self.model.add(tf.keras.layers.Bidirectional(lstm))
        else:
            self.model.add(lstm)
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

        if self.embed_dim is not None:
            x_train = np.array([parse_data.input_vec_for_embedding(x,
                self.context_nt) for x in x_train])

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
        if self.embed_dim is not None:
            x_test = np.array([parse_data.input_vec_for_embedding(x,
                self.context_nt) for x in x_test])
        return self.model.predict(x_test).ravel()

