"""Simple convnet (CNN) for a guide sequence.

This is implemented for classifying Cas9 activity.

This has one convolutional layer, a max pool layer, 2 fully connected hidden
layers, and fully connected output layer passed through sigmoid.
"""

import parse_data

import numpy as np
import tensorflow as tf


# Don't use a random seed
tf.random.set_seed(1)

#####################################################################
# Read and batch input/output
#####################################################################
# Read data
data_parser = parse_data.Doench2016Cas9ActivityParser(
        subset=None,
        context_nt=20,
        split=(0.8, 0.1, 0.1),
        shuffle_seed=1)
data_parser.read()

x_train, y_train = data_parser.train_set()
x_validate, y_validate = data_parser.validate_set()
x_test, y_test = data_parser.test_set()

# Print the size of each data set
data_sizes = 'DATA SIZES - Train: {}, Validate: {}, Test: {}'
print(data_sizes.format(len(x_train), len(x_validate), len(x_test)))

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
        conv_layer_num_filters = 20 # aka, number of output channels
        conv_layer_filter_width = 4
        self.conv = tf.keras.layers.Conv1D(
                conv_layer_num_filters,
                conv_layer_filter_width,
                strides=1,  # stride by 1
                padding='valid',
                activation='relu',
                name='conv')

        # Construct a pooling layer
        # Take the maximum over a window of width max_pool_window, for
        # each output channel of the conv layer (and, of course, for each batch)
        # Stride by max_pool_stride; note that if
        # max_pool_stride = max_pool_window, then the max pooling
        # windows are non-overlapping
        max_pool_window = 4
        max_pool_stride = 2
        self.pool = tf.keras.layers.MaxPooling1D(
                pool_size=max_pool_window,
                strides=max_pool_stride,
                name='maxpool')

        # Flatten the max pooling output from above while preserving
        # the batch axis
        self.flatten = tf.keras.layers.Flatten()

        # Construct 2 fully connected hidden layers
        # Set the dimension of each fully connected layer (i.e., dimension
        # of the output space) to fc_hidden_dim
        fc_hidden_dim = 50
        self.fc_1 = tf.keras.layers.Dense(
                fc_hidden_dim,
                activation='relu',
                name='fc_1')
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

    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return self.fc_final(x)

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

# Also report on the accuracy, with the mean across each batch
train_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
validate_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='validate_accuracy')
test_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

# Define an optimizer
optimizer = tf.keras.optimizers.Adam()

# Train the model using GradientTape; this is called on each batch
@tf.function
def train_step(seqs, labels):
    with tf.GradientTape() as tape:
        # Compute predictions and loss
        predictions = model(seqs)
        loss = bce_per_sample(labels, predictions)
    # Compute gradients and opitmize parameters
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Record metrics
    train_loss_metric(loss)
    train_accuracy_metric(labels, predictions)

# Define functions for computing validation and test metrics; these are
# called on each batch
@tf.function
def validate_step(seqs, labels):
    # Compute predictions and loss
    predictions = model(seqs)
    loss = bce_per_sample(labels, predictions)
    # Record metrics
    validate_loss_metric(loss)
    validate_accuracy_metric(labels, predictions)
@tf.function
def test_step(seqs, labels):
    # Compute predictions and loss
    predictions = model(seqs)
    loss = bce_per_sample(labels, predictions)
    # Record metrics
    test_loss_metric(loss)
    test_accuracy_metric(labels, predictions)

# Train (and validate) for each epoch
num_epochs = 50
for epoch in range(num_epochs):
    # Train on each batch
    for seqs, labels in train_ds:
        train_step(seqs, labels)

    # Validate on each batch
    # Note that we could run the validation data through the model all
    # at once (not batched), but batching may help with memory usage by
    # reducing how much data is run through the network at once
    for seqs, labels in validate_ds:
        validate_step(seqs, labels)

    # Log the metrics; note that these are cumulative
    log = ('Epoch {} - Train loss: {}, Train accuracy: {}, ' +
           'Validate loss: {}, Validate accuracy: {}')
    print(log.format(epoch+1,
                     train_loss_metric.result(),
                     train_accuracy_metric.result(),
                     validate_loss_metric.result(),
                     validate_accuracy_metric.result()))

    # Reset metric states so they are not cumulative over epochs
    train_loss_metric.reset_states()
    validate_loss_metric.reset_states()
    train_accuracy_metric.reset_states()
    validate_accuracy_metric.reset_states()

# Test the model
for seqs, labels in test_ds:
    test_step(seqs, labels)
log = ('TEST - Loss: {}, Accuracy: {}')
print(log.format(test_loss_metric.result(),
                 test_accuracy_metric.result()))
test_loss_metric.reset_states()
test_accuracy_metric.reset_states()
#####################################################################
#####################################################################
