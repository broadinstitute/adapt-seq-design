# Simple convnet (CNN) for a guide sequence.
# This has one convolutional layer, a max pool layer, 2 fully connected hidden
# layers, and fully connected output layer passed through sigmoid.

# Based loosely in part on Killoran et al. 2017:
# https://github.com/co9olguy/Generating-and-designing-DNA/blob/master/scripts/train_predictor.py


import tensorflow as tf
import numpy as np


#####################################################################
# Read and batch input/output
#####################################################################
# TODO: read input/output data

# TODO
batch_size = 64
def train_iter():
    # Yield batches of training data
    pass
def validate_iter():
    # Yield batches of validation data
    pass
def test_iter():
    # Yield batches of test data
    pass
#####################################################################
#####################################################################


#####################################################################
# Construct a convolutional network
#####################################################################

###################################
# Initialize

# TODO: initialize biases to 0
# TODO: for weights, use tf.contrib.layers.xavier_initializer() (uniform=False)

# Don't use a random seed
tf.set_random_seed(1)

# Specify placeholder for inputs
# Use None for batch size to dynamically determine it based on what's given
batched_input = tf.placeholder(tf.float32, shape=(None, 28, 4))

# Specify placeholder for outputs
# Use None for batch size to dynamically determine it based on what's given
batched_true_output = tf.placeholder(tf.float32, (None))
###################################

###################################
# Construct the convolutional layer

# Set an initial value for the weights of the convolutional filters based on a
# truncated normal distribution
# The shape of the conv filters is
# (filter width, number of input channels (4), number of output channels (20))
num_filters = 20    # aka, number of output channels
conv_filters_shape = (4, 4, num_filters)
conv_filters_init = tf.truncated_normal(conv_filter_shape, stddev=1.0)

# Setup the convolution; stride by 1 and do not pad the input
# (`padding='valid'`) because all input sequences should be the same
# length
conv_filters = tf.Variable(conv_filters_init, name='conv_W')
conv = tf.nn.conv1d(batched_input, conv_filters, stride=1, padding='VALID')

# Setup a bias on the convolutional layer
conv_bias_shape = (num_filters)
conv_bias_init = tf.truncated_normal(conv_bias_shape, stddev=1.0)
conv_bias = tf.Variable(conv_bias_init, name='conv_b')

# Add a bias to the convolution, and apply a leaky ReLU activation
conv_layer = tf.nn.leaky_relu(tf.add(conv, conv_bias))
###################################

###################################
# Construct a pooling layer

# tf.nn.max_pool requires a 4D tensor as input (i.e., have a height
# dimension, but the conv layer above (using conv1d) only has 3
# dimensions output (no height); give it an extra dimension for height
# just before the width (axis=1) -- i.e., change the output shape from
# (batch, conv filter width, output channel) to
# (batch, 1, conv filter width, output channel)
max_pool_input = tf.expand_dims(conv_layer, axis=1)

# The input tensor will be 4D and gives the output from a conv filter;
# the 4 dimensions are:
#   (batch, conv filter height, conv filter width, output channel)
# For the max pool the kernel size (ksize) [a, b, c, d]
# says to take the maximum over a batches, b steps in the height, c
# steps in the width, and d output channels
# We don't want to take the maximum over multiple batches or over
# multiple output channels, so a=1 and d=1; we do want to take the
# maximum over a (1 x max_pool_window) window, so b=1 (since there is
# only one value in the height dimension) and c=max_pool_window where
# max_pool_window gives how many conv outputs (in the dimension of
# the sequence, i.e., width) to pool over
max_pool_window = 4
ksize = (1, 1, max_pool_window, 1)

# strides gives the stride of the sliding window for each dimension
# of the input tensor; let strides be [a, b, c, d]
# Look at each training batch at a time, so a=1 (if a>1, we would
# skip over a batch); there's no where to go in the height dimension,
# so b=1; look at each output channel from the conv layer at a time, so
# d=1 (if d>1, we would skip over output channels); and stride in
# the width dimension according to max_pool_stride
# Note that if max_pool_stride = max_pool_window, then the max pooling
# windows are non-overlapping
max_pool_stride = 2
strides = (1, 1, max_pool_stride, 1)

max_pool_layer = tf.nn.max_pool(max_pool_input,
        ksize,
        strides,
        padding='VALID')
###################################

###################################
# Construct 2 fully connected hidden layers

# Determine how many inputs this will receive, as the number
# per max pool times the number of filters 
num_per_max_pool = (20 - max_pool_window) / max_pool_stride + 1
fc_layer1_num_in = num_per_max_pool * num_filters

# Set the dimension of each fully connected layer (number of
# neurons in each layer)
fc_hidden_dim = 50
fc_layer2_num_in = fc_hidden_dim

# Flatten the max pooling output from above; use -1 to dynamically
# calculate the first dimension (number of samples in training batch)
max_pool_output_flattened = tf.reshape(max_pool_layer, (-1, fc_layer1_num_in))

# Set initial weights and bias for each layer
fc_layer1_W_init = tf.truncated_normal((fc_layer1_num_in, fc_hidden_dim),
        stddev=1.0)
fc_layer1_b_init = tf.constant(0.0, shape=(fc_layer1_num_in))
fc_layer2_W_init = tf.truncated_normal((fc_layer2_num_in, fc_hidden_dim),
        stddev=1.0)
fc_layer2_b_init = tf.constant(0.0, shape=(fc_layer2_num_in))

# Set variables for weights and bias
fc_layer1_W = tf.Variable(fc_layer1_W_init, name='fc1_W')
fc_layer1_b = tf.Variable(fc_layer1_b_init, name='fc1_b')
fc_layer2_W = tf.Variable(fc_layer2_W_init, name='fc2_W')
fc_layer2_b = tf.Variable(fc_layer2_b_init, name='fc2_b')

# Compute outputs for each layer
fc_layer1 = tf.nn.leaky_relu(
        tf.matmul(max_pool_output_filtered, fc_layer1_W) + fc_layer1_b)
fc_layer2 = tf.nn.leaky_relu(
        tf.matmul(fc_layer1, fc_layer2_W) + fc_layer2_b)
###################################


###################################
# Construct the final layer (fully connected)

# Determine how many inputs this will receive
fc_final_num_in = fc_hidden_dim

# Set the dimension of this final fully connected layer (number
# of neurons)
fc_final_dim = 1

# Set initial weights and bias
fc_final_W_init = tf.truncated_normal((fc_final_num_in, fc_final_dim),
        stddev=1.0)
fc_final_b_init = tf.constant(0.0, shape=(fc_final_dim))

# Set variables for weights and bias
fc_final_W = tf.Variable(fc_final_W_init, name='fcfinal_W')
fc_final_b = tf.Variable(fc_final_b_init, name='fcfinal_b')

# Compute output
fc_final = tf.nn.leaky_relu(
        tf.matmul(fc_layer1, fc_final_W) + fc_final_b)
###################################

###################################
# Compute activation of the final output

# Use sigmoid activation
output = tf.nn.sigmoid(fc_final)
###################################

#####################################################################
#####################################################################


#####################################################################
# Perform training and testing
#####################################################################
# Setup cross-entropy as the loss function
# This expects logits (the output of the last layer of the network
# before activation -- i.e., not bounded between 0 and 1)
cross_entropy_per_sample = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fc_final,
        labels=batched_true_output)
# Take the mean cross-entropy across the samples
cross_entropy = tf.reduce_mean(cross_entropy_per_sample)

# Define an optimizer and minimize the cross-entropy
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
        name='optimizer')
optimizer_op = optimizer.minimize(cross_entropy)

# Start a session
sess = tf.Session()

# Initialize the variables
sess.run(tf.global_variables_initializer())

# Train (and validate) for each epoch
num_epochs = 10
for epoch in range(num_epochs):
    # Train on each batch
    train_costs = []
    for batch in train_iter():
        batch_in, batch_out = batch
        # Minimize and compute the cross-entropy cost
        _, cost = sess.run([optimizer_op, cross_entropy],
                feed_dict={batched_input: batch_in,
                    batched_true_output: batch_out})
        train_costs += [cost]

    # Validate on each batch
    # Note that we could run the validation data through the model all
    # at once (not batched), but batching may help with memory usage by
    # reducing how much data is run through the network at once
    validate_costs = []
    for batch in validate_iter():
        batch_in, batch_out = batch
        # Compute the cross-entropy cost
        # TODO: also compute accuracy
        cost = sess.run(cross_entropy,
                feed_dict={batched_input: batch_in,
                    batched_true_output: batch_out})
        validate_costs += [cost]

    # TODO: save checkpoints

    # Log the costs
    print("Epoch: %d : training cost = %f, validate cost = %f" %
            (epoch + 1, np.mean(train_costs), np.mean(validate_costs)))

# Test the model
test_costs = []
for batch in test_iter():
    batch_in, batch_out = batch
    # Compute the cross-entropy cost
    cost = sess.run(cross_entropy,
            feed_dict={batched_input: batch_in,
                batched_true_output: batch_out})
    test_costs += [cost]
print("Test cost = %f" % np.mean(test_costs))

# Close the session
sess.close()
#####################################################################
#####################################################################
