'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_sample(f=1.0, t0=None, batch_size=1):
    predict = 50
    sample = 100
    Fs = 100

    T = np.empty((batch_size, sample))
    Y = np.empty((batch_size, sample))
    FT = np.empty((batch_size, predict))
    FY = np.empty((batch_size, predict))

    _t0 = t0
    for i in range(batch_size):
        t = np.arange(0, sample + predict)/Fs
        if _t0 is None:
            t0 = np.random.rand() * 2 * np.pi
        else:
            t0 = _t0 + i/float(batch_size)

        if f is None:
            f = np.random.rand() * 4

        y = np.sin(2 * np.pi * f * (t + t0))

        T[i, :] = t[0:sample]
        Y[i, :] = y[0:sample]

        FT[i, :] = t[sample:sample+predict]
        FY[i, :] = y[sample:sample+predict]

    return T, Y, FT, FY

t, y, t_next, y_next = generate_sample(t0=0, batch_size=1)

#for i in range(t.shape[0]):
#    plt.plot(t[i, :], y[i, :])
#    plt.plot(np.append(t[i, -1], t_next[i, :]), np.append(y[i, -1], y_next[i, :]), color='red', linestyle=':')
#
#plt.xlabel('time [t]')
#plt.ylabel('signal')
#plt.show()
#exit()


import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 50
display_step = 100

# Network Parameters
n_input = 1  # input is sin(x)
n_steps = t.shape[1]  # timesteps
n_hidden = 100  # hidden layer num of features
n_outputs = t_next.shape[1]  # output is sin(x+1)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_outputs])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_outputs]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss (Euclidean distance) and optimizer
individual_losses = tf.reduce_sum(tf.square(tf.sub(pred, y)), reduction_indices=1)
loss = tf.reduce_mean(individual_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        _, batch_x, __, batch_y = generate_sample(f=1.0, t0=None, batch_size=batch_size)

        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape((batch_size, n_outputs))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss
            loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss_value))
        step += 1
    print("Optimization Finished!")

    # Test the prediction
    n_tests = 3
    for i in range(1, n_tests+1):
        plt.subplot(n_tests, 1, i)
        t, y, next_t, expected_y = generate_sample(f=i, t0=None)

        test_input = y.reshape((1, n_steps, n_input))
        prediction = sess.run(pred, feed_dict={x: test_input})

        # remove the batch size dimensions
        t = t.squeeze()
        y = y.squeeze()
        t_next = t_next.squeeze()
        prediction = prediction.squeeze()

        plt.plot(t, y, color='black')
        plt.plot(np.append(t[-1], t_next), np.append(y[-1], prediction), color='red')
        plt.ylim([-1, 1])
        plt.xlabel('time [t]')
        plt.ylabel('signal')

    plt.show()
