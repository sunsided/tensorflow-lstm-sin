"""
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Inspired by https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

from generate_sample import generate_sample
from rnn import RNN

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import seaborn as sns

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_iters = 300000
batch_size = 50
display_step = 100

# Network Parameters
n_input = 1  # input is sin(x)
n_steps = 100  # timesteps
n_hidden = 150  # hidden layer num of features
n_outputs = 50  # output is sin(x+1)

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

pred = RNN(x, weights, biases, n_input, n_steps, n_hidden)

# Define loss (Euclidean distance) and optimizer
individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
loss = tf.reduce_mean(individual_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        _, batch_x, __, batch_y = generate_sample(f=None, t0=None, batch_size=batch_size, samples=n_steps,
                                                  predict=n_outputs)

        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape((batch_size, n_outputs))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss
            loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss_value))
        step += 1
    print("Optimization Finished!")

    # Test the prediction
    n_tests = 3
    for i in range(1, n_tests + 1):
        plt.subplot(n_tests, 1, i)
        t, y, next_t, expected_y = generate_sample(f=i, t0=None, samples=n_steps, predict=n_outputs)

        test_input = y.reshape((1, n_steps, n_input))
        prediction = sess.run(pred, feed_dict={x: test_input})

        # remove the batch size dimensions
        t = t.squeeze()
        y = y.squeeze()
        next_t = next_t.squeeze()
        prediction = prediction.squeeze()

        plt.plot(t, y, color='black')
        plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=':')
        plt.plot(np.append(t[-1], next_t), np.append(y[-1], prediction), color='red')
        plt.ylim([-1, 1])
        plt.xlabel('time [t]')
        plt.ylabel('signal')

    plt.show()
