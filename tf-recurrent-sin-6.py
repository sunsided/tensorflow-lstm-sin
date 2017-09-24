"""
A Recurrent Neural Network (GRU) implementation example using TensorFlow library.
Inspired by https://github.com/aymericdamien/TensorFlow-Examples/ and http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
"""

from __future__ import print_function

from generate_sample import generate_sample

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import seaborn as sns

import tensorflow as tf
from tensorflow.contrib import rnn

# Parameters
learning_rate = 0.005
training_iters = 1000000
training_iter_step_down_every = 250000
batch_size = 64
display_step = 100

# Network Parameters
n_input = 1  # input is sin(x), a scalar
n_steps = 25  # time steps
n_hidden = 10  # hidden layer num of features
n_outputs = 100  # output is a series of sin(x+...)
n_layers = 4  # number of stacked GRU layers

# tf Graph input
lr = tf.placeholder(tf.float32, [])
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_outputs])

# Define weights
weights = {
    'out': tf.Variable(tf.truncated_normal([n_hidden, n_outputs], stddev=1.0))
}
biases = {
    'out': tf.Variable(tf.truncated_normal([n_outputs], stddev=0.1))
}

# Define the GRU cells
gru_cells = [rnn.GRUCell(n_hidden) for _ in range(n_layers)]
stacked_lstm = rnn.MultiRNNCell(gru_cells)
outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs=x, dtype=tf.float32, time_major=False)

h = tf.transpose(outputs, [1, 0, 2])
pred = tf.nn.bias_add(tf.matmul(h[-1], weights['out']), biases['out'])

# Define loss (Euclidean distance) and optimizer
individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
loss = tf.reduce_mean(individual_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    loss_value = float('+Inf')
    target_loss = 0.2

    # Keep training until reach max iterations
    while (step * batch_size < training_iters) and (loss_value > target_loss):
        current_learning_rate = learning_rate
        current_learning_rate *= 0.1 ** ((step * batch_size) // training_iter_step_down_every)

        _, batch_x, __, batch_y = generate_sample(f=None, t0=None, batch_size=batch_size, samples=n_steps,
                                                  predict=n_outputs)

        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape((batch_size, n_outputs))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, lr: current_learning_rate})
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
