"""
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Inspired by https://github.com/aymericdamien/TensorFlow-Examples/
"""

import tensorflow as tf
from tensorflow.contrib import rnn


def RNN(x, weights, biases, n_input, n_steps, n_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, axis=0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.nn.bias_add(tf.matmul(outputs[-1], weights['out']), biases['out'])
