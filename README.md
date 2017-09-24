# TensorFlow LSTM sin(t) Example

Single- and multilayer LSTM networks with no additional output nonlinearity based on 
[aymericdamien's TensorFlow examples](https://github.com/aymericdamien/TensorFlow-Examples/)
and [Sequence prediction using recurrent neural networks](http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html).

Experiments with varying numbers of hidden units, LSTM cells and techniques like gradient clipping were conducted using `static_rnn` and `dynamic_rnn`. All networks have been optimized using [Adam](https://arxiv.org/abs/1412.6980) on the MSE loss function.

## Experiment 1

Given a single LSTM cell with `100` hidden states, predict the next `50` timesteps 
given the last `100` timesteps. 

The network is trained on a sine of `1 Hz` only using random shifts, thus fails on
generalizing to higher frequencies (`2 Hz` and `3 Hz` in the image); in addition, the
network should be able to simply memoize the shape of the input.
It was optimized with a learning rate of `0.001` for `200000` iterations and 
batches of `50` examples.

![](images/tf-recurrent-sin.jpg)

## Experiment 2

Given a single LSTM cell with `150` hidden states, predict the next `50` timesteps 
given the last `100` timesteps. 

The network is trained on sines of random frequencies between `0.5 .. 4 Hz` using 
random shifts. Prediction quality is worse than for the `1 Hz` only experiment above,
but it generalizes to the `2 Hz` and `3 Hz` tests.
It was optimized with a learning rate of `0.001` for `300000` iterations and 
batches of `50` examples.

At loss `0.614914`, the prediction looks like this:

![](images/tf-recurrent-sin-2.jpg)

## Experiment 3

Given a single LSTM cell with `150` hidden states, predict the next `50` timesteps 
given the last `25` timesteps. 

The network is trained on sines of random frequencies between `0.5 Hz` and `4 Hz` using 
random shifts. Prediction quality is worse than for the `1 Hz` only experiment above,
but it generalizes to the `2 Hz` and `3 Hz` tests.
It was optimized with a learning rate of `0.0005` for `500000` iterations and 
batches of `50` examples.

The following image shows the output at loss `0.177742`:

![](images/tf-recurrent-sin-3.jpg)

The following is the network trained to predict the next `100` timesteps
given the previous `25` timesteps; the parameters are otherwise unchanged.

This is the result at loss `0.257725`:

![](images/tf-recurrent-sin-3.1.jpg)

## Experiment 4

Same as the last experiment, however using `500` hidden states and gradient clipping
for the optimizer as described [here](http://stackoverflow.com/a/36501922/195651):

```python
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = adam.compute_gradients(loss)
clipped_gradients = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gradients]
optimizer = adam.apply_gradients(clipped_gradients)
```

Losses get as low as `0.069027` within the given iterations, but vary wildly.
This is at loss `0.422188`:

![](images/tf-recurrent-sin-4.jpg)

## Experiment 5

This time, the `dynamic_rnn()` function is used instead of `rnn()`, drastically improving the 
startup time. In addition, the single LSTM cell has been replaced with `4` stacked 
LSTM cells of `32` hidden states each.

```python
lstm_cells = [rnn.LSTMCell(n_hidden, forget_bias=1.0) 
              for _ in range(n_layers)]
stacked_lstm = rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32, time_major=False)
```

The output still uses linear regression:

```python
output = tf.transpose(outputs, [1, 0, 2])
pred = tf.matmul(output[-1], weights['out']) + biases['out']
```

The network is trained with learning rate `0.001` for at least `300000` iterations
(with the additional criterion that the loss should be below `0.15`).

The following picture shows the performance at loss `0.138701` at iteration `375000`.

![](images/tf-recurrent-sin-5.jpg)

When using only `10` hidden states, training takes much longer given a learning rate of `0.001`
and reaches a loss of about `0.5` after `~1200000` iterations, where convergence effectively stops.

The following used `10` hidden states and a base learning rate of `0.005` in combination with a 
step policy that reduced the learning rate by a factor of `0.1` every `250000` iterations.
Similar to the previous experiment, optimization was stopped after at least `300000` iterations 
have passed and the loss was below `0.2`.

The picture shows the outcome after `510000` iterations at a loss of `0.180995`:

![](images/tf-recurrent-sin-5.1.jpg)

## Experiment 6

Much like the last experiment, this one uses `10` hidden states per layer
in a `4` layer deep recurrent structure. Instead of using LSTM layers, however,
this one uses GRUs.

Because the loss did not go below `0.3`, training was stopped after `1000000`
iterations.
