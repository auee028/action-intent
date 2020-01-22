import tensorflow as tf
from controller import BaseController
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell

"""
A 1-layer LSTM recurrent neural network with 256 hidden units
Note: the state of the LSTM is not saved in a variable becuase we want
the state to reset to zero on every input sequnece
"""

class RecurrentController(BaseController):

    def network_vars(self):
        self.lstm_cell = LayerNormBasicLSTMCell(num_units=256, dropout_keep_prob=self.keep_prob)
        # self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256)

        self.state = tf.Variable(tf.zeros([self.batch_size, 256]), trainable=False)
        self.output = tf.Variable(tf.zeros([self.batch_size, 256]), trainable=False)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state)

    def get_state(self):
        return tf.nn.rnn_cell.LSTMStateTuple(c=self.state, h=self.output)

    def update_state(self, new_state):
        return tf.group(
            self.output.assign(new_state.h),
            self.state.assign(new_state.c)
        )