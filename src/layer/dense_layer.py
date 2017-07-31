# -*- encoding: utf8 -*-
# author: ronniecao
import numpy
import tensorflow as tf


class DenseLayer:
    
    def __init__(self, input_shape, hidden_dim, activation='relu', dropout=False, 
                 keep_prob=None, batch_normal=False, name='dense'):
        # params
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.biasatch_normal = batch_normal
        # weight
        self.weight = tf.Variable(
            initial_value=tf.random_normal(
                shape=[self.input_shape[1], self.hidden_dim],
                mean=0.0,
                stddev=0.01), 
                # stddev=numpy.sqrt(
                #     6.0 / (self.input_shape[1]))),
            name='W_%s' % (name))
        # bias
        self.bias = tf.Variable(
            initial_value=tf.constant(
                0.0, shape=[self.hidden_dim]),
            name='b_%s' % (name))
        # gamma
        if self.biasatch_normal:
            self.epsilon = 1e-5
            self.gamma = tf.Variable(
                initial_value=tf.random_normal(
                    shape=[self.hidden_dim]),
            name='gamma_%s' % (name))
        # keep_prob
        if self.dropout:
            self.keep_prob = keep_prob
        
    def get_output(self, input):
        # calculate input_shape and output_shape
        self.output_shape = [self.input_shape[0], self.hidden_dim]
        # hidden states
        intermediate = tf.matmul(input, self.weight)
        # batch normalization
        if self.biasatch_normal:
            mean, variance = tf.nn.moments(intermediate, axes=[0])
            self.hidden = tf.nn.batch_normalization(
                intermediate, mean, variance, self.bias, self.gamma, self.epsilon)
        else:
            self.hidden = intermediate + self.bias
        # dropout
        if self.dropout:
            self.hidden = tf.nn.dropout(self.hidden, keep_prob=self.keep_prob)
        # activation
        if self.activation == 'relu':
            self.output = tf.nn.relu(self.hidden)
        elif self.activation == 'tanh':
            self.output = tf.nn.tanh(self.hidden)
        elif self.activation == 'softmax':
            self.output = tf.nn.softmax(self.hidden)
        elif self.activation == 'none':
            self.output = self.hidden
        
        return self.output