# -*- coding: utf8 -*-
# author: ronniecao
# time: 20180828
# intro: fully connected layer based on tensorflow.layers
import numpy
import tensorflow as tf


class DenseLayer:
    
    def __init__(self, hidden_dim, activation='relu', name='dense',
                 batch_normal=False, dropout=False, keep_prob=0.0, 
                 weight_decay=None, input_shape=None, prev_layer=None):
        # params
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.batch_normal = batch_normal
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.name = name
        self.lytpe = 'dense'
        if prev_layer:
            self.prev_layer = prev_layer
            self.input_shape = prev_layer.output_shape
        elif input_shape:
            self.input_shape = input_shape
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        with tf.name_scope('%s_def' % (self.name)) as scope:
            weight_init_value = numpy.random.normal(size=[
                self.input_shape[0], self.hidden_dim], 
                loc=0.0, scale=numpy.sqrt(2.0 / self.input_shape[0]))
            bias_init_value = numpy.zeros([self.hidden_dim], dtype='float32')
            
            self.dense = tf.layers.Dense(
                units=self.hidden_dim,
                activation=None,
                use_bias=not self.batch_normal,
                kernel_initializer=tf.constant_initializer(weight_init_value),
                bias_initializer=tf.constant_initializer(bias_init_value),
                trainable=True,
                name='%s_dense' % (self.name))
            
            if self.batch_normal:
                beta_init_value = numpy.zeros([self.hidden_dim], dtype='float32')
                gamma_init_value = numpy.ones([self.hidden_dim], dtype='float32')
                moving_mean_init_value = numpy.zeros([self.hidden_dim], dtype='float32')
                moving_variance_init_value = numpy.ones([self.hidden_dim], dtype='float32')
                
                self.bn = tf.layers.BatchNormalization(
                    axis=-1,
                    momentum=0.9,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    beta_initializer=tf.constant_initializer(beta_init_value),
                    gamma_initializer=tf.constant_initializer(gamma_init_value),
                    moving_mean_initializer=tf.constant_initializer(moving_mean_init_value),
                    moving_variance_initializer=tf.constant_initializer(moving_variance_init_value),
                    trainable=True,
                    name='%s_bn' % (self.name))
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [self.hidden_dim]
        print('%-10s\t%-25s\t%-20s\t%s' % (
            self.name, 
            '(%d)' % (self.hidden_dim),
            '(%d)' % (self.input_shape[0]),
            '(%d)' % (self.output_shape[0])))
        self.calculation = self.input_shape[0] * self.output_shape[0]
        
    def get_output(self, inputs, is_training=tf.constant(True)):
        
        # hidden states
        self.hidden = self.dense(inputs)
            
        # dropout 技术
        if self.dropout:
            self.hidden = tf.nn.dropout(self.hidden, keep_prob=self.keep_prob)
        
        # activation
        if self.activation == 'relu':
            self.output = tf.nn.relu(self.hidden)
        elif self.activation == 'tanh':
            self.output = tf.nn.tanh(self.hidden)
        elif self.activation == 'softmax':
            self.output = tf.nn.softmax(self.hidden)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(self.hidden)
        elif self.activation == 'leaky_relu':
            self.output = self.leaky_relu(self.hidden)
        elif self.activation == 'none':
            self.output = self.hidden
        
        return self.output
    
    def leaky_relu(self, data):
        hidden = tf.cast(data, dtype=tf.float32)
        mask = tf.cast((hidden > 0), dtype=tf.float32)
        output = 1.0 * mask * hidden + 0.1 * (1 - mask) * hidden
        
        return output
