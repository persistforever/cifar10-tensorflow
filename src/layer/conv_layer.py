# -*- coding: utf8 -*-
# author: ronniecao
import numpy
import tensorflow as tf


class ConvLayer:
    
    def __init__(self, input_shape, n_size, n_filter, stride=1, activation='relu',
                 batch_normal=False, weight_decay=None, name='conv'):
        # params
        self.input_shape = input_shape
        self.n_filter = n_filter
        self.activation = activation
        self.stride = stride
        self.batch_normal = batch_normal
        self.weight_decay = weight_decay
        
        # 权重矩阵
        self.weight = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[n_size, n_size, self.input_shape[3], self.n_filter],
                mean=0.0, stddev=numpy.sqrt(
                    2.0 / (self.input_shape[1] * self.input_shape[2] * self.input_shape[3]))),
            name='W_%s' % (name))
        
        # weight decay技术
        if self.weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(self.weight), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)
            
        # 偏置向量
        self.bias = tf.Variable(
            initial_value=tf.constant(
                0.0, shape=[self.n_filter]),
            name='b_%s' % (name))
        
        # batch normalization 技术的参数
        if self.batch_normal:
            self.epsilon = 1e-5
            self.gamma = tf.Variable(
                initial_value=tf.constant(
                    1.0, shape=[self.n_filter]),
            name='gamma_%s' % (name))
        
    def get_output(self, input):
        # calculate input_shape and output_shape
        self.output_shape = [self.input_shape[0], int(self.input_shape[1]/self.stride),
                             int(self.input_shape[2]/self.stride), self.n_filter]
        
        # hidden states
        self.conv = tf.nn.conv2d(
            input=input, filter=self.weight, 
            strides=[1, self.stride, self.stride, 1], padding='SAME')
        
        # batch normalization 技术
        if self.batch_normal:
            mean, variance = tf.nn.moments(self.conv, axes=[0, 1, 2], keep_dims=False)
            self.hidden = tf.nn.batch_normalization(
                self.conv, mean, variance, self.bias, self.gamma, self.epsilon)
        else:
            self.hidden = self.conv + self.bias
            
        # activation
        if self.activation == 'relu':
            self.output = tf.nn.relu(self.hidden)
        elif self.activation == 'tanh':
            self.output = tf.nn.tanh(self.hidden)
        elif self.activation == 'none':
            self.output = self.hidden
        
        return self.output