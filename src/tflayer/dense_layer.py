# -*- coding: utf8 -*-
# author: ronniecao
# time: 20180828
# intro: fully connected layer based on tensorflow.layers
import math
import numpy
import tensorflow as tf


class DenseLayer:
    
    def __init__(self, 
        hidden_dim, 
        activation='relu', batch_normal=False, dropout=False, keep_prob=0.0, 
        input_shape=None, prev_layer=None,
        name='dense'):

        # params
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.batch_normal = batch_normal
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.name = name
        self.ltype = 'dense'
        self.params = {}
        
        if prev_layer:
            self.prev_layer = prev_layer
            self.input_shape = prev_layer.output_shape
        elif input_shape:
            self.input_shape = input_shape
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        with tf.name_scope('%s_def' % (self.name)) as scope:
            weight_initializer = tf.variance_scaling_initializer(
                scale=2.0, mode='fan_in', distribution='normal', dtype=tf.float32)
            bias_initializer = tf.zeros_initializer(dtype=tf.float32)
            
            self.dense = tf.layers.Dense(
                units=self.hidden_dim,
                activation=None,
                use_bias=not self.batch_normal,
                kernel_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                trainable=True,
                name='%s_dense' % (self.name))
            
            if self.batch_normal:
                beta_initializer = tf.zeros_initializer(dtype=tf.float32)
                gamma_initializer = tf.ones_initializer(dtype=tf.float32)
                moving_mean_initializer = tf.zeros_initializer(dtype=tf.float32)
                moving_variance_initializer = tf.ones_initializer(dtype=tf.float32)
                
                self.bn = tf.layers.BatchNormalization(
                    axis=-1,
                    momentum=0.9,
                    epsilon=1e-3,
                    center=True,
                    scale=True,
                    beta_initializer=beta_initializer,
                    gamma_initializer=gamma_initializer,
                    moving_mean_initializer=moving_mean_initializer,
                    moving_variance_initializer=moving_variance_initializer,
                    fused=True,
                    trainable=True,
                    name='%s_bn' % (self.name))
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [self.hidden_dim]
        print('%-30s\t%-25s\t%-20s\t%-20s' % (
            self.name, 
            '(%d)' % (self.hidden_dim),
            '(%d)' % (self.input_shape[0]),
            '(%d)' % (self.output_shape[0])))
        if self.batch_normal:
            print('%-30s\t%-25s\t%-20s\t%-20s' % (
                self.name + '(bn)', '()', 
                '(%d, %d, %d)' % (
                    self.output_shape[0], self.output_shape[1], self.output_shape[2]),
                '(%d, %d, %d)' % (
                    self.output_shape[0], self.output_shape[1], self.output_shape[2])))
        if self.activation != 'none':
            print('%-30s\t%-25s\t%-20s\t%-20s' % (
                self.name + '(%s)' % (self.activation), '()', 
                '(%d, %d, %d)' % (
                    self.output_shape[0], self.output_shape[1], self.output_shape[2]),
                '(%d, %d, %d)' % (
                    self.output_shape[0], self.output_shape[1], self.output_shape[2])))
        self.calculation = self.input_shape[0] * self.output_shape[0]
        
    def get_output(self, input, is_training=tf.constant(True)):
                
        # hidden states
        self.hidden = self.dense(input)
        
        # batch normalization 技术
        if self.batch_normal:
            self.hidden = self.bn(self.hidden, training=tf.constant(True))
            
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
    
        # 获取params
        if self.batch_normal:
            for tensor in self.dense.weights:
                if 'kernel' in tensor.name:
                    self.params['%s#%s' % (self.name, 'weight')] = tensor
            for tensor in self.bn.weights:
                if 'gamma' in tensor.name:
                    self.params['%s#%s' % (self.name, 'gamma')] = tensor
        else:
            for tensor in self.dense.weights:
                if 'kernel' in tensor.name:
                    self.params['%s#%s' % (self.name, 'weight')] = tensor

        return self.output
    
    def leaky_relu(self, data):
        hidden = tf.cast(data, dtype=tf.float32)
        mask = tf.cast((hidden > 0), dtype=tf.float32)
        output = 1.0 * mask * hidden + 0.1 * (1 - mask) * hidden
        
        return output
