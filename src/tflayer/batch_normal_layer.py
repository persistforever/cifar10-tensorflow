# -*- coding: utf8 -*-
# author: ronniecao
# time: 20180911
# intro: batch normalization layer based on tensorflow.layers
import numpy
import math
import tensorflow as tf
import random


class BatchNormLayer:
    
    def __init__(self, 
        activation='relu',
        data_format='channels_first', input_shape=None, prev_layer=None,
        name='bn'):

        # params
        self.activation = activation
        self.data_format = data_format
        self.name = name
        self.ltype = 'bn'
        self.params = {}
        
        if prev_layer:
            self.prev_layer = prev_layer
            self.input_shape = prev_layer.output_shape
        elif input_shape:
            self.prev_layer = None
            self.input_shape = input_shape
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        self.leaky_scale = tf.constant(0.1, dtype=tf.float32)
    
        with tf.name_scope('%s_def' % (self.name)) as scope:
            # 权重矩阵
            beta_initializer = tf.zeros_initializer(dtype=tf.float32)
            gamma_initializer = tf.ones_initializer(dtype=tf.float32)
            moving_mean_initializer = tf.zeros_initializer(dtype=tf.float32)
            moving_variance_initializer = tf.ones_initializer(dtype=tf.float32)
            
            self.bn = tf.layers.BatchNormalization(
                axis=-1 if self.data_format == 'channels_last' else 1,
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

        self.output_shape = self.input_shape
        print('%-30s\t%-25s\t%-20s\t%-20s' % (
            self.name + '(bn)', '()',
            '(%d, %d, %d)' % (
                self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            '(%d, %d, %d)' % (
                self.output_shape[0], self.output_shape[1], self.output_shape[2])))
        if self.activation != 'none':
            print('%-30s\t%-25s\t%-20s\t%-20s' % (
                self.name + '(%s)' % (self.activation), '()', 
                '(%d, %d, %d)' % (
                    self.output_shape[0], self.output_shape[1], self.output_shape[2]),
                '(%d, %d, %d)' % (
                    self.output_shape[0], self.output_shape[1], self.output_shape[2])))
        self.calculation = 0
        
    def get_output(self, input, is_training=True):
        with tf.name_scope('%s_cal' % (self.name)) as scope:
            # hidden states
            if self.data_format == 'channels_first':
                self.hidden = tf.transpose(input, [0,3,1,2])
            else:
                self.hidden = input
                
            # batch normalization 技术
            self.hidden = self.bn(self.hidden, training=tf.constant(True))
            
            # activation
            if self.activation == 'relu':
                self.output = tf.nn.relu(self.hidden)
            elif self.activation == 'tanh':
                self.output = tf.nn.tanh(self.hidden)
            elif self.activation == 'leaky_relu':
                self.output = self.leaky_relu(self.hidden)
            elif self.activation == 'sigmoid':
                self.output = tf.nn.sigmoid(self.hidden)
            elif self.activation == 'none':
                self.output = self.hidden
            
            if self.data_format == 'channels_first':
                self.output = tf.transpose(self.output, [0,2,3,1])
        
        # 获取params
        for tensor in self.bn.weights:
            if 'gamma' in tensor.name:
                self.params['%s#%s' % (self.name, 'gamma')] = tensor
        
        return self.output
    
    def leaky_relu(self, data):
        output = tf.maximum(self.leaky_scale * data, data, name='leaky_relu')
        
        return output
