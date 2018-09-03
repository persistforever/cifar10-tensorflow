# -*- coding: utf8 -*-
# author: ronniecao
# time: 20180828
# intro: pooling layer based on tensorflow.layers
import numpy
import tensorflow as tf


class PoolLayer:
    
    def __init__(self, y_size, x_size, y_stride, x_stride, mode='max', 
                 data_format='channels_last', resp_normal=False, name='pool',
                 input_shape=None, prev_layer=None):
        # params
        self.y_size = y_size
        self.x_size = x_size
        self.y_stride = y_stride
        self.x_stride = x_stride
        self.mode = mode
        self.data_format = data_format
        self.resp_normal = resp_normal
        self.name = name
        self.ltype = 'pool'
        self.params = []
        if prev_layer:
            self.input_shape = prev_layer.output_shape
            self.prev_layer = prev_layer
        elif input_shape:
            self.input_shape = input_shape
            self.prev_layer = None
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        # 计算感受野
        self.feel_field = [1, 1]
        self.feel_field[0] = min(self.input_shape[0], 1 * int(self.y_size))
        self.feel_field[1] = min(self.input_shape[1], 1 * int(self.x_size))
        prev_layer = self.prev_layer
        while prev_layer:
            if prev_layer.ltype == 'conv':
                self.feel_field[0] = min(prev_layer.input_shape[0], 
                    self.feel_field[0] + int((prev_layer.y_size+1)/2))
                self.feel_field[1] = min(prev_layer.input_shape[1], 
                    self.feel_field[1] + int((prev_layer.x_size+1)/2))
            elif prev_layer.ltype == 'pool':
                self.feel_field[0] = min(prev_layer.input_shape[0], 
                    self.feel_field[0] * int(prev_layer.y_size))
                self.feel_field[1] = min(prev_layer.input_shape[1], 
                    self.feel_field[1] * int(prev_layer.x_size))
            prev_layer = prev_layer.prev_layer
        
        with tf.name_scope('%s_def' % (self.name)) as scope:
            if self.mode == 'max':
                self.pool = tf.layers.MaxPooling2D(
                    pool_size=[self.y_size, self.x_size],
                    strides=[self.y_stride, self.x_stride],
                    padding='SAME',
                    data_format=self.data_format,
                    name='%s_pool' % (self.name))
            elif self.mode == 'avg':
                self.pool = tf.layers.AveragePooling2D(
                    pool_size=[self.y_size, self.x_size],
                    strides=[self.y_stride, self.x_stride],
                    padding='SAME',
                    data_format=self.data_format,
                    name='%s_pool' % (self.name))
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [
            int(self.input_shape[0]/self.y_stride),
            int(self.input_shape[1]/self.x_stride), 
            self.input_shape[2]]
        print('%-10s\t%-25s\t%-20s\t%-20s\t%s' % (
            self.name, 
            '((%d, %d) / (%d, %d))' % (
                self.y_size, self.x_size, self.y_stride, self.x_stride),
            '(%d, %d, %d)' % (
                self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            '(%d, %d, %d)' % (
                self.output_shape[0], self.output_shape[1], self.output_shape[2]),
            '(%d, %d)' % (
                self.feel_field[0], self.feel_field[1])))
        self.calculation = self.output_shape[0] * self.output_shape[1] * \
            self.output_shape[2] * self.y_size * self.x_size
        
    def get_output(self, inputs, is_training=True):
        with tf.name_scope('%s_cal' % (self.name)) as scope: 
            self.hidden = self.pool(inputs=inputs)
            
            if self.resp_normal:
                self.hidden = tf.nn.local_response_normalization(
                    self.hidden, depth_radius=7, alpha=0.001, beta=0.75, name='lrn')
            self.output = self.hidden
        
        return self.output
