# -*- coding: utf8 -*-
# author: ronniecao
# time: 20180828
# intro: pooling layer based on tensorflow.layers
import numpy
import tensorflow as tf


class PoolLayer:
    
    def __init__(self, 
        y_size, x_size, y_stride, x_stride, 
        mode='max', resp_normal=False,
        data_format='channels_first', input_shape=None, prev_layer=None,
        name='pool'):
        
        # params
        self.y_size = y_size
        self.x_size = x_size
        self.y_stride = y_stride
        self.x_stride = x_stride
        self.mode = mode
        self.resp_normal = resp_normal
        self.data_format = data_format
        self.name = name
        self.ltype = 'pool'
        self.params = {}

        if prev_layer:
            self.input_shape = prev_layer.output_shape
            self.prev_layer = prev_layer
        elif input_shape:
            self.input_shape = input_shape
            self.prev_layer = None
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [
            int(self.input_shape[0]/self.y_stride),
            int(self.input_shape[1]/self.x_stride), 
            self.input_shape[2]]
        print('%-30s\t%-25s\t%-20s\t%-20s' % (
            self.name, 
            '((%d, %d) / (%d, %d))' % (
                self.y_size, self.x_size, self.y_stride, self.x_stride),
            '(%d, %d, %d)' % (
                self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            '(%d, %d, %d)' % (
                self.output_shape[0], self.output_shape[1], self.output_shape[2])))
        self.calculation = self.output_shape[0] * self.output_shape[1] * \
            self.output_shape[2] * self.y_size * self.x_size
        
    def get_output(self, input, is_training=True):
        with tf.name_scope('%s_cal' % (self.name)) as scope: 
            if self.data_format == 'channels_first':
                input = tf.transpose(input, [0,3,1,2])
            
            self.hidden = self.pool(inputs=input)
            
            if self.resp_normal:
                self.hidden = tf.nn.local_response_normalization(
                    self.hidden, depth_radius=7, alpha=0.001, beta=0.75, name='lrn')
            self.output = self.hidden
        
        if self.data_format == 'channels_first':
            self.output = tf.transpose(self.output, [0,2,3,1])
        
        return self.output
