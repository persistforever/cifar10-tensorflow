# -*- coding: utf8 -*-
# author: ronniecao
# time: 20180910
# intro: residual block based on tensorflow.layers
import numpy
import math
import tensorflow as tf
import random
from src.tflayer.conv_layer import ConvLayer
from src.tflayer.pool_layer import PoolLayer
from src.tflayer.dense_layer import DenseLayer
from src.tflayer.batch_normal_layer import BatchNormLayer


class ResidualBlock:
    
    def __init__(self, 
        y_size, x_size, y_stride, x_stride, n_filter, 
        activation='relu', batch_normal=False,
        data_format='channels_first', prev_layer=None,
        name='conv'):
        
        # params
        self.y_size = y_size
        self.x_size = x_size
        self.y_stride = y_stride
        self.x_stride = x_stride
        self.n_filter = n_filter
        self.activation = activation
        self.batch_normal = batch_normal
        self.data_format = data_format
        self.name = name
        self.ltype = 'residual'
        self.params = {}
        
        if prev_layer:
            self.prev_layer = prev_layer
            self.input_shape = prev_layer.output_shape
        elif input_shape:
            self.prev_layer = None
            self.input_shape = input_shape
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        # 定义网络结构
        with tf.name_scope('%s_def' % (self.name)) as scope:
            
            self.first_layer = ConvLayer(
                x_size=self.x_size, 
                y_size=self.y_size, 
                x_stride=self.x_stride, 
                y_stride=self.y_stride, 
                n_filter=self.n_filter, 
                activation=self.activation, 
                batch_normal=self.batch_normal, 
                data_format=self.data_format, 
                prev_layer=self.prev_layer,
                name='%s_%s' % (self.name, 'conv1'))
            
            self.second_layer = ConvLayer(
                x_size=self.x_size, 
                y_size=self.y_size, 
                x_stride=1, 
                y_stride=1, 
                n_filter=self.n_filter, 
                activation='none', 
                batch_normal=False, 
                data_format=self.data_format, 
                prev_layer=self.first_layer,
                name='%s_%s' % (self.name, 'conv2'))
            
            self.third_layer = BatchNormLayer(
                activation=self.activation, 
                data_format=self.data_format, 
                prev_layer=self.second_layer,
                name='%s_%s' % (self.name, 'bn'))

        self.output_shape = self.third_layer.output_shape
        
    def get_output(self, input):
        
        with tf.name_scope('%s_cal' % (self.name)) as scope:
            
            hidden_conv1 = self.first_layer.get_output(input=input)
            hidden_conv2 = self.second_layer.get_output(input=hidden_conv1)
            
            # 调整input的y和x尺寸
            if self.x_stride != 1 or self.y_stride != 1:
                hidden_pool = tf.nn.avg_pool(
                    input, ksize=[1,self.y_stride,self.x_stride,1], 
                    strides=[1,self.y_stride,self.x_stride,1], padding='VALID')
            else:
                hidden_pool = input

            # 调整input的n尺寸
            prev_hidden_dim = self.input_shape[2]
            pad_dim = int((self.n_filter - prev_hidden_dim) / 2.0)
            if pad_dim > 0:
                hidden_pad = tf.pad(hidden_pool, [[0,0], [0,0], [0,0], [pad_dim, pad_dim]])
            else:
                hidden_pad = hidden_pool
            
            hidden_conv = hidden_pad + hidden_conv2
            hidden_output = self.third_layer.get_output(input=hidden_conv)

            self.output = hidden_output

        # 获取params
        for name, tensor in self.first_layer.params.items():
            self.params[name] = tensor
        for name, tensor in self.second_layer.params.items():
            self.params[name] = tensor
        for name, tensor in self.third_layer.params.items():
            self.params[name] = tensor

        return self.output
