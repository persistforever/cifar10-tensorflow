# -*- coding: utf8 -*-
# author: ronniecao
import numpy
import tensorflow as tf


class PoolLayer:
    
    def __init__(self, n_size=2, stride=2, mode='max', padding='SAME',
                 resp_normal=False, name='pool'):
        # params
        self.n_size = n_size
        self.stride = stride
        self.mode = mode
        self.padding = padding
        self.resp_normal = resp_normal
        
    def get_output(self, input):
        if self.mode == 'max':
            self.pool = tf.nn.max_pool(
                value=input, ksize=[1, self.n_size, self.n_size, 1],
                strides=[1, self.stride, self.stride, 1], padding=self.padding)
        elif self.mode == 'avg':
            self.pool = tf.nn.avg_pool(
                value=input, ksize=[1, self.n_size, self.n_size, 1],
                strides=[1, self.stride, self.stride, 1], padding=self.padding)
        
        # response normal 技术
        if self.resp_normal:
            self.hidden = tf.nn.local_response_normalization(
                self.pool, depth_radius=7, alpha=0.001, beta=0.75)
        else:
            self.hidden = self.pool
            
        self.output = self.hidden
        
        return self.output