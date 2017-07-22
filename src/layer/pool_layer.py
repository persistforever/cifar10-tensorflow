# -*- encoding: utf8 -*-
# author: ronniecao
import numpy
import tensorflow as tf


class PoolLayer:
    
    def __init__(self, n_size=2, stride=2, mode='max', name='pool'):
        # params
        self.n_size = n_size
        self.stride = stride
        self.mode = mode
        
    def get_output(self, input):
        if self.mode == 'max':
            hidden = tf.nn.max_pool(
                value=input, ksize=[1, self.n_size, self.n_size, 1],
                strides=[1, self.stride, self.stride, 1], padding='VALID')
        elif self.mode == 'avg':
            hidden = tf.nn.avg_pool(
                value=input, ksize=[1, self.n_size, self.n_size, 1],
                strides=[1, self.stride, self.stride, 1], padding='VALID')
        self.output = hidden
        
        return self.output