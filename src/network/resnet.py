# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/09/06
# intro: ResNet in image classification
from __future__ import print_function
import sys
import os
import time
import yaml
import numpy
import tensorflow as tf
from src.tflayer.conv_layer import ConvLayer
from src.tflayer.pool_layer import PoolLayer
from src.tflayer.dense_layer import DenseLayer
from src.tflayer.residual_block import ResidualBlock


class Network():
    
    def __init__(self, config_path, network_config_path):
        
        # 读取配置
        self.option = yaml.load(open(config_path, 'r'))
        self.network_option = yaml.load(open(network_config_path, 'r'))
        
        # 初始化graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._network_structure()

    def _network_structure(self):
        # 网络结构
        print('\n' + '='*20 + ' network structure ' + '='*20)
        print('%-30s\t%-25s\t%-20s\t%-20s' % ('Name', 'Filter', 'Input', 'Output')) 
        
        self.layers = {}
        self.conv_lists, self.dense_lists = [], []
        for layer_dict in self.network_option['net']['conv_first']:
            self.layers[layer_dict['name']] = layer = ConvLayer(
                x_size=layer_dict['x_size'], 
                y_size=layer_dict['y_size'], 
                x_stride=layer_dict['x_stride'], 
                y_stride=layer_dict['y_stride'], 
                n_filter=layer_dict['n_filter'], 
                activation=layer_dict['activation'], 
                batch_normal=layer_dict['bn'], 
                data_format=self.option['data_format'], 
                input_shape=(self.option['image_size'], self.option['image_size'], self.option['n_channel']),
                name=layer_dict['name'])
            self.conv_lists.append(layer)
        
        for i in range(self.network_option['n_layers']):
            for layer_dict in self.network_option['net']['conv_layer%d' % (i+1)]:
                if layer_dict['type'] == 'conv':
                    self.layers[layer_dict['name']] = layer = ConvLayer(
                        x_size=layer_dict['x_size'], 
                        y_size=layer_dict['y_size'], 
                        x_stride=layer_dict['x_stride'], 
                        y_stride=layer_dict['y_stride'], 
                        n_filter=layer_dict['n_filter'], 
                        activation=layer_dict['activation'], 
                        batch_normal=layer_dict['bn'], 
                        data_format=self.option['data_format'], 
                        prev_layer=layer, 
                        name=layer_dict['name'])
                elif layer_dict['type'] == 'residual':
                    self.layers[layer_dict['name']] = layer = ResidualBlock(
                        x_size=layer_dict['x_size'], 
                        y_size=layer_dict['y_size'], 
                        x_stride=layer_dict['x_stride'], 
                        y_stride=layer_dict['y_stride'], 
                        n_filter=layer_dict['n_filter'], 
                        activation=layer_dict['activation'], 
                        batch_normal=layer_dict['bn'], 
                        data_format=self.option['data_format'], 
                        prev_layer=layer, 
                        name=layer_dict['name'])
                self.conv_lists.append(layer)
        
        for layer_dict in self.network_option['net']['dense_first']:
            self.layers[layer_dict['name']] = layer = DenseLayer(
                hidden_dim=layer_dict['hidden_dim'], 
                activation=layer_dict['activation'],
                batch_normal=layer_dict['bn'], 
                input_shape=(64, ),
                name=layer_dict['name'])
            self.dense_lists.append(layer)
        
        print('='*20 + ' network structure ' + '='*20 + '\n')
        
    def _inference(self, images):
        # 数据流
        hidden_state = images
        for layer in self.conv_lists:
            hidden_state = layer.get_output(input=hidden_state)

        # global average pooling
        hidden_state = tf.reduce_mean(hidden_state, axis=[1,2])

        # classification
        for layer in self.dense_lists:
            hidden_state = layer.get_output(input=hidden_state)
        logits = hidden_state

        return logits
            
    def get_loss(self, images, labels):
        logits = self._inference(images)

        # 分类目标函数
        self.classify_objective = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
        
        # l2正则化目标函数
        self.l2_objective = 0.0
        for layer_name, layer in self.layers.items():
            for tensor_name, tensor in layer.params.items():
                self.l2_objective += tf.multiply(tf.nn.l2_loss(tensor), self.option['weight_decay'])

        self.avg_loss = self.classify_objective + self.l2_objective
        
        # 观察值
        correct_prediction = tf.equal(labels, tf.cast(tf.argmax(tf.nn.softmax(logits), 1), dtype=tf.int32))
        self.avg_accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        return self.avg_loss, self.avg_accuracy
