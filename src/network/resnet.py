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
        print('%-10s\t%-25s\t%-20s\t%-20s\t%s' % ('Name', 'Filter', 'Input', 'Output', 'Field')) 
        
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
            conv_list = []
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
                conv_list.append(layer)
            self.conv_lists.append(conv_list)
        
        for layer_dict in self.network_option['net']['dense_first']:
            self.layers[layer_dict['name']] = layer = DenseLayer(
                hidden_dim=layer_dict['hidden_dim'], 
                activation=layer_dict['activation'],
                batch_normal=layer_dict['bn'], 
                input_shape=(int(self.option['image_size']/8) * int(self.option['image_size']/8) * 512, ),
                name=layer_dict['name'])
            self.dense_lists.append(layer)
        for layer_dict in self.network_option['net']['dense']:
            self.layers[layer_dict['name']] = layer = DenseLayer(
                hidden_dim=layer_dict['hidden_dim'], 
                activation=layer_dict['activation'],
                batch_normal=layer_dict['bn'],
                prev_layer=layer,
                name=layer_dict['name']) 
            self.dense_lists.append(layer)
        
        print('='*20 + ' network structure ' + '='*20 + '\n')
        
    def _inference(self, images):
        # 数据流
        hidden_conv = self.conv_lists[0].get_output(input=images)
        
        for i in range(0, self.network_option['n_blocks']):
            hidden_conv1 = self.conv_lists[1][2*i].get_output(input=hidden_conv)
            hidden_conv2 = self.conv_lists[1][2*i+1].get_output(input=hidden_conv1)
            hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)
        
        for i in range(2, self.network_option['n_layers']+1):
            hidden_conv1 = self.conv_lists[i][0].get_output(input=hidden_conv)
            hidden_conv2 = self.conv_lists[i][1].get_output(input=hidden_conv1)
            hidden_pool = tf.nn.max_pool(
                hidden_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            hidden_pad = tf.pad(hidden_pool, [[0,0], [0,0], [0,0], [2**(i+3),2**(i+3)]])
            hidden_conv = tf.nn.relu(hidden_pad + hidden_conv2)
            for j in range(1, self.network_option['n_blocks']):
                hidden_conv1 = self.conv_lists[i][2*j].get_output(input=hidden_conv)
                hidden_conv2 = self.conv_lists[i][2*j+1].get_output(input=hidden_conv1)
                hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)
        
        hidden_state = tf.reshape(hidden_conv, [
            -1, int(self.option['image_size']/8) * int(self.option['image_size']/8) * 512])
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
        for name, layer in self.layers.items():
            if layer.ltype == 'conv':
                for weight in layer.conv.weights:
                    self.l2_objective += tf.multiply(tf.nn.l2_loss(weight), self.option['weight_decay'])
                for weight in layer.bn.weights:
                    if 'gamma' in weight.name:
                        self.l2_objective += tf.multiply(tf.nn.l2_loss(weight), self.option['weight_decay'])
            if layer.ltype == 'dense':
                for weight in layer.dense.weights:
                    self.l2_objective += tf.multiply(tf.nn.l2_loss(weight), self.option['weight_decay'])
                if 'bn' in layer.__dict__:
                    for weight in layer.bn.weights:
                        if 'gamma' in weight.name:
                            self.l2_objective += tf.multiply(tf.nn.l2_loss(weight), self.option['weight_decay'])

        self.avg_loss = self.classify_objective + self.l2_objective
        
        # 观察值
        correct_prediction = tf.equal(labels, tf.cast(tf.argmax(logits, 1), dtype=tf.int32))
        self.avg_accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        return self.avg_loss, self.avg_accuracy
