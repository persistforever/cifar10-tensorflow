# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/09/06
from __future__ import print_function
import sys
import os
import time
import yaml
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf


class Manager():
    
    def __init__(self, config_path, backups_dir, logs_dir):
        # 读取配置
        self.option = yaml.load(open(config_path, 'r'))
        self.backups_dir = backups_dir
        self.logs_dir = logs_dir

    def init_module(self, dataloader, network):
        self.dataloader = dataloader
        self.network = network
    
    def _init_place_holders(self):
        with self.network.graph.as_default():
            # 输入变量
            images = tf.placeholder(
                dtype=tf.float32, 
                shape=[None, self.option['image_size'], self.option['image_size'], self.option['n_channel']], 
                name='images')
            labels = tf.placeholder(
                dtype=tf.int32, 
                shape=[None], 
                name='labels')
            self.place_holders = {'images': images, 'labels': labels}
            self.global_step = tf.train.get_or_create_global_step()

    def _init_train(self):
        with self.network.graph.as_default():
            # 构建会话
            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(
                gpu_options=gpu_options,
                allow_soft_placement=True)
            train_sess = tf.Session(config=config)
            
            # 获取变量
            self._init_place_holders()
            with tf.name_scope('cal_loss_and_accuracy'):
                self.avg_loss, self.avg_accuracy = self.network.get_loss(
                    self.place_holders['images'],
                    self.place_holders['labels'])
            
            # 优化器
            lr = tf.cond(
                tf.less(self.global_step, 32000), 
                lambda: tf.constant(0.1),
                lambda: tf.cond(
                    tf.less(self.global_step, 48000),
                    lambda: tf.constant(0.01),
                    lambda: tf.constant(0.001)))
            if self.option['update_function'] == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            elif self.option['update_function'] == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            elif self.option['update_function'] == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
            self.optimizer_handle = self.optimizer.minimize(
                self.avg_loss, global_step=self.global_step)
            
            train_sess.run(tf.global_variables_initializer())

        return train_sess

    def train(self):
        if not os.path.exists(self.backups_dir):
            os.mkdir(self.backups_dir)
        if not os.path.exists(os.path.join(self.backups_dir, 'basic')):
            os.mkdir(os.path.join(self.backups_dir, 'basic'))

        # 构建会话
        self.sess = self._init_train()
        
        # 模型训练
        print('\n' + '='*20 + ' start training ' + '='*20)
        for epoch in range(0, self.option['n_epoch']+1):
            
            # 数据增强
            st = time.time()
            
            train_images = self.dataloader.data_augmentation(
                self.dataloader.train_images, mode='train',
                flip=True, crop=True, whiten=True, noise=False,
                crop_shape=(self.option['image_size'], self.option['image_size'], self.option['n_channel']))
            train_labels = self.dataloader.train_labels
            
            valid_images = self.dataloader.data_augmentation(
                self.dataloader.valid_images, mode='test',
                flip=False, crop=True, whiten=True, noise=False,
                crop_shape=(self.option['image_size'], self.option['image_size'], self.option['n_channel']))
            valid_labels = self.dataloader.valid_labels
            
            et = time.time()
            data_span = et - st
            
            # 开始本轮的训练，并计算目标函数值
            st = time.time()
            
            train_loss, train_accuracy = 0.0, 0.0
            for i in range(0, self.dataloader.n_train, self.option['batch_size']):
                batch_images = train_images[i: i+self.option['batch_size']]
                batch_labels = train_labels[i: i+self.option['batch_size']]
                [_, avg_loss, avg_accuracy, iteration] = self.sess.run(
                    fetches=[self.optimizer_handle, self.avg_loss, self.avg_accuracy, self.global_step], 
                    feed_dict={self.place_holders['images']: batch_images, 
                               self.place_holders['labels']: batch_labels})
                train_loss += avg_loss * batch_images.shape[0]
                train_accuracy += avg_accuracy * batch_images.shape[0]
                
                if i % (100 * self.option['batch_size']) == 0:
                    print('epoch[%d], iter[%d], train loss: %.6f, train precision: %.6f\n' % (
                        epoch, iteration, avg_loss, avg_accuracy))
            
            train_loss = 1.0 * train_loss / self.dataloader.n_train
            train_accuracy = 1.0 * train_accuracy / self.dataloader.n_train
            
            et = time.time()
            train_span = et - st
            
            # 在训练之后，获得本轮的验证集损失值和准确率
            st = time.time()
            
            valid_loss, valid_accuracy = 0.0, 0.0
            for i in range(0, self.dataloader.n_valid, self.option['batch_size']):
                batch_images = valid_images[i: i+self.option['batch_size']]
                batch_labels = valid_labels[i: i+self.option['batch_size']]
                [avg_loss, avg_accuracy] = self.sess.run(
                    fetches=[self.avg_loss, self.avg_accuracy], 
                    feed_dict={self.place_holders['images']: batch_images, 
                               self.place_holders['labels']: batch_labels}) 
                valid_loss += avg_loss * batch_images.shape[0]
                valid_accuracy += avg_accuracy * batch_images.shape[0]
                
            valid_loss = 1.0 * valid_loss / self.dataloader.n_valid
            valid_accuracy = 1.0 * valid_accuracy / self.dataloader.n_valid
            
            et = time.time()
            valid_span = et - st
            
            print('epoch[%d], iter[%d], data time: %.2fs, train time: %.2fs, valid time: %.2fs' % (
                epoch, iteration, data_span, train_span, valid_span))
            print('epoch[%d], iter[%d], train loss: %.6f, train precision: %.6f, '
                'valid loss: %.6f, valid precision: %.6f\n' % (
                epoch, iteration, train_loss, train_accuracy, valid_loss, valid_accuracy))
            
            # 保存模型
            if epoch % 25 == 0:
                with self.network.graph.as_default():
                    saver = tf.train.Saver(
                        var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2)
                    saver_path = saver.save(
                        self.sess, os.path.join(self.backups_dir, 'model_%d.ckpt' % (epoch)))
                
        print('='*20 + ' start training ' + '='*20 + '\n')
        self.sess.close()
                
    def test(self, dataloader, backup_path, epoch, batch_size=128):
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))
        # 在测试集上计算准确率
        accuracy_list = []
        test_images = dataloader.data_augmentation(dataloader.test_images,
            flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
        test_labels = dataloader.test_labels
        for i in range(0, dataloader.n_test, batch_size):
            batch_images = test_images[i: i+batch_size]
            batch_labels = test_labels[i: i+batch_size]
            [avg_accuracy] = self.sess.run(
                fetches=[self.accuracy], 
                feed_dict={self.images:batch_images, 
                           self.labels:batch_labels,
                           self.keep_prob:1.0})
            accuracy_list.append(avg_accuracy)
        print('test precision: %.4f' % (numpy.mean(accuracy_list)))
        self.sess.close()
