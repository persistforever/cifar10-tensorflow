# -*- encoding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import time
import yaml
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.tflayers.conv_layer import ConvLayer
from src.tflayers.pool_layer import PoolLayer
from src.tflayers.dense_layer import DenseLayer

class ConvNet():
    
    def __init__(self, network_path, n_channel=3, n_classes=10, image_size=24):
        self.image_size = image_size

        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, image_size, image_size, n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable( 
            0, dtype=tf.int32, name='global_step')
        
        network_option_path = os.path.join(network_path)
        self.network_option = yaml.load(open(network_option_path, 'r'))
        
        # 网络结构
        print()
        self.conv_lists, self.dense_lists = [], []
        for layer_dict in self.network_option['net']['conv_first']:
            layer = ConvLayer(
                x_size=layer_dict['x_size'], y_size=layer_dict['y_size'], 
                x_stride=layer_dict['x_stride'], y_stride=layer_dict['y_stride'], 
                n_filter=layer_dict['n_filter'], activation=layer_dict['activation'], 
                batch_normal=layer_dict['bn'], weight_decay=1e-4, 
                data_format='channels_last', name=layer_dict['name'], 
                input_shape=(image_size, image_size, n_channel))
            self.conv_lists.append(layer)
        
        for i in range(self.network_option['n_layers']):
            conv_list = []
            for layer_dict in self.network_option['net']['conv_layer%d' % (i+1)]:
                if layer_dict['type'] == 'conv':
                    layer = ConvLayer(
                        x_size=layer_dict['x_size'], y_size=layer_dict['y_size'], 
                        x_stride=layer_dict['x_stride'], y_stride=layer_dict['y_stride'], 
                        n_filter=layer_dict['n_filter'], activation=layer_dict['activation'], 
                        batch_normal=layer_dict['bn'], weight_decay=1e-4, 
                        data_format='channels_last', name=layer_dict['name'], prev_layer=layer)
                conv_list.append(layer)
            self.conv_lists.append(conv_list)
        
        for layer_dict in self.network_option['net']['dense_first']:
            layer = DenseLayer(
                hidden_dim=layer_dict['hidden_dim'], activation=layer_dict['activation'],
                dropout=layer_dict['dropout'], keep_prob=self.keep_prob,
                batch_normal=layer_dict['bn'], weight_decay=1e-4, 
                name=layer_dict['name'],
                input_shape=(int(image_size/8) * int(image_size/8) * 512, ))
            self.dense_lists.append(layer)
        for layer_dict in self.network_option['net']['dense']:
            layer = DenseLayer(
                hidden_dim=layer_dict['hidden_dim'], activation=layer_dict['activation'],
                dropout=layer_dict['dropout'], keep_prob=self.keep_prob,
                batch_normal=layer_dict['bn'], weight_decay=1e-4, 
                name=layer_dict['name'], prev_layer=layer)
            self.dense_lists.append(layer)
        print()

        self._inference(self.images)
        
    def _inference(self, images):
        # 数据流
        hidden_conv = self.conv_lists[0].get_output(inputs=images)
        
        for i in range(0, self.network_option['n_blocks']):
            hidden_conv1 = self.conv_lists[1][2*i].get_output(inputs=hidden_conv)
            hidden_conv2 = self.conv_lists[1][2*i+1].get_output(inputs=hidden_conv1)
            hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)
        
        for i in range(2, self.network_option['n_layers']+1):
            hidden_conv1 = self.conv_lists[i][0].get_output(inputs=hidden_conv)
            hidden_conv2 = self.conv_lists[i][1].get_output(inputs=hidden_conv1)
            hidden_pool = tf.nn.max_pool(
                hidden_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            hidden_pad = tf.pad(hidden_pool, [[0,0], [0,0], [0,0], [2**(i+3),2**(i+3)]])
            hidden_conv = tf.nn.relu(hidden_pad + hidden_conv2)
            for j in range(1, self.network_option['n_blocks']):
                hidden_conv1 = self.conv_lists[i][2*j].get_output(inputs=hidden_conv)
                hidden_conv2 = self.conv_lists[i][2*j+1].get_output(inputs=hidden_conv1)
                hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)
        
        hidden_state = tf.reshape(hidden_conv, [-1, int(self.image_size/8) * int(self.image_size/8) * 512])
        for layer in self.dense_lists:
            hidden_state = layer.get_output(inputs=hidden_state)
        logits = hidden_state
            
        # 目标函数
        self.objective = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels))
        self.avg_loss = self.objective
        
        # 优化器
        lr = tf.cond(tf.less(self.global_step, 50000), 
                     lambda: tf.constant(0.001),
                     lambda: tf.cond(tf.less(self.global_step, 100000), 
                                     lambda: tf.constant(0.0001),
                                     lambda: tf.constant(0.00001)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)
        
        # 观察值
        correct_prediction = tf.equal(self.labels, tf.argmax(logits, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        
    def train(self, dataloader, backup_path, n_epoch=5, batch_size=128):
        if not os.path.exists(backup_path):
            os.mkdir(backup_path)

        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=10)
        
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        
        # 模型训练
        print()
        for epoch in range(0, n_epoch+1):
            
            # 数据增强
            st = time.time()
            train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                flip=True, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            train_labels = dataloader.train_labels
            valid_images = dataloader.data_augmentation(dataloader.valid_images, mode='test',
                flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            valid_labels = dataloader.valid_labels
            et = time.time()
            data_span = et - st
            
            # 开始本轮的训练，并计算目标函数值
            train_loss = 0.0
            st = time.time()
            for i in range(0, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                [_, avg_loss, iteration] = self.sess.run(
                    fetches=[self.optimizer, self.avg_loss, self.global_step], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 0.5})
                train_loss += avg_loss * batch_images.shape[0]
            et = time.time()
            train_span = et - st
            average_loss = 1.0 * train_loss / dataloader.n_train
            
            # 在训练之后，获得本轮的训练集损失值和准确率
            train_accuracy, train_loss = 0.0, 0.0
            for i in range(0, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 1.0})
                train_accuracy += avg_accuracy * batch_images.shape[0]
                train_loss += avg_loss * batch_images.shape[0]
            train_accuracy = 1.0 * train_accuracy / dataloader.n_train
            train_loss = 1.0 * train_loss / dataloader.n_train
            
            # 在训练之后，获得本轮的验证集损失值和准确率
            valid_accuracy, valid_loss = 0.0, 0.0
            for i in range(0, dataloader.n_valid, batch_size):
                batch_images = valid_images[i: i+batch_size]
                batch_labels = valid_labels[i: i+batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 1.0})
                valid_accuracy += avg_accuracy * batch_images.shape[0]
                valid_loss += avg_loss * batch_images.shape[0]
            valid_accuracy = 1.0 * valid_accuracy / dataloader.n_valid
            valid_loss = 1.0 * valid_loss / dataloader.n_valid
            
            print('epoch[%d], iter[%d], data time: %.2fs, train time: %.2fs' % (
                epoch, iteration, data_span, train_span))
            print('epoch[%d], iter[%d], train loss: %.6f, train precision: %.6f, '
                'valid loss: %.6f, valid precision: %.6f\n' % (
                epoch, iteration, train_loss, train_accuracy, valid_loss, valid_accuracy))
            
            # 保存模型
            if epoch <= 1000 and epoch % 100 == 0 or \
                epoch <= 10000 and epoch % 1000 == 0:
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))
                
        self.sess.close()
                
    def test(self, dataloader, backup_path, epoch, batch_size=128):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
            
    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [temp] = sess.run(
            fetches=[self.observe],
            feed_dict={self.images: numpy.random.random(size=[128, 24, 24, 3]),
                       self.labels: numpy.random.randint(low=0, high=9, size=[128,]),
                       self.keep_prob: 1.0})
        print(temp)
        
    def observe_salience(self, batch_size=128, image_h=32, image_w=32, n_channel=3, 
                         num_test=10, epoch=1):
        if not os.path.exists('results/epoch%d/' % (epoch)):
            os.makedirs('results/epoch%d/' % (epoch))
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        sess = tf.Session()
        # 读取模型
        model_path = 'backup/cifar10/model_%d.ckpt' % (epoch)
        assert(os.path.exists(model_path+'.index'))
        saver.restore(sess, model_path)
        print('read model from %s' % (model_path))
        # 获取图像并计算梯度
        for batch in range(num_test):
            batch_image, batch_label = cifar10.test.next_batch(batch_size)
            image = numpy.array(batch_image.reshape([image_h, image_w, n_channel]) * 255,
                                dtype='uint8')
            result = sess.run([self.labels_prob, self.labels_max_prob, self.labels_pred,
                               self.gradient],
                              feed_dict={self.images:batch_image, self.labels:batch_label,
                                         self.keep_prob:0.5})
            print(result[0:3], result[3][0].shape)
            gradient = sess.run(self.gradient, feed_dict={
                self.images:batch_image, self.keep_prob:0.5})
            gradient = gradient[0].reshape([image_h, image_w, n_channel])
            gradient = numpy.max(gradient, axis=2)
            gradient = numpy.array((gradient - gradient.min()) * 255
                                    / (gradient.max() - gradient.min()), dtype='uint8')
            print(gradient.shape)
            # 使用pyplot画图
            plt.subplot(121)
            plt.imshow(image)
            plt.subplot(122)
            plt.imshow(gradient, cmap=plt.cm.gray)
            plt.savefig('results/epoch%d/result_%d.png' % (epoch, batch))
        
    def observe_hidden_distribution(self, batch_size=128, image_h=32, image_w=32, n_channel=3, 
                                    num_test=10, epoch=1):
        if not os.path.exists('results/epoch%d/' % (epoch)):
            os.makedirs('results/epoch%d/' % (epoch))
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        sess = tf.Session()
        # 读取模型
        model_path = 'backup/cifar10/model_%d.ckpt' % (epoch)
        if os.path.exists(model_path+'.index'):
            saver.restore(sess, model_path)
            print('read model from %s' % (model_path))
        else:
            sess.run(tf.global_variables_initializer())
        # 获取图像并计算梯度
        for batch in range(num_test):
            batch_image, batch_label = cifar10.test.next_batch(batch_size)
            result = sess.run([self.nobn_conv1, self.bn_conv1, self.nobn_conv2, self.bn_conv2,
                               self.nobn_conv3, self.bn_conv3, self.nobn_fc1, self.nobn_fc1,
                               self.nobn_softmax, self.bn_softmax],
                              feed_dict={self.images:batch_image, self.labels:batch_label,
                                         self.keep_prob:0.5})
            distribution1 = result[0][:,0].flatten()
            distribution2 = result[1][:,0].flatten()
            distribution3 = result[2][:,0].flatten()
            distribution4 = result[3][:,0].flatten()
            distribution5 = result[4][:,0].flatten()
            distribution6 = result[5][:,0].flatten()
            distribution7 = result[6][:,0].flatten()
            distribution8 = result[7][:,0].flatten()
            plt.subplot(241)
            plt.hist(distribution1, bins=50, color='#1E90FF')
            plt.title('convolutional layer 1')
            plt.subplot(242)
            plt.hist(distribution3, bins=50, color='#1C86EE')
            plt.title('convolutional layer 2')
            plt.subplot(243)
            plt.hist(distribution5, bins=50, color='#1874CD')
            plt.title('convolutional layer 3')
            plt.subplot(244)
            plt.hist(distribution7, bins=50, color='#5CACEE')
            plt.title('full connection layer')
            plt.subplot(245)
            plt.hist(distribution2, bins=50, color='#00CED1')
            plt.title('batch normalized')
            plt.subplot(246)
            plt.hist(distribution4, bins=50, color='#48D1CC')
            plt.title('batch normalized')
            plt.subplot(247)
            plt.hist(distribution6, bins=50, color='#40E0D0')
            plt.title('batch normalized')
            plt.subplot(248)
            plt.hist(distribution8, bins=50, color='#00FFFF')
            plt.title('batch normalized')
            plt.show()
