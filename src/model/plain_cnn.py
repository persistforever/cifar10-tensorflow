# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import yaml
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.tflayers.conv_layer import ConvLayer
from src.tflayers.dense_layer import DenseLayer
from src.tflayers.pool_layer import PoolLayer

class ConvNet():
    
    def __init__(self, n_channel=3, n_classes=10, image_size=24):
        # 设置超参数
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.image_size = image_size
        
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, self.image_size, self.image_size, self.n_channel], 
            name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, name='global_step')
        
        # 网络输出
        self.logits = self.inference(self.images)
        # 目标函数
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels))
        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        # 优化器
        lr = tf.cond(tf.less(self.global_step, 50000), 
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 100000),
                                     lambda: tf.constant(0.005),
                                     lambda: tf.cond(tf.less(self.global_step, 150000),
                                                     lambda: tf.constant(0.0025),
                                                     lambda: tf.constant(0.001))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)
        
        # 观察值
        correct_prediction = tf.equal(self.labels, tf.argmax(self.logits, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        
    def inference(self, images):
        network_option_path = os.path.join('src/network/vgg.yaml')
        self.network_option = yaml.load(open(network_option_path, 'r'))
        
        # 网络结构
        self.layer_lists = []
        for layer_dict in self.network_option['vgg']['conv_first']:
            layer = ConvLayer(
                x_size=layer_dict['x_size'], y_size=layer_dict['y_size'], 
                x_stride=layer_dict['x_stride'], y_stride=layer_dict['y_stride'], 
                n_filter=layer_dict['n_filter'], activation=layer_dict['activation'], 
                batch_normal=layer_dict['bn'], weight_decay=0.0, 
                name=layer_dict['name'], 
                input_shape=(self.image_size, self.image_size, self.n_channel))
            self.layer_lists.append(layer)
        
        for layer_dict in self.network_option['vgg']['conv']:
            if layer_dict['type'] == 'conv':
                layer = ConvLayer(
                    x_size=layer_dict['x_size'], y_size=layer_dict['y_size'], 
                    x_stride=layer_dict['x_stride'], y_stride=layer_dict['y_stride'], 
                    n_filter=layer_dict['n_filter'], activation=layer_dict['activation'], 
                    batch_normal=layer_dict['bn'], weight_decay=0.0, 
                    name=layer_dict['name'], prev_layer=layer)
            elif layer_dict['type'] == 'pool':
                layer = PoolLayer(
                    x_size=layer_dict['x_size'], y_size=layer_dict['y_size'], 
                    x_stride=layer_dict['x_stride'], y_stride=layer_dict['y_stride'], 
                    mode=layer_dict['mode'], resp_normal=False, 
                    name=layer_dict['name'], prev_layer=layer)
            self.layer_lists.append(layer)
        
        for layer_dict in self.network_option['vgg']['dense_first']:
            layer = DenseLayer(
                hidden_dim=layer_dict['hidden_dim'], activation=layer_dict['activation'],
                dropout=layer_dict['dropout'], keep_prob=self.keep_prob,
                batch_normal=layer_dict['bn'], weight_decay=0.0, 
                name=layer_dict['name'],
                input_shape=(int(self.image_size/8) * int(self.image_size/8) * 256, ))
            self.layer_lists.append(layer)
        for layer_dict in self.network_option['vgg']['dense']:
            layer = DenseLayer(
                hidden_dim=layer_dict['hidden_dim'], activation=layer_dict['activation'],
                dropout=layer_dict['dropout'], keep_prob=self.keep_prob,
                batch_normal=layer_dict['bn'], weight_decay=0.0, 
                name=layer_dict['name'], prev_layer=layer)
            self.layer_lists.append(layer)
        
        # 数据流
        hidden_state = images
        for layer in self.layer_lists[0:21]:
            hidden_state = layer.get_output(inputs=hidden_state)
        hidden_state = tf.reshape(hidden_state, (-1, int(self.image_size/8) * int(self.image_size/8) * 256))
        for layer in self.layer_lists[21:]:
            hidden_state = layer.get_output(inputs=hidden_state)
        logits = hidden_state
        
        return logits
        
    def train(self, dataloader, backup_path, n_epoch=5, batch_size=128):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=5)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        # 模型训练
        for epoch in range(0, n_epoch+1):
            # 数据增强
            train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                flip=True, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            train_labels = dataloader.train_labels
            valid_images = dataloader.data_augmentation(dataloader.valid_images, mode='test',
                flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            valid_labels = dataloader.valid_labels
            
            # 开始本轮的训练，并计算目标函数值
            train_loss = 0.0
            for i in range(0, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                [_, avg_loss, iteration] = self.sess.run(
                    fetches=[self.optimizer, self.avg_loss, self.global_step], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 0.5})
                
                train_loss += avg_loss * batch_images.shape[0]
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
            
            print('epoch{%d}, iter[%d], train loss: %.6f, '
                  'valid precision: %.6f, valid loss: %.6f' % (
                epoch, iteration, train_loss, valid_accuracy, valid_loss))
            sys.stdout.flush()
            
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
