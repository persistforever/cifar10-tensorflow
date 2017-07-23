# -*- encoding: utf8 -*-
# author: ronniecao
import os
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import src.data.cifar10 as cifar10
from src.model.network import Network
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer

cifar10 = cifar10.read_data_sets('data/CIFAR10_data', one_hot=True)

class ConvNet(Network):
    
    def construct_model(self, batch_size=128, n_channel=3, n_classes=10):
        # input variable
        self.image = tf.placeholder(
            dtype=tf.float32, shape=[batch_size, 32, 32, n_channel], name='image')
        self.label = tf.placeholder(
            dtype=tf.float32, shape=[batch_size, n_classes], name='label')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        
        # network structure
        conv_layer1 = ConvLayer(
            input_shape=(batch_size, 32, 32, n_channel), n_size=3, n_filter=64, 
            stride=1, activation='relu', batch_normal=False, name='conv1')
        pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', name='pool1')
        conv_layer2 = ConvLayer(
            input_shape=(batch_size, 16, 16, 64), n_size=3, n_filter=128, 
            stride=1, activation='relu', batch_normal=False, name='conv2')
        pool_layer2 = PoolLayer(
            n_size=2, stride=2, mode='max', name='pool2')
        conv_layer3 = ConvLayer(
            input_shape=(batch_size, 8, 8, 128), n_size=3, n_filter=256, 
            stride=1, activation='relu', batch_normal=False, name='conv3')
        pool_layer3 = PoolLayer(
            n_size=2, stride=2, mode='max', name='pool3')
        dense_layer1 = DenseLayer(
            input_shape=(batch_size, 4 * 4 * 256), hidden_dim=1024, 
            activation='relu', dropout=True, keep_prob=self.keep_prob, 
            batch_normal=False, name='dense1')
        dense_layer2 = DenseLayer(
            input_shape=(batch_size, 1024), hidden_dim=n_classes, 
            activation='softmax', dropout=False, keep_prob=None, 
            batch_normal=False, name='dense2')
        
        # data flow
        hidden_conv1 = conv_layer1.get_output(input=self.image)
        hidden_pool1 = pool_layer1.get_output(input=hidden_conv1)
        hidden_conv2 = conv_layer2.get_output(input=hidden_pool1)
        hidden_pool2 = pool_layer2.get_output(input=hidden_conv2)
        hidden_conv3 = conv_layer3.get_output(input=hidden_pool2)
        hidden_pool3 = pool_layer3.get_output(input=hidden_conv3)
        input_dense1 = tf.reshape(hidden_pool3, [batch_size, 4 * 4 * 256])
        hidden_dense1 = dense_layer1.get_output(input=input_dense1)
        self.label_prob = dense_layer2.get_output(input=hidden_dense1)
        self.observe = self.label_prob
        # objective function and optimizer
        self.objective = - tf.reduce_sum(self.label * tf.log(self.label_prob + 1e-6))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.objective)
        # observation
        correct_prediction = tf.equal(tf.argmax(self.label_prob, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        self.label_pred = tf.argmax(self.label_prob, 1)
        self.label_max_prob = tf.reduce_max(self.label_prob)
        self.gradient = tf.gradients(self.label_max_prob, self.image)
        
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
            result = sess.run([self.label_prob, self.label_max_prob, self.label_pred,
                               self.gradient],
                              feed_dict={self.image:batch_image, self.label:batch_label,
                                         self.keep_prob:0.5})
            print(result[0:3], result[3][0].shape)
            gradient = sess.run(self.gradient, feed_dict={
                self.image:batch_image, self.keep_prob:0.5})
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
                              feed_dict={self.image:batch_image, self.label:batch_label,
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