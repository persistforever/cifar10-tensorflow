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
            input_shape=(batch_size, 32, 32, n_channel), n_size=3, n_filter=16, 
            stride=1, activation='relu', batch_normal=True, name='conv1')
        conv_layer2 = ConvLayer(
            input_shape=(batch_size, 32, 32, 16), n_size=3, n_filter=16, 
            stride=1, activation='relu', batch_normal=True, name='conv2')
        conv_layer3 = ConvLayer(
            input_shape=(batch_size, 32, 32, 16), n_size=3, n_filter=16, 
            stride=1, activation='relu', batch_normal=True, name='conv3')
        conv_layer4 = ConvLayer(
            input_shape=(batch_size, 32, 32, 16), n_size=3, n_filter=16, 
            stride=1, activation='relu', batch_normal=True, name='conv4')
        conv_layer5 = ConvLayer(
            input_shape=(batch_size, 32, 32, 16), n_size=3, n_filter=16, 
            stride=1, activation='relu', batch_normal=True, name='conv5')
        conv_layer6 = ConvLayer(
            input_shape=(batch_size, 32, 32, 16), n_size=3, n_filter=32, 
            stride=2, activation='relu', batch_normal=True, name='conv6')
        conv_layer7 = ConvLayer(
            input_shape=(batch_size, 16, 16, 32), n_size=3, n_filter=32, 
            stride=1, activation='relu', batch_normal=True, name='conv7')
        conv_layer8 = ConvLayer(
            input_shape=(batch_size, 16, 16, 32), n_size=3, n_filter=32, 
            stride=1, activation='relu', batch_normal=True, name='conv8')
        conv_layer9 = ConvLayer(
            input_shape=(batch_size, 16, 16, 32), n_size=3, n_filter=32, 
            stride=1, activation='relu', batch_normal=True, name='conv9')
        conv_layer10 = ConvLayer(
            input_shape=(batch_size, 16, 16, 32), n_size=3, n_filter=64, 
            stride=2, activation='relu', batch_normal=True, name='conv10')
        conv_layer11 = ConvLayer(
            input_shape=(batch_size, 8, 8, 64), n_size=3, n_filter=64, 
            stride=1, activation='relu', batch_normal=True, name='conv11')
        conv_layer12 = ConvLayer(
            input_shape=(batch_size, 8, 8, 64), n_size=3, n_filter=64, 
            stride=1, activation='relu', batch_normal=True, name='conv12')
        conv_layer13 = ConvLayer(
            input_shape=(batch_size, 8, 8, 64), n_size=3, n_filter=64, 
            stride=1, activation='relu', batch_normal=True, name='conv13')
        pool_layer14 = PoolLayer(
            n_size=2, stride=2, mode='avg', name='pool14')
        dense_layer15 = DenseLayer(
            input_shape=(batch_size, 4 * 4 * 64), hidden_dim=10, activation='softmax', 
            dropout=False, keep_prob=None, batch_normal=True, name='dense15')
        
        # data flow
        output_conv1 = conv_layer1.get_output(input=self.image)
        output_conv2 = conv_layer2.get_output(input=output_conv1)
        output_conv3 = conv_layer3.get_output(input=output_conv2)
        # conv2 and conv3 residual
        hidden_conv3 = conv_layer3.hidden
        output_conv3 = tf.nn.relu(hidden_conv3 + output_conv1)
        output_conv4 = conv_layer4.get_output(input=output_conv3)
        output_conv5 = conv_layer5.get_output(input=output_conv4)
        # conv4 and conv5 residual
        hidden_conv5 = conv_layer5.hidden
        output_conv5 = tf.nn.relu(hidden_conv5 + output_conv3)
        output_conv6 = conv_layer6.get_output(input=output_conv5)
        output_conv7 = conv_layer7.get_output(input=output_conv6)
        # conv6 and conv7 residual
        hidden_conv7 = conv_layer7.hidden
        output_conv5 = PoolLayer(n_size=2, stride=2, mode='avg').get_output(output_conv5)
        output_conv5 = tf.pad(output_conv5, [[0,0], [0,0], [0,0], [8,8]])
        output_conv7 = tf.nn.relu(hidden_conv7 + output_conv5)
        output_conv8 = conv_layer8.get_output(input=output_conv7)
        output_conv9 = conv_layer9.get_output(input=output_conv8)
        # conv8 and conv9 residual
        hidden_conv9 = conv_layer9.hidden
        output_conv9 = tf.nn.relu(hidden_conv9 + output_conv7)
        output_conv10 = conv_layer10.get_output(input=output_conv9)
        output_conv11 = conv_layer11.get_output(input=output_conv10)
        # conv10 and conv11 residual
        hidden_conv11 = conv_layer11.hidden
        output_conv9 = PoolLayer(n_size=2, stride=2, mode='avg').get_output(output_conv9)
        output_conv9 = tf.pad(output_conv9, [[0,0], [0,0], [0,0], [16,16]])
        output_conv11 = tf.nn.relu(hidden_conv11 + output_conv9)
        output_conv12 = conv_layer12.get_output(input=output_conv11)
        output_conv13 = conv_layer13.get_output(input=output_conv12)
        # conv12 and conv13 residual
        h_conv13 = conv_layer13.hidden
        output_conv13 = tf.nn.relu(h_conv13 + output_conv11)
        hidden_pool14 = pool_layer14.get_output(output_conv13)
        input_dense15 = tf.reshape(hidden_pool14, [batch_size, 4 * 4 * 64])
        self.label_prob = dense_layer15.get_output(input=input_dense15)
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