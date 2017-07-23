# -*- encoding: utf8 -*-
import os
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tensorflow.python.ops.gradient_checker import _compute_gradient


mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

class ConvNet:
    
    def construct_model(self, batch_size=128):
        # input variable
        self.image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784], name='image')
        self.label = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10], name='label')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        image_tensor = tf.reshape(tensor=self.image, shape=[batch_size, 28, 28, 1], 
                                  name='image_tensor')
        
        # convolutional layer 1
        # filter
        W_conv1 = tf.Variable(
            initial_value=tf.random_normal(shape=[5, 5, 1, 16], mean=0.0, stddev=0.01), 
            name='W_conv1')
        # bias
        b_conv1 = tf.Variable(
            initial_value=tf.zeros(shape=[16]), 
            name='b_conv1')
        # hidden states
        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(input=image_tensor, filter=W_conv1, 
                         strides=[1, 2, 2, 1], padding='SAME') + b_conv1, 
            name='h_conv1')
        
        # convolutional layer 2
        # filter
        W_conv2 = tf.Variable(
            initial_value=tf.random_normal(shape=[5, 5, 16, 16], mean=0.0, stddev=0.01), 
            name='W_conv2')
        # bias
        b_conv2 = tf.Variable(
            initial_value=tf.zeros(shape=[16]), 
            name='b_conv2')
        # hidden states
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(input=h_conv1, filter=W_conv2, 
                         strides=[1, 2, 2, 1], padding='SAME') + b_conv2, 
            name='h_conv2')
        h_conv2_flat = tf.reshape(tensor=h_conv2, shape=[-1, 7 * 7 * 16], name='h_conv2_flat')
        
        # fully-connected layer
        # weight
        W_fc1 = tf.Variable(
            initial_value=tf.random_normal(shape=[7 * 7 * 16, 64], mean=0.0, stddev=0.01), 
            name='W_fc1')
        # bias
        b_fc1 = tf.Variable(
            initial_value=tf.zeros(shape=[64]), 
            name='b_fc1')
        # hidden_states
        h_fc1 = tf.nn.relu(
            tf.matmul(h_conv2_flat, W_fc1) + b_fc1, 
            name='h_fc1')
        h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob, name='h_fc1_dropout')
        
        # softmax layer
        # weight
        W_softmax = tf.Variable(
            initial_value=tf.random_normal(shape=[64, 10], mean=0.0, stddev=0.01), 
            name='W_softmax')
        # bias
        b_softmax = tf.Variable(
            initial_value=tf.zeros(shape=[10]), 
            name='b_softmax')
        # softmax
        self.label_prob = tf.nn.softmax(
            logits=tf.matmul(h_fc1_dropout, W_softmax) + b_softmax)
        
        # objective function and optimizer
        self.objective = - tf.reduce_sum(self.label * tf.log(self.label_prob + 1e-6))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.objective)
        # observation
        correct_prediction = tf.equal(tf.argmax(self.label_prob, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        self.label_pred = tf.argmax(self.label_prob, 1)
        self.label_max_prob = tf.reduce_max(self.label_prob)
        self.gradient = tf.gradients(self.label_max_prob, self.image)
        
    def train(self, batch_size=128, epochs=5):
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=epochs)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs+1):
            # 保存模型
            # saver_path = saver.save(sess, '../backup/mnist/model.ckpt')
            if epoch <= 50 or epoch % 20 == 0:
                saver_path = saver.save(sess, '../backup/mnist/model_%d.ckpt' % (epoch))
            # 在训练之前，在验证集上计算准确率
            precision = []
            for batch in range(int(mnist.validation.num_examples / batch_size)):
                batch_image, batch_label = mnist.validation.next_batch(batch_size)
                precision_onebatch = sess.run(self.accuracy, feed_dict={
                    self.image:batch_image, self.label:batch_label, self.keep_prob:0.5})
                precision.append(precision_onebatch)
            print('epoch: %d, valid precision: %.4f' % (epoch, numpy.mean(precision)))
            # 开始本轮的训练
            for batch in range(int(mnist.train.num_examples / batch_size)):
                batch_image, batch_label = mnist.train.next_batch(batch_size)
                sess.run(self.optimizer, feed_dict={
                    self.image:batch_image, self.label:batch_label, self.keep_prob:0.5})
                if (batch+1) % int(mnist.train.num_examples / batch_size / 10) == 0:
                    objective = sess.run(self.objective, 
                        feed_dict={self.image:batch_image, self.label:batch_label, 
                                   self.keep_prob:0.5})
                    print(('loss: %.4f') % (objective))
                
    def test(self, batch_size=128):
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        sess = tf.Session()
        # 读取模型
        model_path = '../backup/mnist/model_18.ckpt'
        assert(os.path.exists(model_path+'.index'))
        saver.restore(sess, model_path)
        print('read model from %s' % (model_path))
        # 在测试集上计算准确率
        precision = []
        for batch in range(int(mnist.test.num_examples / batch_size)):
            batch_image, batch_label = mnist.test.next_batch(batch_size)
            precision_onebatch = sess.run(self.accuracy, feed_dict={
                self.image:batch_image, self.label:batch_label, self.keep_prob:0.5})
            precision.append(precision_onebatch)
        print('test precision: %.4f' % (numpy.mean(precision)))
        
    def observe_salience(self, batch_size=128, num_test=10, epoch=1):
        if not os.path.exists('../results/epoch%d/' % (epoch)):
            os.makedirs('../results/epoch%d/' % (epoch))
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        sess = tf.Session()
        # 读取模型
        model_path = '../backup/mnist/model_%d.ckpt' % (epoch)
        assert(os.path.exists(model_path+'.index'))
        saver.restore(sess, model_path)
        print('read model from %s' % (model_path))
        # 获取图像并计算梯度
        for batch in range(num_test):
            batch_image, batch_label = mnist.test.next_batch(batch_size)
            image = numpy.array(batch_image.reshape([28, 28]) * 255, dtype='uint8')
            result = sess.run([self.label_prob, self.label_max_prob, self.label_pred,
                               self.gradient],
                              feed_dict={self.image:batch_image, self.label:batch_label,
                                         self.keep_prob:0.5})
            print(result[0:3], result[3][0].shape)
            gradient = sess.run(self.gradient, feed_dict={
                self.image:batch_image, self.keep_prob:0.5})
            gradient = gradient[0].reshape([28, 28])# * batch_image.reshape([28, 28])
            gradient = numpy.array((gradient - gradient.min()) * 255
                                    / (gradient.max() - gradient.min()), dtype='uint8')
            # 使用pyplot画图
            plt.subplot(121)
            plt.imshow(image, cmap=plt.cm.gray)
            plt.subplot(122)
            plt.imshow(gradient, cmap=plt.cm.gray)
            plt.savefig('../results/epoch%d/result_%d.png' % (epoch, batch))
        

convnet = ConvNet()
convnet.construct_model(batch_size=1)
# convnet.train(batch_size=128, epochs=50)
# convnet.test(batch_size=1)
convnet.observe_salience(batch_size=1, num_test=10, epoch=20)