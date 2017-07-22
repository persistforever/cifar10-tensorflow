# -*- encoding: utf8 -*-
# author: ronniecao
import os
import numpy
import tensorflow as tf
import src.data.cifar10 as cifar10
import matplotlib.pyplot as plt

cifar10 = cifar10.read_data_sets('data/CIFAR10_data', one_hot=True)

class ConvNet:
    
    def construct_model(self, batch_size=128, n_channel=3, n_classes=10):
        # input variable
        self.image = tf.placeholder(dtype=tf.float32, 
                                    shape=[batch_size, 32, 32, n_channel], 
                                    name='image')
        self.label = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_classes], name='label')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        
        # convolutional layer 1
        # filter
        W_conv1 = tf.Variable(
            initial_value=tf.random_normal(
                shape=[5, 5, n_channel, 64],
                mean=0.0, stddev=0.01), 
            name='W_conv1')
        # gamma
        gamma_conv1 = tf.Variable(
            initial_value=tf.random_normal(
                shape=[64]),
            name='gamma_conv1')
        # bias
        b_conv1 = tf.Variable(
            initial_value=tf.zeros(shape=[64]), 
            name='b_conv1')
        # hidden states
        conv_conv1 = tf.nn.conv2d(
            input=self.image, filter=W_conv1,
            strides=[1, 1, 1, 1], padding='SAME', 
            name='conv_conv1')
        pool_conv1 = tf.nn.max_pool(
            value=conv_conv1, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='VALID', 
            name='pool_conv1')
        # batch normalization
        mean_conv1, variance_conv1 = tf.nn.moments(pool_conv1, axes=[0, 1, 2])
        normal_pre = tf.reshape(pool_conv1, [batch_size*16*16, 64])
        self.nobn_conv1 = normal_pre
        mean_conv1 = tf.expand_dims(mean_conv1, dim=0)
        variance_conv1 = tf.expand_dims(variance_conv1, dim=0)
        normal_conv1 = (normal_pre - mean_conv1) / variance_conv1
        bn_conv1 = normal_conv1 * tf.expand_dims(gamma_conv1, axis=0) + b_conv1
        self.bn_conv1 = bn_conv1
        bn_conv1 = tf.reshape(bn_conv1, [batch_size, 16, 16, 64])
        # hidden_states
        h_conv1 = tf.nn.relu(bn_conv1, name='h_conv1')
        
        # convolutional layer 2
        # filter
        W_conv2 = tf.Variable(
            initial_value=tf.random_normal(shape=[5, 5, 64, 128], 
                                           mean=0.0, stddev=0.01), 
            name='W_conv2')
        # gamma
        gamma_conv2 = tf.Variable(
            initial_value=tf.random_normal(
                shape=[128]),
            name='gamma_conv2')
        # bias
        b_conv2 = tf.Variable(
            initial_value=tf.zeros(shape=[128]), 
            name='b_conv2')
        # hidden states
        conv_conv2 = tf.nn.conv2d(
            input=h_conv1, filter=W_conv2,
            strides=[1, 1, 1, 1], padding='SAME', 
            name='conv_conv2')
        pool_conv2 = tf.nn.max_pool(
            value=conv_conv2, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='VALID', 
            name='pool_conv2')
        # batch normalization
        mean_conv2, variance_conv2 = tf.nn.moments(pool_conv2, axes=[0, 1, 2])
        normal_pre = tf.reshape(pool_conv2, [batch_size*8*8, 128])
        self.nobn_conv2 = normal_pre
        mean_conv2 = tf.expand_dims(mean_conv2, dim=0)
        variance_conv2 = tf.expand_dims(variance_conv2, dim=0)
        normal_conv2 = (normal_pre - mean_conv2) / variance_conv2
        bn_conv2 = normal_conv2 * tf.expand_dims(gamma_conv2, axis=0) + b_conv2
        self.bn_conv2 = bn_conv2
        bn_conv2 = tf.reshape(bn_conv2, [batch_size, 8, 8, 128])
        # hidden_states
        h_conv2 = tf.nn.relu(bn_conv2, name='h_conv2')
        
        # convolutional layer 3
        # filter
        W_conv3 = tf.Variable(
            initial_value=tf.random_normal(shape=[5, 5, 128, 256], 
                                           mean=0.0, stddev=0.01), 
            name='W_conv2')
        # gamma
        gamma_conv3 = tf.Variable(
            initial_value=tf.random_normal(
                shape=[256]),
            name='gamma_conv3')
        # bias
        b_conv3 = tf.Variable(
            initial_value=tf.zeros(shape=[256]), 
            name='b_conv3')
        # hidden states
        conv_conv3 = tf.nn.conv2d(
            input=h_conv2, filter=W_conv3,
            strides=[1, 1, 1, 1], padding='SAME', 
            name='conv_conv3')
        pool_conv3 = tf.nn.max_pool(
            value=conv_conv3, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='VALID', 
            name='pool_conv3')
        # batch normalization
        mean_conv3, variance_conv3 = tf.nn.moments(pool_conv3, axes=[0, 1, 2])
        normal_pre = tf.reshape(pool_conv3, [batch_size*4*4, 256])
        self.nobn_conv3 = normal_pre
        mean_conv3 = tf.expand_dims(mean_conv3, dim=0)
        variance_conv3 = tf.expand_dims(variance_conv3, dim=0)
        normal_conv3 = (normal_pre - mean_conv3) / variance_conv3
        bn_conv3 = normal_conv3 * tf.expand_dims(gamma_conv3, axis=0) + b_conv3
        self.bn_conv3 = bn_conv3
        bn_conv3 = tf.reshape(bn_conv3, [batch_size, 4, 4, 256])
        # hidden_states
        h_conv3 = tf.nn.relu(bn_conv3, name='h_conv3')
        h_conv3_flat = tf.reshape(tensor=h_conv3, shape=[batch_size, 4*4*256], 
                                  name='h_conv3_flat')
        
        # fully-connected layer
        # weight
        W_fc1 = tf.Variable(
            initial_value=tf.random_normal(shape=[4*4*256, 1024], 
                                           mean=0.0, stddev=0.01), 
            name='W_fc1')
        # gamma
        gamma_fc1 = tf.Variable(
            initial_value=tf.random_normal(
                shape=[1024]),
            name='gamma_fc1')
        # bias
        b_fc1 = tf.Variable(
            initial_value=tf.zeros(shape=[1024]), 
            name='b_fc1')
        # hidden_states
        input_fc1 = tf.matmul(h_conv3_flat, W_fc1)
        self.nobn_fc1 = input_fc1
        # batch normalization
        mean_fc1, variance_fc1 = tf.nn.moments(input_fc1, axes=[0])
        mean_fc1 = tf.expand_dims(mean_fc1, dim=0)
        variance_fc1 = tf.expand_dims(variance_fc1, dim=0)
        normal_fc1 = (input_fc1 - mean_fc1) / variance_fc1
        bn_fc1 = normal_fc1 * tf.expand_dims(gamma_fc1, axis=0) + b_fc1
        self.bn_fc1 = bn_fc1
        # hidden_states
        h_fc1 = tf.nn.relu(bn_fc1, name='h_conv3')
        h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob, name='h_fc1_dropout')
        
        # softmax layer
        # weight
        W_softmax = tf.Variable(
            initial_value=tf.random_normal(shape=[1024, n_classes],
                                           mean=0.0, stddev=0.1), 
            name='W_softmax')
        # gamma
        gamma_softmax = tf.Variable(
            initial_value=tf.random_normal(
                shape=[n_classes]),
            name='gamma_softmax')
        # bias
        b_softmax = tf.Variable(
            initial_value=tf.zeros(shape=[n_classes]), 
            name='b_softmax')
        # hidden_states
        input_softmax = tf.matmul(h_fc1_dropout, W_softmax)
        self.nobn_softmax = input_softmax
        # batch normalization
        mean_softmax, variance_softmax = tf.nn.moments(input_softmax, axes=[0])
        mean_softmax = tf.expand_dims(mean_softmax, dim=0)
        variance_softmax = tf.expand_dims(variance_softmax, dim=0)
        normal_softmax = (input_softmax - mean_softmax) / variance_softmax
        bn_softmax = normal_softmax * tf.expand_dims(gamma_softmax, axis=0) + b_softmax
        self.bn_softmax = bn_softmax
        # softmax
        self.label_prob = tf.nn.softmax(logits=bn_softmax)
        
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
        for epoch in range(0, epochs+1):
            # 保存模型
            saver_path = saver.save(sess, 'backup/cifar10-v2/model.ckpt')
            if epoch <= 50 or epoch % 20 == 0:
                saver_path = saver.save(sess, 'backup/cifar10-v2/model_%d.ckpt' % (epoch))
            # 在训练之前，在验证集上计算准确率
            precision = []
            for batch in range(int(cifar10.validation.num_examples / batch_size)):
                batch_image, batch_label = cifar10.validation.next_batch(batch_size)
                precision_onebatch = sess.run(self.accuracy, feed_dict={
                    self.image:batch_image, self.label:batch_label, self.keep_prob:0.5})
                precision.append(precision_onebatch)
            print('epoch: %d, valid precision: %.4f' % (epoch, numpy.mean(precision)))
            # 开始本轮的训练
            for batch in range(int(cifar10.train.num_examples / batch_size)):
                batch_image, batch_label = cifar10.train.next_batch(batch_size)
                sess.run(self.optimizer, feed_dict={
                    self.image:batch_image, self.label:batch_label, self.keep_prob:0.5})
                if (batch+1) % int(cifar10.train.num_examples / batch_size / 10) == 0:
                    objective = sess.run(self.objective, 
                        feed_dict={self.image:batch_image, self.label:batch_label, 
                                   self.keep_prob:0.5})
                    print(('loss: %.4f') % (objective))
                
    def test(self, batch_size=128):
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        sess = tf.Session()
        # 读取模型
        model_path = 'backup/cifar10/model_18.ckpt'
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
            
    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        temp = sess.run(self.observe, feed_dict={
            self.image: numpy.random.random(size=[1, 32, 32, 3]),
            self.keep_prob: 0.1})
        print(temp.shape)