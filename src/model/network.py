# -*- encoding: utf8 -*-
# author: ronniecao
import os
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from abc import abstractmethod
import src.data.cifar10 as cifar10
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer

cifar10 = cifar10.read_data_sets('data/CIFAR10_data', one_hot=True)

class Network:
    
    @abstractmethod
    def construct_model(self):
        pass
        
    def train(self, backup_path, n_epoch=5, batch_size=128):
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=n_epoch)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, n_epoch+1):
            # 保存模型
            saver_path = saver.save(
                sess, os.path.join(backup_path, 'model.ckpt'))
            if epoch <= 100 and epoch % 10 == 0 or epoch <= 1000 and epoch % 100 == 0 or \
                epoch <= 10000 and epoch % 1000 == 0:
                saver_path = saver.save(
                    sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))
            # 在训练之前，在验证集上计算准确率
            precision = []
            for batch in range(int(cifar10.validation.num_examples / batch_size)):
                batch_image, batch_label = cifar10.validation.next_batch(batch_size)
                [precision_onebatch] = sess.run(
                    fetches=[self.accuracy], 
                    feed_dict={self.image:batch_image, 
                               self.label:batch_label, 
                               self.keep_prob:1.0})
                precision.append(precision_onebatch)
            print('epoch: %d, valid precision: %.4f' % (epoch, numpy.mean(precision)))
            # 开始本轮的训练
            for batch in range(int(cifar10.train.num_examples / batch_size)):
                batch_image, batch_label = cifar10.train.next_batch(batch_size)
                [_, objective] = sess.run(
                    fetches=[self.optimizer, self.objective], 
                    feed_dict={self.image:batch_image, 
                               self.label:batch_label, 
                               self.keep_prob:0.5})
                if (batch+1) % int(cifar10.train.num_examples / batch_size / 10) == 0:
                    print(('loss: %.4f') % (objective))
                
    def test(self, backup_path, epoch, batch_size=128):
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 读取模型
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        saver.restore(sess, model_path)
        print('read model from %s' % (model_path))
        # 在测试集上计算准确率
        precision = []
        for batch in range(int(cifar10.test.num_examples / batch_size)):
            batch_image, batch_label = cifar10.test.next_batch(batch_size)
            [precision_onebatch] = sess.run(
                fetches=[self.accuracy], 
                feed_dict={self.image:batch_image, 
                           self.label:batch_label,
                           self.keep_prob:1.0})
            precision.append(precision_onebatch)
        print('test precision: %.4f' % (numpy.mean(precision)))
            
    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [temp] = sess.run(
            fetches=[self.observe],
            feed_dict={self.image: numpy.random.random(size=[128, 32, 32, 3]),
                       self.keep_prob: 1.0})
        print(temp.shape)