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

class Network:
    
    @abstractmethod
    def construct_model(self):
        pass
        
    @abstractmethod
    def train(self):
        pass
                
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