# -*- encoding: utf8 -*-
import pickle
import numpy
import random
import matplotlib.pyplot as plt
import platform


class Corpus:
    
    def __init__(self):
        self.load_cifar10('data/CIFAR10_data')
        self.split_train_valid(valid_rate=0.9)
        self.n_train = self.train_images.shape[0]
        self.n_valid = self.valid_images.shape[0]
        self.n_test = self.test_images.shape[0]
        
    def split_train_valid(self, valid_rate=0.9):
        images, labels = self.train_images, self.train_labels 
        thresh = int(images.shape[0] * valid_rate)
        self.train_images, self.train_labels = images[0:thresh,:,:,:], labels[0:thresh]
        self.valid_images, self.valid_labels = images[thresh:,:,:,:], labels[thresh:]
    
    def load_cifar10(self, directory):
        # 读取训练集
        images, labels = [], []
        for filename in ['%s/data_batch_%d' % (directory, j) for j in range(1, 6)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo)
            for i in range(len(cifar10[b"labels"])):
                image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                image = numpy.transpose(image, (1, 2, 0))
                image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        self.train_images, self.train_labels = images, labels
        # 读取测试集
        images, labels = [], []
        for filename in ['%s/test_batch' % (directory)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo)
            for i in range(len(cifar10[b"labels"])):
                image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                image = numpy.transpose(image, (1, 2, 0))
                image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        self.test_images, self.test_labels = images, labels