# -*- encoding: utf8 -*-
import pickle
import numpy
import random
import matplotlib.pyplot as plt
import platform
import cv2


class Corpus:
    
    def __init__(self):
        self.load_cifar10('data/CIFAR10_data')
        self._split_train_valid(valid_rate=0.9)
        self.n_train = self.origin_train_images.shape[0]
        self.n_valid = self.origin_valid_images.shape[0]
        self.n_test = self.origin_test_images.shape[0]
        # set train_images, valid_images, test_images
        self.train_images = self.origin_train_images
        self.valid_images = self.origin_valid_images
        self.test_images = self.origin_test_images
        
    def _split_train_valid(self, valid_rate=0.9):
        images, labels = self.origin_train_images, self.train_labels 
        thresh = int(images.shape[0] * valid_rate)
        self.origin_train_images, self.train_labels = images[0:thresh,:,:,:], labels[0:thresh]
        self.origin_valid_images, self.valid_labels = images[thresh:,:,:,:], labels[thresh:]
    
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
        self.origin_train_images, self.train_labels = images, labels
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
        self.origin_test_images, self.test_labels = images, labels
        
    def data_augmentation(self, flip=False, crop=False, shape=(24,24,3), whiten=False):
        self.train_images = self.origin_train_images
        self.valid_images = self.origin_valid_images
        self.test_images = self.origin_test_images
        # 图像切割
        if crop:
            self.train_images = self._image_crop(self.train_images, shape=shape)
            self.valid_images = self._image_crop_test(self.valid_images, shape=shape)
            self.test_images = self._image_crop_test(self.test_images, shape=shape)
        # 图像翻转
        if flip:
            self.train_images = self._image_flip(self.train_images)
        # 图像白化
        if whiten:
            self.train_images = self._image_whitening(self.train_images)
            self.valid_images = self._image_whitening(self.valid_images)
            self.test_images = self._image_whitening(self.test_images)
    
    def _image_crop(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = numpy.random.randint(old_image.shape[0] - shape[0] + 1)
            top = numpy.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left+shape[0], top: top+shape[1], :]
            new_images.append(new_image)
        
        return numpy.array(new_images)
    
    def _image_crop_test(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            new_image = old_image[left: left+shape[0], top: top+shape[1], :]
            new_images.append(new_image)
        
        return numpy.array(new_images)
    
    def _image_flip(self, images):
        # 图像翻转
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            if numpy.random.random() < 0.5:
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
            images[i,:,:,:] = new_image
        
        return images
    
    def _image_whitening(self, images):
        # 图像白化
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
            images[i,:,:,:] = new_image
        
        return images