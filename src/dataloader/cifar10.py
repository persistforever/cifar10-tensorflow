# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/09/06
# intro: dataloader of image classification on cifar-10 datasets
from __future__ import print_function
import os
import platform
import yaml
import pickle
import numpy
import cv2


class Dataloader:
    
    def __init__(self, data_path, config_path):
        # 读取配置
        option = yaml.load(open(config_path, 'r'))
        
        self.load_cifar10(data_path)
        self._split_train_valid(valid_rate=option['valid_rate'])
        self.n_train = self.train_images.shape[0]
        self.n_valid = self.valid_images.shape[0]
        self.n_test = self.test_images.shape[0]
        print('\n' + '='*20 + ' load data ' + '='*20)
        print('# train data: %d' % (self.n_train))
        print('# valid data: %d' % (self.n_valid))
        print('# test data: %d' % (self.n_test))
        print('='*20 + ' load data ' + '='*20 + '\n')
        
    def _split_train_valid(self, valid_rate=0.9):
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
        
    def data_augmentation(self, images, mode='train', flip=False, 
        crop=False, crop_shape=(24,24,3), whiten=False, 
        noise=False, noise_mean=0, noise_std=0.01):
        # 图像切割
        if crop:
            if mode == 'train':
                images = self._image_crop(images, shape=crop_shape)
            elif mode == 'test':
                images = self._image_crop_test(images, shape=crop_shape)
        # 图像翻转
        if flip:
            images = self._image_flip(images)
        # 图像白化
        if whiten:
            images = self._image_whitening(images)
        # 图像噪声
        if noise:
            images = self._image_noise(images, mean=noise_mean, std=noise_std)
            
        return images
    
    def _image_crop(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            old_image = numpy.pad(old_image, [[4,4], [4,4], [0,0]], 'constant')
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
            old_image = numpy.pad(old_image, [[4,4], [4,4], [0,0]], 'constant')
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
            mean = numpy.mean(old_image)
            std = numpy.max([numpy.std(old_image), 
                1.0/numpy.sqrt(images.shape[1]*images.shape[2]*images.shape[3])])
            new_image = (old_image - mean) / std
            images[i,:,:,:] = new_image
        
        return images
    
    def _image_noise(self, images, mean=0, std=0.01):
        # 图像噪声
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = old_image
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        new_image[i, j, k] += random.gauss(mean, std)
            images[i,:,:,:] = new_image
        
        return images
