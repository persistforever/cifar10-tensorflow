# -*- encoding: utf8 -*-
import pickle
import numpy
import random
import matplotlib.pyplot as plt
import platform


class Dataset:
    
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        self.index = 0
        self.num_examples = dataset.shape[0]
        self._shuffle_data()
        # self._image_whitening()
        
    def _shuffle_data(self):
        index = list(range(self.num_examples))
        random.shuffle(index)
        self.dataset = self.dataset[index]
        self.label = self.label[index]
        
    def next_batch(self, batch_size):
        assert(batch_size is not None)
        if self.index + batch_size < self.num_examples:
            end_index = self.index + batch_size
            batch_dataset = self.dataset[self.index:end_index,:]
            batch_label = self.label[self.index:end_index]
            self.index = end_index
        elif self.index + batch_size == self.num_examples:
            batch_dataset = self.dataset[self.index:,:]
            batch_label = self.label[self.index:]
            self.index = 0
        else:
            end_index = batch_size - (self.num_examples - self.index)
            tail_dataset = self.dataset[self.index:,:]
            tail_label = self.label[self.index:]
            head_dataset = self.dataset[:end_index, :]
            head_label = self.label[:end_index]           
            batch_dataset = numpy.concatenate([tail_dataset, head_dataset], axis=0)
            batch_label = numpy.concatenate([tail_label, head_label], axis=0)
            self.index = end_index
            
        return [batch_dataset, batch_label]
    
    def _image_whitening(self):
        for i in range(self.dataset.shape[0]):
            if numpy.random.random() < 0.5:
                old_image = self.dataset[i,:,:,:]
                new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
                self.dataset[i,:,:,:] = new_image
        
        return new_image
    

class Corpus:
    
    def __init__(self):
        self.train = None
        self.validation = None
        self.test = None
        
    def set_train_valid(self, dataset, label, valid_rate=0.1):
        thred = int(dataset.shape[0] * valid_rate)
        self.validation = Dataset(dataset[:thred,:], label[:thred])
        self.train = Dataset(dataset[thred:,:], label[thred:])
        
    def set_test(self, dataset, label):
        self.test = Dataset(dataset, label)

    
def read_data_sets(directory, one_hot=True):
    cifar10 = Corpus()
    # 训练集和验证集
    dataset, label = None, []
    for i in range(1, 6):
        with open('%s/data_batch_%d' % (directory, i), 'rb') as fo:
            if 'Windows' in platform.platform():
                dict = pickle.load(fo, encoding='bytes')
            elif 'Linux' in platform.platform():
                dict = pickle.load(fo)
            if dataset is None:
                dataset = dict[b'data'].reshape([10000, 3, 32, 32])
            else:
                dataset = numpy.concatenate([dataset, dict[b'data'].reshape([10000, 3, 32, 32])],
                                      axis=0)
            label.extend(dict[b'labels'])
    dataset = dataset.transpose([0, 2, 3, 1])
    label = numpy.array(label, dtype='int32')
    if one_hot:
        label_onehot = numpy.zeros((label.shape[0], 10), dtype='int32')
        for i in range(label.shape[0]):
            label_onehot[i, label[i]] = 1
        label = label_onehot
    cifar10.set_train_valid(dataset, label, valid_rate=0.1)
    # 测试集
    dataset, label = None, []
    with open('%s/test_batch' % (directory), 'rb') as fo:
        if 'Windows' in platform.platform():
            dict = pickle.load(fo, encoding='bytes')
        elif 'Linux' in platform.platform():
            dict = pickle.load(fo)
        if dataset is None:
            dataset = dict[b'data'].reshape([10000, 3, 32, 32])
        else:
            dataset = numpy.concatenate([dataset, dict[b'data'].reshape([10000, 3, 32, 32])],
                                  axis=0)
        label.extend(dict[b'labels'])
    dataset = dataset.transpose([0, 2, 3, 1])
    label = numpy.array(label, dtype='int32')
    if one_hot:
        label_onehot = numpy.zeros((label.shape[0], 10), dtype='int32')
        for i in range(label.shape[0]):
            label_onehot[i, label[i]] = 1
        label = label_onehot
    cifar10.set_test(dataset, label)
    
    return cifar10
    

# cifar10 = read_data_sets('../data/CIFAR10_data', one_hot=True)
# print(cifar10.train.num_examples, cifar10.test.num_examples, cifar10.valid.num_examples)