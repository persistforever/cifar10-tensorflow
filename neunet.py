# -*- encoding: utf8 -*-
# author: ronniecao
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def basic_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet()
    convnet.construct_model(batch_size=128, n_channel=3, n_classes=10)
    # convnet.debug()
    convnet.train(batch_size=128, epochs=30, backup_path='backup/cifar10-v1/')
    # convnet.test(batch_size=1)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)

def basic_cnn_cifar():
    from src.model.basic_cnn_cifar import ConvNet
    convnet = ConvNet()
    convnet.construct_model(batch_size=128, n_channel=3, n_classes=10)
    # convnet.debug()
    convnet.train(batch_size=128, epochs=30)
    # convnet.test(batch_size=1)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def batchnormal_cnn():
    from src.model.batchnormal_cnn import ConvNet
    convnet = ConvNet()
    convnet.construct_model(batch_size=128, n_channel=3, n_classes=10)
    # convnet.debug()
    convnet.train(batch_size=128, epochs=1000, backup_path='backup/cifar10-v2/')
    # convnet.test(batch_size=1)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def plain_cnn():
    from src.model.plain_cnn import ConvNet
    convnet = ConvNet()
    convnet.construct_model(batch_size=128, n_channel=3, n_classes=10)
    # convnet.debug()
    convnet.train(batch_size=128, epochs=1000, backup_path='backup/cifar10-v3')
    # convnet.test(batch_size=1)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)

plain_cnn()