# -*- coding: utf8 -*-
# author: ronniecao
import os
from src.data.cifar10 import Corpus

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

cifar10 = Corpus()

def basic_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24)
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backup/cifar10-v14/', batch_size=128, n_epoch=500)
    # convnet.test(dataloader=cifar10, backup_path='backup/cifar10-v2/', epoch=5000, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def plain_cnn():
    from src.model.plain_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24)
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backup/cifar10-v13/', batch_size=128, n_epoch=1000)
    # convnet.test(backup_path='backup/cifar10-v3/', epoch=0, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def residual_net():
    from src.model.residual_net import ConvNet
    convnet = ConvNet()
    convnet.construct_model(batch_size=128, n_channel=3, n_classes=10)
    # convnet.debug()
    convnet.train(backup_path='backup/cifar10-v4/', batch_size=128, n_epoch=200)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)

basic_cnn()