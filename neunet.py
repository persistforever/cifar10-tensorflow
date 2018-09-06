# -*- coding: utf8 -*-
# author: ronniecao
import os
from src.data.cifar10 import Corpus

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cifar10 = Corpus()

def basic_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/basic.yaml')
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backups/cifar10-v1/', batch_size=128, n_epoch=500)
    # convnet.test(dataloader=cifar10, backup_path='backup/cifar10-v2/', epoch=5000, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def vgg_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/vgg.yaml')
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backups/cifar10-v2/', batch_size=128, n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v3/', epoch=0, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def resnet():
    from src.model.resnet import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/resnet.yaml')
    convnet.train(dataloader=cifar10, backup_path='backups/cifar10-v5/', batch_size=128, n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)


vgg_cnn()
