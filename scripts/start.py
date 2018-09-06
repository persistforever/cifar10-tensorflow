# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/09/06
# intro: start class of image classification
from __future__ import print_function
import sys
import os
import argparse
import yaml

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

 
class Starter:
    
    def __init__(self, method, gpus=''):
        from src.dataloader.cifar10 import Dataloader
        from src.model.resnet import Model
        from src.network.resnet import Network
            
        # 读取配置
        config_path = os.path.join(config.project_root, 'src/config/params/resnet.yaml')
        option = yaml.load(open(config_path, 'r'))
        
        # 实例化data模块
        dataloader = Corpus()

        # 实例化network模块
        network = 
        
        # 实例化model模块
        model = ConvNet(
            n_channel=option['n_channel'], 
            n_classes=option['n_classes'], 
            image_size=option['image_size'], 
            network_path='src/config/network/resnet.yaml')
    
    def main(method='train'):
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = option['train_gpus']
        
        convnet.train(dataloader=cifar10, backup_path='backups/cifar10-v5/', batch_size=128, n_epoch=500)
        # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)


if __name__ == '__main__':
    print('current process id: %d' % (os.getpid()))
    parser = argparse.ArgumentParser(description='parsing command parameters')
    parser.add_argument('-method')
    parser.add_argument('-name')
    arg = parser.parse_args()
    method = arg.method
    
    resnet(method='train')
