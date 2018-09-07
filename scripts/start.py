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
    
    def __init__(self):
        from src.dataloader.cifar10 import Dataloader
        from src.network.resnet import Network
        from src.manager.resnet import Manager
            
        # 读取配置
        config_path = os.path.join('src/config/options/resnet.yaml')
        self.option = yaml.load(open(config_path, 'r'))
        
        # 实例化data模块
        self.dataloader = Dataloader(
            data_path=self.option['data_path'],
            config_path=config_path)
        
        # 实例化network模块
        self.network = Network(
            config_path=config_path,
            network_config_path=self.option['network_path'])
       
        # 实例化model模块
        self.manager = Manager(
            config_path=config_path,
            backups_dir=os.path.join('backups', self.option['seq']),
            logs_dir=os.path.join('logs', self.option['seq']))
        self.manager.init_module(
            dataloader=self.dataloader, 
            network=self.network)
    
    def main(self, method='train'):
        
        if method == 'train':
            # 设置GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = self.option['train_gpus']
            self.manager.train()


if __name__ == '__main__':
    print('current process id: %d' % (os.getpid()))
    parser = argparse.ArgumentParser(description='parsing command parameters')
    parser.add_argument('-method')
    parser.add_argument('-name')
    arg = parser.parse_args()
    method = arg.method
    
    starter = Starter()
    starter.main(method=method)
