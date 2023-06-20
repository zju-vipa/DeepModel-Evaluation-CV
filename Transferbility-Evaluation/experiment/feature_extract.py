import argparse
import os
import sys 
import pathlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.optim as optim

import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models

from torch.autograd import Variable

from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

import global_settings as settings
from utils import get_network, get_pretrain_network, get_training_transform, get_test_transform, WarmUpLR
from dataset import tinyfood101, tinycaltech, tinydomainnet, tinydtd, imagenet50

from pytorch_pretrained_vit import ViT
# import timm


class HookTool: 
    def __init__(self):
        self.in_feature = [] 
        self.out_feature = []

    def hook_fun(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.in_feature.append(fea_in[0].detach().cpu())
        
def deep_vit_feature_extract(dataloader, class_num, sample_num):
    # model = ViT('L_32', pretrained=True).cuda().eval()
    model = ViT('B_16', pretrained=True).cuda().eval()
    # model = timm.create_model('tresnet_m_miil_in21k', pretrained=True).eval().cuda()
    hooker = HookTool()
    # print(extractor)
    for (name, module) in model.named_modules():
        if name == 'fc':
            module.register_forward_hook(hook=hooker.hook_fun)

    total_labels = []
    tqdm_loader = tqdm(dataloader)
    for i, (imgs, labels) in enumerate(tqdm_loader):
        imgs = Variable(imgs).cuda()
        labels = Variable(labels)
        model(imgs)
        total_labels.append(labels)


    src_feature = torch.cat(hooker.in_feature, dim=0)
    total_label = torch.cat(total_labels)

    feature_dim = src_feature.shape[1]
    unique_labels, label_counts = torch.unique(total_label, return_counts=True)
    label_counts = label_counts.unsqueeze(0)
    label_counts = label_counts.view(label_counts.shape[1], -1)

    features = torch.zeros(class_num, feature_dim, device='cpu')
    assert class_num == len(unique_labels)
    print(src_feature, src_feature.shape)
    features.scatter_add_(0, total_label.squeeze().unsqueeze(-1).repeat(1, src_feature.size(-1)), src_feature.squeeze())

    print(features, features.shape)
    features = torch.div(features, label_counts)
    print(features)

    return features


# 返回dataloader(train and test)
def get_dataset_and_loader(args, class_num, sample_num):
    train_loader, test_loader = '', ''
    size = 224
    
    transform_train = get_training_transform(args.dataset, size)
    transform_test = get_test_transform(args.dataset, size)
    
    if args.dataset == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    # 采用默认数据增强来进行
    elif args.dataset == 'tinydtd':
        train_ds = tinydtd.TinyDTD(root='./data', split='train', download=True, transform=transform_train, class_num=class_num, sample_num=sample_num)
        test_ds = tinydtd.TinyDTD(root='./data', split='test', download=True, transform=transform_test, class_num=class_num,sample_num=sample_num)
    elif args.dataset == 'tinyfood101':
        train_ds = tinyfood101.TinyFood101(root='./data', split='train', download=True, transform=transform_train, class_num=class_num,sample_num=sample_num)
        test_ds = tinyfood101.TinyFood101(root='./data', split='test', download=True, transform=transform_train, class_num=class_num,sample_num=sample_num)
    elif args.dataset == 'tinycaltech256':
        train_ds = tinycaltech.TinyCaltech256(root='./data', split='train', download=True, transform=transform_train, class_num=class_num,sample_num=sample_num)
        test_ds = tinycaltech.TinyCaltech256(root='./data', split='test', download=True, transform=transform_test, class_num=class_num,sample_num=sample_num)
    elif args.dataset == 'tinydomainnet':
        assert args.target_domain_label is not None
        train_ds = tinydomainnet.TinyDomainnet(root='./data',  domain_label=args.target_domain_label, split='train', transform=transform_train, class_num=class_num,sample_num=sample_num)
        test_ds = tinydomainnet.TinyDomainnet(root='./data', domain_label=args.target_domain_label, split='test', transform=transform_test, class_num=class_num,sample_num=sample_num)
    elif args.dataset == 'imagenet50':
        train_ds = imagenet50.Imagenet50(split='trainval', transform=transform_train)
        test_ds = imagenet50.Imagenet50(split='val', transform=transform_train)
    else:
        print("Dataset Not Available")
        exit(0)
    
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    test_loader = DataLoader(test_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    return train_loader, test_loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='imagenet50')
parser.add_argument('--target_domain_label', type=str, default='real')
parser.add_argument('--bs', type=int, default=8)
parser.add_argument('--class_num', type=int) 
parser.add_argument('--sample_num', type=int) 
if __name__ == '__main__':
    
    args = parser.parse_args()

    loader, _= get_dataset_and_loader(args, args.class_num, args.sample_num)
    features = deep_vit_feature_extract(loader, args.class_num, args.sample_num)
    print(features.shape)

    if not args.dataset == 'tinydomainnet':
        save_path = f'/nfs4-p1/wjx/transferbility/experiment/features/{args.dataset}_{args.class_num}_{args.sample_num}'
    else:
        save_path = f'/nfs4-p1/wjx/transferbility/experiment/features/{args.dataset}_{args.target_domain_label}_{args.class_num}_{args.sample_num}'
    torch.save(features, save_path)
    print('Done')


    
