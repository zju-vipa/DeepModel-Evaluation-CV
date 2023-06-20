import argparse
import os
import sys 
import pathlib
import time
from pathlib import Path
import logging

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
from dataset import tinycaltech, tinyfood101, tinydtd, tinydomainnet

from metrics.gbc import get_gbc_score
from sklearn.decomposition import PCA
from metrics import leep


seed=10
torch.manual_seed(seed)
# 返回dataloader(train and test)
def get_dataset_and_loader(target_dataset, class_num, sample_num, target_domain_label=None):
    train_loader, test_loader = '', ''
    size = 112
    
    print(class_num, sample_num)
    transform_train = get_training_transform(target_dataset, size)
    transform_test = get_test_transform(target_dataset, size)
    
    if target_dataset == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    # 采用默认数据增强来进行
    elif target_dataset == 'tinydtd':
        train_ds = tinydtd.TinyDTD(root='./data', split='train', download=True, transform=transform_test, class_num=class_num, sample_num=sample_num)
        test_ds = tinydtd.TinyDTD(root='./data', split='test', download=True, transform=transform_test, class_num=class_num, sample_num=sample_num)
    elif target_dataset == 'tinyfood101':
        train_ds = tinyfood101.TinyFood101(root='./data', split='train', download=True, transform=transform_test, class_num=class_num, sample_num=sample_num)
        test_ds = tinyfood101.TinyFood101(root='./data', split='test', download=True, transform=transform_train, class_num=class_num, sample_num=sample_num)
    elif target_dataset == 'tinycaltech256':
        train_ds = tinycaltech.TinyCaltech256(root='./data', split='train', download=True, transform=transform_test, class_num=class_num, sample_num=sample_num)
        test_ds = tinycaltech.TinyCaltech256(root='./data', split='test', download=True, transform=transform_test, class_num=class_num, sample_num=sample_num)
    elif target_dataset == 'tinydomainnet':
        assert target_domain_label is not None
        train_ds = tinydomainnet.TinyDomainnet(root='./data',  domain_label=target_domain_label, split='train', transform=transform_test, class_num=class_num, sample_num=sample_num)
        test_ds = tinydomainnet.TinyDomainnet(root='./data', domain_label=target_domain_label, split='test', transform=transform_test, class_num=class_num, sample_num=sample_num)
    else:
        logging.error("Dataset Not Available")
        exit(0)
    
    train_loader = DataLoader(train_ds, shuffle=False, num_workers=8, batch_size=args.bs)
    test_loader = DataLoader(test_ds, shuffle=False, num_workers=8, batch_size=args.bs)
    return train_loader, test_loader

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='tinydomainnet')
parser.add_argument('--source_domain_label', type=str, default='real')
# parser.add_argument('--target_domain_label', type=str, default='real')
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--class_num', type=int) 
parser.add_argument('--sample_num', type=int) 
parser.add_argument('--log_path', type=str, default='log/evaluate/leep_result.log')
if __name__ == '__main__':
    
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    with torch.no_grad():

        archs = ['vgg16', 'resnet50', 'mobilenet_v3_small', 'densenet161']
        model_paths = {
            'vgg16' : '/nfs4-p1/wjx/transferbility/experiment/checkpoints/domainnet/real/vgg16/2023-05-03T17:11:57.481693/vgg16-300-regular.pth',
            'resnet50' : '/nfs4-p1/wjx/transferbility/experiment/checkpoints/domainnet/real/resnet50/2023-05-03T16:31:05.388789/resnet50-300-regular.pth',
            'mobilenet_v3_small' : '/nfs4-p1/wjx/transferbility/experiment/checkpoints/domainnet/real/mobilenet_v3_small/2023-05-03T17:48:31.490878/mobilenet_v3_small-300-regular.pth',
            'densenet161' : '/nfs4-p1/wjx/transferbility/experiment/checkpoints/domainnet/real/densenet161/2023-05-06T03:33:45.857828/densenet161-300-regular.pth'
        }
        # archs = ['vgg16']
        arch_scores = {}
        target_scores = {}
        for arch in archs:
            # model src
            if args.source_dataset == 'imagenet':
                model, _ = get_pretrain_network(arch)
            else:
                model = get_network(arch).eval()

                num_classes = 40
                if arch == 'mobilenet_v3_small':
                    feature_dim = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(feature_dim, num_classes).cuda()
                elif arch == 'inception_v3':
                    Aux_feature_dim = model.AuxLogits.fc.in_features
                    feature_dim = model.fc.in_features
                    model.AuxLogits.fc = nn.Linear(Aux_feature_dim, num_classes).cuda()
                    model.fc = nn.Linear(feature_dim, num_classes).cuda()
                elif arch == 'vgg16':
                    feature_dim = model.classifier[6].in_features
                    model.classifier[6] = nn.Linear(feature_dim, num_classes).cuda()
                else:
                    fc = getattr(model, settings.CLASSIFICATION_HEAD_NAME[arch])
                    feature_dim = fc.in_features
                    setattr(model, settings.CLASSIFICATION_HEAD_NAME[arch], nn.Linear(feature_dim, num_classes).cuda())
            
            if not args.source_dataset == 'imagenet':
                source_model_path = Path(model_paths[arch])
                if not source_model_path.is_file() or source_model_path.suffix != '.pth':
                    logging.error("Error: Source model path is wrong!")
                    exit(0)
                pre_weights = torch.load(str(source_model_path))
                model.load_state_dict(pre_weights, strict=True)
            
            model_scores = []
            target_dataset = 'tinydomainnet'
            target_domain_labels = ['quickdraw', 'infograph', 'clipart', 'sketch']
            # target_datasets = ['tinydtd', 'tinycaltech256', 'tinydomainnet']
            # target_domain_label = 'real'
            # for target_domain_label in target_domain_labels:
            for target_domain_label in target_domain_labels:
                if target_dataset == 'domainnet' or target_dataset == 'tinydomainnet':
                    assert target_domain_label is not None
                    print(f'Load Domainnet {target_domain_label}')
                    tgt_train_loader, _ = get_dataset_and_loader(target_dataset=target_dataset, class_num=args.class_num, sample_num=args.sample_num, target_domain_label=target_domain_label)
                else:
                    print(f'Load {target_dataset}')
                    tgt_train_loader, _ = get_dataset_and_loader(target_dataset=target_dataset, class_num=args.class_num, sample_num=args.sample_num)
            
                score = leep.leep(model, tgt_train_loader, args.class_num)
                model_scores.append(score)
                if not args.source_dataset == 'tinydomainnet':
                    logging.info(f'LEEP: {args.source_dataset}->{target_dataset}_{args.class_num}_{args.sample_num} {arch}: ' + '{:.16f}'.format(score))
                else:
                    logging.info(f'LEEP: domainnet {args.source_domain_label}->{target_domain_label} {arch}: ' + '{:.16f}'.format(score))
                if target_domain_label not in target_scores.keys():
                    target_scores[target_domain_label] = [score]
                else:
                    target_scores[target_domain_label].append(score)

            
            arch_scores[arch] = model_scores
        
        logging.info(f'LEEP: {target_domain_labels}')
        for arch, scores in arch_scores.items():
            logging.info(f'LEEP: {arch}: {scores}')
            
        for target, scores in target_scores.items():
            logging.info(f'LEEP: {target}: {scores}')
    
    

    

    


