import argparse
import os
import sys 
import pathlib
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
from utils import get_network, get_pretrain_network, get_training_transform, get_test_transform
from dataset import tinyfood101, tinycaltech, tinydomainnet, tinydtd, imagenet50

from metrics import leep

# 返回dataloader(train and test)
def get_dataset_and_loader(target_dataset, class_num, sample_num, target_domain_label=None):
    train_loader, test_loader = '', ''
    size = 112
    
    transform_train = get_training_transform(target_dataset, size)
    transform_test = get_test_transform(target_dataset, size)
    
    if target_dataset == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    # 采用默认数据增强来进行
    elif target_dataset == 'tinydtd':
        train_ds = tinydtd.TinyDTD(root='./data', split='train', download=True, transform=transform_test, sample_num=sample_num)
        test_ds = tinydtd.TinyDTD(root='./data', split='test', download=True, transform=transform_test, sample_num=sample_num)
    elif target_dataset == 'tinyfood101':
        train_ds = tinyfood101.TinyFood101(root='./data', split='train', download=True, transform=transform_test, sample_num=sample_num)
        test_ds = tinyfood101.TinyFood101(root='./data', split='test', download=True, transform=transform_train, sample_num=sample_num)
    elif target_dataset == 'tinycaltech256':
        train_ds = tinycaltech.TinyCaltech256(root='./data', split='train', download=True, transform=transform_test, sample_num=sample_num)
        test_ds = tinycaltech.TinyCaltech256(root='./data', split='test', download=True, transform=transform_test, sample_num=sample_num)
    elif target_dataset == 'tinydomainnet':
        assert target_domain_label is not None
        train_ds = tinydomainnet.TinyDomainnet(root='./data',  domain_label=target_domain_label, split='train', transform=transform_test, sample_num=sample_num)
        test_ds = tinydomainnet.TinyDomainnet(root='./data', domain_label=target_domain_label, split='test', transform=transform_test, sample_num=sample_num)
    else:
        logging.error("Dataset Not Available")
        exit(0)
    
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    test_loader = DataLoader(test_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    return train_loader, test_loader

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='imagenet')
parser.add_argument('--target_dataset', type=str)
parser.add_argument('--source_domain_label', type=str, default='real')
parser.add_argument('--target_domain_label', type=str, default='quickdraw')
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--class_num', type=int) 
parser.add_argument('--log_path', type=str, default='log/evaluate/leep_result.log')
if __name__ == '__main__':

    args = parser.parse_args()

    path_list = args.log_path.split('/')
    if not os.path.exists(os.path.join(*path_list[:-1])):
        os.makedirs(os.path.join(*path_list[:-1]))
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    
    with torch.no_grad():
        archs = ['vgg16', 'resnet50', 'mobilenet_v3_small', 'densenet161']
        arch_scores = {}
        sample_num_scores = {}
        for arch in archs:
            if args.source_dataset == 'imagenet' or args.source_dataset == 'imagenet50':
                model, _ = get_pretrain_network(arch)
            else:
                model = get_network(arch).eval()
                # num_classes = settings.NUM_CLASSES[args.source_dataset]
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

                source_model_path = Path(args.model_path)
                if not source_model_path.is_file() or source_model_path.suffix != '.pth':
                    logging.error("Error: Source model path is wrong!")
                    exit(0)
                pre_weights = torch.load(str(source_model_path))
                model.load_state_dict(pre_weights, strict=True)

            sample_nums =  [2, 5, 10, 20, 35, 50, 75]
            # sample_nums = 2
            scores = []

            for sample_num in sample_nums:
                if args.target_dataset == 'domainnet' or args.target_dataset == 'tinydomainnet':
                    assert args.target_domain_label is not None
                    train_loader, _ = get_dataset_and_loader(target_dataset=args.target_dataset, class_num=args.class_num, sample_num=sample_num, target_domain_label=args.target_domain_label)
                else:
                    train_loader, _ = get_dataset_and_loader(target_dataset=args.target_dataset, class_num=args.class_num, sample_num=sample_num)
                
                score = leep.leep(model, train_loader, args.class_num)
                scores.append(score)
                if not args.source_dataset == 'tinydomainnet' and not args.source_dataset == 'domainnet':
                    logging.info(f'LEEP: {args.source_dataset}->{args.target_dataset}_{args.class_num}_{sample_num} {arch}: ' + '{:.16f}'.format(score))
                else:
                    logging.info(f'LEEP: domainnet {args.source_domain_label}->{args.target_dataset}_{args.class_num}_{sample_num} {arch}: ' + '{:.16f}'.format(score))

                if sample_num not in sample_num_scores.keys():
                    sample_num_scores[sample_num] = [score]
                else:
                    sample_num_scores[sample_num].append(score)
            arch_scores[arch] = scores
    
    logging.info(f'LEEP: {sample_nums}')
    for arch, scores in arch_scores.items():
        logging.info(f'LEEP: {arch}: {scores}')
        
    for sample_num, scores in sample_num_scores.items():
            logging.info(f'GBC: {sample_num}: {scores}')

