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
from utils import get_network, get_pretrain_network, get_training_transform, get_test_transform, WarmUpLR
from dataset import caltech, food101, tinyfood101, domainnet

# 返回dataloader(train and test)
def get_dataset_and_loader(args):
    train_loader, test_loader = '', ''
    size = 112
    if args.arch == 'inception_v3':
        size = 299
    
    transform_train = get_training_transform(args.source_dataset, size)
    transform_test = get_test_transform(args.source_dataset, size)
    
    if args.source_dataset == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    # 采用默认数据增强来进行
    elif args.source_dataset == 'dtd':
        train_ds = torchvision.datasets.DTD(root='./data', split='train', download=True, transform=transform_train)
        test_ds = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform_test)
    elif args.source_dataset == 'food101':
        train_ds = food101.Food101(root='./data', split='train', download=True, transform=transform_train)
        test_ds = food101.Food101(root='./data', split='test', download=True, transform=transform_test)
    elif args.source_dataset == 'tinyfood101':
        train_ds = tinyfood101.TinyFood101(root='./data', split='train', download=True, transform=transform_train)
        test_ds = tinyfood101.TinyFood101(root='./data', split='test', download=True, transform=transform_train)
    elif args.source_dataset == 'caltech256':
        train_ds = caltech.Caltech256(root='./data', split='train', download=True, transform=transform_train)
        test_ds = caltech.Caltech256(root='./data', split='test', download=True, transform=transform_test)
    elif args.source_dataset == 'domainnet' or args.source_dataset == 'tinydomainnet':
        assert args.source_domain_label is not None
        train_ds = domainnet.DomainNet(image_root='./data/DomainNet',  dataset=args.source_domain_label, dataset_name=args.source_dataset, split='train', transform=transform_train)
        test_ds = domainnet.DomainNet(image_root='./data/DomainNet', dataset=args.source_domain_label, dataset_name=args.source_dataset, split='test', transform=transform_test)
    else:
        print("Dataset Not Available")
        exit(0)
    
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    test_loader = DataLoader(test_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    return train_loader, test_loader

def eval_training(model, test_loader, loss_function):
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0 # top1 error

    tqdm_test_loader = tqdm(test_loader)
    for (images, labels) in tqdm_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    return correct.float() / len(test_loader.dataset)

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='cifar100')
parser.add_argument('--source_domain_label', type=str, default='real')
parser.add_argument('--arch', type=str)
parser.add_argument('--model_path', type=str, default='') # 如果resume，则表示model的路径；不是resume，则表示迁移学习源模型的路径（若源数据集是imagenet则使用torchvision直接加载，此参数无效）
parser.add_argument('--bs', type=int, default=16)
if __name__ == '__main__':
    args = parser.parse_args()
    _, test_loader = get_dataset_and_loader(args)

    if args.source_dataset == 'imagenet':
            model, weights = get_pretrain_network(args.arch)
    else:
        model = get_network(args.arch)
    num_classes = settings.NUM_CLASSES[args.source_dataset]
    # 更换分类头
    if args.arch == 'mobilenet_v3_small':
        if args.source_dataset == 'imagenet':
            feature_dim = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(feature_dim, num_classes).cuda()
        else:
            feature_dim = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(feature_dim, num_classes).cuda()
        
    elif args.arch == 'inception_v3':
        Aux_feature_dim = model.AuxLogits.fc.in_features
        feature_dim = model.fc.in_features
        model.AuxLogits.fc = nn.Linear(Aux_feature_dim, num_classes).cuda()
        model.fc = nn.Linear(feature_dim, num_classes).cuda()
    elif args.arch == 'vgg16':
        feature_dim = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(feature_dim, num_classes).cuda()
    else:
        fc = getattr(model, settings.CLASSIFICATION_HEAD_NAME[args.arch])
        feature_dim = fc.in_features
        setattr(model, settings.CLASSIFICATION_HEAD_NAME[args.arch], nn.Linear(feature_dim, num_classes).cuda())

    print(model)
    source_model_path = Path(args.model_path)
    if not source_model_path.is_file() or source_model_path.suffix != '.pth':
        logging.error("Error: Source model path is wrong!")
        exit(0)
    pre_weights = torch.load(str(source_model_path))
    model.load_state_dict(pre_weights, strict=True)
    file_name = source_model_path.name

    model.eval()
    loss_function = nn.CrossEntropyLoss()
    acc = eval_training(model, test_loader, loss_function)
    print(acc)
    

