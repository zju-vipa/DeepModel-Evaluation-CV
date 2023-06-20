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
from utils import get_network, get_dataset_and_loader, get_pretrain_network, get_torchvision_network

from metrics.pge import pge, get_dis


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='imagenet50')
parser.add_argument('--domain_label', type=str, default='real')
parser.add_argument('--arch', type=str)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--pge_grad_path', type=str, default='checkpoints/pge_grad')

if __name__ == '__main__':

    args = parser.parse_args()
    if not os.path.exists(args.pge_grad_path):
        os.makedirs(args.pge_grad_path)
    
    if args.dataset == 'imagenet' or args.dataset == 'imagenet50':
        model, _ = get_torchvision_network(args.arch, pretrain=False)
    else:
        model = get_network(args.arch)
    

    
    num_classes = settings.NUM_CLASSES[args.dataset]
    if args.arch == 'mobilenet_v3_small':
        if args.dataset == 'imagenet' or args.dataset == 'imagenet50':
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

    if args.dataset == 'domainnet' or args.dataset == 'tinydomainnet':
        assert args.domain_label is not None
        train_loader, _ = get_dataset_and_loader(args.dataset, args.bs, args.domain_label)
    else:
        train_loader, _ = get_dataset_and_loader(args.dataset, args.bs)
    
    if args.dataset == 'domainnet' or args.dataset == 'tinydomainnet':
        dataset_name = f'{args.dataset}_{args.domain_label}'
    else:
        dataset_name = args.dataset

    criterion = nn.CrossEntropyLoss()
    save_path = os.path.join(args.pge_grad_path, '{0}-{1}'.format(args.arch, dataset_name))
    pge_grad = pge(model, args.arch, train_loader, args.dataset, criterion, save_path=save_path)


