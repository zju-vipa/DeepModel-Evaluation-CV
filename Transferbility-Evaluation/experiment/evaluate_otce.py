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

from metrics.otce import compute_coupling
# 返回dataloader(train and test)
def get_dataset_and_loader(dataset, bs, domain_label=None):
    train_loader, test_loader = '', ''
    size = 112
    
    transform_train = get_training_transform(dataset, size)
    transform_test = get_test_transform(dataset, size)
    
    
    if dataset == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    # 采用默认数据增强来进行
    elif dataset == 'dtd':
        train_ds = torchvision.datasets.DTD(root='./data', split='train', download=True, transform=transform_train)
        test_ds = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform_test)
    elif dataset == 'food101':
        train_ds = food101.Food101(root='./data', split='train', download=True, transform=transform_train)
        test_ds = food101.Food101(root='./data', split='test', download=True, transform=transform_test)
    elif dataset == 'caltech256':
        train_ds = caltech.Caltech256(root='./data', split='train', download=True, transform=transform_train)
        test_ds = caltech.Caltech256(root='./data', split='test', download=True, transform=transform_test)
    elif dataset == 'domainnet' or dataset == 'tinydomainnet':
        assert domain_label is not None
        train_ds = domainnet.DomainNet(image_root='./data/DomainNet',  dataset=domain_label, dataset_name=dataset, split='train', transform=transform_train)
        test_ds = domainnet.DomainNet(image_root='./data/DomainNet', dataset=domain_label, dataset_name=dataset, split='test', transform=transform_test)
    elif dataset == 'imagenet':
        train_ds = torchvision.datasets.ImageNet(root='/datasets/ILSVRC2012', split='train')
        test_ds = torchvision.datasets.ImageNet(root='/datasets/ILSVRC2012', split='test')
    else:
        logging.error("Dataset Not Available")
        exit(0)
    
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=2, batch_size=bs)
    test_loader = DataLoader(test_ds, shuffle=True, num_workers=2, batch_size=bs)
    return train_loader, test_loader

src_features = []
tgt_features = []


parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='tinydomainnet')
parser.add_argument('--target_dataset', type=str, default='tinydomainnet')
parser.add_argument('--arch', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--source_domain_label', type=str, default='real')
parser.add_argument('--target_domain_label', type=str, default='quickdraw')
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--log_path', type=str, default='log/evaluate/otce_result.log')
if __name__ == '__main__':

    def src_hook(module, fea_in, fea_out):
        src_features.append(fea_in[0])
        # src_features_out_hook.append(fea_out)
        return None
    def tgt_hook(module, fea_in, fea_out):
        tgt_features.append(fea_in[0])
        # tgt_features_out_hook.append(fea_out)

    
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    with torch.no_grad():
        # model1 src, model2 tgt
        model1 = get_network(args.arch).eval()
        model2 = get_network(args.arch).eval()
        num_classes = settings.NUM_CLASSES[args.target_dataset]
        if args.arch == 'mobilenet_v3_small':
            if args.source_dataset == 'imagenet' or args.source_dataset == 'imagenet50':
                feature_dim = model1.classifier[3].in_features
                model1.classifier[3] = nn.Linear(feature_dim, num_classes).cuda()
                model2.classifier[3] = nn.Linear(feature_dim, num_classes).cuda()
            else:
                feature_dim = model1.classifier[1].in_features
                model1.classifier[1] = nn.Linear(feature_dim, num_classes).cuda()
                model2.classifier[1] = nn.Linear(feature_dim, num_classes).cuda()
        elif args.arch == 'vgg16':
            feature_dim = model1.classifier[6].in_features
            model1.classifier[6] = nn.Linear(feature_dim, num_classes).cuda()
            model2.classifier[6] = nn.Linear(feature_dim, num_classes).cuda()
        else:
            fc = getattr(model1, settings.CLASSIFICATION_HEAD_NAME[args.arch])
            feature_dim = fc.in_features
            setattr(model1, settings.CLASSIFICATION_HEAD_NAME[args.arch], nn.Linear(feature_dim, num_classes).cuda())
            setattr(model2, settings.CLASSIFICATION_HEAD_NAME[args.arch], nn.Linear(feature_dim, num_classes).cuda())
        source_model_path = Path(args.model_path)
        if not source_model_path.is_file() or source_model_path.suffix != '.pth':
            logging.error("Error: Source model path is wrong!")
            exit(0)
        pre_weights = torch.load(str(source_model_path))
        model1.load_state_dict(pre_weights, strict=False)
        model2.load_state_dict(pre_weights, strict=False)

        src_train_loader, _ = get_dataset_and_loader(args.source_dataset, args.bs, args.source_domain_label)
        tgt_train_loader, _ = get_dataset_and_loader(args.target_dataset, args.bs, args.target_domain_label)

        for (name, module) in model1.named_modules():
            if name == settings.HOOK_MODULE_NAME[args.arch]:
                module.register_forward_hook(hook=src_hook)
        
        for (name, module) in model2.named_modules():
            if name == settings.HOOK_MODULE_NAME[args.arch]:
                module.register_forward_hook(hook=tgt_hook)
        
        tqdm_src_train_loader = tqdm(src_train_loader, file=sys.stdout, position=0)
        for batch_index, (images, _) in enumerate(tqdm_src_train_loader):
            images = Variable(images)
            images = images.cuda()
            model1(images)
        
        tqdm_tgt_train_loader = tqdm(tgt_train_loader, file=sys.stdout, position=0)
        for batch_index, (images, _) in enumerate(tqdm_tgt_train_loader):
            images = Variable(images)
            images = images.cuda()
            model2(images)
        
        src_feature = torch.cat(src_features, dim=0)
        tgt_feature = torch.cat(tgt_features, dim=0)
        
        _, W = compute_coupling(src_feature, tgt_feature)
        if not args.source_dataset == 'tinydomainnet':
            logging.info(f'OTCE: {args.source_dataset}->{args.target_dataset} {args.arch}: ' + '{:.16f}'.format(W))
        else:
            logging.info(f'OTCE: domainnet {args.source_domain_label}->{args.target_domain_label} {args.arch}: ' + '{:.16f}'.format(W))
    
    

    

    


