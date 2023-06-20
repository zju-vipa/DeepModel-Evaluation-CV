import argparse
import os
import sys 
import logging
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
from utils import get_torchvision_network, get_training_transform, get_test_transform, WarmUpLR
from dataset import caltech, food101, tinyfood101, domainnet

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='cifar100')
parser.add_argument('--arch', type=str)
parser.add_argument('--domain_label', type=str, default='real')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--model_path', type=str, default='') # 如果resume，则表示model的路径
parser.add_argument('--resume',action='store_true', default=False) # resume表示预训练没有完成的模型进一步训练
parser.add_argument('--lr_factor', type=float, default=0.2)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--lr_step', type=int, default=50)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--tensorboard_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--checkpoint_save_step', type=int, default=5)
parser.add_argument('--log_path', type=str)
parser.add_argument('--end_epoch', type=str, default=200)
# parser.add_argument('--gpu_id', type=str, default=0)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
#预训练
def pretrain_learning(args):
    #获取数据集和dataloader
    train_loader, test_loader = get_dataset_and_loader(args)
    logging.info(f'Train sample num: {len(train_loader.dataset)}' )
    logging.info(f'Test sample num: {len(test_loader.dataset)}' )
    start_epoch = 1
    #获取模型
    num_classes = settings.NUM_CLASSES[args.source_dataset]
    model = get_torchvision_network(args.arch, class_num=num_classes, pretrain=False).cuda()

    logging.info(model)
    if args.resume:
        source_model_path = Path(args.model_path)
        if not source_model_path.is_file() or source_model_path.suffix != '.pth':
            logging.error("Error: Source model path is wrong!")
            exit(0)
        pre_weights = torch.load(str(source_model_path))
        model.load_state_dict(pre_weights, strict=False)
        file_name = source_model_path.name
        
        l = file_name.split('-')
        assert len(l) == 3
        start_epoch = int(l[1]) + 1
    
    
    #预训练参数
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # train_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=args.lr_factor)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.end_epoch - start_epoch + 1)

    # # #使用预热学习率
    # if args.source_dataset == 'cifar100':
    #     iter_per_epoch = len(train_loader)
    #     warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
    
    #tensorboard设置
    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_path, settings.TIME_NOW))

    #预训练checkpoint设置
    checkpoint_path = os.path.join(args.checkpoint_path, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth')
    #训练
    best_acc = 0.0
    for epoch in range(start_epoch, args.end_epoch + 1):
        
        train(epoch, model, train_loader, optimizer, loss_function, writer)
        acc, test_loss = eval_training(epoch, model, test_loader, loss_function, writer)

        logging.info('Epoch {:d}, Test set: LR: {:.8f},  Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            epoch, optimizer.param_groups[0]['lr'], test_loss, acc 
        ))
        
        #start to save best performance model after learning rate decay to 0.01 
        if best_acc < acc:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % args.checkpoint_save_step:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular'))

        train_scheduler.step()
    
    writer.close()
    return model


def train(epoch, model, train_loader, optimizer, loss_function, writer, warmup_scheduler=None):

    model.train()

    tqdm_train_loader = tqdm(train_loader, file=sys.stdout, position=0)
    for batch_index, (images, labels) in enumerate(tqdm_train_loader):
        if warmup_scheduler and epoch <= 1:
            warmup_scheduler.step()
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        last_layer = list(model.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        tqdm_train_loader.set_description('Training Epoch: {epoch}, Loss: {:0.4f}, LR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * 128 + len(images),
            total_samples=len(train_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

def eval_training(epoch, model, test_loader, loss_function, writer):
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0 # top1 error

    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    #add informations to tensorboard
    writer.add_scalar('Test/Average_loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset), test_loss / len(test_loader.dataset)

# 返回dataloader(train and test)
def get_dataset_and_loader(args):
    train_loader, test_loader = '', ''
    transform_train = get_training_transform(args.source_dataset)
    transform_test = get_test_transform(args.source_dataset)
    if args.source_dataset == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif args.source_dataset == 'cifar10':
        train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # 采用默认数据增强来进行
    elif args.source_dataset == 'dtd':
        train_ds = torchvision.datasets.DTD(root='./data', split='train', download=True, transform=transform_train)
        test_ds = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform_test)
    elif args.source_dataset == 'food101':
        train_ds = food101.Food101(root='./data', split='train', download=True, transform=transform_train)
        test_ds = food101.Food101(root='./data', split='test', download=True, transform=transform_test)
    elif args.source_dataset == 'caltech256':
        train_ds = caltech.Caltech256(root='./data', split='train', download=True, transform=transform_train)
        test_ds = caltech.Caltech256(root='./data', split='test', download=True, transform=transform_test)
    elif args.source_dataset == 'domainnet' or args.source_dataset == 'tinydomainnet':
        assert args.domain_label is not None
        train_ds = domainnet.DomainNet(image_root='./data/DomainNet',  dataset=args.domain_label, dataset_name=args.source_dataset, split='train', transform=transform_train)
        test_ds = domainnet.DomainNet(image_root='./data/DomainNet', dataset=args.domain_label, dataset_name=args.source_dataset, split='test', transform=transform_test)
    else:
        logging.error("Dataset Not Available")
        exit(0)

    train_loader = DataLoader(train_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    test_loader = DataLoader(test_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    return train_loader, test_loader


if __name__ == '__main__':

    #设置随机种子
    seed=10
    torch.manual_seed(seed)
    
    args = parser.parse_args()
    
    path_list = args.log_path.split('/')
    if not os.path.exists(os.path.join(*path_list[:-1])):
        os.makedirs(os.path.join(*path_list[:-1]))
    # 配置日志格式
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logging.info('Description: 预训练日志，参数如下')
    logging.info(args.__dict__)
    
    model = pretrain_learning(args)
        


    