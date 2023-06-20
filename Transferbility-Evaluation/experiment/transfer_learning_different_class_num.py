import argparse
import os
import sys 
import pathlib
from pathlib import Path
import logging
import random

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
from utils import get_network, get_torchvision_network, get_training_transform, get_test_transform, WarmUpLR
from dataset import caltech, food101, tinyfood101_old, domainnet, cifar100

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='imagenet')
parser.add_argument('--target_dataset', type=str, default='cifar100')
parser.add_argument('--class_num', type=int)
# parser.add_argument('--source_domain_label', type=str, default='real')
# parser.add_argument('--target_domain_label', type=str)
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--model_path', type=str, default='') # 如果resume，则表示model的路径；不是resume，则表示迁移学习源模型的路径（若源数据集是imagenet则使用torchvision直接加载，此参数无效）
parser.add_argument('--resume',action='store_true', default=False) # resume表示迁移学习没有完成的模型进一步训练
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_factor', type=float, default=0.2)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--tensorboard_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--checkpoint_save_step', type=int, default=5)
parser.add_argument('--log_path', type=str)
parser.add_argument('--end_epoch', type=str, default=200)

seed=10
torch.manual_seed(seed)
#迁移学习
def transfer_learning(args):

    num_classes = args.class_num
    # 生成类别
    if num_classes != 100:
        classes = random.sample(range(0, 100), num_classes)
    else:
        classes = range(0, 100)
    logging.info(f'Class_num: {num_classes}')
    logging.info(f'Classes: {classes}')

    #获取数据集和dataloader
    train_loader, test_loader = get_dataset_and_loader(args, classes=classes)
    start_epoch = 1

    #获取预训练模型
    #直接加载torchvision提供的模型，及预训练权重（在后续实验中一直用到）
    if args.source_dataset == 'imagenet':
        model = get_torchvision_network(args.arch, class_num=1000, pretrain=True)
    else:
        model = get_network(args.arch)
    
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
    
    # 不是resume
    if not args.resume and not args.source_dataset == 'imagenet':
        source_model_path = Path(args.model_path)
        if not source_model_path.is_file() or source_model_path.suffix != '.pth':
            logging.error("Error: Source model path is wrong!")
            exit(0)

        pre_weights = torch.load(str(source_model_path))
        pre_dict = {k:v for k, v in pre_weights.items() if settings.CLASSIFICATION_HEAD_NAME[args.arch] not in k}
        model.load_state_dict(pre_dict, strict=False)
    # resume
    elif not args.source_dataset == 'imagenet':
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
        
    
    #设置fine-tune参数
    #fine-tune采用全部微调的方式
    loss_function = nn.CrossEntropyLoss()
    
    # SGD
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 余弦退火学习率
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=(args.end_epoch - start_epoch + 1))
    # StepLR
    # train_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=args.lr_factor)
    # ReduceLROnPlateau
    # train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',  factor=0.1, patience=10, verbose=False, threshold=0.001, threshold_mode='rel', cooldown=10, eps=1e-08)

    #tensorboard设置
    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_path, settings.TIME_NOW))
    
    #迁移学习checkpoint设置
    checkpoint_path = os.path.join(args.checkpoint_path, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth')

    #开始迁移
    best_acc = 0.0
    for epoch in range(start_epoch, args.end_epoch+1):
        train(epoch, model, train_loader, optimizer, loss_function, writer, args.arch)
        acc, test_loss = eval_training(epoch, model, test_loader, loss_function, writer)

        logging.info('Epoch {:d} Test set: LR: {:.8f},  Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            epoch, optimizer.param_groups[0]['lr'], test_loss, acc 
        ))
        
        if best_acc < acc:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % args.checkpoint_save_step:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular'))
        
        train_scheduler.step()

    writer.close()
    return model


def train(epoch, model, train_loader, optimizer, loss_function, writer, model_name=None):

    model.train()
    tqdm_train_loader = tqdm(train_loader, file=sys.stdout, position=0)
    for batch_index, (images, labels) in enumerate(tqdm_train_loader):
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        if model_name == 'inception_v3':
            outputs, aux_val_output = model(images)
        else:
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

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

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
def get_dataset_and_loader(args, classes=None):
    train_loader, test_loader = '', ''
    size = 112
    if args.arch == 'inception_v3':
        size = 299
    
    transform_train = get_training_transform(args.target_dataset, size)
    transform_test = get_test_transform(args.target_dataset, size)
    
    if args.target_dataset == 'cifar100':
        train_ds = cifar100.Cifar100(root='./data', split='train',  transform=transform_train, classes=classes)
        test_ds = cifar100.Cifar100(root='./data', split='test', transform=transform_test, classes=classes)
    # 采用默认数据增强来进行
    elif args.target_dataset == 'dtd':
        train_ds = torchvision.datasets.DTD(root='./data', split='train', download=True, transform=transform_train)
        test_ds = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform_test)
    elif args.target_dataset == 'food101':
        train_ds = food101.Food101(root='./data', split='train', download=True, transform=transform_train)
        test_ds = food101.Food101(root='./data', split='test', download=True, transform=transform_test)
    elif args.target_dataset == 'tinyfood101':
        train_ds = tinyfood101_old.TinyFood101(root='./data', split='train', download=True, transform=transform_train)
        test_ds = tinyfood101_old.TinyFood101(root='./data', split='test', download=True, transform=transform_train)
    elif args.target_dataset == 'caltech256':
        train_ds = caltech.Caltech256(root='./data', split='train', download=True, transform=transform_train)
        test_ds = caltech.Caltech256(root='./data', split='test', download=True, transform=transform_test)
    elif args.target_dataset == 'domainnet' or args.target_dataset == 'tinydomainnet':
        assert args.target_domain_label is not None
        train_ds = domainnet.DomainNet(image_root='./data/DomainNet',  dataset=args.target_domain_label, dataset_name=args.target_dataset, split='train', transform=transform_train)
        test_ds = domainnet.DomainNet(image_root='./data/DomainNet', dataset=args.target_domain_label, dataset_name=args.target_dataset, split='test', transform=transform_test)
    else:
        logging.error("Dataset Not Available")
        exit(0)
    
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    test_loader = DataLoader(test_ds, shuffle=True, num_workers=8, batch_size=args.bs)
    return train_loader, test_loader

#获取对应模型，默认使用gpu
# def get_model(name, num_classes=100, cuda=True):
#     model = ''
#     if name == 'resnet18':
#         model = models.resnet18(num_classes=num_classes, weights=None)
#     elif name == 'resnet34':
#         model = models.resnet34(num_classes=num_classes, weights=None)
#     elif name == 'resnet50':
#         model = models.resnet50(num_classes=num_classes, weights=None)
#     elif name == 'resnet101':
#         model = models.resnet101(num_classes=num_classes, weights=None)
#     elif name == 'resnet152':
#         model = models.resnet152(num_classes=num_classes, weights=None)
#     else:
#         print("Not Available")
#         exit(0)
#     return model.cuda()


if __name__ == '__main__':

    args = parser.parse_args()
    
    path_list = args.log_path.split('/')
    if not os.path.exists(os.path.join(*path_list[:-1])):
        os.makedirs(os.path.join(*path_list[:-1]))
    # 配置日志格式
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logging.info('Description: 迁移学习日志，参数如下')
    logging.info(args.__dict__)
    
    model = transfer_learning(args)


    