
import argparse
import os
import sys 

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
from utils import WarmUpLR, get_network

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='cifar100')
parser.add_argument('--target_dataset', type=str, default='mnist')
parser.add_argument('--arch', type=str)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=128)

def pre_train(args):
     #获取数据集和dataloader
    train_loader, test_loader = get_dataset_and_loader(args)

    #获取模型
    model = get_network(args)

    #设置预训练参数
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    
    #使用预热学习率
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
    
    #使用 tensorboard可视化
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
        
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.source_dataset, args.arch, settings.TIME_NOW))

    #保存checkpoint
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.source_dataset, args.arch, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth')

    #开始预训练
    #每10个epoch保存一个checkpoint
    best_acc = 0.0
    for epoch in range(1, settings.TOTAL_EPOCH):
        train(epoch, model, train_loader, optimizer, loss_function, writer, warmup_scheduler)
        acc = eval_training(epoch, model, test_loader, loss_function, writer)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular'))

        if epoch > 1:
            train_scheduler.step(epoch)
    
    writer.close()

def train(epoch, model, train_loader, optimizer, loss_function, writer, warmup_scheduler):

    model.train()
    tqdm_train_loader = tqdm(train_loader, file=sys.stdout, position=0)
    for batch_index, (images, labels) in enumerate(tqdm_train_loader):
        if epoch <= 1:
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

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

# 返回dataloader(train and test)
def get_dataset_and_loader(args):
    train_loader, test_loader = '', ''
    if args.source_dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
        train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_ds, shuffle=False, num_workers=2, batch_size=args.bs)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])

        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_ds, shuffle=False, num_workers=2, batch_size=args.bs)
    else:
        print("Not Available")
        exit(0)

    return train_loader, test_loader

#获取对应模型，默认使用gpu
def get_model(name, num_classes=100, cuda=True):
    model = ''
    if name == 'resnet18':
        model = models.resnet18(num_classes=num_classes, weights=None)
    elif name == 'resnet34':
        model = models.resnet34(num_classes=num_classes, weights=None)
    elif name == 'resnet50':
        model = models.resnet50(num_classes=num_classes, weights=None)
    elif name == 'resnet101':
        model = models.resnet101(num_classes=num_classes, weights=None)
    elif name == 'resnet152':
        model = models.resnet152(num_classes=num_classes, weights=None)
    else:
        print("Not Available")
        exit(0)
    return model.cuda()


if __name__ == '__main__':

    args = parser.parse_args()

    pre_train(args)


    