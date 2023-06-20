import argparse
import os
import sys 
import random


import numpy as np
import torch
from torch import nn
import torch.optim as optim

import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models

from torch.autograd import Variable

from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

import global_settings as settings
from utils import WarmUpLR, get_network

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='cifar100')
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--lr', type=float, default=0.1)

def main(args):
    
    ratio_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

    train_loader_dict = get_train_loader_different_ratio(args.source_dataset, ratio_list)
    test_loader = get_test_loader(args.source_dataset)
    
    #使用 tensorboard可视化
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
        
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, 'sample_num', args.source_dataset, args.arch, settings.TIME_NOW))
        
    for ratio, train_loader in train_loader_dict.items():
        #获取模型
        model = get_network(args.arch)
        num_classes = settings.NUM_CLASSES[args.source_dataset]
        fc = getattr(model, settings.CLASSIFICATION_HEAD_NAME[args.arch])
        feature_dim = fc.in_features
        setattr(model, settings.CLASSIFICATION_HEAD_NAME[args.arch], nn.Linear(feature_dim, num_classes).cuda())

        #设置预训练参数
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
        
        #使用预热学习率
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
        
        #保存checkpoint
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'sample_num', args.source_dataset, args.arch, str(ratio), settings.TIME_NOW)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth')

        #开始预训练
        #每10个epoch保存一个checkpoint
        best_acc = 0.0
        for epoch in range(1, settings.TOTAL_EPOCH):
            train(epoch, model, train_loader, optimizer, loss_function, writer, warmup_scheduler, ratio)
            acc = eval_training(epoch, model, test_loader, loss_function, writer, ratio)

            #start to save best performance model after learning rate decay to 0.01 
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='best'))
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:
                torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular'))

            if epoch > 1:
                train_scheduler.step(epoch)
                
        print(f'Complish ratio {ratio}')
        
    writer.close()



def train(epoch, model, train_loader, optimizer, loss_function, writer, warmup_scheduler, ratio):

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
                writer.add_scalar('LastLayerGradients/grad_norm2_weights, sample ratio: ' + str(ratio), para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias, sample ratio: ' + str(ratio), para.grad.norm(), n_iter)

        tqdm_train_loader.set_description('Training Epoch: {epoch}, Loss: {:0.4f}, LR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * 128 + len(images),
            total_samples=len(train_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss, sample ratio: ' + str(ratio), loss.item(), n_iter)

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch, model, test_loader, loss_function, writer, ratio):
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
    writer.add_scalar('Test/Average_loss, sample ratio: ' + str(ratio), test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy, sample ratio: ' + str(ratio), correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

def get_test_loader(dataset_name):
    test_loader = ''
    if dataset_name == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
         
        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_ds, shuffle=False, num_workers=2, batch_size=128)
    
    return test_loader
    

# 返回包含不同比例训练数据的dataloader
def get_train_loader_different_ratio(dataset_name, sample_ratio: list):
    dataset_dict = {}
    if dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
        for r in sample_ratio:
            ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            random_indexs = random.sample(range(len(ds)), int(r * len(ds)))
            ds_r = Subset(ds, random_indexs)
            dl_r = DataLoader(ds_r, shuffle=False, num_workers=4, batch_size=128)
            dataset_dict[r] = dl_r
            print(f"Create dataset with ratio {r}, Size: {len(dl_r.dataset)}")
    
    print(f"Generated smaller {dataset_name} datasets with ratio: ", sample_ratio)
    return dataset_dict
            
    
# 返回dataloader(train and test)
def get_dataset_and_loader(dataset_name, transfer=False, transfer_train_transform=None, transfer_test_transform=None):
    train_loader, test_loader = '', ''
    if not transfer:
        if dataset_name == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
            ])
            train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(train_ds, shuffle=False, num_workers=2, batch_size=128)

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
            ])
            
            test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            test_loader = DataLoader(test_ds, shuffle=False, num_workers=2, batch_size=128)
        else:
            print("Not Available")
            exit(0)
    else:
        #暂时先假定source一定是cifar100，后面再改
        if dataset_name == "mnist":
            transfer_train_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
            ])
            train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transfer_train_transform)
            train_loader = DataLoader(train_ds, shuffle=False, num_workers=2, batch_size=128)

            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
            ])
            test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transfer_train_transform)
            test_loader = DataLoader(train_ds, shuffle=False, num_workers=2, batch_size=128)
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

    main(args)


    