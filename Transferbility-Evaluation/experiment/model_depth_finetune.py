import argparse
import os
import sys
import logging
import math

import numpy as np
import torch
from torch import nn
import torch.optim as optim

import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import global_settings as settings
from utils import get_network

from dataset.tinyfood101 import TinyFood101

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='cifar100')
parser.add_argument('--target_dataset', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0003)
parser.add_argument('--pretrained_epoch', type=int)
parser.add_argument('--log_name', type=str)
parser.add_argument('--sample_ratio', type=float, default=1.0)
parser.add_argument('--class_ratio', type=float, default=1.0)

def main(args):
     #获取数据集和dataloader
    train_loader, test_loader = get_dataset_and_loader(args.target_dataset, class_ratio=args.class_ratio, sample_ratio=args.sample_ratio)
    
    model_dict = {}
    logging.info('Load test models')
    num_classes = math.floor(settings.NUM_CLASSES[args.target_dataset] * args.class_ratio)
    # 一共选择6个模型
    # 全部选择训练190epoch的模型
    for model_name in tqdm(args.model_name_list):
        logging.info('Load model ' + model_name)
        #获取目标数据集 类别数
        args.arch = model_name
        model = get_network(args.arch)
        fc = getattr(model, settings.CLASSIFICATION_HEAD_NAME[args.arch])
        feature_dim = fc.in_features
        setattr(model, settings.CLASSIFICATION_HEAD_NAME[args.arch], nn.Linear(feature_dim, num_classes).cuda())
        
        checkpoint_path = os.path.join('checkpoints', args.source_dataset, args.arch)
        source_checkpoint_path_regular = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path))[-1], "{model}-{ep}-{type}.pth".format(model=args.arch, ep=args.pretrained_epoch, type='regular'))
        source_checkpoint_path_best = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path))[-1], "{model}-{ep}-{type}.pth".format(model=args.arch, ep=args.pretrained_epoch, type='best'))
        if os.path.exists(source_checkpoint_path_regular):
            pred_weights = torch.load(source_checkpoint_path_regular)
        elif os.path.exists(source_checkpoint_path_best):
            pred_weights = torch.load(source_checkpoint_path_best)
        else:
            logging.error('Error: ' + source_checkpoint_path_regular + ' not exists!')
            exit(0)
        
        # 这里fc是resnet网络的
        pre_dict = {k:v for k, v in pred_weights.items() if settings.CLASSIFICATION_HEAD_NAME[args.arch] not in k}
        model.load_state_dict(pre_dict, strict=False)
        model_dict[model_name] = model
    
    writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, 'model_depth', args.source_dataset + '->' + f"{args.target_dataset}-{args.class_ratio}-{args.sample_ratio}", settings.TIME_NOW))

    
    #遍历model进行迁移(选取收敛的最好结果)
    for name, model in model_dict.items():
        #设置训练参数
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #learning rate decay
    
        
        #保存checkpoint
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'model_depth', args.source_dataset + '->' + f"{args.target_dataset}-{args.class_ratio}-{args.sample_ratio}", name, settings.TIME_NOW)
        if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
        target_checkpoint_path = os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth')

        #开始迁移学习
        #每5个epoch保存一个checkpoint
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(1, 50):

            train(epoch, model, train_loader, optimizer, loss_function, writer)
            acc = eval_training(epoch, model, test_loader, loss_function, writer)

            #start to save best performance model after learning rate decay to 0.01 
            if best_acc < acc:
                torch.save(model.state_dict(), target_checkpoint_path.format(arch=name, epoch=epoch, type='best'))
                best_acc = acc
                best_epoch = epoch
                continue

            torch.save(model.state_dict(), target_checkpoint_path.format(arch=name, epoch=epoch, type='regular'))
                
            train_scheduler.step()
        
        logging.info('{arch} pre-trained on {source_dataset}  : Best Acc={best_acc}, Epoch: {best_ep}'.format(arch=name, source_dataset=args.source_dataset,  best_acc=best_acc, best_ep=best_epoch))
        
        writer.add_scalar(args.source_dataset + '_' + f"{args.target_dataset}-{args.class_ratio}-{args.sample_ratio}" + f' BestAcc{args.pretrained_epoch}/depth', best_acc, "".join(list(filter(str.isdigit, name))))
        
    writer.close()
        
        
    


def train(epoch, model, train_loader, optimizer, loss_function, writer):

    model.train()
    tqdm_train_loader = tqdm(train_loader, file=sys.stdout, position=0)
    for batch_index, (images, labels) in enumerate(tqdm_train_loader):
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

    logging.info('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))

    #add informations to tensorboard
    writer.add_scalar('Test/Average_loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

# 返回dataloader(train and test)
def get_dataset_and_loader(dataset_name, class_ratio, sample_ratio):
    train_loader, test_loader = '', ''
    
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
        
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.Resize(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
        train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_ds, shuffle=False, num_workers=2, batch_size=128)

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])

        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_ds, shuffle=False, num_workers=2, batch_size=128)
        
    elif dataset_name == 'tinyfood101':
        transform_train = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(settings.FOOD101_TRAIN_MEAN, settings.FOOD101_TRAIN_STD)
        ])
        train_ds = TinyFood101(root='./data', split='train', class_ratio=class_ratio, sample_ratio=sample_ratio, transform=transform_train)
        train_loader = DataLoader(train_ds, shuffle=False, num_workers=2, batch_size=128)

        transform_test = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(settings.FOOD101_TRAIN_MEAN, settings.FOOD101_TRAIN_STD)
        ])

        test_ds = TinyFood101(root='./data', split='test', class_ratio=class_ratio, transform=transform_test)
        test_loader = DataLoader(test_ds, shuffle=False, num_workers=2, batch_size=128)
    else:
        logging.error("Dataset " + dataset_name + " Not Available")
        exit(0)
    
            
    return train_loader, test_loader

# #获取对应模型，默认使用gpu
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
#         logging.error("Arch " + name + " Not Available")
#         exit(0)
#     return model.cuda()


if __name__ == '__main__':

    args = parser.parse_args()
    if not os.path.exists('log/model_depth'):
        os.makedirs('log/model_depth')
    # 配置日志格式
    logging.basicConfig(filename=f'log/model_depth/{args.log_name}', level=logging.INFO)
    
    logging.info('Description: 模型深度和迁移学习性能关系的实验, 选择TinyFood101, 计算快一些, 取前0.4 * 101 = 25 个类别，每个类别选择 750 * 0.6 = 450个样本来训练，测试选择全量250张图片，分辨率统一128*128，无数据增强，其他参数如下所示')
    
    args.model_name_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    
    logging.info(args.__dict__)
    main(args)


    