#coding=utf-8
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

from torch.autograd import Variable

from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import global_settings as settings
from utils import get_network

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='cifar100')
parser.add_argument('--target_dataset', type=str, default='mnist')
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--pretrained_epoch', type=int)

def main(args):
    #获取数据集和dataloader
    train_loader, test_loader = get_dataset_and_loader(args.target_dataset, transfer=True)

    #导入预训练模型参数
    #共200epoch，按10为间隔选取模型

    #找到文件夹
    checkpoint_path = os.path.join('checkpoints', 'sample_num', args.source_dataset, args.arch)
    if not os.path.exists(checkpoint_path):
        print('Error: Model not exists!')
        exit(0)
    sample_portion_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    model_dict = {}
    print("Load test models:")
    num_classes = settings.NUM_CLASSES[args.target_dataset]
    # 一共选择6个模型
    # 全部选择训练190epoch的模型
    for p in tqdm(sample_portion_list):
        #获取目标数据集 类别数
        model = get_network(args.arch)
        fc = getattr(model, settings.CLASSIFICATION_HEAD_NAME[args.arch])
        feature_dim = fc.in_features
        setattr(model, settings.CLASSIFICATION_HEAD_NAME[args.arch], nn.Linear(feature_dim, num_classes).cuda())
        
        source_checkpoint_path = os.path.join(checkpoint_path, str(p))
        source_checkpoint_path_regular = os.path.join(source_checkpoint_path, os.listdir(source_checkpoint_path)[0], "{model}-{ep}-{type}.pth".format(model=args.arch, ep=args.pretrained_epoch, type='regular'))
        source_checkpoint_path_best = os.path.join(source_checkpoint_path, os.listdir(source_checkpoint_path)[0], "{model}-{ep}-{type}.pth".format(model=args.arch, ep=args.pretrained_epoch, type='best'))
        if os.path.exists(source_checkpoint_path_regular):
            pred_weights = torch.load(source_checkpoint_path_regular)
        elif os.path.exists(source_checkpoint_path_best):
            pred_weights = torch.load(source_checkpoint_path_best)
        else:
            print('Error: ' + source_checkpoint_path_regular + ' not exists!')
            exit(0)
        
        # 这里fc是resnet网络的
        pre_dict = {k:v for k, v in pred_weights.items() if settings.CLASSIFICATION_HEAD_NAME[args.arch] not in k}
        model.load_state_dict(pre_dict, strict=False)
        model_dict[str(p)] = model
    
    writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, 'sample_num', args.source_dataset + '->' + args.target_dataset, args.arch, settings.TIME_NOW))

    
    index = 1
    #遍历model进行迁移(选取收敛的最好结果)
    for p, model in model_dict.items():
        #设置训练参数
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #learning rate decay
    
        
        #保存checkpoint
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'sample_num', args.source_dataset + '->' + args.target_dataset, args.arch, str(p), settings.TIME_NOW)
        if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
        target_checkpoint_path = os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth')

        #开始预训练
        #每5个epoch保存一个checkpoint
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(1, settings.TR_TOTAL_EPOCH):

            train(epoch, model, train_loader, optimizer, loss_function, writer)
            acc = eval_training(epoch, model, test_loader, loss_function, writer)

            #start to save best performance model after learning rate decay to 0.01 
            if best_acc < acc:
                torch.save(model.state_dict(), target_checkpoint_path.format(arch=args.arch, epoch=epoch, type='best'))
                best_acc = acc
                best_epoch = epoch
                continue

            if not epoch % settings.SAVE_EPOCH:
                torch.save(model.state_dict(), target_checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular'))
                
            train_scheduler.step()
        
        print('{arch} pre-trained on {source_dataset} of {p} samples: Best Acc={best_acc}, Epoch: {best_ep}'.format(arch=args.arch, source_dataset=args.source_dataset, p=p, best_acc=best_acc, best_ep=best_epoch))
        
        writer.add_scalar(args.source_dataset + '->' + args.target_dataset + f' BestAcc{args.pretrained_epoch}/sample_num_portion', best_acc, index)
        index += 1
        
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
def get_dataset_and_loader(dataset_name, transfer=False, transfer_train_transform=None, transfer_test_transform=None):
    train_loader, test_loader = '', ''
    if not transfer:
        if dataset_name == 'cifar100':
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
            print("Dataset " + dataset_name + " Not Available")
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
        print("Arch " + name + " Not Available")
        exit(0)
    return model.cuda()


if __name__ == '__main__':

    args = parser.parse_args()

    main(args)


    