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

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='cifar100')
parser.add_argument('--target_dataset', type=str, default='mnist')
parser.add_argument('--arch', type=str)
parser.add_argument('--lr', type=float, default=0.1)

def main(args):
     #获取数据集和dataloader
    train_loader, test_loader = get_dataset_and_loader(args.source_dataset)

    #获取模型
    model = get_model(args.arch, 100)

    #设置预训练参数
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    
    #使用预热学习率
    # iter_per_epoch = len(train_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)
    
    #使用 tensorboard可视化
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
        
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.source_dataset, args.arch, settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    # 可视化网络结构
    writer.add_graph(model, Variable(input_tensor, requires_grad=True))


    #保存checkpoint
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.source_dataset, args.arch, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth')

    #开始预训练
    #每10个epoch保存一个checkpoint
    best_acc = 0.0
    for epoch in range(1, settings.TOTAL_EPOCH):
        train(epoch, model, train_loader, optimizer, loss_function, writer)
        acc = eval_training(epoch, model, test_loader, loss_function, writer)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular'))

        train_scheduler.step(epoch)
    writer.close()
    
    
    #迁移学习
    tr_train_loader, tr_test_loader = get_dataset_and_loader(args.target_dataset, transfer=True)

    #模型使用刚预训练到固定epoch的模型，作为预训练模型(ResNet)
    #更改分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10).cuda()

    #设置fine-tune参数
    #fine-tune采用全部微调的方式
    tr_loss_function = nn.CrossEntropyLoss()
    tr_optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    tr_train_scheduler = optim.lr_scheduler.StepLR(tr_optimizer, 10, gamma=0.1)

    #tensorboard设置
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.source_dataset+'->'+args.target_dataset, args.arch, settings.TIME_NOW))
    
    
    #迁移学习checkpoint设置
    tr_checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.source_dataset+'->'+args.target_dataset, args.arch, settings.TIME_NOW)
    if not os.path.exists(tr_checkpoint_path):
        os.makedirs(tr_checkpoint_path)
    tr_checkpoint_path = os.path.join(tr_checkpoint_path, '{arch}-{epoch}-{type}.pth')

    #开始迁移
    #每10个epoch保存一个checkpoint
    best_acc = 0.0
    for epoch in range(1, settings.TR_TOTAL_EPOCH):
        train(epoch, model, tr_train_loader, tr_optimizer, tr_loss_function, writer)
        acc = eval_training(epoch, model, tr_test_loader, tr_loss_function, writer)

        #start to save best performance model after learning rate decay to 0.01 
        if best_acc < acc:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular'))

        tr_train_scheduler.step(epoch)
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
def get_model(name,num_classes=100, cuda=True):
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


    