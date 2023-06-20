import sys
import logging
import os
import numpy as np
from tqdm import tqdm
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

from dataset import caltech, domainnet, food101, tinyfood101, imagenet50

import global_settings as settings

#from dataset import CIFAR100Train, CIFAR100Test

def get_network(arch, use_gpu=True):
    """ return given network
    """

    if arch == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif arch == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif arch == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif arch == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif arch == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif arch == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif arch == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif arch == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif arch == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif arch == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif arch == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif arch == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif arch == 'xception':
        from models.xception import xception
        net = xception()
    elif arch == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif arch == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif arch == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif arch == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif arch == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif arch == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif arch == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif arch == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif arch == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif arch == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif arch == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif arch == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif arch == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif arch == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif arch == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif arch == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif arch == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif arch == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif arch == 'mobilenet_v3_small':
        from models.mobilenetv3 import mobilenetv3
        net = mobilenetv3(mode='small')
    elif arch == 'mobilenet_v3_large':
        from models.mobilenetv3 import mobilenetv3
        net = mobilenetv3(mode='large')
    elif arch == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif arch == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif arch == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif arch == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif arch == 'seresnet34':
        from models.senet import seresnet34 
        net = seresnet34()
    elif arch == 'seresnet50':
        from models.senet import seresnet50 
        net = seresnet50()
    elif arch == 'seresnet101':
        from models.senet import seresnet101 
        net = seresnet101()
    elif arch == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
        net = net.cuda()

    return net

#获取torchvision提供的预训练模型，和权重(imagenet1k)
def get_pretrain_network(arch, cuda=True):
    model = ''
    weights = ''
    if arch == 'resnet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
    elif arch == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    elif arch == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    elif arch == 'densenet161':
        weights = models.DenseNet161_Weights.IMAGENET1K_V1
        model = models.densenet161(weights=weights)
    elif arch == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    elif arch == 'vgg16':
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1
        model = models.vgg16_bn(weights=weights)
    elif arch == 'inception_v3':
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights)
    else:
        raise RuntimeError(f"Not model available for {arch} on imagenet")

    if cuda:
        model.cuda()
    return model, weights

def get_torchvision_network(arch, class_num, pretrain=False, cuda=True):
    model = ''
    weights = ''
    if pretrain:
        if arch == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            model = models.resnet34(weights=weights, num_classes=class_num)
        elif arch == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights, num_classes=class_num)
        elif arch == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights, num_classes=class_num)
        elif arch == 'densenet161':
            weights = models.DenseNet161_Weights.IMAGENET1K_V1
            model = models.densenet161(weights=weights, num_classes=class_num)
        elif arch == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = models.mobilenet_v3_small(weights=weights, num_classes=class_num)
        elif arch == 'vgg16':
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1
            model = models.vgg16_bn(weights=weights, num_classes=class_num)
        elif arch == 'inception_v3':
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            model = models.inception_v3(weights=weights, num_classes=class_num)
        else:
            raise RuntimeError(f"Not model available for {arch} on imagenet")
    else:
        if arch == 'resnet34':
            model = models.resnet34(weights=None, num_classes=class_num)
        elif arch == 'resnet18':
            model = models.resnet18(weights=None, num_classes=class_num)
        elif arch == 'resnet50':
            model = models.resnet50(weights=None, num_classes=class_num)
        elif arch == 'densenet161':
            model = models.densenet161(weights=None, num_classes=class_num)
        elif arch == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights=None, num_classes=class_num)
        elif arch == 'vgg16':
            model = models.vgg16_bn(weights=None, num_classes=class_num)
        elif arch == 'inception_v3':
            model = models.inception_v3(weights=None, num_classes=class_num)
        else:
            raise RuntimeError(f"Not model available for {arch} from torchvision")

    if cuda:
        model.cuda()
    return model
    
def get_training_transform(dataset, size=112):
    transform = ''
    if dataset == 'cifar100' or dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomResizedCrop([32, 32]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
    elif dataset == 'dtd' or dataset == 'tinydtd':
        transform = transforms.Compose([
            transforms.RandomResizedCrop([size, size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.DTD_TRAIN_MEAN, settings.DTD_TRAIN_STD)
        ])
    elif dataset == 'food101' or dataset == 'tinyfood101':
        transform = transforms.Compose([
            transforms.RandomResizedCrop([size, size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.FOOD101_TRAIN_MEAN, settings.FOOD101_TRAIN_STD)
        ])
    elif dataset == 'caltech256' or dataset == 'tinycaltech256':
        transform = transforms.Compose([
            transforms.RandomResizedCrop([size, size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.CALTECH256_TRAIN_MEAN, settings.CALTECH256_TRAIN_STD)
        ])
    elif dataset == 'domainnet' or dataset == 'tinydomainnet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop([size, size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.DOMAINNET_TRAIN_MEAN, settings.DOMAINNET_TRAIN_STD)
        ])
    elif dataset == 'imagenet50':
        transform = transforms.Compose([
            transforms.RandomResizedCrop([size, size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.IMAGENET_TRAIN_MEAN, settings.IMAGENET_TRAIN_STD)
        ])
    else:
        raise RuntimeError(f"Transform: Dataset {dataset} not exists")
    return transform
    
def get_test_transform(dataset, size=112):
    transform = ''
    if dataset == 'cifar100' or dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
    elif dataset == 'dtd' or dataset == 'tinydtd':
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop([size, size]),
            transforms.ToTensor(),
            transforms.Normalize(settings.DTD_TRAIN_MEAN, settings.DTD_TRAIN_STD)
        ])
    elif dataset == 'food101' or dataset == 'tinyfood101':
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop([size, size]),
            transforms.ToTensor(),
            transforms.Normalize(settings.FOOD101_TRAIN_MEAN, settings.FOOD101_TRAIN_STD)
        ])
    elif dataset == 'caltech256' or dataset == 'tinycaltech256':
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop([size, size]),
            transforms.ToTensor(),
            transforms.Normalize(settings.CALTECH256_TRAIN_MEAN, settings.CALTECH256_TRAIN_STD)
        ])
    elif dataset == 'domainnet' or dataset == 'tinydomainnet':
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop([size, size]),
            transforms.ToTensor(),
            transforms.Normalize(settings.DOMAINNET_TRAIN_MEAN, settings.DOMAINNET_TRAIN_STD)
        ])
    elif dataset == 'imagenet50':
       transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop([size, size]),
            transforms.ToTensor(),
            transforms.Normalize(settings.IMAGENET_TRAIN_MEAN, settings.IMAGENET_TRAIN_STD)
        ])
    else:
        raise RuntimeError(f"Transform: Dataset {dataset} not exists")
    
    
    return transform

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
    elif dataset == 'imagenet50':
        train_ds = imagenet50.Imagenet50(split='trainval', transform=transform_train)
        test_ds = imagenet50.Imagenet50(split='val', transform=transform_train)
    else:
        logging.error("Dataset Not Available")
        exit(0)
    
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=4, batch_size=bs)
    test_loader = DataLoader(test_ds, shuffle=True, num_workers=4, batch_size=bs)
    return train_loader, test_loader


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def general_class_file(class_file_path, save_path, class_num, sample_num):
    class_names = os.listdir(class_file_path)
    classes = []
    for class_name in class_names:
        if math.floor(len(os.listdir(os.path.join(class_file_path, class_name))) * 0.8) >= sample_num:
            classes.append(class_name)
        if len(classes) == class_num:
            break
    if len(classes) < class_num:
        print('Error: Not Enough images')
        exit(1)
    f = open(save_path, 'w')
    for n in classes:
        f.write(n + '\n')
    f.close()
    print('Success ' + save_path)

def general_class_file_temp():
    class_index = [2, 21, 38, 54, 55, 57, 67, 79, 83, 89, 95, 110, 115, 117, 119, 128, 131, 137, 139, 154, 161, 162, 165, 174, 179, 182, 203, 210, 243, 246, 269, 274, 277, 292, 303, 308, 309, 317, 324, 332]
    class_file_path = '/nfs4-p1/wjx/transferbility/experiment/data/DomainNet/clipart'
    save_path = '/nfs4-p1/wjx/transferbility/experiment/data/DomainNet/traintest/real_40_class'
    class_names = os.listdir(class_file_path)
    select_class_names = []
    for i in class_index:
        select_class_names.append(class_names[i])
    f = open(save_path, 'w')
    for n in select_class_names:
        f.write(n + '\n')
    f.close()
    print('Success ' + save_path)

def compute_pearson(Mat1, Mat2):
    return np.corrcoef(Mat1, Mat2)


def get_acc_from_log(log_path, save_path='/nfs4/wjx/transferbility/experiment/log/temp.log'):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        acc = []
        for line in lines:
            if 'Accuracy:' in line:
                acc.append(float(line[-7:-1].strip()))
        
    f_w = open(save_path, 'a')
    f_w.write(log_path + '\n')
    f_w.write(str(acc))
    f_w.write('\n')
    f_w.close()

def norm(x):
    x = 200 * math.atan(x)
    x /= math.pi
    if x <= 0:
        x += 100
    return x


# if __name__ == '__main__':
    # class_file_path = '/nfs4-p1/wjx/transferbility/experiment/data/DomainNet/quickdraw'
    # class_num = 45
    # sample_num = 80
    # domain_label = 'quickdraw'
    # # save_path = '/nfs4-p1/wjx/transferbility/experiment/data/DomainNet/traintest/' + f'{class_num}_class'
    # save_path = '/nfs4-p1/wjx/transferbility/experiment/data/DomainNet/traintest/' + f'{domain_label}_{class_num}_class'
    # general_class_file(class_file_path, save_path, class_num, sample_num)

    # general_class_file_temp()
    
    # food101
    # GT = np.array([0.5383, 0.5444, 0.4177, 0.5639], dtype='float32')
    
    # caltech256
    # GT = np.array([0.8830, 0.8750, 0.7750, 0.8884], dtype='float32')
    
    # dtd
    # GT = np.array([0.6503, 0.6374, 0.5236, 0.6633])
    
    # domainnet_real
    # GT = np.array([0.7651, 0.7737, 0.6714, 0.7680])
    
    
    
    # 有数据增强
    # GBC = {
    #     2 : np.array([-0.5927016735076904, -0.6161342859268188, -1.4384171962738037, -0.46756190061569214], dtype='float32'),
    #     5:  np.array([-27.339984893798828, -18.82773208618164, -34.248600006103516, -21.009408950805664], dtype='float32'),
    #     10: np.array([-110.62705993652344, -62.70158386230469, -76.05364990234375, -65.2789306640625], dtype='float32'),
    #     20: np.array([-155.30470275878906, -102.28736877441406, -143.67324829101562, -110.91399383544922], dtype='float32'),
    #     35: np.array([-213.33705139160156, -132.53802490234375, -199.5028839111328, -135.34283447265625], dtype='float32'),
    #     50: np.array([-212.14471435546875, -148.77456665039062, -216.64659118652344, -162.05113220214844], dtype='float32'),
    #     75: np.array([-232.35963439941406, -172.74253845214844, -236.4102020263672, -170.1233367919922], dtype='float32'),
    # }
    
    
    # 无数据增强
    # caltech256
    # GBC = {
    #     2 : np.array([-0.048024293035268784, -0.009696952998638153, -0.13651971518993378, -0.01436699740588665], dtype='float32'),
    #     5:  np.array([-0.2269451767206192, -0.09134134650230408, -1.8296130895614624, -0.12004079669713974], dtype='float32'),
    #     10: np.array([-0.5362597703933716, -0.2685742676258087, -5.2464070320129395, -0.13879333436489105], dtype='float32'),
    #     20: np.array([-0.9462781548500061, -0.43727490305900574, -9.99323558807373, -0.27690589427948], dtype='float32'),
    #     35: np.array([-1.3551806211471558, -0.8453568816184998, -16.358314514160156, -0.48182037472724915], dtype='float32'),
    #     50: np.array([-1.4800089597702026, -0.9603784680366516, -18.283199310302734, -0.5944811105728149], dtype='float32'),
    #     75: np.array([-1.4483342170715332, -1.1077405214309692, -20.974271774291992, -0.7068461179733276], dtype='float32'),
    # }
    
    # dtd
    # GBC = {
    #     2 : np.array([-0.3710222840309143, -0.4250386953353882, -0.9057834148406982, -0.6981114149093628], dtype='float32'),
    #     5:  np.array([-8.685672760009766, -6.105576515197754, -9.426580429077148, -10.150843620300293], dtype='float32'),
    #     10: np.array([-24.434368133544922, -19.328941345214844, -31.883445739746094, -25.39607810974121], dtype='float32'),
    #     20: np.array([-28.62234115600586, -26.981708526611328, -46.30844497680664, -34.852928161621094], dtype='float32'),
    #     35: np.array([-36.36018371582031, -34.67959213256836, -59.06226348876953, -44.6309700012207], dtype='float32'),
    #     50: np.array([-37.972164154052734, -36.39981460571289, -61.610748291015625, -46.20335006713867], dtype='float32'),
    #     75: np.array([-39.20087432861328, -39.06902313232422, -66.4333267211914, -49.501853942871094], dtype='float32'),
    # }
     # domainnet_real
    # GBC = {
    #     2 : np.array([-0.014720160514116287, -0.010564716532826424, -0.09470690786838531, -0.0073592024855315685], dtype='float32'),
    #     5:  np.array([-0.5102369785308838, -0.3210894763469696, -1.02501380443573, -0.325858473777771], dtype='float32'),
    #     10: np.array([-0.9574849605560303, -0.30697864294052124, -3.2374343872070312, -0.47888973355293274], dtype='float32'),
    #     20: np.array([-1.273423433303833, -0.37903767824172974, -5.949032783508301, -0.6518117785453796], dtype='float32'),
    #     35: np.array([-0.9199018478393555, -0.3486481010913849, -7.063453197479248, -0.5227451324462891], dtype='float32'),
    #     50: np.array([-1.2833399772644043, -0.5077776312828064, -8.904624938964844, -0.7034992575645447], dtype='float32'),
    #     75: np.array([-1.4538559913635254, -0.6033698916435242, -10.061837196350098, -0.795966625213623], dtype='float32'),
    # }
    
    
    # food101
    # LEEP = {
    #     2 : np.array([-1.4198287725448608, -1.2838808298110962, -1.8722316026687622, -1.1423434019088745], dtype='float32'),
    #     5:  np.array([-1.8355298042297363, -1.5757489204406738, -2.352278709411621, -1.524027705192566], dtype='float32'),
    #     10: np.array([-2.1064279079437256, -1.9172114133834839, -2.6016552448272705, -1.8039271831512451], dtype='float32'),
    #     20: np.array([-2.3221616744995117, -2.1667826175689697, -2.837067127227783, -2.1237173080444336], dtype='float32'),
    #     35: np.array([-2.458050489425659, -2.344268798828125, -2.9736557006835938, -2.2880969047546387], dtype='float32'),
    #     50: np.array([-2.505713939666748, -2.4159445762634277, -3.035611629486084, -2.3569014072418213], dtype='float32'),
    #     75: np.array([-2.5702099800109863, -2.4883556365966797, -3.0891261100769043, -2.462486743927002], dtype='float32'),
    # }
    
    # dtd
    # LEEP = {
    #     2 : np.array([-0.7785151600837708, -0.7867864966392517, -1.7239112854003906, -0.5442280769348145], dtype='float32'),
    #     5:  np.array([-1.1498234272003174, -1.1142451763153076, -2.078871011734009, -0.8217195868492126], dtype='float32'),
    #     10: np.array([-1.4181221723556519, -1.4033410549163818, -2.3252153396606445, -1.0058624744415283], dtype='float32'),
    #     20: np.array([-1.667333722114563, -1.6274328231811523, -2.5943984985351562, -1.2668116092681885], dtype='float32'),
    #     35: np.array([-1.847398042678833, -1.7715518474578857, -2.71509051322937, -1.5017768144607544], dtype='float32'),
    #     50: np.array([-1.9499651193618774, -1.8777318000793457, -2.7670140266418457, -1.609323263168335], dtype='float32'),
    #     75: np.array([-2.0350589752197266, -1.9941984415054321, -2.8200273513793945, -1.7440390586853027], dtype='float32'),
    # }
    
    # caltech256
    # LEEP = {
    #     2 : np.array([-0.11741828918457031, -0.11735725402832031, -0.7602846026420593, -0.08964068442583084], dtype='float32'),
    #     5:  np.array([-0.2797502279281616, -0.23462876677513123, -1.1174936294555664, -0.1569584757089615], dtype='float32'),
    #     10: np.array([-0.4002099633216858, -0.3173302710056305, -1.3070448637008667, -0.21217672526836395], dtype='float32'),
    #     20: np.array([-0.5406638979911804, -0.4414609968662262, -1.4985483884811401, -0.32905998826026917], dtype='float32'),
    #     35: np.array([-0.6502856612205505, -0.5694165825843811, -1.6598988771438599, -0.44420114159584045], dtype='float32'),
    #     50: np.array([-0.7194045186042786, -0.6459112167358398, -1.7360389232635498, -0.5419591069221497], dtype='float32'),
    #     75: np.array([-0.8040042519569397, -0.7433145046234131, -1.8161660432815552, -0.6410917639732361], dtype='float32'),
    # }
    
    # domainnet_real
    # LEEP = {
    #     2 : np.array([-0.08213944733142853, -0.06891479343175888, -0.558681309223175, -0.040365494787693024], dtype='float32'),
    #     5:  np.array([-0.26953840255737305, -0.18480908870697021, -0.953008770942688, -0.14709828794002533], dtype='float32'),
    #     10: np.array([-0.37002190947532654, -0.29208123683929443, -1.1490813493728638, -0.20539073646068573], dtype='float32'),
    #     20: np.array([-0.4681621491909027, -0.378055602312088, -1.295548915863037, -0.2948218882083893], dtype='float32'),
    #     35: np.array([-0.5556614398956299, -0.47667741775512695, -1.4215823411941528, -0.3894674479961395], dtype='float32'),
    #     50: np.array([-0.6281460523605347, -0.547954797744751, -1.4915170669555664, -0.4807109534740448], dtype='float32'),
    #     75: np.array([-0.692151665687561, -0.6228830814361572, -1.55868399143219, -0.5605493783950806], dtype='float32'),
    # }
    
    # scores = []
    # for sample_num, GBC_scores in LEEP.items():
    #     scores.append(compute_pearson(GT, GBC_scores)[0][1])
    
    # for i in scores:
    #     print(i)


    # get_acc_from_log('/nfs4/wjx/transferbility/experiment/log/resnet50/imagenet_food101/log1.log')
    
    
# if __name__ == '__main__':
    
#     GT_VGG = [0.5383, 0.8830, 0.6503, 0.7651]
#     GT_RESNET = [0.5444, 0.8750, 0.6374, 0.7737]
#     GT_MOBILENET = [0.4177, 0.7750, 0.5236, 0.6714]
#     GT_DENSENET = [0.5639, 0.8884, 0.6633, 0.7680]
#     # VGG
#     GBC_VGG = {
#         2 : np.array([-0.3936136066913605, -0.048024293035268784, -0.3710222840309143, -0.014720160514116287], dtype='float32'),
#         5:  np.array([-17.936397552490234, -0.2269451767206192, -8.685672760009766, -0.5102369785308838], dtype='float32'),
#         10: np.array([-57.7987174987793, -0.5362597703933716, -24.434368133544922, -0.9574849605560303], dtype='float32'),
#         20: np.array([-91.38324737548828, -0.9462781548500061, -28.62234115600586, -1.273423433303833], dtype='float32'),
#         35: np.array([-119.41167449951172, -1.3551806211471558, -36.36018371582031, -0.9199018478393555], dtype='float32'),
#         50: np.array([-129.5592498779297, -1.4800089597702026, -37.972164154052734, -1.2833399772644043], dtype='float32'),
#         75: np.array([-134.4385223388672, -1.4483342170715332, -39.20087432861328, -1.4538559913635254], dtype='float32'),
#     }
    
#     # Resnet
#     GBC_RESNET = {
#         2 : np.array([-0.5463927984237671, -0.009696952998638153, -0.4250386953353882, -0.010564716532826424], dtype='float32'),
#         5:  np.array([-12.423501014709473, -0.09134134650230408, -6.105576515197754, -0.3210894763469696], dtype='float32'),
#         10: np.array([-46.59213638305664, -0.2685742676258087, -19.328941345214844, -0.30697864294052124], dtype='float32'),
#         20: np.array([-73.87225341796875, -0.43727490305900574, -26.981708526611328, -0.37903767824172974], dtype='float32'),
#         35: np.array([-104.4665298461914, -0.8453568816184998, -34.67959213256836, -0.3486481010913849], dtype='float32'),
#         50: np.array([-116.03933715820312, -0.9603784680366516, -36.39981460571289, -0.5077776312828064], dtype='float32'),
#         75: np.array([-132.10580444335938, -1.1077405214309692, -39.06902313232422, -0.6033698916435242], dtype='float32'),
#     }
    
#     # MobileNet
#     GBC_MOBILENET = {
#         2 : np.array([-1.1867444515228271, -0.13651971518993378, -0.9057834148406982, -0.010564716532826424], dtype='float32'),
#         5:  np.array([-20.938838958740234, -1.8296130895614624, -9.426580429077148, -0.3210894763469696], dtype='float32'),
#         10: np.array([-68.18294525146484, -5.2464070320129395, -31.883445739746094, -0.30697864294052124], dtype='float32'),
#         20: np.array([-117.84859466552734, -9.99323558807373, -46.30844497680664, -0.37903767824172974], dtype='float32'),
#         35: np.array([-156.13052368164062, -16.358314514160156, -59.06226348876953, -0.3486481010913849], dtype='float32'),
#         50: np.array([-180.6627197265625, -18.283199310302734, -61.610748291015625, -0.5077776312828064], dtype='float32'),
#         75: np.array([-195.9774627685547, -20.974271774291992, -66.4333267211914, -0.6033698916435242], dtype='float32'),
#     }
    
#     # Densenet
#     GBC_DENSENET = {
#         2 : np.array([-0.4647059440612793, -0.01436699740588665, -0.6981114149093628, -0.0073592024855315685], dtype='float32'),
#         5:  np.array([-13.843973159790039, -0.12004079669713974, -10.150843620300293, -0.325858473777771], dtype='float32'),
#         10: np.array([-46.540809631347656, -0.13879333436489105, -25.39607810974121, -0.47888973355293274], dtype='float32'),
#         20: np.array([-82.20672607421875, -0.27690589427948, -34.852928161621094, -0.6518117785453796], dtype='float32'),
#         35: np.array([-107.53746795654297, -0.48182037472724915, -44.6309700012207, -0.5227451324462891], dtype='float32'),
#         50: np.array([-119.7061767578125, -0.5944811105728149, -46.20335006713867, -0.7034992575645447], dtype='float32'),
#         75: np.array([-132.2081298828125, -0.7068461179733276, -49.501853942871094, -0.795966625213623], dtype='float32'),
#     }
    
#     #LEEP
#     LEEP_VGG = {
#         2 : np.array([-1.4198287725448608, -0.11741828918457031, -0.7785151600837708, -0.08213944733142853], dtype='float32'),
#         5:  np.array([-1.8355298042297363, -0.2797502279281616, -1.1498234272003174, -0.26953840255737305], dtype='float32'),
#         10: np.array([-2.1064279079437256, -0.4002099633216858, -1.4181221723556519, -0.37002190947532654], dtype='float32'),
#         20: np.array([-2.3221616744995117, -0.5406638979911804, -1.667333722114563, -0.4681621491909027], dtype='float32'),
#         35: np.array([-2.458050489425659, -0.6502856612205505, -1.847398042678833, -0.5556614398956299], dtype='float32'),
#         50: np.array([-2.505713939666748, -0.7194045186042786, -1.9499651193618774, -0.6281460523605347], dtype='float32'),
#         75: np.array([-2.5702099800109863, -0.8040042519569397, -2.0350589752197266, -0.692151665687561], dtype='float32'),
#     }
    
#     LEEP_RESNET = {
#         2 : np.array([-1.2838808298110962, -0.11735725402832031, -0.7867864966392517, -0.06891479343175888], dtype='float32'),
#         5:  np.array([-1.5757489204406738, -0.23462876677513123, -1.1142451763153076, -0.18480908870697021], dtype='float32'),
#         10: np.array([-1.9172114133834839, -0.3173302710056305, -1.4033410549163818, -0.29208123683929443], dtype='float32'),
#         20: np.array([-2.1667826175689697, -0.4414609968662262, -1.6274328231811523, -0.378055602312088], dtype='float32'),
#         35: np.array([-2.344268798828125, -0.5694165825843811, -1.7715518474578857, -0.47667741775512695], dtype='float32'),
#         50: np.array([-2.4159445762634277, -0.6459112167358398, -1.8777318000793457, -0.547954797744751], dtype='float32'),
#         75: np.array([-2.4883556365966797, -0.7433145046234131, -1.9941984415054321, -0.6228830814361572], dtype='float32'),
#     }
    
#     LEEP_MOBILENET = {
#         2 : np.array([-1.8722316026687622, -0.7602846026420593, -1.7239112854003906, -0.558681309223175], dtype='float32'),
#         5:  np.array([-2.352278709411621, -1.1174936294555664, -2.078871011734009, -0.953008770942688], dtype='float32'),
#         10: np.array([-2.6016552448272705, -1.3070448637008667, -2.3252153396606445, -1.1490813493728638], dtype='float32'),
#         20: np.array([-2.837067127227783, -1.4985483884811401, -2.5943984985351562, -1.295548915863037], dtype='float32'),
#         35: np.array([-2.9736557006835938, -1.6598988771438599, -2.71509051322937, -1.4215823411941528], dtype='float32'),
#         50: np.array([-3.035611629486084, -1.7360389232635498, -2.7670140266418457, -1.4915170669555664], dtype='float32'),
#         75: np.array([-3.0891261100769043, -1.8161660432815552, -2.8200273513793945, -1.55868399143219], dtype='float32'),
#     }
    
#     LEEP_DENSENET = {
#         2 : np.array([-1.1423434019088745, -0.08964068442583084, -0.5442280769348145, -0.040365494787693024], dtype='float32'),
#         5:  np.array([-1.524027705192566, -0.1569584757089615, -0.8217195868492126, -0.14709828794002533], dtype='float32'),
#         10: np.array([-1.8039271831512451, -0.21217672526836395, -1.0058624744415283, -0.20539073646068573], dtype='float32'),
#         20: np.array([-2.1237173080444336, -0.32905998826026917, -1.2668116092681885, -0.2948218882083893], dtype='float32'),
#         35: np.array([-2.2880969047546387, -0.44420114159584045, -1.5017768144607544, -0.3894674479961395], dtype='float32'),
#         50: np.array([-2.3569014072418213, -0.5419591069221497, -1.609323263168335, -0.4807109534740448], dtype='float32'),
#         75: np.array([-2.462486743927002, -0.6410917639732361, -1.7440390586853027, -0.5605493783950806], dtype='float32'),
#     }
#     scores = []
#     for sample_num, GBC_scores in LEEP_VGG.items():
#         scores.append(compute_pearson(GT_VGG, GBC_scores)[0][1])
    
#     for i in scores:
#         print(i)
    
if __name__ == '__main__':
    a = [0.147 ,0.263, 0.145, 0.329]
    b = [-3.119,-3.187 ,-3.364 ,-3.228]
    print(compute_pearson(a, b)[0][1])
    
# if __name__ == '__main__':
#     # ImageNet GT
#     ImageNet_GT = np.array([
#         [0.5383, 0.5444, 0.4177, 0.5639],
#         [0.8830, 0.8750, 0.7750, 0.8884],
#         [0.6503, 0.6374, 0.5236, 0.6633],
#         [0.7651, 0.7737, 0.6714, 0.7680]
#     ], dtype='float32')
    
#     Domainnet_GT = np.array([
#         [0.1465, 0.2633, 0.1450, 0.3288],
#         [0.2242, 0.2893, 0.1646, 0.2929],
#         [0.5154, 0.5461, 0.3186, 0.5384],
#         [0.3897, 0.4338, 0.2296, 0.4079]
#     ], dtype='float32')
    
#     ImageNet_LEEP = np.array([
#         [-2.505713939666748, -2.4159445762634277, -3.035611629486084, -2.3569014072418213],
#         [-0.7194045186042786, -0.6459112167358398, -1.7360389232635498, -0.5419591069221497],
#         [-1.9499651193618774, -1.8777318000793457, -2.7670140266418457, -1.609323263168335],
#         [-0.6281460523605347, -0.547954797744751, -1.4915170669555664, -0.4807109534740448]
#     ], dtype='float32')
    
#     Domainnet_LEEP = np.array([
#         [-3.118582487, -3.187020779, -3.364232779, -3.228020191],
#         [-2.58503294, -2.587373495, -2.874448776, -2.585079193],
#         [-1.593101501, -1.828739285, -2.114179373, -1.785196543],
#         [-2.164670944, -2.302207232, -2.611359119, -2.316084146]
#     ], dtype='float32')
    
#     ImageNet_GBC = np.array([
#         [-129.5592499, -123.0393372, -180.6627197, -119.7061768],
#         [-1.48000896, -0.960378468, -18.28319931, -0.594481111],
#         [-37.97216415, -36.39981461, -61.61074829, -46.20335007],
#         [-1.283339977, -0.507777631, -8.904624939, -0.703499258]
#     ], dtype='float32')
    
#     Domainnet_GBC = np.array([
#         [-37.436859130859375, -38.71722412109375, -49.757423400878906,-35.045806884765625],
#         [-131.9421234, -100.2569504, -141.8555603, -107.3612289],
#         [-14.71665001, -16.02541924, -22.18327332, -15.74227047],
#         [-39.67108345, -36.42839813, -64.97670746, -47.50025177]
#     ], dtype='float32')
    
#     ImageNet_EMD = np.array([
#         [2.569456996, 3.351689655, 3.145636379, 3.509466477]
#     ], dtype='float32')
    
#     Domainnet_EMD = np.array([
#         [1.033354093, 3.323431794, 4.160627962, 3.744796814]
#     ], dtype='float32')
    
    # ImageNet
    # for i in range(0, 4):
    #     a = Domainnet_GT[i]
    #     b = Domainnet_GBC[i]
    #     print(compute_pearson(a, b)[0][1])
    
    # for i in range(0, 4):
    #     a = Domainnet_GT[:,i:i+1].reshape(4)
    #     b = Domainnet_GBC[:, i:i+1].reshape(4)
    #     print(compute_pearson(a, b)[0][1])
    
    # ImageNet_GT_mean = np.mean(ImageNet_GT, 1)
    # Domainnet_GT_mean = np.mean(Domainnet_GT, 1)
    
    # for i in range(0, 4):
    #     a = ImageNet_GT[:,i:i+1].reshape(4)
    #     b = ImageNet_EMD[0]
    #     print(compute_pearson(a, b)[0][1])
    # print(compute_pearson(ImageNet_EMD[0], ImageNet_GT_mean)[0][1])
    # print(compute_pearson(Domainnet_EMD[0], Domainnet_GT_mean)[0][1])