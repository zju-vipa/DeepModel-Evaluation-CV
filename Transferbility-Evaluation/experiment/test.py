from dataset.tinyfood101 import TinyFood101
from dataset.caltech import Caltech256
from utils import get_mean_std, get_network
import logging
import os

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torchvision.datasets as datasets

from utils import get_dataset_and_loader, get_network, get_torchvision_network

from dataset import domainnet, imagenet50, caltech, food101, tinyfood101
import argparse
import time
# from torchstat import stat
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_ds = Caltech256(root='./data', split='train', download=True, transform=transforms.Compose([
#     transforms.Resize([224,224]),
#     transforms.ToTensor()
# ]))
# train_loader = DataLoader(dataset=train_ds, batch_size=64, shuffle=True)

# mean, std = get_mean_std(train_loader)
# print(mean)
# print(std)

# parser = argparse.ArgumentParser()
# parser.add_argument('--arch', type=str)
# args = parser.parse_args()
# model = get_network(args)
# for name, param in model.named_parameters():
#     if not param.requires_grad:
#         print(name)


# 得到模型已有的权重枚举类型名称
# model_list = models.list_models(module=torchvision.models)
# for m in model_list:
#     print(m)
# model_list = ['resnet34','densenet161', 'mobilenet_v3_small', 'inception_v3']
# weights_list = [models.ResNet34_Weights.IMAGENET1K_V1, models.DenseNet161_Weights.IMAGENET1K_V1, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, models.Inception_V3_Weights.IMAGENET1K_V1]
# logging.basicConfig(filename='log/model_weight/log.log', level=logging.INFO)
# for w in weights_list:
#     transform = w.transforms()
#     logging.info(transform)

# name = 'resnet18'
# for i in models.get_model_weights(name):
#     print(i)

# model = models.mobilenet_v3_small()
# model = models.resnet34()
# model = models.densenet161()
# model = models.inception_v3()
# for name, m in model.named_children():
#     print(name, " >>> ", m)


# print(len(os.listdir('/nfs4-p1/wjx/transferbility/experiment/data/caltech256/256_ObjectCategories/056.dog')))
# for name in os.listdir('/nfs4-p1/wjx/transferbility/experiment/data/caltech256/256_ObjectCategories/056.dog'):
#     if 'jpg' not in name:
#         print(name)

# ds = domainnet.DomainNet(image_root='./data/DomainNet',  dataset='real', dataset_name='tinydomainnet', split='train')

# model = get_network(args)
# print(model.classifier[6])

# t1 = time.time()
# ds = datasets.ImageNet(root='/datasets/ILSVRC2012', split='val')
# print('Done')
# t2 = time.time()
# print((t2-t1) / 60)

# source_model_path = Path('/nfs4-p1/wjx/transferbility/experiment/checkpoints/imagenet_caltech256/resnet50/2023-05-03T19:08:29.942223/resnet50-180-regular.pth')\

# print(source_model_path.is_file())
# print(source_model_path.suffix)
# file_name = source_model_path.name
# print(file_name.split('-'))

# model = get_network('densenet161')
# for (name, module) in model.named_modules():
#     print(name)
# print(torch.get_default_dtype())

# s_f = torch.nn.Softmax(dim=0)
# a = torch.Tensor([[[1.0, 2.0, 3.0, 4.0]]])
# b = torch.Tensor([5.0, 6.0, 7.0, 8.0])
# print(s_f(a))
# print(s_f(b))
# a = torch.randn([3, 3])
# b = torch.Tensor([0, 1, 0])
# b = torch.tensor(b, dtype=torch.int64)
# print(a)
# a[b] += torch.ones([3, 3])
# print(a)

# train, test = get_dataset_and_loader('imagenet50', 10)
# for image, label in train:
#     continue

# p = Path('/nfs4-p1/wjx/transferbility/experiment/checkpoints/domainnet/real/vgg16/2023-05-03T17:11:57.481693/vgg16-300-regular.pth')
# print(p.is_file())
# print(p.suffix == '.pth')

# print(torch.cat([torch.Tensor([1.0, 2.0]),torch.Tensor([1.0, 2.0])] ))

# a = [True, False, True]
# b = torch.randn(3, 3)
# print(b)
# print(b[a])
# a = torch.Tensor([1.0])
# b = torch.Tensor([2.0, 3.0])
# d = torch.Tensor([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
# e = torch.Tensor([2.0, 3.0, 2.0])
# # c = [a, b, d]
# # e = [b, a, d]
# # print(torch.cat(c))
# # print(torch.cat(e))
# print(torch.div(d, e))

# caltech.Caltech256(root='./data', split='train', download=True)
# caltech.Caltech256(root='./data', split='train', download=True)
# domainnet.DomainNet(image_root='./data/DomainNet',  dataset='real', dataset_name='domainnet', split='train')

# from dataset import tinycaltech, tinydomainnet
# domain_label = 'painting'
# tinydomainnet.TinyDomainnet(root='./data', domain_label=domain_label, split='train', class_num=40, sample_num=12)
# tinydomainnet.TinyDomainnet(root='./data', domain_label=domain_label, split='test', class_num=40, sample_num=12)

# from dataset import tinycaltech
# tinycaltech.TinyCaltech256(root='./data', split='train', class_num=45, sample_num=80)
# tinycaltech.TinyCaltech256(root='./data', split='test', class_num=45, sample_num=80)

# from dataset import tinydtd
# tinydtd.TinyDTD(root='./data', split='train', class_num=45, sample_num=5)
# tinydtd.TinyDTD(root='./data', split='test', class_num=45, sample_num=5)

# from dataset import tinyfood101
# tinyfood101.TinyFood101(root='./data', split='train', class_num=45, sample_num=5)
# tinyfood101.TinyFood101(root='./data', split='test', class_num=45, sample_num=5)

# model1 = get_network('densenet161')
# model2 = get_torchvision_network('densenet161')
# print(model1)
# print(model2)
# densenet_torch, _ = get_torchvision_network('resnet50')
# densenet_torch.cpu()
# stat(densenet, (3, 112, 112))
# stat(densenet_torch, (3, 112, 112))

# a = torch.ones(2, 3)

# b = torch.Tensor([1, 2])
# b = b.unsqueeze(0)
# print(b.shape)

# print(a)
# print(b.view(b.shape[1], -1))
# print(b)

# a = [-0.5935959815979004, -27.533113479614258, -109.94434356689453, -154.84161376953125, -212.82379150390625, -212.24581909179688, -232.3478240966797]

# b = [-0.6182572841644287, -18.83757781982422, -62.58692932128906, -102.34619140625, -132.7967529296875, -148.45416259765625, -172.69325256347656]

# c = [-1.4513880014419556, -34.361183166503906, -75.9618148803711, -143.59327697753906, -199.22740173339844, -216.92063903808594, -236.4709930419922]

# d = [-0.45988133549690247, -20.94731330871582, -64.9219970703125, -110.95137023925781, -135.3296661376953, -162.07362365722656, -170.06137084960938]

# logging.basicConfig(filename='log/evaluate/gbc_result.log', level=logging.INFO)
# scores = {}
# sample_nums = [2, 5, 10, 20, 35, 50, 75]
# for at, bt, ct, dt, sample_num in zip(a, b, c, d, sample_nums):
#     scores = [at, bt, ct, dt]
#     logging.info(f'GBC: {sample_num}: {scores}')

# food101
a = [-1.1423434019088745, -1.524027705192566, -1.8039271831512451, -2.1237173080444336, -2.2880969047546387, -2.3569014072418213, -2.462486743927002]

# caltech
b = [-0.08964068442583084, -0.1569584757089615, -0.21217672526836395, -0.32905998826026917, -0.44420114159584045, -0.5419591069221497, -0.6410917639732361]

#dtd
c = [-0.5442280769348145, -0.8217195868492126, -1.0058624744415283, -1.2668116092681885, -1.5017768144607544, -1.609323263168335, -1.7440390586853027]

#domainnet
d = [-0.040365494787693024, -0.14709828794002533, -0.20539073646068573, -0.2948218882083893, -0.3894674479961395, -0.4807109534740448, -0.5605493783950806]

for at, bt, ct, dt in zip(a, b, c, d):
    scores = [at, bt, ct, dt]
    print(scores)
