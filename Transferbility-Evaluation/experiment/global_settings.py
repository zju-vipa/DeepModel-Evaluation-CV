import os
from datetime import datetime
from torchvision import transforms

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
FOOD101_TRAIN_MEAN = (0.5451, 0.4436, 0.3437)
FOOD101_TRAIN_STD = (0.2668, 0.2689, 0.2737)
DTD_TRAIN_MEAN = (0.5292, 0.4738, 0.4258)
DTD_TRAIN_STD = (0.2623, 0.2529, 0.2606)
CALTECH256_TRAIN_MEAN = (0.485, 0.456, 0.406)
CALTECH256_TRAIN_STD = (0.229, 0.224, 0.225)
DOMAINNET_TRAIN_MEAN = (0.485, 0.456, 0.406)
DOMAINNET_TRAIN_STD = (0.229, 0.224, 0.225)
IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)
#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoints'

#total training epoches
TOTAL_EPOCH = 200 
MILESTONES = [60, 120, 160]

#transfer epochs
TR_TOTAL_EPOCH = 100
#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#log file dir
LOG_ROOT = 'log'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
TR_SAVE_EPOCH = 5

NUM_CLASSES = {
    'mnist' : 10,
    'cifar100' : 100,
    'cifar10' : 10,
    'food101': 101,
    'caltech256': 257,
    'dtd': 47,
    'domainnet' : 345,
    'tinydomainnet' : 40,
    'imagenet' : 1000,
    'imagenet50' : 1000,
    'tinyfood101' : 101
}

CLASSIFICATION_HEAD_NAME = {
    'resnet18' : 'fc',
    'resnet34' : 'fc',
    'resnet50' : 'fc',
    'resnet101' : 'fc',
    'resnet152' : 'fc',
    'densenet161' : 'classifier',
    'mobilenet_v3_small' : 'classifier',
    'inception_v3' : 'fc',
    'vgg16' : 'classifier.6'
}

HOOK_MODULE_NAME = {
    'resnet50' : 'fc', 
    'vgg16' : 'classifier.0',
    'mobilenet_v3_small' : 'classifier.0',
    'densenet161' : 'classifier'
}





