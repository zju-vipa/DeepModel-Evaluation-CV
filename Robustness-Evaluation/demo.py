import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from transformers import DeiTModel, DeiTConfig, DeiTForImageClassification
from transformers import SwinConfig, SwinForImageClassification
import configparser

import sys
sys.path.append('EvalBox/Classification/')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from EvalBox.Classification.EBD import EBD
from EvalBox.Classification.NS import NS


import torch
import torch.nn as nn
from torchvision import models


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, help='Enter off-the-shelf model name')

config = configparser.ConfigParser()

if __name__ == "__main__":


    args = parser.parse_args()
    name = args.name
    assert name != None, "Please select model name"

    print("Model name is {}".format(name))

    config.read('config.ini')

    torchvisionWeights = {
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
        "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "googlenet": models.GoogLeNet_Weights.IMAGENET1K_V1,
        "inception_v3": models.Inception_V3_Weights.IMAGENET1K_V1,
        "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1, 
        "mnasnet0_5": models.MNASNet0_5_Weights.IMAGENET1K_V1,
        "regnet_x_16gf": models.RegNet_X_16GF_Weights.IMAGENET1K_V1,
        "shufflenet_v2_x1_0": models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
        "maxvit_t": models.MaxVit_T_Weights.IMAGENET1K_V1,
        "regnet_x_800mf": models.RegNet_X_800MF_Weights.IMAGENET1K_V1,
        "resnext50_32x4d": models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
        "shufflenet_v2_x0_5": models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
        "squeezenet1_0": models.SqueezeNet1_0_Weights.IMAGENET1K_V1,
        "vgg11": models.VGG11_Weights.IMAGENET1K_V1,
        "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1
    }


    torchvisonModels = {
        "resnet50": models.resnet50,
        "densenet121": models.densenet121,
        "efficientnet_b0": models.efficientnet_b0,
        "googlenet": models.googlenet,
        "inception_v3": models.inception_v3,
        "mobilenet_v2": models.mobilenet_v2,
        "mnasnet0_5": models.mnasnet0_5,
        "regnet_x_16gf": models.regnet_x_16gf,
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
        "maxvit_t": models.maxvit_t,
        "regnet_x_800mf": models.regnet_x_800mf,
        "resnext50_32x4d": models.resnext50_32x4d,
        "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5,
        "squeezenet1_0": models.squeezenet1_0,
        "vgg11": models.vgg11,
        "vit_b_16": models.vit_b_16
    }

    huggingfaceModels = {
        'deit': DeiTForImageClassification,
        'swin': SwinForImageClassification,
    }

    huggingfaceConfig = {
        'deit': DeiTConfig(),
        'swin': SwinConfig(),
    }

    huggingfaceTransform = {
        'deit': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406]),
        ]),
        'swin': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406]),
        ]),
    }

    transform = None
    model = None
   

    if name in torchvisonModels:
        weights = torchvisionWeights[name]
        model = torchvisonModels[name](weights=weights)
        transform = weights.transforms()

    if name in huggingfaceModels:
        model = huggingfaceModels[name](huggingfaceConfig[name])
        transform = huggingfaceTransform[name]


    assert model != None, "Model is not loaded correctly!"
    print("model loaded")

    # test_dataset = datasets.ImageNet(root=config['Dataset-ImageNet-Original']['path'], split='val', transform=transform)

    # print("dataset prepared")

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=512,
    #     shuffle=False,
    # )

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # ebd = EBD(model, 'Log/Classification/2023-05-25 13-54-45 resnet50-torchvision-imagenet-predicted.log')
    # # ebd = EBD(model, platform="huggingface")
    # # ebd.apply(eval_loader=test_loader)
    ns = NS(model=model, device=config['Global']['device'])
    ns.apply()

# from tqdm import tqdm
# import datetime
# print("Start evaluation")


# now = datetime.datetime.now()
# time_str = now.strftime("%Y-%m-%d %H-%M-%S")
# f = open("Log/Classification/" + time_str + " resnet50-torchvision-imagenet-predicted.log", "w")
# with torch.no_grad():
#     # 在测试集上进行推理
#     model.eval()
#     correct = 0
#     total = 0
#     for images, labels in tqdm(test_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # print(outputs.shape)
#         _, predicted = torch.max(outputs.data, 1)
#         # print(predicted)
#         total += labels.size(0)

#         # is_correct_list = (predicted == labels).to_list()

#         correct += (predicted == labels).sum().item()
#         predicted = predicted.cpu().numpy().flatten()

#         for predicted_label in predicted:
#             f.write("{}\n".format(predicted_label))

# #     # 输出准确率
#     print('准确率: {:.2f}%'.format(100 * correct / total))

# f.close()






































