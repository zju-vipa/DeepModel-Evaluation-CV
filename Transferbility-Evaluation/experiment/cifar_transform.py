import os
from PIL import Image
import torchvision as tv
import numpy as np
import torch
import _pickle
 
# 解压缩，返回解压后的字典
# def unpickle(file):
#     fo = open(file, 'rb')
#     dict = pickle.load(fo, encoding='latin1')
#     fo.close()
#     return dict
def index_to_label():
    #精细类别的序号与名称 序号:名称
    fineLabelNameDict={}
    #精细类别对应的粗糙类别 精细序号：粗糙序号-粗糙名称
    fineLableToCoraseLabelDict={}
    # 解压数据
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = _pickle.load(fo, encoding='bytes')
        return dict
    # 给定路径添加数据
    def Dealdata(meta, train):
        for fineLabel, coarseLabel in zip(train[b'fine_labels'], train[b'coarse_labels']):
            if fineLabel not in fineLabelNameDict.keys():
                fineLabelNameDict[fineLabel]=meta[b'fine_label_names'][fineLabel].decode('utf-8')
            if fineLabel not in fineLableToCoraseLabelDict.keys():
                fineLableToCoraseLabelDict[fineLabel]=str(coarseLabel)+"-"+meta[b'coarse_label_names'][coarseLabel].decode('utf-8')
    # 解压后meta的路径
    metaPath = '/nfs4/wjx/transferbility/experiment/data/cifar-100-python/meta'
    # 解压后train的路径
    trainPath = '/nfs4/wjx/transferbility/experiment/data/cifar-100-python/train'
    
    meta = unpickle(metaPath)
    train = unpickle(trainPath)
    Dealdata(meta, train)
    
    # print(len(fineLabelNameDict))
    # print(len(fineLableToCoraseLabelDict))
    # print(fineLabelNameDict)
    # print(fineLableToCoraseLabelDict)
    return fineLabelNameDict
 
 

def cifar100_to_images(root, label_dict):

    character_train = [[] for i in range(100)]
    character_test = [[] for i in range(100)]

    train_set = tv.datasets.CIFAR100(root, train=True, download=False)
    test_set = tv.datasets.CIFAR100(root, train=False, download=False)

    trainset = []
    testset = []
    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        trainset.append(list((np.array(X), np.array(Y))))
    for i, (X, Y) in enumerate(test_set):  # 将test_set的数据和label读入列表
        testset.append(list((np.array(X), np.array(Y))))

    for X, Y in trainset:
        character_train[Y].append(X)  # 32*32*3

    for X, Y in testset:
        character_test[Y].append(X)  # 32*32*3

    os.mkdir(os.path.join(root, 'train'))
    os.mkdir(os.path.join(root, 'test'))

    for i, per_class in enumerate(character_train):
        character_path = os.path.join(root, 'train', str(i)+ '-' + label_dict[i])
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img_path = character_path + '/' + str(j) + ".jpg"
            img = Image.fromarray(img)
            img.save(img_path)

    for i, per_class in enumerate(character_test):
        character_path = os.path.join(root, 'test', str(i)+ '-' + label_dict[i])
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img_path = character_path + '/' + str(j) + ".jpg"
            img = Image.fromarray(img)
            img.save(img_path)


if __name__ == '__main__':
    index_to_label_name_dict = index_to_label()
    cifar100_to_images(root='/nfs4/wjx/transferbility/experiment/data/', label_dict = index_to_label_name_dict)