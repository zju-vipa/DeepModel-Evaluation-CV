import argparse
import os
import sys 
import pathlib
from pathlib import Path
import logging
import torch
from torch import nn
import torch.optim as optim

import numpy as np
import ot
import geomloss
import global_settings as settings
from utils import get_network, get_pretrain_network, get_training_transform, get_test_transform, WarmUpLR
from dataset import tinyfood101, tinycaltech, tinydomainnet, tinydtd, imagenet50

# parser = argparse.ArgumentParser()
# parser.add_argument('--feature_path1', type=str)
# parser.add_argument('--feature_path2', type=str)
np.set_printoptions(threshold = np.inf) 
if __name__ == '__main__':
    feature_paths1 = ['/nfs4-p1/wjx/transferbility/experiment/features/imagenet50_1000_50', '/nfs4-p1/wjx/transferbility/experiment/features/tinydomainnet_real_40_300']
    feature_paths2 = [
        '/nfs4-p1/wjx/transferbility/experiment/features/tinydtd_45_50',
        '/nfs4-p1/wjx/transferbility/experiment/features/tinycaltech256_45_50',
        '/nfs4-p1/wjx/transferbility/experiment/features/tinyfood101_45_50',
        '/nfs4-p1/wjx/transferbility/experiment/features/tinydomainnet_real_45_50',
        '/nfs4-p1/wjx/transferbility/experiment/features/tinydomainnet_sketch_40_10', 
        '/nfs4-p1/wjx/transferbility/experiment/features/tinydomainnet_clipart_40_10', 
        '/nfs4-p1/wjx/transferbility/experiment/features/tinydomainnet_infograph_40_10',
        '/nfs4-p1/wjx/transferbility/experiment/features/tinydomainnet_quickdraw_40_10'
    ]
    save_path = '/nfs4-p1/wjx/transferbility/experiment/log/evaluate/EMD_result.log'
    logging.basicConfig(filename=save_path, level=logging.INFO)
    for i, feature_path1 in enumerate(feature_paths1):
        if i == 0:
            index = range(0, 4)
        else:
            index = range(4, 8)
        
        for i in index:
            feature_path2 = feature_paths2[i]
        
            dataset1 = feature_path1.split('/')[-1]
            dataset2 = feature_path2.split('/')[-1]
            feature1 = torch.load(feature_path1)
            feature2 = torch.load(feature_path2)

            cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
            C = cost_function(feature1, feature2).cpu().numpy()
            # C = np.sqrt(C)

            P = ot.emd(ot.unif(feature1.shape[0]), ot.unif(feature2.shape[0]), C, numItermax=1000000)
            # P = ot.sinkhorn(ot.unif(feature1.shape[0]), ot.unif(feature2.shape[0]), C, reg= .5, numItermax=1000000, method='sinkhorn')
            EMD = np.sum(P*np.array(C)) / np.sum(P)

            result = 10*np.exp(-0.1*EMD)
            logging.info(f'EMD: {dataset1}->{dataset2}: ' + '{:.16f}'.format(result))
    
    
    
    
    
    
