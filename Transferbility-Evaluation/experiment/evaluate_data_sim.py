import argparse
import os
import sys 
import pathlib
from pathlib import Path
import logging

import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
def get_images(dataset):
    root_path = './data'
    if dataset == 'caltech256':
        root_path = os.path.join(root_path, 'caltech256')
        catagory_list = os.listdir(root_path)



if __name__ == '__main__':
