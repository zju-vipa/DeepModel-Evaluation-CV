# This script eval model robustness on different data distributions.

import torchvision.datasets as datasets
from model_fetcher import model_fetcher
import argparse
import torch

import sys
sys.path.append("../Dataset/Classification")

from imagenetc import ImageNetC



def main(args):
    '''
        Main process
    '''
    device = 'cpu'
    model_name = args.name
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = "cuda:{}".format(args.gpu)
    else:
        print('GPU is not available.')




    mf = model_fetcher(device=device)
    model, transform, platform = mf.get(model_name)
    print(model, transform, platform)


    test_dataset = datasets.ImageNet(root=args.path, split='val', transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
    )





if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Enter off-the-shelf model name')
    parser.add_argument('-g', '--gpu', type=int, metavar='', help='Index of the GPU to be used')
    parser.add_argument('-p', '--path', type=str, metavar='', required=True, help='Path to the datasets')
    args = parser.parse_args()
    main(args)