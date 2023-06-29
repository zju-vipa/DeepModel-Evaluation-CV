# This script eval model robustness on different data distributions.

import torchvision.datasets as datasets
from model_fetcher import model_fetcher
import argparse
import torch
from tqdm import tqdm


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

    if device == 'cpu':
        print('Use cpu to infer, which may cost too much time')


    mf = model_fetcher(device=device)
    model, transform, platform = mf.get(model_name)
    # print(model, transform, platform)

    test_dataset = ImageNetC(root=args.path, transform=transform)
    print("ImageNet-C prepared")

    # test_dataset = datasets.ImageNet(root=args.path, split='val', transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )


    with torch.no_grad():
    # 在测试集上进行推理

        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += labels.size(0)

            # is_correct_list = (predicted == labels).to_list()

            correct += (predicted == labels).sum().item()
            predicted = predicted.cpu().numpy().flatten()
            # print('准确率: {:.2f}%'.format(100 * correct / total))

            # for predicted_label in predicted:
            #     f.write("{}\n".format(predicted_label))

    #     # 输出准确率
        print('准确率: {:.2f}%'.format(100 * correct / total))





if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Enter off-the-shelf model name')
    parser.add_argument('-g', '--gpu', type=int, metavar='', help='Index of the GPU to be used')
    parser.add_argument('-b', '--batch_size', type=int, metavar='', default=1024, help='Batch size of test dataset')
    parser.add_argument('-p', '--path', type=str, metavar='', required=True, help='Path to the datasets')
    args = parser.parse_args()
    main(args)