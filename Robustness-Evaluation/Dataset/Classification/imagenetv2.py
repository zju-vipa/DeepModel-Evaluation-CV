import json
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder

class ImageNetV2(data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)