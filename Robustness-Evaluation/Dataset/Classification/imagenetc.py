import json
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm


class ImageNetC(data.Dataset):
    def __init__(self, root, transform=None):
        self.root=root
        self.transform=transform


        index_to_class = json.load(open("imagenet_class_index.json"))
        class_to_index = {}
        for index_ in index_to_class:
            class_to_index[index_to_class[index_][0]] = int(index_)

        self.class_to_index = class_to_index




        self.img_path_list = []
        self.img_offset_list = []
        self.img_key_dict = {}
        self.corruption_types = ['blur', 'digital', 'extra', 'noise', 'weather']
        self.corruption_levels= ['1', '2', '3', '4', '5']

        for corruption_type in tqdm(self.corruption_types):
            sub_path = os.path.join(root, corruption_type)
            sub_corruption_types = os.listdir(sub_path)
            for sub_corruption_type in sub_corruption_types:
                for corruption_level in self.corruption_levels: 
                    sub_sub_path = os.path.join(sub_path, sub_corruption_type, corruption_level)
                    classes = os.listdir(sub_sub_path)
                    for class_ in classes:
                        sub_sub_sub_path = os.path.join(sub_sub_path, class_)
                        tmp_img_list = os.listdir(sub_sub_sub_path)
                        self.img_key_dict[sub_sub_sub_path] = {
                            "class_index" : self.class_to_index[class_],
                            "image_list" : tmp_img_list
                        }
                        len_tmp_img_list = len(tmp_img_list)
                        self.img_path_list.extend([sub_sub_sub_path] * len_tmp_img_list)
                        self.img_offset_list.extend(range(len_tmp_img_list))


    def __getitem__(self, index):

        sub_sub_sub_path = self.img_path_list[index]
        offset =  self.img_offset_list[index]

        tmp_dict = self.img_key_dict[sub_sub_sub_path]

        target = tmp_dict["class_index"]

        path = os.path.join(sub_sub_sub_path, tmp_dict["image_list"][offset])

        img=Image.open(path).convert("RGB")

        if self.transform is not None:
            img=self.transform(img)

        return img, target

    def __len__(self):
        return len(self.img_offset_list)