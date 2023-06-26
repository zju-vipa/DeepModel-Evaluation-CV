import json
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



class ImageNetC(data.Dataset):
    def __init__(self, root, class_to_index_path, transform=None):
        self.root=root
        self.transform=transform


        index_to_class = json.load("imagenet_class_index.json")
        class_to_index = {}
        for index_ in index_to_class:
            class_to_index[index_to_class[index_][0]] = int(index_)

        self.class_to_index = class_to_index




        self.imgs=[]
        self.corruption_types = ['snow', 'brightness', 'fog', 'frost', 'blur']
        self.corruption_levels= ['1', '2', '3', '4', '5']

        for corruption_type in self.corruption_types:
            for corruption_level in self.corruption_levels:
                sub_path = os.path.join(root, corruption_type, corruption_level)
                classes = os.listdir(sub_path)
                for class_ in classes:
                    subsubpath = os.path.join(sub_path, class_)
                    for img in os.listdir(subsubpath):
                        self.imgs.append((os.path.join(subsubpath, img), self.class_to_index[class_]))

    def __getitem__(self, index):
        path, target=self.imgs[index]
        img=Image.open(path).convert("RGB")

        if self.transform is not None:
            img=self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)





# import torch
# import torch.nn.functional as F

# from torchvision import models

# # 加载预训练的ResNet50模型，并将其调整为我们的数据集
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# # Set the model to evaluation mode
# model.eval()

# # Specify the device to run the model on
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Create data loader for the dataset
# batch_size = 512
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# # Initialize variables
# num_correct = 0
# total_images = 0

# from tqdm import tqdm
# with torch.no_grad():
#     for batch in tqdm(data_loader):
#         # Move the input data to the specified device
#         inputs, targets = batch
#         inputs = inputs.to(device)
#         targets = targets.to(device)

#         # Forward pass through the model
#         outputs = model(inputs)

#         # Convert the output probabilities to class predictions
#         _, preds = torch.max(F.softmax(outputs, dim=1), 1)

#         # Update the number of correctly predicted images
#         num_correct += (preds == targets).sum().item()

#         # Update the total number of images
#         total_images += targets.size(0)

# # Compute the accuracy
# accuracy = num_correct / total_images

# # Print the accuracy
# print(f"Accuracy: {accuracy:.2%}")