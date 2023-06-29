# Initialize models which are pre-build or uploaded (the latter is to be implemented)

import torch
import torchvision.models as models
import torchvision.transforms as transforms
# import transformers


class model_fetcher:
    '''
        This class return models prebuild and transforms
    '''

    def __init__(self, device=torch.device('cpu')):
        self.device = device

        self.prebuild_models = {
            "alexnet": {
                "platform": "torchvision",
                "model": models.alexnet,
                "weights": models.AlexNet_Weights.IMAGENET1K_V1
            },
            "resnet50": {
                "platform": "torchvision",
                "model": models.resnet50,
                "weights": models.ResNet50_Weights.IMAGENET1K_V1
            },
            "densenet121": {
                "platform": "torchvision",
                "model": models.densenet121,
                "weights": models.DenseNet121_Weights.IMAGENET1K_V1
            },
            "efficientnet_b0": {
                "platform": "torchvision",
                "model": models.efficientnet_b0,
                "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1
            },
            "googlenet": {
                "platform": "torchvision",
                "model": models.googlenet,
                "weights": models.GoogLeNet_Weights.IMAGENET1K_V1
            },
            "inception_v3": {
                "platform": "torchvision",
                "model": models.inception_v3,
                "weights": models.Inception_V3_Weights.IMAGENET1K_V1
            },
            "mobilenet_v2": {
                "platform": "torchvision",
                "model": models.mobilenet_v2,
                "weights": models.MobileNet_V2_Weights.IMAGENET1K_V1
            },
            "mnasnet0_5": {
                "platform": "torchvision",
                "model": models.mnasnet0_5,
                "weights": models.MNASNet0_5_Weights.IMAGENET1K_V1
            },
            "regnet_x_16gf": {
                "platform": "torchvision",
                "model": models.regnet_x_16gf,
                "weights": models.RegNet_X_16GF_Weights.IMAGENET1K_V1
            },
            "shufflenet_v2_x1_0": {
                "platform": "torchvision",
                "model": models.shufflenet_v2_x1_0,
                "weights": models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
            },
            "maxvit_t": {
                "platform": "torchvision",
                "model": models.maxvit_t,
                "weights": models.MaxVit_T_Weights.IMAGENET1K_V1
            },
            "regnet_x_800mf": {
                "platform": "torchvision",
                "model": models.regnet_x_800mf,
                "weights": models.RegNet_X_800MF_Weights.IMAGENET1K_V1
            },
            "resnext50_32x4d": {
                "platform": "torchvision",
                "model": models.resnext50_32x4d,
                "weights": models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
            },
            "shufflenet_v2_x0_5": {
                "platform": "torchvision",
                "model": models.shufflenet_v2_x0_5,
                "weights": models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
            },
            "squeezenet1_0": {
                "platform": "torchvision",
                "model": models.squeezenet1_0,
                "weights": models.SqueezeNet1_0_Weights.IMAGENET1K_V1
            },
            "vgg11": {
                "platform": "torchvision",
                "model": models.vgg11,
                "weights": models.VGG11_Weights.IMAGENET1K_V1
            },
            "vit_b_16": {
                "platform": "torchvision",
                "model": models.vit_b_16,
                "weights": models.ViT_B_16_Weights.IMAGENET1K_V1
            },
            # "deit": {
            #     "platform": "huggingface",
            #     "model": transformers.DeiTForImageClassification,
            #     "config": transformers.DeiTConfig(),
            #     "transform": transforms.Compose([
            #         transforms.ToTensor(),
            #         transforms.Resize(
            #             256, transforms.InterpolationMode.BICUBIC),
            #         transforms.CenterCrop(224),
            #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #                              0.485, 0.456, 0.406]),
            #     ])
            # },
            # "swin": {
            #     "platform": "huggingface",
            #     "model": transformers.SwinForImageClassification,
            #     "config": transformers.SwinConfig(),
            #     "transform": transforms.Compose([
            #         transforms.ToTensor(),
            #         transforms.Resize(
            #             256, transforms.InterpolationMode.BICUBIC),
            #         transforms.CenterCrop(224),
            #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #                              0.485, 0.456, 0.406]),
            #     ])
            # }
        }

    def get(self, model_name):

        model = None
        transform = None
        platform = None

        if model_name in self.prebuild_models:

            model_info = self.prebuild_models[model_name]
            platform = model_info["platform"]
            if platform == "torchvision":
                model = model_info["model"](weights=model_info["weights"])
                transform = model_info["weights"].transforms()

            if platform == "huggingface":
                model = model_info["model"](model_info["config"])
                transform = model_info["transform"]

        assert model != None, "Model is not loaded correctly!"

        model.to(self.device)
        model.eval()

        return model, transform, platform
