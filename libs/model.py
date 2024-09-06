import os
from logging import Logger
import os.path as osp
from typing import Literal, Optional
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.resnet import \
    ResNet50_Weights, ResNet101_Weights, \
    ResNeXt101_32X8D_Weights, ResNeXt50_32X4D_Weights, ResNeXt101_64X4D_Weights


def remove_module_prefix(state_dict:dict):
    """
    Remove the 'module.' prefix from the state dictionary keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove the 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

class ResNet_CLS_Contrastive_Model(nn.Module):
    
    def __init__(self, ncls:int=4, grayscale:bool=True, depth:Literal["r50", "r101","rx50_32", "rx101_32", "rx101_64"]="rx50_32"):
        
        super().__init__()
        
        self.name = depth
        self.fdim = 1024
        self.ncls = ncls
    
        self.backbone:models.ResNet = None
        match depth:
            case "r50":
                self.backbone = models.resnet50(weights = ResNet50_Weights.DEFAULT)
            case "r101":
                self.backbone = models.resnet101(weights = ResNet101_Weights.DEFAULT)
            case "rx50_32":
                self.backbone = models.resnext50_32x4d(weights = ResNeXt50_32X4D_Weights.DEFAULT)
            case "rx101_32":
                self.backbone = models.resnext101_32x8d(weights = ResNeXt101_32X8D_Weights.DEFAULT)
            case "rx101_64":
                self.backbone = models.resnext101_64x4d(weights = ResNeXt101_64X4D_Weights.DEFAULT)
            case _:
                raise KeyError("Not support")
        
        if grayscale:
            self.swap_conv1_to_gray()
        
        out_f = self.backbone.fc.in_features
        
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        self.cls_head = torch.nn.Linear(out_f, self.ncls)
        
        self.feature_net = torch.nn.Sequential(
            *[
                torch.nn.Linear(out_f, out_f), 
                torch.nn.ReLU(inplace=True), 
                torch.nn.BatchNorm1d(out_f),
                torch.nn.Linear(out_f, self.fdim)
            ]
        )
        self.proto_net = torch.nn.Sequential(
            *[
                torch.nn.Linear(out_f, out_f), 
                torch.nn.ReLU(inplace=True), 
                torch.nn.BatchNorm1d(out_f),
                torch.nn.Linear(out_f, self.fdim)
            ]
        ) 

    def swap_conv1_to_gray(self):
        self.backbone.conv1 = nn.Conv2d(
            1, self.backbone.conv1.out_channels, 
            kernel_size=self.backbone.conv1.kernel_size, 
            stride=self.backbone.conv1.stride, 
            padding=self.backbone.conv1.padding, 
            bias=self.backbone.conv1.bias
        )
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(self.backbone.conv1.weight.data.mean(dim=1, keepdim=True))

    def forward(self, x:torch.Tensor, features:bool=False) -> torch.Tensor|tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        f0:torch.Tensor = self.backbone(x).squeeze((-1, -2))
        logit = self.cls_head(f0)
        
        if features:
            f = F.normalize(self.feature_net(f0), p=2, dim=1)
            prototype = F.normalize(self.proto_net(self.cls_head.weight), p=2, dim=1)
            return logit, f, prototype
        
        return logit

    def __repr__(self):
        return self.name

    @classmethod
    def get_model(cls, ncls:int=4, grayscale:bool=True, depth:str="rx50_32", pretrained:Optional[str]=None, logger:Optional[Logger]=None)->"ResNet_CLS_Contrastive_Model":

        a_model = cls(ncls=ncls, grayscale=grayscale, depth=depth)
        msg = f"build {a_model} from torchvision defalut weights"
        if pretrained is not None:
            if osp.exists(pretrained):
                a_model.load_state_dict(remove_module_prefix(torch.load(pretrained, map_location="cpu")))
                msg = f"load pretrained {a_model} from {pretrained}"
            else:
                msg = f"{pretrained} not found, {msg}"

        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
        return a_model


if __name__ == "__main__":
    import os
    from PIL import Image
    from torchvision import transforms
    from dataset import G_normalizer
    ner = transforms.Compose(G_normalizer)
    def read_batch(batch:list[os.PathLike]) -> torch.Tensor:
        a_batch = [ner(Image.open(_).convert("L")) for _ in batch]
        return torch.stack(a_batch)
    
    test_imgs = read_batch([
        "../dataset/source/Type1/0.jpg",
        "../dataset/source/Type1/1.jpg"
    ])
    print(test_imgs.size())

    resnext101_model = ResNet_CLS_Contrastive_Model(depth="r101")
    logit, features, prototype = resnext101_model(x=test_imgs, features=True)
    print(logit.size(), features.size(), prototype.size())
