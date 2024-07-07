from typing import Literal
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F 

class _ResNet_with_coo(nn.Module):
    
    def __init__(self, backbone:nn.Module, ncls:int, deep:int):
        super(_ResNet_with_coo, self).__init__()
        # Copy all layers except the fully connected layer
        self.deep = deep
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc:nn.Linear = nn.Linear(in_features=backbone.fc.in_features, out_features=ncls, bias=True)
 
    def single_forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1, -1)
        x = x.mean(dim=0, keepdim=True)
        return x
        
    def forward(self, x:list[torch.Tensor]) -> torch.Tensor:
        
        xj:torch.Tensor = None
        if len(x) == 1:
            # single image
            xj = self.single_forward(x[0])
        else:
            # a batch
            xj = torch.vstack([self.single_forward(xi) for xi in x])
        return self.fc(xj)
    
    def __repr__(self):
        return f"resnet{self.deep} with coo"



_ResNet_map = {
    '18':models.resnet18(weights='DEFAULT'),
    '34':models.resnet34(weights='DEFAULT'),
    '50':models.resnet50(weights='DEFAULT')
}

def resnet(grayscale=True, ncls:int=4, coo:bool=False, encoder_depth:Literal['18','34','50'] = '50'):
    
    pretrained = _ResNet_map[encoder_depth]
 
    in_channels = 1 if grayscale else 3
    if coo:
        in_channels += 4
    
    pretrained.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    if not coo:
        # simple classification model 
        pretrained.fc = nn.Linear(in_features=pretrained.fc.in_features, out_features=ncls, bias=True)
        return pretrained
    pretrained = _ResNet_with_coo(ncls=ncls, backbone=pretrained, deep=int(encoder_depth))
    return pretrained