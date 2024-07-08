from typing import Literal
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torchvision import models
import torch.nn.functional as F 

class _ResNet_with_Coo_Channel(nn.Module):
    
    def __init__(self, backbone:models.ResNet, ncls:int, deep:int):
        super(_ResNet_with_Coo_Channel, self).__init__()
        # Copy all layers except the fully connected layer
        self.deep = deep
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.f_dim = backbone.fc.in_features
        self.patch_agg = MultiheadAttention(embed_dim=self.f_dim, num_heads=1)
        self.fc:nn.Linear = nn.Linear(in_features=self.f_dim, out_features=ncls, bias=True)
 
    def single_forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1, -1)
        x = self.patch_agg()
        return torch.max(x, dim=0, keepdim=True).values
        
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
        return f"resnet{self.deep} with coo channels"


class _ResNet_with_Coo_Pos(nn.Module):
    
    def __init__(self, backbone:models.ResNet, ncls:int, deep:int):
        super(_ResNet_with_Coo_Pos, self).__init__()
        
        self.deep = deep
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.f_dim = backbone.fc.in_features
        self.patch_agg = MultiheadAttention(embed_dim=self.f_dim, num_heads=2, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, self.f_dim))

        self.pos_head = nn.Sequential(
            *[
                nn.Linear(4, backbone.fc.in_features),
                nn.ReLU()
            ]
        )
    
        self.fc:nn.Linear = nn.Linear(in_features=backbone.fc.in_features, out_features=ncls, bias=True)
 
    def single_forward(self, x:torch.Tensor, coor:torch.Tensor) -> torch.Tensor:
    
        x = self.features(x).flatten(1, -1) + self.pos_head(coor)
        cls_token = self.cls_token + self.pos_head(torch.zeros(1, 4).to(coor.device))
        x = torch.cat((cls_token, x), dim=0)
        x, _ = self.patch_agg(query=x, key=x, value=x, need_weights=False)

        return x[0, :].unsqueeze(0) 
    
    def forward(self, x:list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        
        xj:torch.Tensor = None
        if len(x) == 1:
            # single image
            xj = self.single_forward(x[0][0], x[0][1])
        else:
            # a batch
            xj = torch.vstack([self.single_forward(xi[0], xi[1]) for xi in x])
        return self.fc(xj)
    
    def __repr__(self):
        return f"resnet{self.deep} with coo pos"



class _ResNet_with_Coo_Pos_NoAgg(nn.Module):
    
    def __init__(self, backbone:models.ResNet, ncls:int, deep:int):
        super(_ResNet_with_Coo_Pos_NoAgg, self).__init__()
        # Copy all layers except the fully connected layer
        self.deep = deep
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.f_dim = backbone.fc.in_features
   
        self.pos_head = nn.Sequential(
            *[
                nn.Linear(4, self.f_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.f_dim)
            ]
        )
    
        self.fc:nn.Linear = nn.Linear(in_features=backbone.fc.in_features, out_features=ncls, bias=True)
    
    def forward(self, x:torch.Tensor, coo:torch.Tensor)->torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        coo = self.pos_head(coo)
        return self.fc(x+coo)
    
    def __repr__(self):
        return f"resnet{self.deep} with coo pos no agg"



_ResNet_map = {
    '18':models.resnet18(weights='DEFAULT'),
    '34':models.resnet34(weights='DEFAULT'),
    '50':models.resnet50(weights='DEFAULT')
}

def resnet(grayscale=True, ncls:int=4, coo:bool=False, encoder_depth:Literal['18','34','50'] = '50', pos_emd:bool=False, patch_agg:bool=False):
    
    pretrained = _ResNet_map[encoder_depth]
 
    in_channels = 1 if grayscale else 3
    if coo and not pos_emd:
        in_channels += 4
    
    pretrained.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    pretrained.fc = nn.Linear(in_features=pretrained.fc.in_features, out_features=ncls, bias=True)
    if coo:
        if pos_emd:
            if patch_agg:
                pretrained = _ResNet_with_Coo_Pos(ncls=ncls, backbone=pretrained, deep=int(encoder_depth))
            else:
                pretrained = _ResNet_with_Coo_Pos_NoAgg(ncls=ncls, backbone=pretrained, deep=int(encoder_depth))  
            
        else:
            if patch_agg:
                pretrained = _ResNet_with_Coo_Channel(ncls=ncls, backbone=pretrained, deep=int(encoder_depth))
    
    return pretrained