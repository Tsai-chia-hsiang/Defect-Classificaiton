import torch
import torch.nn as nn
from torchvision import models

_split_line = "="*15

class _ResNet_with_coo(nn.Module):
    
    def __init__(self, backbone:nn.Module, fout:int, ncls:int, deep:int):
        super(_ResNet_with_coo, self).__init__()
        # Copy all layers except the fully connected layer
        self.deep = deep
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc:nn.Linear = nn.Linear(in_features=fout, out_features=ncls, bias=True)
        self.pos_head = nn.Sequential(
            *[
                nn.Linear(4, fout),
                nn.ReLU(),
                nn.BatchNorm1d(fout)
            ]
        )
    
    def forward(self, x:torch.Tensor, coo:torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        coo = self.pos_head(coo)
        return self.fc(x+coo)
        #self.fc(torch.concat((x, coo), dim=1))
    
    def __repr__(self):
        return f"resnet{self.deep} with coo"

def resnet_18(grayscale=True, ncls:int=4, coo:bool=False):
    model = models.resnet18(weights='DEFAULT')

    # Modify the first convolutional layer to accept a single channel (grayscale) input
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if not coo:
        model.fc = nn.Linear(in_features=512, out_features=ncls, bias=True)
        print(f"{_split_line} model : Resnet18 {_split_line}")
    else:
        model = _ResNet_with_coo(backbone=model, deep=18, fout=512, ncls=ncls)

        print(f"{_split_line} model : {model} {_split_line}")
    return model

def resnet_34(grayscale=True, ncls:int=4, coo:bool=False):
    model = models.resnet34(weights='DEFAULT')

    # Modify the first convolutional layer to accept a single channel (grayscale) input
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    if not coo:
        model.fc = nn.Linear(in_features=512, out_features=ncls, bias=True)
        print(f"{_split_line} model : Resnet34 {_split_line}")
    else:
        model = _ResNet_with_coo(backbone=model, deep=34, fout=512, ncls=ncls)
        print(f"{_split_line} model : {model} {_split_line}")
    return model

def resnet_50(grayscale=True, ncls:int=4, coo:bool=False):
    model = models.resnet50(weights='DEFAULT')

    # Modify the first convolutional layer to accept a single channel (grayscale) input
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if not coo:
        model.fc = nn.Linear(in_features=2048, out_features=ncls, bias=True)
        print(f"{_split_line} model : Resnet50 {_split_line}")
    else:
        model = _ResNet_with_coo(backbone=model, deep=50, fout=2048, ncls=ncls)
        print(f"{_split_line} model : {model} {_split_line}")
    
    return model


if __name__ == "__main__":
    rn18 = resnet_18(grayscale=True, ncls=4)
    print(rn18)