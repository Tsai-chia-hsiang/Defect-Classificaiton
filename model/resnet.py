import torch
import torch.nn as nn
from torchvision import models

def resnet_18(grayscale=True, ncls:int=4):
    model = models.resnet18(weights='DEFAULT')

    # Modify the first convolutional layer to accept a single channel (grayscale) input
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=ncls, bias=True)
    return model

def resnet_34(grayscale=True, ncls:int=4):
    model = models.resnet34(weights='DEFAULT')

    # Modify the first convolutional layer to accept a single channel (grayscale) input
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=ncls, bias=True)
    return model

def resnet_50(grayscale=True, ncls:int=4):
    model = models.resnet50(weights='DEFAULT')

    # Modify the first convolutional layer to accept a single channel (grayscale) input
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=ncls, bias=True)
    return model

if __name__ == "__main__":
    rn18 = resnet_18(grayscale=True, ncls=4)
    print(rn18)