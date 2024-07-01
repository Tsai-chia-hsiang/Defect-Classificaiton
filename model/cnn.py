import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import  models

class CNN(nn.Module):
    def __init__(self, grayscale:bool=True, ncls:int=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1 if grayscale else 3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, ncls)

        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x:torch.Tensor):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
    
            
        return x
    
def vgg_11(grayscale:bool=True, ncls:int=4):
    model = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
    if grayscale:
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Change input channels to 1
    model.classifier[6] = nn.Linear(4096, ncls)  # Change output to 3 classes
    return model