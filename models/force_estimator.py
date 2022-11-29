import torch
import torch.nn as nn
from models.utils import ResBlock2d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import torch.nn.functional as F

class ResNet18(nn.Module):

    def __init__(self, in_channels):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResBlock2d(in_features=64, kernel_size=3, padding=1, stride=False),
            ResBlock2d(in_features=64, kernel_size=3, padding=1, stride=False)
        )

        self.layer2 = nn.Sequential(
            ResBlock2d(in_features=64, kernel_size=3, padding=1, stride=True, out_channels=128),
            ResBlock2d(in_features=128, kernel_size=3, padding=1, stride=False)
        )

        self.layer3 = nn.Sequential(
            ResBlock2d(in_features=128, kernel_size=3, padding=1, stride=True, out_channels=256),
            ResBlock2d(in_features=256, kernel_size=3, padding=1, stride=False)
        )

        self.layer4 = nn.Sequential(
            ResBlock2d(in_features=256, kernel_size=3, padding=1, stride=True, out_channels=512),
            ResBlock2d(in_features=512, kernel_size=3, padding=1, stride=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        
    def forward(self, image):
        
        out = self.maxpool(self.bn1(self.conv1(image)))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out
