import torch
import torch.nn as nn
from models.utils import ResBlock2d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import torch.nn.functional as F
from models.utils import FcBlock

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


class ForceEstimatorVS(nn.Module):
    def __init__(self, rs_size):
        super(ForceEstimatorVS, self).__init__()

        self.encoder = ResNet18(in_channels=3)
        self.output = FcBlock(512*8*8, 30)

        self.linear1 = FcBlock(30 + rs_size, 84)
        self.linear2 = FcBlock(84, 180)
        self.linear3 = FcBlock(180, 50)
        self.final = nn.Linear(50, 3)
    
    def forward(self, x, robot_state=None):

        out = self.encoder(x)
        out_flatten = out.view(-1)
        out_flatten = self.output(out_flatten)
        out = torch.cat([out_flatten, robot_state], dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.final(out)
        return out


class ForceEstimatorV(nn.Module):
    def __init__(self):
        super(ForceEstimatorV, self).__init__()

        self.encoder = ResNet18(in_channels=3)

        self.linear = FcBlock(512*8*8, 30)
        self.final = nn.Linear(30, 3)

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(-1)
        out = self.linear(out)
        out = self.final(out)
        return out


class ForceEstimatorS(nn.Module):
    def __init__(self, rs_size):
        super(ForceEstimatorS, self).__init__()

        self.linear1 = FcBlock(rs_size, 500)
        self.linear2 = FcBlock(500, 1000)
        self.linear3 = FcBlock(1000, 1000)
        self.linear4 = FcBlock(1000, 1000)
        self.linear5 = FcBlock(1000, 500)
        self.linear6 = FcBlock(500, 50)
        self.final = nn.Linear(50, 3)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
        out = self.linear6(out)
        return out
