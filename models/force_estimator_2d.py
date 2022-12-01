import torch
import torch.nn as nn
from models.utils import ResBlock2d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import torch.nn.functional as F
from models.utils import FcBlock

class ResNet18(nn.Module):

    """
    Original architecture of the ResNet18 network. "Deep Residual Learning for Image Recognition"
    by Kaimin He et al. (doi: https://doi.org/10.48550/arXiv.1512.03385)
    """

    def __init__(self, in_channels: int = 3, final_features: int = 512):
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

        self.final = nn.Linear(in_features=512*7*7, out_features=final_features)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        
        out = self.maxpool(self.bn1(self.conv1(image)))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out_flatten = out.view(out.shape[0], -1)
        out = self.final(out_flatten)
        
        return out


class ForceEstimatorVS(nn.Module):

    """
    Vision + State network architecture from the following paper: "Towards Force Estimation in Robot-Assisted Surgery using Deep Learning
    with Vision and Robot State" by Zonghe Chua et al. (https://doi.org/10.48550/arXiv.2011.02112)
    """
    def __init__(self, rs_size: int):
        super(ForceEstimatorVS, self).__init__()

        self.encoder = ResNet18(in_channels=3)

        self.linear1 = FcBlock(30 + rs_size, 84)
        self.linear2 = FcBlock(84, 180)
        self.linear3 = FcBlock(180, 50)
        self.final = nn.Linear(50, 3)
    
    def forward(self, x, robot_state=None):

        out_flatten = self.encoder(x)
        out = torch.cat([out_flatten, robot_state], dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.final(out)
        return out


class ForceEstimatorV(nn.Module):
    """
    Vision only network from the paper: "Towards Force Estimation in Robot-Assisted Surgery using Deep Learning
    with Vision and Robot State" by Zonghe Chua et al. (doi: https://doi.org/10.48550/arXiv.2011.02112)
    """
    def __init__(self):
        super(ForceEstimatorV, self).__init__()

        self.encoder = ResNet18(in_channels=3)
        self.final = nn.Linear(30, 3)

    def forward(self, x):
        out = self.encoder(x)
        out = self.final(out)
        return out


class ForceEstimatorS(nn.Module):
    """
    State only network from the paper: "Towards Force Estimation in Robot-Assisted Surgery using Deep Learning
    with Vision and Robot State" by Zhongue Chua et al. (doi: https://doi.org/10.48550/arXiv.2011.02112)
    """
    def __init__(self, rs_size: int):
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
        out = self.final(out)
        return out


class RecurrentCNN(nn.Module):

    """
    Adaptation of the recurrent neural network with 2 LSTM blocks presented in multiple papers.
    - "Camera Configuration Models for Machine Vision Based Force Estimation in Robot-Assisted Soft Body Manipulation" by Wenjun Liu et al. (doi: https://doi.org/10.1109/ISMR48347.2022.9807587)
    - "A recurrent convolutional neural network approach for sensorless force estimation in robotic surgery" by Arturo Marban et al. (doi: https://doi.org/10.1016/j.bspc.2019.01.011)
    """
    def __init__(self, embed_dim: int, hidden_size: int, num_layers: int, num_classes: int):
        super(RecurrentCNN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.encoder = ResNet18(in_channels=3, final_features=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = x.reshape(batch_size, -1, self.embed_dim)
        #lstm part
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        x, _ = self.lstm(x, (h_0, c_0))
        x = x[:, -1, :]
        x = self.fc(x)
        return x
