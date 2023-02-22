import torch
import torch.nn as nn
from models.utils import FcBlock
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from models.bam import BAM


class ResNetMultiImageInput(models.ResNet):

    def __init__(self, block, layers, num_classes=1000, num_input_images=1, att_type=None):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if att_type=='BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, att_type=None):
    """Constructs a ResNet model.

    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Defaults to False.
        num_input_images (int, optional): Number of frames stacked as input. Defaults to 1.
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, att_type=att_type)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, att_type=None):
        super(ResnetEncoder, self).__init__()

        self.att_type = att_type

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        
        if att_type is not None:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, att_type)
        else:
            self.encoder = resnets[num_layers](pretrained)
        
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        
    def forward(self, input_image):
        x = input_image

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)

        if self.att_type is not None:
            if not self.encoder.bam1 is None:
                x = self.encoder.bam1(x)

        x = self.encoder.layer2(x)
        if self.att_type is not None:
            if not self.encoder.bam2 is None:
                x = self.encoder.bam2(x)

        x = self.encoder.layer3(x)
        if self.att_type is not None:
            if not self.encoder.bam3 is None:
                x = self.encoder.bam3(x)

        x = self.encoder.layer4(x)

        return x


class ForceEstimatorVS(nn.Module):

    """
    Vision + State network architecture from the following paper: "Towards Force Estimation in Robot-Assisted Surgery using Deep Learning
    with Vision and Robot State" by Zonghe Chua et al. (https://doi.org/10.48550/arXiv.2011.02112)
    """
    def __init__(self, rs_size: int, num_layers: int = 18, pretrained: bool = True, att_type: str = None):
        super(ForceEstimatorVS, self).__init__()

        self.encoder = ResnetEncoder(num_layers, pretrained, att_type=att_type)

        self.linear1 = FcBlock(2048 * 8 * 8, 1000)
        self.linear2 = FcBlock(1000 + rs_size, 84)
        self.linear3 = FcBlock(84, 180)
        self.linear4 = FcBlock(180, 50)
        self.final = nn.Linear(50, 3)
    
    def forward(self, x, robot_state=None):

        out = self.encoder(x)
        out_flatten = out.view(out.shape[0], -1)
        out = self.linear1(out_flatten)
        out = torch.cat([out, robot_state], dim=1)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.final(out)
        return out


class ForceEstimatorV(nn.Module):
    """
    Vision only network from the paper: "Towards Force Estimation in Robot-Assisted Surgery using Deep Learning
    with Vision and Robot State" by Zonghe Chua et al. (doi: https://doi.org/10.48550/arXiv.2011.02112)
    """
    def __init__(self, num_layers: int = 18, pretrained: bool = True, att_type: str = None):
        super(ForceEstimatorV, self).__init__()

        self.encoder = ResnetEncoder(num_layers, pretrained, att_type=att_type)

        self.linear1 = FcBlock(2048 * 8 * 8, 500)
        self.final = nn.Linear(500, 3)

    def forward(self, x):
        out = self.encoder(x)
        out_flatten = out.view(out.shape[0], -1)
        out = self.linear1(out_flatten)
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

        self.encoder = ResnetEncoder(in_channels=3, final_features=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor, robot_state: torch.Tensor = None) -> torch.Tensor:
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

