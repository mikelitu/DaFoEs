import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from models.bam import BAM


class ResNetMultiImageInput(models.ResNet):

    def __init__(self, block, layers, num_classes=1000, num_input_images=1,
                 input_channel=3, att_type=None):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if att_type=='BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 =  None # BAM(128*block.expansion)
            self.bam3 = None # BAM(256*block.expansion)
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

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1,
                            include_depth=True, att_type=None):
    """Constructs a ResNet model.

    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Defaults to False.
        num_input_images (int, optional): Number of frames stacked as input. Defaults to 1.
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, 
                                  input_channel=4 if include_depth else 3, att_type=att_type)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, include_depth=True, att_type=None):
        super(ResnetEncoder, self).__init__()

        self.att_type = att_type
        self.gradients = None


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
        
        if include_depth or att_type is not None:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, include_depth, att_type)
        else:
            self.encoder = resnets[num_layers](pretrained)
        
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        
    def forward(self, input_image: torch.Tensor):
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