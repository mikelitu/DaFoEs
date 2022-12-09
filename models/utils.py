from torch import nn
import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d

import torch.nn.utils.spectral_norm as spectral_norm
import re


def make_coordinate_grid_2d(features: torch.Tensor) -> torch.Tensor: 
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    _, _, h, w = features.shape
    x = torch.arange(w).type_as(features)
    y = torch.arange(h).type_as(features)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def make_coordinate_grid(features: torch.Tensor) -> torch.Tensor:
    _, _, d, h, w = features.shape
    x = torch.arange(w).type_as(features)
    y = torch.arange(h).type_as(features)
    z = torch.arange(d).type_as(features)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
   
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed


def world_2_camera(position, intrinsics):

    h_position_i = intrinsics @ position
    
    h_position_i[:, 0] = h_position_i[:, 0] / h_position_i[:, 2]
    h_position_i[:, 1] = h_position_i[:, 1] / h_position_i[:, 2]

    position_i = h_position_i[:, :2]

    return position_i


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features: int, kernel_size: int, padding: int, stride: bool = False, out_channels: int = None):
        super(ResBlock2d, self).__init__()
        if out_channels is None:
            out_channels = in_features
        
        if stride:    

            self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_channels, kernel_size=kernel_size,
                                padding=padding, stride=2)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_channels, kernel_size=1, stride=2),
                BatchNorm2d(out_channels, affine=True)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(out_channels, affine=True)
        self.norm2 = BatchNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out += shortcut
        out = F.relu(out)
        return out


class FcBlock(nn.Module):
    """
    Dense layer block with batchnormalization and relu activation
    """
    def __init__(self, in_features, out_features):
        super(FcBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = BatchNorm1d(out_features)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out)
        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
    

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

