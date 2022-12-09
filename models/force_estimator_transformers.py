from torch import nn
import torch
import torch.nn.functional as F

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from vit_pytorch.learnable_memory_vit import ViT, Adapter

class ViTForceDetector(nn.Module):
    def __init__(self, 
                image_size: int = 256, 
                patch_size: int = 16, 
                num_classes: int = 1000,
                dim: int = 1024,
                depth: int = 6,
                heads: int = 8,
                mlp_dim: int = 2048,
                dropout: float = 0.1,
                emb_dropout: float = 0.1) -> None:

        super(ViTForceDetector, self).__init__()
        self.v = ViT(image_size = image_size, 
                patch_size = patch_size,
                num_classes = num_classes,
                dim = dim,
                depth = depth,
                heads = heads,
                mlp_dim = mlp_dim,
                dropout = dropout,
                emb_dropout = emb_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.v(out)
        return out
