from torch import nn
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d


class FcBlock(nn.Module):
    """
    Dense layer block with batchnormalization and relu activation
    """
    def __init__(self, in_features, out_features):
        super(FcBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
