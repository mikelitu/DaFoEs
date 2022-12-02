import torch
import torch.nn as nn

class EuclideanDist(nn.Module):
    """
    Calculate the Euclidean distance between two keypoint batches
    """
    def __init__(self):
        super(EuclideanDist, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (target - pred).pow(2).sum(2).sqrt().mean()


class RMSE(nn.Module):
    """
    Adaptation of the torch implementation of the Mean Squared Error to Root Mean Squared Error
    """
    def __init__(self):
        super(RMSE, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(pred, target))
        return loss