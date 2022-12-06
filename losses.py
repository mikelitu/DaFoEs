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
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(pred, target))
        return loss

class GDL(nn.Module):
    """
    Gradient Difference Loss implementation in Pytorch
    """
    def __init__(self):
        super(GDL, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Equation for GDL:
        L_GDL = sum(|real_i - real_i-1| - |pred_i - pred_i-1|)
        """
        loss = 0
        for i in range(len(pred)):
            loss += torch.abs(torch.norm(target[i, 1::] - target[i, :-1:], dim=-1) - torch.norm(pred[i, 1::] - pred[i, :-1:], dim=-1)).mean()
        return loss
