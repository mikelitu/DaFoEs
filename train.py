import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(n_epochs, model: nn.Module, optimizer, dataset, args, ckp = None):

    if ckp is not None:
        load_checkpoint(ckp, model, optimizer)