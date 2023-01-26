from __future__ import division
import shutil
import torch
from torch.utils.data import Dataset
from path import Path

def save_checkpoint(save_path: Path, model_state, is_best:bool, filename='checkpoint.pth.tar'):
    torch.save(model_state, save_path/'{}'.format(filename))

    if is_best:
        shutil.copyfile(save_path/'{}'.format(filename),
                        save_path/'model_best.pth.tar')

