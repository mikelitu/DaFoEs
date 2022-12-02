from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
from datasets.vision_state_dataset import VisionStateDataset
from datasets.state_dataset import StateDataset



def save_checkpoint(save_path, model_state, is_best, filename='checkpoint.pth.tar'):
    torch.save(model_state, save_path/'{}'.format(filename))

    if is_best:
        shutil.copyfile(save_path/'{}'.format(filename),
                        save_path/'model_best.pth.tar')

def assert_type(type, args, transforms):
    relations_train_dataset = {'2d_v': ,'2d_vs': , '2d_s': '2d_rnn': , }