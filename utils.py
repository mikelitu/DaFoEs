from __future__ import division
import shutil
import torch
from torch.utils.data import Dataset
from path import Path
import numpy as np
import torch.nn as nn
import datetime

def save_checkpoint(save_path: Path, model_state, is_best:bool, filename='checkpoint.pth.tar'):
    torch.save(model_state, save_path/'{}'.format(filename))

    if is_best:
        shutil.copyfile(save_path/'{}'.format(filename),
                        save_path/'model_best.pth.tar')

def none_or_str(value):
    if value=="None":
        return None
    return value

def create_saving_dir(root: Path, 
                      experiment_name: str, 
                      architecture: str,
                      dataset: str, 
                      recurrency: bool, 
                      att_type: str = None, 
                      occ_param: str = None):
    
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")

    if recurrency:
        architecture = "r" + architecture
    
    if att_type is not None:
        architecture = architecture + att_type.lower()
    
    if architecture == "fc":
        if occ_param is not None:
            save_path = root/"{}".format(dataset)/architecture/occ_param/experiment_name/timestamp
        else:
            save_path = root/"{}".format(dataset)/architecture/experiment_name/timestamp
    else:
        if occ_param is not None:
            save_path = root/"{}".format(dataset)/architecture/occ_param/experiment_name/timestamp
        else:
            save_path = root/"{}".format(dataset)/architecture/experiment_name/timestamp
    
    return save_path
