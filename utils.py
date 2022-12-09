from __future__ import division
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
from path import Path
from datasets.vision_state_dataset import VisionStateDataset
from datasets.state_dataset import StateDataset
from datasets.vision_dataset import VisionDataset
from typing import Tuple
from models.force_estimator_2d import ForceEstimatorV, ForceEstimatorS, ForceEstimatorVS, RecurrentCNN
from models.force_estimator_transformers import KPDetector

Datasets = Tuple[Dataset]

def save_checkpoint(save_path, model_state, is_best, filename='checkpoint.pth.tar'):
    torch.save(model_state, save_path/'{}'.format(filename))

    if is_best:
        shutil.copyfile(save_path/'{}'.format(filename),
                        save_path/'model_best.pth.tar')

def assert_dataset_type(args, train_transforms, val_transforms):
    relations_train_dataset = {'2d_v': create_vision_datasets(args, train_transforms, val_transforms),
                                '2d_vs': create_vision_state_datasets(args, train_transforms, val_transforms), 
                                '2d_s': create_state_datasets(args),
                                '2d_rnn': create_vision_state_datasets(args, train_transforms, val_transforms), 
                                '3d_cnn': create_vision_state_datasets(args, train_transforms, val_transforms)}
    
    train_dataset, val_dataset = relations_train_dataset[args.type]

    return train_dataset, val_dataset


def create_vision_datasets(args, train_transforms: object, val_transforms: object) -> Datasets:

    train_dataset = VisionDataset(
        args.data,
        is_train=True,
        transform=train_transforms,
        seed=args.seed,
    )

    val_dataset = VisionDataset(
        args.data,
        is_train=False,
        transform=val_transforms,
        seed=args.seed,
        occlude_params=args.occ
    )

    print("Created ------ VISION DATASET ---------")

    return train_dataset, val_dataset

def create_state_datasets(args) -> Datasets:
    train_dataset = StateDataset(
        args.data,
        is_train=True,
        seed=args.seed
    )

    val_dataset = StateDataset(
        args.data,
        is_train=False,
        seed=args.seed,
        occlude_params=args.occ
    )

    print("Created ------ STATEDATASET ---------")

    return train_dataset, val_dataset

def create_vision_state_datasets(args, train_transforms: object, val_transforms: object) -> Datasets:
    train_dataset = VisionStateDataset(
        args.data,
        is_train=True,
        transform=train_transforms,
        seed=args.seed,
    )

    val_dataset = VisionStateDataset(
        args.data,
        is_train=False,
        transform=val_transforms,
        seed=args.seed,
        occlude_params=args.occ
    )

    print("Created ------ VISION & STATE DATASET ---------")

    return train_dataset, val_dataset

def assert_model_type(args):

    relations_model = {'2d_v': force_vision(),
                        '2d_s': force_state(),
                        '2d_vs': force_state_vision(),
                        '2d_rnn': force_rnn()}
    
    model, model_name = relations_model[args.type]

    return model, model_name

def force_vision():

    model = ForceEstimatorV(final_layer=500)
    model_name = "Vision Force Estimator"
    return model, model_name

def force_state():

    model = ForceEstimatorS(rs_size="to be decided")
    model_name = "State Force Estimator"
    return model, model_name

def force_state_vision():

    model = ForceEstimatorVS(final_layer=500, rs_size="to be determined")
    model_name = "Force & State Force Estimator"
    return model, model_name

def force_rnn():

    model = RecurrentCNN(embed_dim=512, hidden_size=500, num_layers=2, num_classes=3)
    model_name = "Recurrent Force Estimator"
    return model, model_name