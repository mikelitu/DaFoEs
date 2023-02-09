import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset
import imageio
import numpy as np
import random
from path import Path
import pandas as pd
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_as_float(path: Path) -> np.ndarray:
    return  imageio.imread(path).astype(np.float32)

def normalize_labels(labels: np.ndarray, eps=1e-10) -> np.ndarray:
    return (labels - labels.mean(axis=0)) / (labels.std(axis=0) + eps) 



class VisionStateDataset(Dataset):
    """A dataset to load data from dfferent folders that are arranged this way:
        root/scene_1/000.png
        root/scene_1/001.png
        ...
        root/scene_1/labels.csv
        root/scene_2/000.png
        .
        transform functions takes in a list images and a numpy array representing the intrinsics of the camera and the robot state
    """

    def __init__(self, root, is_train=True, transform=None, seed=0, train_type="random"):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)

        train_files = {"random": "train.txt", 
                    "geometry": "train_geometry.txt", 
                    "color": "train_color.txt", 
                    "structure": "train_structure.txt",
                    "stiffness": "train_stiffness.txt",
                    "position": "train_position.txt"}

        val_files = {"random": "val.txt", 
                    "geometry": "val_geometry.txt", 
                    "color": "val_color.txt", 
                    "structure": "val_structure.txt",
                    "stiffness": "val_stiffness.txt",
                    "position": "val_position.txt"}

        scene_list_path = self.root/train_files[train_type] if is_train else self.root/val_files[train_type]
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders()
        
        
    def crawl_folders(self):
        
        samples = []

        for scene in self.scenes:
            labels = np.array(pd.read_csv(scene/'labels.csv')).astype(np.float32)
            norm_labels = normalize_labels(labels)
            images = sorted(scene.files("*.png"))
            n_labels = len(norm_labels) // len(images)
            step = 7

            for i in range(len(images)):
                if i < 25: continue
                if i > len(images) - 25: break
                sample = {}
                sample['img'] = images[i]
                sample['label'] = norm_labels[n_labels*i: (n_labels*i) + step]
                sample['forces'] = 0.25 * labels[n_labels*i:(n_labels*i) + step, -6:]
                samples.append(sample)
        
        random.shuffle(samples)
        self.samples = samples
    
    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        img = load_as_float(sample['img'])
        label = sample['label']

        if self.transform is not None:
            img = self.transform([img])
            img = img[0]
        
        return {'img': img, 'robot_state': label[:, :-6], 'forces': sample['forces']}
    
    def __len__(self):
        return len(self.samples)
