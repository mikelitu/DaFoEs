import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset
import imageio
import numpy as np
import random
from path import Path

def load_as_float(path):
    return  imageio.v3.imread(path).astype(np.float32)


class VisionDataset(Dataset):
    """A dataset to load data from dfferent folders that are arranged this way:
        root/scene_1/000.png
        root/scene_1/001.png
        ...
        root/scene_1/labels.txt
        root/scene_2/000.png
        .
        transform functions takes in a list images and a numpy array representing the intrinsics of the camera and the robot state
    """

    def __init__(self, root, is_train=True, transform=None, seed=0, occlude_params=[]):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/"train.txt" if is_train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders()
        
        
    def crawl_folders(self):
        
        samples = []

        for scene in self.scenes:
            images = sorted(scene.files("*.png"))

            for i in range(len(images)):
                sample = {}
                sample['img'] = images[i]
                samples.append(sample)
        
        random.shuffle(samples)
        self.samples = samples
    
    def __getitem__(self, index: int):
        sample = self.samples[index]
        img = load_as_float(sample['img'])

        if self.transform is not None:
            img, _ = self.transform(img, None)
        
        return img
    
    def __len__(self):
        return len(self.samples)