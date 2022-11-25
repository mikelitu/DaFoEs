import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset
import imageio
import numpy as np
import random
from path import Path


class VisionStateDataset(Dataset):
    """A dataset to load data from dfferent folders that are arranged this way:
        root/scene_1/000.png
        root/scene_1/001.png
        ...
        root/scene_1/labels.txt
        root/scene_2/000.png
        .
        transform functions takes in a list images and a numpy array representing the intrinsics of the camera and the robot state
    """

    def __init__(self, root, is_train=True, transform=None, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/"train.txt" if is_train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.cam = np.fromfile(self.root/"cam.txt").astype(np.float32).reshape(3,3)
        self.crawl_folders()
        
        
    def crawl_folders(self):
        
        samples = []

        for scene in self.scenes:
            images = sorted(scene.files("*.png"))
