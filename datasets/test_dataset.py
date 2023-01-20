import torch
from torch.utils.data import Dataset
import imageio
from path import Path
from PIL import ImageFile
import numpy as np
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_as_float(path: Path) -> np.ndarray:
    return  imageio.imread(path).astype(np.float32)

def normalize_labels(labels: np.ndarray, eps=1e-10) -> np.ndarray:
    return (labels - labels.mean(axis=0)) / (labels.std(axis=0) + eps) 


class TestDataset(Dataset):

    def __init__(self, test_folder: Path) -> None:
        super().__init__()
