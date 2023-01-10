from __future__ import division
import torch
import random
import numpy as np
from PIL import Image
from typing import List, Tuple


'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms: object):
        self.transforms = transforms

    def __call__(self, images: List[np.ndarray], intrinsics: np.ndarray = None) -> Tuple[torch.Tensor]:
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, images: List[torch.Tensor], intrinsics: np.ndarray) -> Tuple[torch.Tensor]:
        for t, m, s in zip(images, self.mean, self.std):
            t.sub_(m).div_(s)
        return images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images: List[np.ndarray], intrinsics: np.ndarray = None) -> Tuple[torch.Tensor]:
        # put it from HWC to CHW format
        images = [np.transpose(im, (2, 0, 1)) for im in images]
        # handle numpy array
        tensor = [torch.from_numpy(im).float()/255 for im in images]
        return tensor, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images: List[np.ndarray], intrinsics: np.ndarray = None) -> Tuple[np.ndarray]:
        output_intrinsics = None
        if random.random() < 0.5:
            if intrinsics is not None:
                output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(image)) for image in images]
            w = output_images[0].shape[1]
            if intrinsics is not None:
                output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_images = images
            if intrinsics is not None:
                output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images: List[np.ndarray], intrinsics: np.ndarray = None) -> Tuple[np.ndarray]:
        output_intrinsics = None
        if intrinsics is not None:
            output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
        if intrinsics is not None:
            output_intrinsics[0] *= x_scaling
            output_intrinsics[1] *= y_scaling

        scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        if intrinsics is not None:
            output_intrinsics[0, 2] -= offset_x
            output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics

class CentreCrop(object):
    def __call__(self, images: List[np.ndarray], intrinsics: np.ndarray = None):
        if intrinsics is not None:
            output_intrinsics = np.copy(intrinsics)
        else:
            output_intrinsics = None
        
        in_h, in_w, _ = images[0].shape
        c_h, c_w = in_h // 2 , in_w // 2
        cropped_images = [im[c_h - 150:c_h + 150, c_w - 150:c_w + 150] for im in images]
        if intrinsics is not None:
            output_intrinsics[0, 2] -= 300
            output_intrinsics[1, 2] -= 300
        return cropped_images, output_intrinsics

class SquareResize(object):
    """Resize the image to a square of 256x256"""
    def __call__(self, images: List[np.ndarray], intrinsics: np.ndarray = None):
        if intrinsics is not None:
            output_intrinsics = np.copy(intrinsics)
        else:
            output_intrinsics = None

        new_size = (256, 256)

        if intrinsics is not None:
            in_h, in_w, _ = images[0].shape
            scaling_h, scaling_w = (new_size[1] / in_h), (new_size[0] / in_w)
            output_intrinsics[0] *= scaling_w
            output_intrinsics[1] *= scaling_h

        scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize(new_size)).astype(np.float32) for im in images]

        return scaled_images, output_intrinsics



         