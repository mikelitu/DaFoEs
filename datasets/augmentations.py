from __future__ import division
import torch
import random
import numpy as np
from PIL import Image
from typing import List, Tuple
import torch.nn.functional as F
from datasets.utils import get_reflection
import cv2


'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms: object):
        self.transforms = transforms

    def __call__(self, images: List[np.ndarray], depths: List[np.ndarray] = None, states: List[np.ndarray] = None, forces: List[np.ndarray] = None) -> List[torch.Tensor]:
        for t in self.transforms:
            images, depths, states, forces = t(images, depths, states, forces)
        return images, depths, states, forces


class Normalize(object):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, images: List[torch.Tensor], depths: List[torch.Tensor] = None, states: List[np.ndarray] = None, forces: List[np.ndarray] = None) -> List[torch.Tensor]:
        for t, m, s in zip(images, self.mean, self.std):
            t.sub_(m).div_(s)
        return images, depths, states, forces


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images: List[np.ndarray], depths: List[np.ndarray] = None, states: List[np.ndarray] = None, forces: List[np.ndarray] = None) -> List[torch.Tensor]:
        # put it from HWC to CHW format
        images = [np.transpose(im, (2, 0, 1)) for im in images]
        if depths is not None:
            depths = [np.transpose(depth, (2, 0, 1)) for depth in depths]
            depth_tensor = [torch.from_numpy(depth).float() for depth in depths]
        else:
            depth_tensor = None
        # handle numpy array
        im_tensor = [torch.from_numpy(im).float()/255. for im in images]
        
        return im_tensor, depth_tensor, states, forces


class RandomHorizontalFlip(object):

    def __call__(self, images: List[np.ndarray], depths: List[np.ndarray] = None, states: List[np.ndarray] = None, forces: List[np.ndarray] = None) -> np.ndarray:
        
        if random.random() < 0.5:
            output_images = [np.copy(np.fliplr(img)) for img in images]
            if depths is not None:
                output_depths = [np.copy(np.fliplr(depth)) for depth in depths]
            else:
                output_depths = None

            if states is not None:
                reflected_states, reflected_forces = get_reflection(states, forces, mode='horizontal')
        else:
            output_images = images
            output_depths = depths
            reflected_states = states
            reflected_forces = forces
        
        return output_images, output_depths, reflected_states, reflected_forces


class RandomVerticalFlip(object):

    def __call__(self, images: List[np.ndarray], depths: List[np.ndarray] = None, states: List[np.ndarray] = None, forces: List[np.ndarray] = None) -> np.ndarray:
        
        if random.random() < 0.5:
            output_images = [np.copy(np.flipud(img)) for img in images]
            
            if depths is not None:
                output_depths = [np.copy(np.flipud(depth)) for depth in depths]
            else:
                output_depths = None
            
            if states is not None:
                reflected_states, reflected_forces = get_reflection(states, forces, mode='vertical')
        else:
            output_images = images
            output_depths = depths
            reflected_states = states
            reflected_forces = forces
        
        return output_images, output_depths, reflected_states, reflected_forces


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images: List[np.ndarray], depths: List[np.ndarray] = None, states: List[np.ndarray] = None, forces: List[np.ndarray] = None) -> np.ndarray:

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32) for im in images]
        
        if depths is not None:
            scaled_depths = [cv2.resize(depth.astype(np.uint16), (scaled_w, scaled_h)).astype(np.float32) for depth in depths]
        else:
            scaled_depths = None

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)

        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]
        
        if depths is not None:
            cropped_depths = [depth[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for depth in scaled_depths]
        else:
            cropped_depths = None

        return cropped_images, cropped_depths, states, forces

class CentreCrop(object):
    def __call__(self, images: List[np.ndarray], depths: List[np.ndarray] = None, states: List[np.ndarray] = None, forces: List[np.ndarray] = None) -> List[np.ndarray]:
    
        in_h, in_w, _ = images[0].shape
        c_h, c_w = in_h // 2 , in_w // 2
        cropped_images = [im[c_h - 150:c_h + 150, c_w - 150:c_w + 150] for im in images]
        if depths is not None:
            cropped_depths = [depth[c_h - 150:c_h + 150, c_w - 150:c_w + 150] for depth in depths]
        else:
            cropped_depths = None

        return cropped_images, cropped_depths, states, forces


class SquareResize(object):
    """Resize the image to a square of 256x256"""
    def __call__(self, images: List[np.ndarray], depths: List[np.ndarray] = None, states: List[np.ndarray] = None, forces: List[np.ndarray] = None) -> List[np.ndarray]:

        new_size = (256, 256)
        scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize(new_size)).astype(np.float32) for im in images]
        if depths is not None:
            scaled_depths = [cv2.resize(depth.astype(np.uint16), new_size).astype(np.float32) for depth in depths]
        else:
            scaled_depths = None
        return scaled_images, scaled_depths, states, forces


class GaussianNoise(object):

    def __init__(self, noise_factor = 0.3) -> None:
        self.noise_factor = noise_factor
    
    def __call__(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        noise_factor = self.noise_factor * random.random()
        noisy_imgs = [img + torch.randn_like(img) * noise_factor for img in images]
        noisy_imgs = [torch.clip(img, 0., 1.) for img in noisy_imgs]
        return noisy_imgs
