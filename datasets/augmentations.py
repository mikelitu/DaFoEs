from __future__ import division
import torch
import random
import numpy as np
from PIL import Image
from skimage.transform import resize
import torchvision
from skimage import img_as_float, img_as_ubyte
import warnings
from typing import List, Tuple

Params = Tuple[List[float]]


def get_scaling(image: np.ndarray, size: Tuple[int, int]) -> Tuple[float]:

    h, w, _ = image.shape

    new_w, new_h = size

    scaling_w = new_w / w
    scaling_h = new_h / h

    return scaling_w, scaling_h


'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms: object):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray]:
        for t in self.transforms:
            image, intrinsics = t(image, intrinsics)
        return image, intrinsics


class Normalize(object):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray]:
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, image: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray]:
        # put it from HWC to CHW format
        im = np.transpose(image, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(im).float()/255
        return tensor, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, image: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray]:
        output_intrinsics = None
        if random.random() < 0.5:
            if intrinsics is not None:
                output_intrinsics = np.copy(intrinsics)
            output_image = np.copy(np.fliplr(image))
            w = output_image.shape[1]
            if intrinsics is not None:
                output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_image = image
            if intrinsics is not None:
                output_intrinsics = intrinsics
        return output_image, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, image: np.ndarray, intrinsics: np.ndarray = None) -> Tuple[np.ndarray]:
        output_intrinsics = None
        if intrinsics is not None:
            output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = image.shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
        if intrinsics is not None:
            output_intrinsics[0] *= x_scaling
            output_intrinsics[1] *= y_scaling

        scaled_image = np.array(Image.fromarray(image.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32)

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = scaled_image[offset_y:offset_y + in_h, offset_x:offset_x + in_w]

        if intrinsics is not None:
            output_intrinsics[0, 2] -= offset_x
            output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics


class Resize(object):

    def __init__(self, size: Tuple[int] = (256, 256), interp: str = 'nearest'):

        assert isinstance(size, tuple), 'The new size is not valid, it must be a tuple'
        assert len(size) == 2, 'The new size is not valid, only size 2 accepted'
        self.size = size
        self.interp = interp

    def __call__(self, image: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray]:
        output_intrinsics = intrinsics
        if intrinsics is not None:
            output_intrinsics = np.copy(intrinsics)

        scaled = resize(image, self.size, order=1 if self.interp=='bilinear' else 0, preserve_range=True,
                        mode_constant=True, anti_aliasing=True)
        
        if intrinsics is not None:
            scaling_w, scaling_h = get_scaling(image, self.size)
            output_intrinsics[0] *= scaling_w
            output_intrinsics[1] *= scaling_h

        return scaled, output_intrinsics


class ColorJitter(object):

    def __init__(self, brightness: float = 0, contrast: float = 0, saturation: float = 0, hue: float = 0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def get_params(self, brightness: float, contrast: float, saturation: float, hue: float) -> Params:
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness
            )
        else:
            brightness_factor = None
        
        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast
            )
        else:
            contrast_factor = None
        
        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation
            )
        else:
            saturation_factor = None
        
        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        
        return brightness_factor, contrast_factor, saturation_factor, hue_factor
    
    def __call__(self, image: np.ndarray, intrinsics: np.ndarray = None) -> Tuple[np.ndarray]:
        assert isinstance(image, np.ndarray), "The image has to be a numpy array"
        new_intrinsics = None
        if intrinsics is not None:
            new_intrinsics = np.copy(intrinsics)

        brightness, contrast, saturation, hue = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        #Create a img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)
        img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array, img_as_float]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jittered_img = image
            for func in img_transforms:
                jittered_img = func(jittered_img)
        
        return jittered_img, new_intrinsics

         