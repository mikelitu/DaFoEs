from __future__ import division
import torch
import torch.nn.functional as F

def set_id_grid(image: torch.Tensor):

    """
    Create the initial identity grid. Sets a global variable to be accessed by the rest of the functions in this script.
    Args:
        image: Image 
    """
    global pixel_coords
    b, _, d, h, w = image.shape
    i_range = torch.arange(0, h).view(1, 1, h, 1).expand(
        1, h, w).type_as(image) # [1, D, H, W]
    j_range = torch.arange(0, w).view(1, 1, 1, w).expand(
        1, h, w).type_as(image) # [1, D, H, W]
    ones = torch.ones(1, h, w).as_type(image) # [1, H, W]

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1) # [1, 3, H, W]

