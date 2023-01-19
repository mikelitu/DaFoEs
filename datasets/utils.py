import torch
from typing import List, Tuple

def check_key(key: str) -> List[int]:
    indices_dict = {'robot_pos': ["values to be decided"], 'joint_pos': ["values to be decided"], 'geo_com': ["values to be decided"]}
    return indices_dict[key]

def check_params(keys: List[str], params: List[str]):
    occluded = []
    for param in params:
        if param in keys:
           occluded.append(param)

    return occluded 

def generate_keys_rs() -> List[str]:
    return ["robot_pos", "joint_pos", "geo_com"]

def _get_gaussian_kernel1d(kernel_size: int, sigma: float):
        ksize_half = (kernel_size - 1) * 0.5

        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d
    
def _get_gaussian_kernel2d(kernel_size: Tuple[int], sigma: Tuple[float]) -> torch.Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0])
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1])
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d