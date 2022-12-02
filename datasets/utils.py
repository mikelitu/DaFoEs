import torch
from typing import List

def check_key(key: str):
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