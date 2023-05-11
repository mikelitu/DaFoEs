import torch
from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

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


def get_reflection(states: List[np.ndarray], forces: List[np.ndarray], mode: str = 'horizontal'):
    assert mode in ['horizontal', 'vertical'], "The mode must be horizontal or vertical"

    if mode == 'horizontal':
        T = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    
    # Robot state
    robot_position = [state[:3] for state in states]
    robot_orientation = [state[3:7] for state in states]
    robot_joints = [state[7:13] for state in states]

    # Command state
    haptic_position = [state[13:16] for state in states]
    haptic_orientation = [state[16:20] for state in states]
    haptic_joints = [state[20:26] for state in states]

    reflected_states, reflected_forces = [], []

    for (r_pos, r_or, r_joints, h_pos, h_or, h_joints, f) in zip(robot_position, robot_orientation, 
                                                              robot_joints, haptic_position, 
                                                              haptic_orientation, haptic_joints,
                                                              forces):
        reflected_robot_pos, reflected_robot_or = reflect_cartesian(r_pos, r_or, T)
        reflected_robot_joints = reflect_joints(r_joints, mode=mode)
        reflected_robot_state = np.append(np.append(reflected_robot_pos, reflected_robot_or), reflected_robot_joints)
        reflected_haptic_pos, reflected_haptic_or = reflect_cartesian(h_pos, h_or, T)
        reflected_haptic_joints = reflect_joints(h_joints, mode=mode)
        reflected_haptic_state = np.append(np.append(reflected_haptic_pos, reflected_haptic_or), reflected_haptic_joints)
        reflected_state = np.append(reflected_robot_state, reflected_haptic_state)
        reflected_states.append(reflected_state)
        reflected_forces.append(T @ f)

    return reflected_states, reflected_forces


def reflect_cartesian(position: np.ndarray, orientation: np.ndarray, T: np.ndarray):
    rot_mat = R.from_quat(orientation).as_matrix()
    reflected_rot_mat = T @ rot_mat
    reflected_orientation = R.from_matrix(reflected_rot_mat).as_quat()

    reflected_position = T @ position

    return reflected_position, reflected_orientation

def reflect_joints(joints: np.ndarray, mode: str = 'horizontal'):
    reflected_joints = np.copy(joints)
    if mode == 'horizontal':
        reflected_joints[0] = -joints[0]
        reflected_joints[5] = -joints[5]
    else:
        reflected_joints[1] = -joints[1]
    
    return reflected_joints


def transform_cartesian(position: np.ndarray, orientation: np.ndarray, T: np.ndarray):

    rot_mat = R.from_quat(orientation).as_matrix()
    transformed_rot_mat = T @ rot_mat
    transformed_orientation = R.from_matrix(transformed_rot_mat).as_quat()

    transformed_position = T @ position

    return transformed_position, transformed_orientation

def transform_joints(joints: np.ndarray, angle: int):
    joints[3] += angle
    return joints 

def transform_state(states: List[np.ndarray], forces: List[np.ndarray], angle: int):
    # Rotation around the Z axis to rotate the image coordinate system
    T = np.ndarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    robot_position = [state[:3] for state in states]
    robot_orientation = [state[3:7] for state in states]
    robot_joints = [state[7:13] for state in states]

    # Command state
    haptic_position = [state[13:16] for state in states]
    haptic_orientation = [state[16:20] for state in states]
    haptic_joints = [state[20:26] for state in states]

    transformed_states, transformed_forces = [], []

    for (r_pos, r_or, r_joint, h_pos, h_or, h_joint, f) in zip(robot_position, robot_orientation, robot_joints,
                                                               haptic_position, haptic_orientation, haptic_joints,
                                                               forces):
        
        transformed_r_pos, transformed_r_or = transform_cartesian(r_pos, r_or, T)
        transformed_r_joints = transform_joints(r_joint, angle)
        transformed_r_state = np.append(np.append(transformed_r_pos, transformed_r_or), transformed_r_joints)
        transformed_h_pos, transformed_h_or = transform_cartesian(h_pos, h_or, T)
        transformed_h_joints = transform_joints(h_joint, angle)
        transformed_h_state = np.append(np.append(transformed_h_pos, transformed_h_or), transformed_h_joints)
        transformed_state = np.append(transformed_r_state, transformed_h_state)
        transformed_states.append(transformed_state)
        transformed_forces.append(T @ f)

    return transformed_states, transformed_forces

@torch.jit.script
def RGBtoD(r, g, b) -> float:
    if (b + g + r < 255):
        return 0.
    elif (r >= g) and (r >= b):
        if (g >= b):
            return float(g) - float(b)
        else:
            return (float(g) - float(b)) + 1529.
    elif (g >= r) and (g >= b):
        return float(b) - float(r) + 510.
    elif (b >= g) and (b >= r):
        return float(r) - float(g) + 1020.
    
    return 0.

    

    

