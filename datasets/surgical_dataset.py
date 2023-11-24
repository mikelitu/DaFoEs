import torch
import os
from typing import Dict, List, Union
from torch.utils.data import Dataset
import imageio
import numpy as np
import random
from path import Path
import cv2
from PIL import ImageFile
from surgical_video_processing import crop_right_tool
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

class VideoLoader(object):
    def __init__(self,
                 filename: Union[Path, str],
                 ) -> None:
        
        self.cap = cv2.VideoCapture(str(filename))
        cv2.destroyAllWindows()


def generate_random_kinematics(mu: float, sigma: float):
    """
    Generate a random kinematics for the robot. The kinematics is a 7 dimensional vector with the following information:
    | 0 -> Time (t)
    | 1 to 6 -> Force sensor reading (fx, fy, fz, tx, ty, tz)
    | 7 to 19 -> Dvrk estimation task position, orientation, linear and angular velocities (px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz)
    | 20 to 26 -> Joint angles (q1, q2, q3, q4, q5, q6, q7)
    | 27 to 33 -> Joint velocities (vq1, vq2, vq3, vq4, vq5, vq6, vq7)
    | 44 to 40 -> Joint torque (tq1, tq2, tq3, tq4, tq5, tq6, tq7)
    | 41 to 47 -> Desired joint angle (q1d, q2d, q3d, q4d, q5d, q6d, q7d)
    | 48 to 54 -> Desired joint torque (tq1d, tq2d, tq3d, tq4d, tq5d, tq6d, tq7d)
    | 55 to 57 -> Estimated end effector force (psm_fx, psm_fy, psm
    """
    kinematic_chain = np.random.normal(mu, sigma, 54)
    return kinematic_chain / np.linalg.norm(kinematic_chain)

def load_as_float(path: Path) -> np.ndarray:
    img = imageio.imread(path)[:,:, :3].astype(np.float32)
    return crop_right_tool(img)


class SurgicalDataset(Dataset):
    """A dataset to load data from dfferent folders that are arranged this way:
        root/scene_1/000.png
        root/scene_1/001.png
        ...
        root/scene_1/labels.csv
        root/scene_2/000.png
        .
        transform functions takes in a list images and a numpy array representing the intrinsics of the camera and the robot state
    """

    def __init__(self, root, recurrency_size=5, transform=None, seed=0, extension=".png"):

        np.random.seed(seed)
        random.seed(seed)

        self.root = Path(root)
        self.extension = extension

        self.transform = transform
        self.recurrency_size = recurrency_size
        self._generate_samples()

    def _open_video(self, video) -> cv2.VideoCapture:
        
        # api = cv2.CAP_OPENCV_HDFS
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            return cap
        else:
            raise FileNotFoundError("Video {} not found".format(video))
    
    def _generate_samples(self) -> List[Dict[str, List[Path]]]:
        files = self.root.files("*" + self.extension)
        samples = []
        for i in range(len(files) - self.recurrency_size):
            sample = {}
            sample['img'] = files[i:i+self.recurrency_size]
            sample['label'] = np.array([generate_random_kinematics(0, 1) for _ in range(self.recurrency_size)])
            samples.append(sample)
        
        self.samples = samples
        
    
    def __getitem__(self, index: int) -> Dict[str, List[torch.Tensor]]:
        sample = self.samples[index]
        imgs = [load_as_float(img) for img in sample['img']]
        
        labels = sample['label']

        if self.transform is not None:
            imgs, depths, labels, forces = self.transform(imgs, depths, labels, forces, sample["dataset"])
    

        return {'img': imgs[0] if self.recurrency_size==1 else imgs, 'robot_state': labels}
    
    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    root = Path("/home/mikel/surgical_videos_analysis/videos/capture1")
    dataset = SurgicalDataset(root=root,recurrency_size=5, transform=None, seed=0, extension=".png")
    data = dataset[0]
    img = data['img'][0]
    print(img.shape)
    img = img / 255.
    plt.imshow(img)
    plt.show()
