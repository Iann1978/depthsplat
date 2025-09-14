from dataclasses import dataclass
from typing import Literal

from pathlib import Path

from ..dataset.types import BatchedViews
from .context_provider import ContextProvider, ContextProviderCfgCommon
import cv2
import os
import json
import numpy as np
import torch

@dataclass
class ImagesContextProviderCfg(ContextProviderCfgCommon):
    name: Literal["images"]
    root_path: Path
    debug: bool

def load_camera_params(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        K = np.array(data['K'])
        R = np.array(data['R'])
        t = np.array(data['t'])
        return K, R, t

class ImagesContextProvider(ContextProvider):
    def __init__(self, cfg:ImagesContextProviderCfg):
        self.cfg = cfg
        self.get_images()
        self.get_camera_params()
        pass

    def get_images(self):
        print('get_images')

        self.frame0 = cv2.imread(os.path.join(self.cfg.root_path, 'camera0.jpg'))
        self.frame1 = cv2.imread(os.path.join(self.cfg.root_path, 'camera1.jpg'))
        if self.frame0 is None:
            print("Failed to read camera0.jpg")
        else:
            print("Read camera0.jpg successfully, shape:", self.frame0.shape)
        if self.frame1 is None:
            print("Failed to read camera1.jpg")
        else:
            print("Read camera1.jpg successfully, shape:", self.frame1.shape)
        # if self.debug:
        #     self.show_images()
        return self.frame0, self.frame1
    
    def show_images(self):
        import matplotlib.pyplot as plt

        if self.frame0 is not None and self.frame1 is not None:
            # Convert BGR to RGB for matplotlib
            frame0_rgb = cv2.cvtColor(self.frame0, cv2.COLOR_BGR2RGB)
            frame1_rgb = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(frame0_rgb)
            plt.title('Camera 0')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(frame1_rgb)
            plt.title('Camera 1')
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            print("One or both frames are None, cannot display images.")
    
    def get_camera_params(self):
        print('get_camera_params')
        self.K0, self.R0, self.t0 = load_camera_params(os.path.join(self.cfg.root_path, 'camera0.json'))
        self.K1, self.R1, self.t1 = load_camera_params(os.path.join(self.cfg.root_path, 'camera1.json'))
        if self.cfg.debug:
            self.show_camera_params()
        return self.K0, self.R0, self.t0, self.K1, self.R1, self.t1
    
    def show_camera_params(self):
        print('self.K0, self.R0, self.t0', self.K0, self.R0, self.t0)
        print('self.K1, self.R1, self.t1', self.K1, self.R1, self.t1)


    
    def get_context(self) -> BatchedViews:
        b, v = 1, 2
        
        # Resize images to reduce memory usage
        target_height, target_width = self.cfg.image_shape  # [240, 320]
        frame0_resized = cv2.resize(self.frame0, (target_width, target_height))
        frame1_resized = cv2.resize(self.frame1, (target_width, target_height))
        
        # Convert numpy arrays to PyTorch tensors
        frame0_tensor = torch.from_numpy(frame0_resized).float()
        frame1_tensor = torch.from_numpy(frame1_resized).float()
        image = torch.stack([frame0_tensor, frame1_tensor], dim=0)
        image = image.permute(0, 3, 1, 2)
        image = image.unsqueeze(0)
        image = image/255.0
        print("image's min, max", image.min(), image.max())
        print(f"Resized images to: {target_height}x{target_width}")

        # Convert camera intrinsics from numpy to PyTorch tensors
        # Scale intrinsics to match resized images
        original_height, original_width = self.frame0.shape[:2]  # Original image size
        scale_x = 1 / original_width
        scale_y = 1 / original_height
        
        K0_tensor = torch.from_numpy(self.K0).float()
        K0_tensor[0, 0] *= scale_x  # fx
        K0_tensor[1, 1] *= scale_y  # fy
        K0_tensor[0, 2] *= scale_x  # cx
        K0_tensor[1, 2] *= scale_y  # cy
        print('K0_tensor', K0_tensor)
        
        K1_tensor = torch.from_numpy(self.K1).float()
        K1_tensor[0, 0] *= scale_x  # fx
        K1_tensor[1, 1] *= scale_y  # fy
        K1_tensor[0, 2] *= scale_x  # cx
        K1_tensor[1, 2] *= scale_y  # cy
        print('K1_tensor', K1_tensor)
        intrinsics = torch.stack([K0_tensor, K1_tensor], dim=0)
        intrinsics = intrinsics.unsqueeze(0)


        # Convert rotation and translation from numpy to PyTorch tensors
        R0_tensor = torch.from_numpy(self.R0).float()
        t0_tensor = torch.from_numpy(self.t0).float()
        R1_tensor = torch.from_numpy(self.R1).float()
        t1_tensor = torch.from_numpy(self.t1).float()
        
        E0_tensor = torch.eye(4).float()
        E0_tensor[:3, :3] = R0_tensor
        E0_tensor[:3, 3] = t0_tensor.flatten()
        E1_tensor = torch.eye(4).float()
        E1_tensor[:3, :3] = R1_tensor
        E1_tensor[:3, 3] = t1_tensor.flatten()
        extrinsics = torch.stack([E0_tensor, E1_tensor], dim=0)
        extrinsics = extrinsics.unsqueeze(0)
        
        return BatchedViews(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image=image,
            near=torch.tensor([[1, 1]]),  # Shape: (b=1, v=2)
            far=torch.tensor([[100, 100]]),  # Shape: (b=1, v=2)
            index=torch.tensor(0).unsqueeze(0),
        )

def test_images_context_provider():
    print('test_images_context_provider')
    cfg = ImagesContextProviderCfg(name='images', root_path='camsets/sets0', image_shape=[320, 240], debug=True)
    images_context_provider = ImagesContextProvider(cfg)
    context = images_context_provider.get_context()
    print("context's shape", context['image'].shape)
    print("context's min, max", context['image'].min(), context['image'].max())
    print("context's extrinsics shape", context['extrinsics'].shape)
    print("context's intrinsics shape", context['intrinsics'].shape)
    print("context's near shape", context['near'].shape)
    print("context's far shape", context['far'].shape)
    print("context's index shape", context['index'].shape)

if __name__ == '__main__':
    print(os.getcwd())
    test_images_context_provider()
        