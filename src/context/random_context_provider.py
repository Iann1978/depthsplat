from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from ..dataset.types import BatchedViews
from .context_provider import ContextProvider, ContextProviderCfgCommon, debug_output_context

@dataclass
class RandomContextProviderCfg(ContextProviderCfgCommon):
    name: Literal["random"]
    num_views: int
    


class RandomContextProvider(ContextProvider):
    cfg: RandomContextProviderCfg

    def __init__(self, cfg: RandomContextProviderCfg) -> None:
        self.cfg = cfg

    def get_context(self) -> BatchedViews:
        b, v, h, w = 1, self.cfg.num_views, self.cfg.image_shape[0], self.cfg.image_shape[1]

        extrinsics = self._generate_proper_extrinsics(b, v)
        intrinsics = self._generate_proper_intrinsics(b, v, h, w)

        context = BatchedViews(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image=torch.randn(b, v, 3, h, w),
            near=torch.randn(b, v),
            far=torch.randn(b, v)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.to_device(context, device)

    def _generate_proper_extrinsics(self, b: int, v: int) -> torch.Tensor:
        """Generate proper camera extrinsics matrices (world-to-camera transformation)"""
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(b, v, 4, 4)
        
        # Generate random camera positions around a sphere
        for batch_idx in range(b):
            for view_idx in range(v):
                # Random position on a sphere
                radius = 2.0 + torch.rand(1) * 2.0  # Distance from origin
                theta = torch.rand(1) * 2 * torch.pi  # Azimuth
                phi = torch.rand(1) * torch.pi  # Elevation
                
                x = radius * torch.sin(phi) * torch.cos(theta)
                y = radius * torch.sin(phi) * torch.sin(theta)
                z = radius * torch.cos(phi)
                
                camera_pos = torch.tensor([x, y, z])
                
                # Look at origin
                look_at = torch.tensor([0.0, 0.0, 0.0])
                up = torch.tensor([0.0, 1.0, 0.0])
                
                # Create rotation matrix
                z_axis = F.normalize(look_at - camera_pos, dim=0)
                x_axis = F.normalize(torch.cross(z_axis, up), dim=0)
                y_axis = torch.cross(z_axis, x_axis)
                
                rotation = torch.stack([x_axis, y_axis, z_axis], dim=1)
                
                # Create extrinsics matrix (world-to-camera)
                extrinsics[batch_idx, view_idx, :3, :3] = rotation
                extrinsics[batch_idx, view_idx, :3, 3] = camera_pos
                extrinsics[batch_idx, view_idx, 3, :3] = 0.0
                extrinsics[batch_idx, view_idx, 3, 3] = 1.0
        
        return extrinsics
    
    def _generate_proper_intrinsics(self, b: int, v: int, h: int, w: int) -> torch.Tensor:
        """Generate proper camera intrinsics matrices"""
        intrinsics = torch.zeros(b, v, 3, 3)
        
        for batch_idx in range(b):
            for view_idx in range(v):
                # Reasonable focal lengths
                fx = fy = max(h, w) * (0.7 + torch.rand(1) * 0.3)  # 0.7-1.0 of image size
                
                # Principal point at image center
                cx = w / 2.0
                cy = h / 2.0
                
                intrinsics[batch_idx, view_idx, 0, 0] = fx
                intrinsics[batch_idx, view_idx, 1, 1] = fy
                intrinsics[batch_idx, view_idx, 0, 2] = cx
                intrinsics[batch_idx, view_idx, 1, 2] = cy
                intrinsics[batch_idx, view_idx, 2, 2] = 1.0
                
        return intrinsics

def test_random_context_provider():
    cfg = RandomContextProviderCfg(
        name="random",
        num_views=10,
        image_shape=[256, 256]
    )
    provider = RandomContextProvider(cfg)
    context = provider.get_context()
    debug_output_context(context)

if __name__ == "__main__":
    test_random_context_provider()