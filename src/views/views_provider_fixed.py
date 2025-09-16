from typing import Literal
from dataclasses import dataclass
from pathlib import Path
import torch
from .views_provider import ViewsProviderCfgCommon, ViewsProvider, BatchedRenderViews
from ..misc.camera_utils import load_camera_intrinsics_and_image_shape

@dataclass
class ViewsProviderFixedCfg(ViewsProviderCfgCommon):
    name: Literal["fixed"]
    intrinsics_file_path: Path


class ViewsProviderFixed(ViewsProvider[ViewsProviderFixedCfg]):
    def get_views(self) -> BatchedRenderViews:
        intrinsics, image_shape = load_camera_intrinsics_and_image_shape(self.cfg.intrinsics_file_path)
        print('image_shape', image_shape)
        print('intrinsics', intrinsics)
        h, w = image_shape
        intrinsics[0,:] /= w
        intrinsics[1,:] /= h
        print('normalized_intrinsics', intrinsics)
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).float().cuda()
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda()
        near = torch.tensor([[1.0]]).float().cuda()
        far = torch.tensor([[100.0]]).float().cuda()
 
        return BatchedRenderViews(
            intrinsics=intrinsics,
            image_shape=image_shape,
            extrinsics=extrinsics,
            near=near,
            far=far,
        )