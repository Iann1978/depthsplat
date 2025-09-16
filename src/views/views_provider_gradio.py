from typing import Literal
from dataclasses import dataclass
from pathlib import Path
import torch
from .views_provider import ViewsProviderCfgCommon, ViewsProvider, BatchedRenderViews
from ..misc.camera_utils import load_camera_intrinsics_and_image_shape

@dataclass
class ViewsProviderGradioCfg(ViewsProviderCfgCommon):
    name: Literal["gradio"]
    intrinsics_file_path: Path



class ViewsProviderGradio(ViewsProvider[ViewsProviderGradioCfg]):
    def get_views(self) -> BatchedRenderViews:
        print('get_views')
        print('kwargs', self.kwargs)
        print('ui', self.kwargs["ui"])
        print('ui.x', self.kwargs["ui"].x)
        print('ui.y', self.kwargs["ui"].y)

        intrinsics, image_shape = load_camera_intrinsics_and_image_shape(self.cfg.intrinsics_file_path)
        print('image_shape', image_shape)
        print('intrinsics', intrinsics)
        h, w = image_shape
        intrinsics[0,:] /= w
        intrinsics[1,:] /= h
        print('normalized_intrinsics', intrinsics)
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).float().cuda()

        x = self.kwargs["ui"].x
        y = self.kwargs["ui"].y
        print('x', x)
        print('y', y)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda()
        extrinsics[0, 0, 0, 3] = x  # x translation
        extrinsics[0, 0, 1, 3] = y  # y translation
        print('extrinsics', extrinsics)


        near = torch.tensor([[1.0]]).float().cuda()
        far = torch.tensor([[100.0]]).float().cuda()
 
        return BatchedRenderViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            near=near,
            far=far,
            image_shape=image_shape,
        )