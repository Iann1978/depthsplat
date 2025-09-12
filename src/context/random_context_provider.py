from dataclasses import dataclass
from typing import Literal

import torch

from ..dataset.types import BatchedViews
from .context_provider import ContextProvider, ContextProviderCfgCommon

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
        return BatchedViews(
            extrinsics=torch.randn(b, v, 4, 4),
            intrinsics=torch.randn(b, v, 3, 3),
            image=torch.randn(b, v, 3, h, w),
            near=torch.randn(b, v),
            far=torch.randn(b, v)
        )

def test_random_context_provider():
    cfg = RandomContextProviderCfg(
        name="random",
        num_views=10,
        image_shape=[256, 256]
    )
    provider = RandomContextProvider(cfg)
    context = provider.get_context()
    print('extrinsics shape: ', context["extrinsics"].shape)
    print('intrinsics shape: ', context["intrinsics"].shape)
    print('image shape: ', context["image"].shape)
    print('near shape: ', context["near"].shape)
    print('far shape: ', context["far"].shape)

if __name__ == "__main__":
    test_random_context_provider()