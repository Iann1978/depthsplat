from dataclasses import dataclass
from typing import Literal

import torch

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
        context = BatchedViews(
            extrinsics=torch.randn(b, v, 4, 4),
            intrinsics=torch.randn(b, v, 3, 3),
            image=torch.randn(b, v, 3, h, w),
            near=torch.randn(b, v),
            far=torch.randn(b, v)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.to_device(context, device)
        
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