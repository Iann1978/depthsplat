from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import torch

from ..dataset.types import BatchedViews


@dataclass
class ContextProviderCfgCommon:
    image_shape: list[int]



class ContextProvider(ABC):
    @abstractmethod
    def get_context(self) -> BatchedViews:
        pass

    def to_device(self, context: BatchedViews, device: torch.device) -> BatchedViews:
        device_context = BatchedViews(
            extrinsics=context["extrinsics"].to(device),
            intrinsics=context["intrinsics"].to(device),
            image=context["image"].to(device),
            near=context["near"].to(device),
            far=context["far"].to(device)
        )
        return device_context
        
def debug_output_context(context: BatchedViews):
    print('-'*100)
    print('extrinsics shape: ', context["extrinsics"].shape)
    print('intrinsics shape: ', context["intrinsics"].shape)
    print('image shape: ', context["image"].shape)
    print('near shape: ', context["near"].shape)
    print('far shape: ', context["far"].shape)
    print('image\'s device: ', context["image"].device)
    print('extrinsics\'s device: ', context["extrinsics"].device)
    print('intrinsics\'s device: ', context["intrinsics"].device)
    print('near\'s device: ', context["near"].device)
    print('far\'s device: ', context["far"].device)
    print('-'*100)