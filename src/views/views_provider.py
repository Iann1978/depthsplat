from typing import TypedDict, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod
from jaxtyping import Float, Int64
from torch import Tensor

T = TypeVar("T")

class BatchedRenderViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view

@dataclass
class ViewsProviderCfgCommon:
    image_shape: list[int]


class ViewsProvider(ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        self.cfg = cfg

    @abstractmethod
    def get_views(self) -> BatchedRenderViews:
        pass