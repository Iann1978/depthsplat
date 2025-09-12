from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ..dataset.types import BatchedViews


@dataclass
class ContextProviderCfgCommon:
    image_shape: list[int]



class ContextProvider(ABC):
    @abstractmethod
    def get_context(self) -> BatchedViews:
        pass