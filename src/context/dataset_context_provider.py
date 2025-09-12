import warnings
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning)
from dataclasses import dataclass
from typing import Iterator, Literal

from ..dataset import Dataset, DatasetRE10kCfg
from ..dataset import get_dataset
from .context_provider import ContextProvider, ContextProviderCfgCommon
from ..dataset import DatasetCfg
from ..dataset.types import BatchedViews
from omegaconf import DictConfig, OmegaConf
import hydra
from src.config import load_typed_config
from .context_provider import debug_output_context
import torch

@dataclass
class DatasetContextProviderCfg(ContextProviderCfgCommon):
    name: Literal["dataset"]
    dataset: DatasetCfg


class DatasetContextProvider(ContextProvider):
    cfg: DatasetContextProviderCfg
    dataset: Dataset
    dataset_iterator: Iterator

    def __init__(self, cfg: DatasetContextProviderCfg) -> None:
        self.cfg = cfg
        self.dataset = get_dataset(cfg.dataset, "train", None)
        self.dataset_iterator = iter(self.dataset)

    def get_context(self) -> BatchedViews:
        context = next(self.dataset_iterator)["context"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.to_device(context, device)

@hydra.main(config_path="../../config/dataset", config_name="re10k", version_base=None)
def test_dataset_context_provider(cfg_dict: DictConfig):
    print(OmegaConf.to_yaml(cfg_dict))
    dataset_cfg = load_typed_config(cfg_dict, DatasetRE10kCfg)
    print(dataset_cfg)

    print("get dataset context provider")
    provider_cfg = DatasetContextProviderCfg(
        name="dataset",
        dataset=dataset_cfg,
        image_shape=[256, 256]
    )
    context_provider = DatasetContextProvider(provider_cfg)
    context = context_provider.get_context()
    print("context")
    # print(context)
    debug_output_context(context)
if __name__ == "__main__":
    test_dataset_context_provider()