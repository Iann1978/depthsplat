
from dataclasses import dataclass
from typing import Literal
import warnings
import torch
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning)

import hydra
from omegaconf import DictConfig, OmegaConf
from src.config import load_typed_config
from src.context import ContextProviderCfg, get_context_provider, debug_output_context


@dataclass
class InferCfg:
    type: Literal["infer"]
    context_provider: ContextProviderCfg

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="infer",
)
def infer(cfg_dict: DictConfig):
    print("infer")
    print(OmegaConf.to_yaml(cfg_dict))

    cfg = load_typed_config(cfg_dict, InferCfg)
    print(cfg)

    context_provider = get_context_provider(cfg.context_provider)
    print(context_provider)

    context = context_provider.get_context()
    debug_output_context(context)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")


    torch.set_float32_matmul_precision('high')

    infer()