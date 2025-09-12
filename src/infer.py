
from dataclasses import dataclass
from typing import Literal
import warnings
import torch
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning)

import hydra
from omegaconf import DictConfig, OmegaConf
from src.config import load_typed_config
from src.context import ContextProviderCfg, get_context_provider, debug_output_context
from src.model.encoder import get_encoder
from src.config import EncoderCfg, ModelCfg
from src.model.types import debug_output_gaussians


@dataclass
class InferCfg:
    type: Literal["infer"]
    context_provider: ContextProviderCfg
    model: ModelCfg

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

    print("get context provider")
    context_provider = get_context_provider(cfg.context_provider)
    print(context_provider)

    print("get encoder")
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    encoder.eval()
    encoder.cuda()
    print(encoder)
    print(encoder_visualizer)

    context = context_provider.get_context()
    debug_output_context(context)

    gaussians = encoder(context, 0, False)["gaussians"]
    debug_output_gaussians(gaussians)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")


    torch.set_float32_matmul_precision('high')

    infer()