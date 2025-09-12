from dataclasses import dataclass
from typing import Literal

import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass
class InferCfg:
    type: Literal["infer"]

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="infer",
)
def infer(cfg_dict: DictConfig):
    print("infer")
    print(OmegaConf.to_yaml(cfg_dict))

if __name__ == "__main__":
    infer()