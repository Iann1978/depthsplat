import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning)
import copy

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.plugins.environments import LightningEnvironment
from src.dataset.data_module import worker_init_fn



# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.resume_ckpt import find_latest_ckpt
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.misc.image_io import save_image
    from src.dataset.types import debug_output_sample
    from src.dataset.data_module import get_dataset
    from src.model.types import debug_output_gaussians
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="infer2",
)
def train(cfg_dict: DictConfig):
    print("infer2")
    print(OmegaConf.to_yaml(cfg_dict))


    eval_cfg = None

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []

    print("disabled wandb")
    logger = LocalLogger()


    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()


    torch.manual_seed(cfg_dict.seed + 0)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)




    strict_load = not cfg.checkpointing.no_strict_load

    print("load pretrained model")
    pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
    if 'state_dict' in pretrained_model:
        pretrained_model = pretrained_model['state_dict']
    
    # Extract encoder weights by removing "encoder." prefix
    encoder_weights = {}
    for key, value in pretrained_model.items():
        if key.startswith("encoder."):
            encoder_key = key[8:]  # Remove "encoder." prefix
            encoder_weights[encoder_key] = value
    
    encoder.load_state_dict(encoder_weights)
    print(
        cyan(
            f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
        )
    )


    dataset = get_dataset(
        cfg.dataset,
        "test",
        step_tracker,
    )
    # dataset = data_module.dataset_shim(dataset, "test")
    loader = DataLoader(
        dataset,
        cfg.data_loader.test.batch_size,
        num_workers=cfg.data_loader.test.num_workers,
        # generator=data_module.get_generator(cfg.data_loader.test),
        worker_init_fn=worker_init_fn,
        # persistent_workers=data_module.get_persistent(cfg.data_loader.test),
        shuffle=False,
    )

    print(cyan("sample"))
    sample = next(iter(loader))
    debug_output_sample(sample)
    exit()
    sample["context"]["image"] = sample["context"]["image"].cuda()
    sample["context"]["extrinsics"] = sample["context"]["extrinsics"].cuda()
    sample["context"]["intrinsics"] = sample["context"]["intrinsics"].cuda()
    sample["context"]["near"] = sample["context"]["near"].cuda()
    sample["context"]["far"] = sample["context"]["far"].cuda()
    sample["target"]["image"] = sample["target"]["image"].cuda()
    sample["target"]["extrinsics"] = sample["target"]["extrinsics"].cuda()
    sample["target"]["intrinsics"] = sample["target"]["intrinsics"].cuda()
    sample["target"]["near"] = sample["target"]["near"].cuda()
    sample["target"]["far"] = sample["target"]["far"].cuda()
    # print(f"sample: {sample}")


    encoder.eval()
    encoder.cuda()
    # print(encoder)
    # print(encoder_visualizer)

    decoder.eval()
    decoder.cuda()
    print(decoder)

    print(cyan("encoding gaussians"))
    gaussians = encoder(sample["context"], 0, False)["gaussians"]
    debug_output_gaussians(gaussians)
    # print(f"gaussians: {gaussians}")

    print(cyan("rendering gaussians"))
    image_shape = (sample["target"]["image"].shape[-2], sample["target"]["image"].shape[-1])  # (height, width)
    output = decoder(gaussians, sample["target"]["extrinsics"], sample["target"]["intrinsics"], sample["target"]["near"], sample["target"]["far"], image_shape, depth_mode=None)
    # print(f"output: {output}")

    print(cyan("saving images"))
    save_image(sample["context"]["image"][0,0], "context_image0.jpg")
    save_image(sample["context"]["image"][0,1], "context_image1.jpg")
    # save_image(sample["target"]["image"][0], "target_image.jpg")
    save_image(output.color[0,0], "color256-1.jpg")

    exit()




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
