
from dataclasses import dataclass
from typing import Literal
import warnings
import torch
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning)

import hydra
from omegaconf import DictConfig, OmegaConf
from src.config import load_typed_config
from src.context import ContextProviderCfg, get_context_provider, debug_output_context
from src.model.decoder import get_decoder, debug_output_decoder_output
from src.model.encoder import get_encoder
from src.config import EncoderCfg, ModelCfg
from src.model.types import debug_output_gaussians
from src.misc.image_io import save_image
from colorama import Fore
from torch.utils.data import DataLoader
from src.dataset.data_module import get_dataset, worker_init_fn
from src.dataset.types import debug_output_sample
from src.dataset import DatasetCfg

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@dataclass
class InferCfg:
    type: Literal["infer"]
    context_provider: ContextProviderCfg
    model: ModelCfg
    dataset: DatasetCfg 

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="infer",
)
def infer(cfg_dict: DictConfig):
    print("infer")
    print(OmegaConf.to_yaml(cfg_dict))
    # exit()

    cfg = load_typed_config(cfg_dict, InferCfg)
    print(cfg)

    print("get context provider")
    context_provider = get_context_provider(cfg.context_provider)
    print(context_provider)

    print("get encoder")
    print('cfg.model.encoder', cfg.model.encoder)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    pretrained_model = torch.load('pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth', map_location='cpu')
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
            f"Loaded pretrained weights: pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth"
        )
    )
    encoder.eval()
    encoder.cuda()
    print(encoder)
    print(encoder_visualizer)

    print("get decoder")
    decoder = get_decoder(cfg.model.decoder, cfg.context_provider.dataset)
    decoder.eval()
    decoder.cuda()
    print(decoder)

    dataset = get_dataset(
        cfg.dataset,
        "test",
        None,
    )
    # dataset = data_module.dataset_shim(dataset, "test")
    loader = DataLoader(
        dataset,
        1,
        # num_workers=cfg.data_loader.test.num_workers,
        # generator=data_module.get_generator(cfg.data_loader.test),
        # worker_init_fn=worker_init_fn,
        # persistent_workers=data_module.get_persistent(cfg.data_loader.test),
        shuffle=False,
    )

    sample = next(iter(loader))
    debug_output_sample(sample) 
    context = sample["context"]
    context["image"] = context["image"].cuda()
    context["extrinsics"] = context["extrinsics"].cuda()
    context["intrinsics"] = context["intrinsics"].cuda()
    context["near"] = context["near"].cuda()
    context["far"] = context["far"].cuda()

    # context = context_provider.get_context()
    debug_output_context(context)
    save_image(context["image"][0, 0], 'context_image0.jpg')
    save_image(context["image"][0, 1], 'context_image1.jpg')
    print("Saved context images to context_image0.jpg and context_image1.jpg")


    print(cyan("encoding gaussians"))
    gaussians = encoder(context, 0, False)["gaussians"]
    debug_output_gaussians(gaussians)
    
    # Clear cache after encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after encoder: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Use smaller image size to reduce memory usage
    image_shape = (256, 256)  # Reduced from (256, 256)
    print(f"Using image shape: {image_shape}")
    
    # Option to skip depth rendering if memory is tight
    render_depth = False  # Set to False if you want to skip depth rendering
    depth_mode = "depth" if render_depth else None
    print(f"Depth rendering: {'enabled' if render_depth else 'disabled'}")
    
    output = decoder(gaussians, context["extrinsics"], context["intrinsics"], context["near"], context["far"], image_shape, depth_mode=depth_mode)
    debug_output_decoder_output(output)
    
    # Clear cache after decoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after decoder: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    save_image(output.color[0, 0], 'color256.jpg')
    print("Saved rendered color to color256.jpg")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Memory optimization settings
    torch.set_float32_matmul_precision('high')
    
    # Set environment variable for better memory management
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    infer()