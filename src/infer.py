
from dataclasses import dataclass
from typing import Literal
import warnings
import torch
from pathlib import Path
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
from src.model.types import Gaussians
from src.dataset.types import BatchedViews
from src.context.context_provider import ContextProvider
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.decoder import DecoderCfg
from src.views import BatchedRenderViews, ViewsProviderCfg, get_views_provider, ViewsProvider

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

def load_encoder_from_pretrained(encoder, pretrained_model):
    if isinstance(pretrained_model, str) or isinstance(pretrained_model, Path):
        pretrained_model = torch.load(str(pretrained_model), map_location='cpu')
    if 'state_dict' in pretrained_model:
        pretrained_model = pretrained_model['state_dict']
    encoder_weights = {}
    for key, value in pretrained_model.items():
        if key.startswith("encoder."):
            encoder_key = key[8:]  # Remove "encoder." prefix
            encoder_weights[encoder_key] = value
    encoder.load_state_dict(encoder_weights)
    return encoder

@dataclass
class InferCfg:
    type: Literal["infer"]
    context_provider: ContextProviderCfg
    views_provider: ViewsProviderCfg
    model: ModelCfg
    # dataset: DatasetCfg 


class InferApp:
    def __init__(self, cfg: InferCfg):
        self.cfg = cfg
        self.context_provider = self.load_context_provider(cfg.context_provider)
        self.views_provider = self.load_views_provider(cfg.views_provider)
        self.encoder, self.encoder_visualizer = self.load_encoder(cfg.model.encoder)
        self.decoder = self.load_decoder(cfg.model.decoder)
        self.ui = self.load_gradio()

    def load_gradio(self):
        print("load gradio")
        import gradio as gr
        with gr.Blocks() as ui:
            x = gr.State(0.0)
            y = gr.State(0.0)
            with gr.Row():
                gr.Interface(self.infer,
                    inputs=None,
                    outputs="image")
            with gr.Row():
                x_display = gr.Number(value=0.0, label="x", interactive=False)
                y_display = gr.Number(value=0.0, label="y", interactive=False)

            with gr.Row():
                btn_a = gr.Button("A")
                btn_s = gr.Button("S")
                btn_d = gr.Button("D")
                btn_w = gr.Button("W")
            
            def increment(x):
                x = x + 0.01
                return x, x
            def decrement(x):
                x = x - 0.01
                return x, x

            ui.x = x
            ui.y = y

            btn_a.click(decrement, inputs=x, outputs=[x, x_display])
            btn_s.click(decrement, inputs=y, outputs=[y, y_display])
            btn_d.click(increment, inputs=x, outputs=[x, x_display])
            btn_w.click(increment, inputs=y, outputs=[y, y_display])

        return ui

    def load_context_provider(self, cfg: ContextProviderCfg) -> ContextProvider:
        print("load context provider")
        return get_context_provider(cfg)
    
    def load_views_provider(self, cfg: ViewsProviderCfg) -> ViewsProvider:
        print("load views provider")
        return get_views_provider(cfg)
    
    def load_encoder(self, cfg: EncoderCfg) -> Encoder:
        print("load encoder")
        encoder, encoder_visualizer = get_encoder(cfg)
        encoder = load_encoder_from_pretrained(encoder, 'pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth')
        print(cyan(f"Loaded pretrained weights: pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth"))
        encoder.eval()
        encoder.cuda()
        return encoder, encoder_visualizer
    
    def load_decoder(self, cfg: DecoderCfg) -> Decoder:
        print("load decoder")
        background_color = [0.0, 0.0, 0.0]
        decoder = get_decoder(cfg, background_color)
        decoder.eval()
        decoder.cuda()
        return decoder

    def load_context(self) -> BatchedViews:
        print("load context")
        context = self.context_provider.get_context()
        debug_output_context(context)
        save_image(context["image"][0, 0], 'context_image0.jpg')
        save_image(context["image"][0, 1], 'context_image1.jpg')
        print("Saved context images to context_image0.jpg and context_image1.jpg")
        # Move context data to GPU before passing to encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        context = {
            "extrinsics": context["extrinsics"].to(device),
            "intrinsics": context["intrinsics"].to(device),
            "image": context["image"].to(device),
            "near": context["near"].to(device),
            "far": context["far"].to(device),
            "index": context["index"].to(device),
        }
        print(f"Moved context data to device: {device}")
        return context
    

    
    def encode(self, context: BatchedViews) -> Gaussians:
        return self.encoder(context, 0, False)["gaussians"]
    
    def decode(self, gaussians: Gaussians, views: BatchedRenderViews):
        print("")
        print("rendering gaussians")
        image_shape = (256, 256)  # Reduced from (256, 256)
        print(f"Using image shape: {image_shape}")

        output = self.decoder(gaussians, views["extrinsics"], views["intrinsics"], views["near"], views["far"], image_shape, depth_mode=None)
        # debug_output_decoder_output(output)
        
        # Clear cache after decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory after decoder: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        print("")
        print("saving results")
        save_image(output.color[0, 0], 'color256.jpg')
        print("Saved rendered color to color256.jpg")
        return output.color[0, 0]
    
    def infer(self):
        context = self.load_context()
        # views = BatchedRenderViews(
        #     extrinsics=context["extrinsics"],
        #     intrinsics=context["intrinsics"],
        #     near=context["near"],
        #     far=context["far"],
        # )
        views = self.views_provider.get_views()
        gaussians = self.encode(context)
        color = self.decode(gaussians, views)
        color = color.permute(1, 2, 0).detach().cpu().numpy()
        return color

    def run(self):
        print("run")
        self.ui.launch()



@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="infer",
)
def infer(cfg_dict: DictConfig):
    print("infer")
    # print(OmegaConf.to_yaml(cfg_dict))
    # exit()

    cfg = load_typed_config(cfg_dict, InferCfg)

    app = InferApp(cfg)
    app.run()
    exit()






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
        
        # Force garbage collection
        import gc
        gc.collect()

    infer()