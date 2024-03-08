# Prepare the models to speed up loading them later
import torch
from torch import nn, Tensor
import os
from tqdm import tqdm
import json

from .utils import load


model_name_map = {
    "RN50": "resnet50",
    "RN101": "resnet101",
    "RN50x4": "resnet50x4",
    "RN50x16": "resnet50x16",
    "RN50x64": "resnet50x64",
    "ViT-B/32": "vit_b_32",
    "ViT-B/16": "vit_b_16",
    "ViT-L/14": "vit_l_14",
    "ViT-L/14@336px": "vit_l_14_336px",
}


class CLIPTextEncoderTemp(nn.Module):
    def __init__(
        self,
        clip: nn.Module,
    ) -> None:
        super().__init__()
        self.context_length = clip.context_length
        self.vocab_size = clip.vocab_size
        self.dtype = clip.dtype
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection

    def forward(self, text: Tensor) -> None:
        pass


def prepare() -> None:
    print("Preparing CLIP models...")
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    weight_dir = os.path.join(curr_dir, "weights")
    config_dir = os.path.join(curr_dir, "configs")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    device = torch.device("cpu")

    for model_name in tqdm(["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]):
        model = load(model_name, device=device).to(device)
        image_encoder = model.visual.to(device)
        text_encoder = CLIPTextEncoderTemp(model).to(device)
        torch.save(model.state_dict(), os.path.join(weight_dir, f"clip_{model_name_map[model_name]}.pth"))
        torch.save(image_encoder.state_dict(), os.path.join(weight_dir, f"clip_image_encoder_{model_name_map[model_name]}.pth"))
        torch.save(text_encoder.state_dict(), os.path.join(weight_dir, f"clip_text_encoder_{model_name_map[model_name]}.pth"))
        model_config = {
            "embed_dim": model.embed_dim,
            # vision
            "image_resolution": model.image_resolution,
            "vision_layers": model.vision_layers,
            "vision_width": model.vision_width,
            "vision_patch_size": model.vision_patch_size,
            # text
            "context_length": model.context_length,
            "vocab_size": model.vocab_size,
            "transformer_width": model.transformer_width,
            "transformer_heads": model.transformer_heads,
            "transformer_layers": model.transformer_layers,
        }
        image_encoder_config = {
            "embed_dim": model.embed_dim,
            "image_resolution": model.image_resolution,
            "vision_layers": model.vision_layers,
            "vision_width": model.vision_width,
            "vision_patch_size": model.vision_patch_size,
            "vision_heads": model.vision_heads,
        }
        text_encoder_config = {
            "embed_dim": model.embed_dim,
            "context_length": model.context_length,
            "vocab_size": model.vocab_size,
            "transformer_width": model.transformer_width,
            "transformer_heads": model.transformer_heads,
            "transformer_layers": model.transformer_layers,
        }
        with open(os.path.join(config_dir, f"clip_{model_name_map[model_name]}.json"), "w") as f:
            json.dump(model_config, f, indent=4)
        with open(os.path.join(config_dir, f"clip_image_encoder_{model_name_map[model_name]}.json"), "w") as f:
            json.dump(image_encoder_config, f, indent=4)
        with open(os.path.join(config_dir, f"clip_text_encoder_{model_name_map[model_name]}.json"), "w") as f:
            json.dump(text_encoder_config, f, indent=4)
    print("Done!")
