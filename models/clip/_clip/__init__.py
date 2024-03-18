import torch
import os
from typing import Tuple, Optional, Any, Union
import json

from .utils import tokenize, transform
from .prepare import prepare
from .text_encoder import CLIPTextEncoder
from .image_encoder import ModifiedResNet, VisionTransformer
from .model import CLIP


curr_dir = os.path.dirname(os.path.abspath(__file__))

clip_model_names = [
    "clip_resnet50",
    "clip_resnet101",
    "clip_resnet50x4",
    "clip_resnet50x16",
    "clip_resnet50x64",
    "clip_vit_b_32",
    "clip_vit_b_16",
    "clip_vit_l_14",
    "clip_vit_l_14_336px",
]

clip_image_encoder_names = [f"clip_image_encoder_{name[5:]}" for name in clip_model_names]
clip_text_encoder_names = [f"clip_text_encoder_{name[5:]}" for name in clip_model_names]


for name in clip_model_names + clip_image_encoder_names + clip_text_encoder_names:
    model_weights_path = os.path.join(curr_dir, "weights", f"{name}.pth")
    model_config_path = os.path.join(curr_dir, "configs", f"{name}.json")
    if not os.path.exists(os.path.join(curr_dir, "weights", f"{name}.pth")) or not os.path.exists(os.path.join(curr_dir, "configs", f"{name}.json")):
        prepare()
        break


for name in clip_model_names + clip_image_encoder_names + clip_text_encoder_names:
    assert os.path.exists(os.path.join(curr_dir, "weights", f"{name}.pth")), f"Missing {name}.pth in weights folder. Please run models/clip/prepare.py to download the weights."
    assert os.path.exists(os.path.join(curr_dir, "configs", f"{name}.json")), f"Missing {name}.json in configs folder. Please run models/clip/prepare.py to download the configs."


def _clip(name: str, input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    with open(os.path.join(curr_dir, "configs", f"clip_{name}.json"), "r") as f:
        config = json.load(f)

    model = CLIP(
        embed_dim=config["embed_dim"],
        # vision
        image_resolution=config["image_resolution"],
        vision_layers=config["vision_layers"],
        vision_width=config["vision_width"],
        vision_patch_size=config["vision_patch_size"],
        # text
        context_length=config["context_length"],
        vocab_size=config["vocab_size"],
        transformer_width=config["transformer_width"],
        transformer_heads=config["transformer_heads"],
        transformer_layers=config["transformer_layers"]
    )
    state_dict = torch.load(os.path.join(curr_dir, "weights", f"clip_{name}.pth"), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    if input_size is not None:
        input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        if name.startswith("vit"):
            model.visual.adjust_pos_embed(*input_size)

    return model


def _resnet(
    name: str,
    reduction: int = 32,
    features_only: bool = False,
    out_indices: Optional[Tuple[int, ...]] = None,
    **kwargs: Any
) -> ModifiedResNet:
    with open(os.path.join(curr_dir, "configs", f"clip_image_encoder_{name}.json"), "r") as f:
        config = json.load(f)
    model = ModifiedResNet(
        layers=config["vision_layers"],
        output_dim=config["embed_dim"],
        input_resolution=config["image_resolution"],
        width=config["vision_width"],
        heads=config["vision_heads"],
        features_only=features_only,
        out_indices=out_indices,
        reduction=reduction
    )
    state_dict = torch.load(os.path.join(curr_dir, "weights", f"clip_image_encoder_{name}.pth"), map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        print(f"All keys matched successfully.")

    return model


def _vit(name: str, features_only: bool = False, input_size: Optional[Union[int, Tuple[int, int]]] = None, **kwargs: Any) -> VisionTransformer:
    with open(os.path.join(curr_dir, "configs", f"clip_image_encoder_{name}.json"), "r") as f:
        config = json.load(f)
    model = VisionTransformer(
        input_resolution=config["image_resolution"],
        patch_size=config["vision_patch_size"],
        output_dim=config["embed_dim"],
        width=config["vision_width"],
        layers=config["vision_layers"],
        heads=config["vision_heads"],
        features_only=features_only
    )
    state_dict = torch.load(os.path.join(curr_dir, "weights", f"clip_image_encoder_{name}.pth"), map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        print(f"All keys matched successfully.")

    if input_size is not None:
        input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        model.adjust_pos_embed(*input_size)
    return model


def _text_encoder(name: str) -> CLIPTextEncoder:
    with open(os.path.join(curr_dir, "configs", f"clip_text_encoder_{name}.json"), "r") as f:
        config = json.load(f)
    model = CLIPTextEncoder(
        embed_dim=config["embed_dim"],
        context_length=config["context_length"],
        vocab_size=config["vocab_size"],
        transformer_width=config["transformer_width"],
        transformer_heads=config["transformer_heads"],
        transformer_layers=config["transformer_layers"]
    )
    state_dict = torch.load(os.path.join(curr_dir, "weights", f"clip_text_encoder_{name}.pth"), map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        print(f"All keys matched successfully.")

    return model



# CLIP models
def resnet50_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("resnet50", input_size)

def resnet101_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("resnet101", input_size)

def resnet50x4_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("resnet50x4", input_size)

def resnet50x16_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("resnet50x16", input_size)

def resnet50x64_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("resnet50x64", input_size)

def vit_b_32_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("vit_b_32", input_size)

def vit_b_16_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("vit_b_16", input_size)

def vit_l_14_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("vit_l_14", input_size)

def vit_l_14_336px_clip(input_size: Optional[Union[int, Tuple[int, int]]] = None) -> CLIP:
    return _clip("vit_l_14_336px", input_size)


# CLIP image encoders
def resnet50_img(features_only: bool = False, out_indices: Optional[Tuple[int, ...]] = None, **kwargs: Any) -> ModifiedResNet:
    return _resnet("resnet50", features_only=features_only, out_indices=out_indices, **kwargs)

def resnet101_img(features_only: bool = False, out_indices: Optional[Tuple[int, ...]] = None, **kwargs: Any) -> ModifiedResNet:
    return _resnet("resnet101", features_only=features_only, out_indices=out_indices, **kwargs)

def resnet50x4_img(features_only: bool = False, out_indices: Optional[Tuple[int, ...]] = None, **kwargs: Any) -> ModifiedResNet:
    return _resnet("resnet50x4", features_only=features_only, out_indices=out_indices, **kwargs)

def resnet50x16_img(features_only: bool = False, out_indices: Optional[Tuple[int, ...]] = None, **kwargs: Any) -> ModifiedResNet:
    return _resnet("resnet50x16", features_only=features_only, out_indices=out_indices, **kwargs)

def resnet50x64_img(features_only: bool = False, out_indices: Optional[Tuple[int, ...]] = None, **kwargs: Any) -> ModifiedResNet:
    return _resnet("resnet50x64", features_only=features_only, out_indices=out_indices, **kwargs)

def vit_b_32_img(features_only: bool = False, input_size: Optional[Union[int, Tuple[int, int]]] = None, **kwargs: Any) -> VisionTransformer:
    return _vit("vit_b_32", features_only=features_only, input_size=input_size, **kwargs)

def vit_b_16_img(features_only: bool = False, input_size: Optional[Union[int, Tuple[int, int]]] = None, **kwargs: Any) -> VisionTransformer:
    return _vit("vit_b_16", features_only=features_only, input_size=input_size, **kwargs)

def vit_l_14_img(features_only: bool = False, input_size: Optional[Union[int, Tuple[int, int]]] = None, **kwargs: Any) -> VisionTransformer:
    return _vit("vit_l_14", features_only=features_only, input_size=input_size, **kwargs)

def vit_l_14_336px_img(features_only: bool = False, input_size: Optional[Union[int, Tuple[int, int]]] = None, **kwargs: Any) -> VisionTransformer:
    return _vit("vit_l_14_336px", features_only=features_only, input_size=input_size, **kwargs)


# CLIP text encoders
def resnet50_txt() -> CLIPTextEncoder:
    return _text_encoder("resnet50")

def resnet101_txt() -> CLIPTextEncoder:
    return _text_encoder("resnet101")

def resnet50x4_txt() -> CLIPTextEncoder:
    return _text_encoder("resnet50x4")

def resnet50x16_txt() -> CLIPTextEncoder:
    return _text_encoder("resnet50x16")

def resnet50x64_txt() -> CLIPTextEncoder:
    return _text_encoder("resnet50x64")

def vit_b_32_txt() -> CLIPTextEncoder:
    return _text_encoder("vit_b_32")

def vit_b_16_txt() -> CLIPTextEncoder:
    return _text_encoder("vit_b_16")

def vit_l_14_txt() -> CLIPTextEncoder:
    return _text_encoder("vit_l_14")

def vit_l_14_336px_txt() -> CLIPTextEncoder:
    return _text_encoder("vit_l_14_336px")


__all__ = [
    # utils
    "tokenize",
    "transform",
    # clip models
    "resnet50_clip",
    "resnet101_clip",
    "resnet50x4_clip",
    "resnet50x16_clip",
    "resnet50x64_clip",
    "vit_b_32_clip",
    "vit_b_16_clip",
    "vit_l_14_clip",
    "vit_l_14_336px_clip",
    # clip image encoders
    "resnet50_img",
    "resnet101_img",
    "resnet50x4_img",
    "resnet50x16_img",
    "resnet50x64_img",
    "vit_b_32_img",
    "vit_b_16_img",
    "vit_l_14_img",
    "vit_l_14_336px_img",
    # clip text encoders
    "resnet50_txt",
    "resnet101_txt",
    "resnet50x4_txt",
    "resnet50x16_txt",
    "resnet50x64_txt",
    "vit_b_32_txt",
    "vit_b_16_txt",
    "vit_l_14_txt",
    "vit_l_14_336px_txt",
]
