import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import os
import json
from typing import List, Tuple, Union, Optional

from . import _clip
from ..utils import _init_weights, make_resnet_layers, Bottleneck, BasicBlock
from .utils import format_count

curr_dir = os.path.abspath(os.path.dirname(__file__))


# resnet50: reduction, channels, embed_dim = 32, 2048, 1024
# resnet101: reduction, channels, embed_dim = 32, 2048, 512
# resnet50x4: reduction, channels, embed_dim = 32, 2560, 640
# resnet50x16: reduction, channels, embed_dim = 32, 3072, 768
# resnet50x64: reduction, channels, embed_dim = 32, 4096, 1024
# vit_b_32: reduction, channels, embed_dim = 32, 768, 512
# vit_b_16: reduction, channels, embed_dim = 16, 768, 512
# vit_l_14: reduction, channels, embed_dim = 14, 1024, 768
# vit_l_14_336px: reduction, channels, embed_dim = 14, 1024, 768


class VanillaCLIP(nn.Module):
    def __init__(
        self,
        backbone: str,
        input_size: int,
        bins: List[Tuple[float, float]],
        anchor_points: List[float],
        reduction: Optional[int] = None,
        decoder_block: Optional[nn.Module] = None,
        decoder_cfg: Optional[List[Union[str, int]]] = None,
        freeze_text_encoder: bool = True,
        prompt_type: str = "number",
    ) -> None:
        super().__init__()
        assert prompt_type in ["number", "word"], f"Expected prompt_type to be 'number' or 'word', got {prompt_type}"
        self.prompt_type = prompt_type

        self.image_encoder = getattr(_clip, f"{backbone}_img")(input_size=input_size, features_only=True, out_indices=(-1,), reduction=reduction)
        self.text_encoder = getattr(_clip, f"{backbone}_txt")()
        self.freeze_text_encoder = freeze_text_encoder
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.encoder_reduction = self.image_encoder.reduction
        self.reduction = self.encoder_reduction if reduction is None else reduction
        self.channels = self.image_encoder.channels
        self.clip_embed_dim = self.image_encoder.clip_embed_dim

        self.bins = bins
        self.anchor_points = torch.tensor(anchor_points, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)

        if decoder_cfg is not None:
            self.image_decoder = make_resnet_layers(decoder_block, decoder_cfg, in_channels=self.channels, expansion=1, dilation=1)
            self.image_decoder.apply(_init_weights)
            self.channels = decoder_cfg[-1]
        else:
            self.image_decoder = nn.Identity()

        if self.channels != self.clip_embed_dim:
            self.projection = nn.Conv2d(in_channels=self.channels, out_channels=self.clip_embed_dim, kernel_size=1)
            self.projection.apply(_init_weights)
        else:
            self.projection = nn.Identity()

        self._get_text_prompts()
        self._tokenize_text_prompts()

        if self.freeze_text_encoder:
            self._extract_text_features()
        else:
            self.text_features = None

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

    def _get_text_prompts(self) -> None:
        bins = [b[0] if b[0] == b[1] else b for b in self.bins]
        self.text_prompts = [format_count(b, self.prompt_type) for b in bins]
        print(f"Initialized model with text prompts: {self.text_prompts}")

    def _tokenize_text_prompts(self) -> None:
        self.text_prompts = _clip.tokenize(self.text_prompts)

    def _extract_text_features(self) -> None:
        with torch.no_grad():
            self.text_features = self.text_encoder(self.text_prompts)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        device = x.device

        x = self.image_encoder(x)
        if self.reduction != self.encoder_reduction:
            x = F.interpolate(x, scale_factor=self.encoder_reduction / self.reduction, mode="bilinear")
        x = self.image_decoder(x)
        x = self.projection(x)

        image_features = x.permute(0, 2, 3, 1)  # shape (B, H, W, C)
        text_features = self.text_encoder(self.text_prompts.to(device)) if self.text_features is None else self.text_features.to(device)  # shape (N, C)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # (B, H, W, N), logits per image
        logits = logits.permute(0, 3, 1, 2)  # (B, N, H, W)

        probs = logits.softmax(dim=1)
        exp = (probs * self.anchor_points.to(x.device)).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        if self.training:
            return logits, exp
        else:
            return exp


def _vanilla_clip(
    backbone: str,
    input_size: int,
    bins: List[Tuple[float, float]],
    anchor_points: List[float],
    reduction: Optional[int] = None,
    decoding: bool = True,
    decoder_cfg: Optional[List[Union[str, int]]] = None,
    freeze_text_encoder: bool = True,
    prompt_type: str = "number",
) -> VanillaCLIP:
    resnets = ["resnet50", "resnet50x4", "resnet50x16", "resnet50x64", "resnet101"]
    vits = ["vit_b_16", "vit_b_32", "vit_l_14"]
    assert backbone in resnets + vits, f"Backbone should be in {resnets + vits}, got {backbone}"

    if decoding:
        if backbone in resnets:
            decoder_block = Bottleneck
            if decoder_cfg is None:
                if backbone == "resnet50":
                    decoder_cfg = [2048]
                elif backbone == "resnet50x4":
                    decoder_cfg = [1280]
                elif backbone == "resnet50x16":
                    decoder_cfg = [1536]
                elif backbone == "resnet50x64":
                    decoder_cfg = [2048]
                else:  # backbone == "resnet101"
                    decoder_cfg = [2048, 1024]
        else:
            decoder_block = BasicBlock
            if decoder_cfg is None:
                if backbone == "vit_b_16":
                    decoder_cfg = [768]
                elif backbone == "vit_b_32":
                    decoder_cfg = [768]
                else:  # backbone == "vit_l_14"
                    decoder_cfg = [1024]
    else:
        decoder_block = None
        decoder_cfg = None

    return VanillaCLIP(
        backbone=backbone,
        input_size=input_size,
        bins=bins,
        anchor_points=anchor_points,
        reduction=reduction,
        decoder_block=decoder_block,
        decoder_cfg=decoder_cfg,
        freeze_text_encoder=freeze_text_encoder,
        prompt_type=prompt_type,
    )
