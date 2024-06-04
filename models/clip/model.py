import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import os
import math
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

resnet_backbones = ["resnet50", "resnet101", "resnet50x4", "resnet50x16", "resnet50x64"]
vit_backbones = ["vit_b_16", "vit_b_32", "vit_l_14", "vit_l_14_336px"]


class CLIP_EBC(nn.Module):
    def __init__(
        self,
        backbone: str,
        bins: List[Tuple[float, float]],
        anchor_points: List[float],
        reduction: Optional[int] = None,
        freeze_text_encoder: bool = True,
        prompt_type: str = "number",
        input_size: Optional[int] = None,
        num_vpt: Optional[int] = None,
        deep_vpt: Optional[bool] = None,
        vpt_drop: Optional[float] = None,
        decoder_block: Optional[nn.Module] = None,
        decoder_cfg: Optional[List[Union[str, int]]] = None,
    ) -> None:
        super().__init__()
        assert backbone in resnet_backbones + vit_backbones, f"Backbone should be in {resnet_backbones + vit_backbones}, got {backbone}"
        self.backbone = backbone

        # Image encoder
        if backbone in resnet_backbones:
            self.image_encoder = getattr(_clip, f"{backbone}_img")(features_only=True, out_indices=(-1,), reduction=reduction)

        else:
            assert input_size is not None, "Expected input_size to be an integer, got None."
            assert num_vpt is not None, "Expected num_vpt to be an integer, got None."
            assert deep_vpt is not None, "Expected deep_vpt to be a boolean, got None."
            assert vpt_drop is not None, "Expected vpt_drop to be a float, got None."

            self.image_encoder = getattr(_clip, f"{backbone}_img")(features_only=True, input_size=input_size)
            self.image_encoder_depth = len(self.image_encoder.transformer.resblocks)

            # Use VPT. Freeze the image encoder.
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            self.num_vpt = num_vpt
            self.deep_vpt = deep_vpt

            patch_size = self.image_encoder.patch_size[0]
            val = math.sqrt(6. / float(3 * patch_size + self.image_encoder.channels))

            for idx in range(self.image_encoder_depth if self.deep_vpt else 1):
                setattr(self, f"vpt_{idx}", nn.Parameter(torch.empty(self.num_vpt, self.image_encoder.channels)))
                nn.init.uniform_(getattr(self, f"vpt_{idx}"), -val, val)
                setattr(self, f"vpt_drop_{idx}", nn.Dropout(vpt_drop) if vpt_drop > 0 else nn.Identity())

        self.encoder_reduction = self.image_encoder.reduction
        self.reduction = self.encoder_reduction if reduction is None else reduction
        self.channels = self.image_encoder.channels
        self.clip_embed_dim = self.image_encoder.clip_embed_dim

        if decoder_cfg is not None:
            assert decoder_block is not None, "Expected decoder_block to be a nn.Module, got None."
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

        # Text encoder
        assert prompt_type in ["number", "word"], f"Expected prompt_type to be 'number' or 'word', got {prompt_type}"
        self.prompt_type = prompt_type
        self.text_encoder = getattr(_clip, f"{backbone}_txt")()
        self.freeze_text_encoder = freeze_text_encoder
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.bins = bins
        self.anchor_points = torch.tensor(anchor_points, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)

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

    def _prepare_vpt(self, layer: int, batch_size: int, device: torch.device) -> Tensor:
        if not self.deep_vpt:
            assert layer == 0, f"Expected layer to be 0 when using Shallow Visual Prompt Tuning, got {layer}"

        vpt = getattr(self, f"vpt_{layer}").to(device)
        vpt = vpt.unsqueeze(0).expand(batch_size, -1, -1)
        vpt = getattr(self, f"vpt_drop_{layer}")(vpt)
        vpt = vpt.permute(1, 0, 2)  # (num_vpt, batch_size, hidden_dim)
        assert vpt.shape[1] == batch_size, f"Expected the VPT to have the shape [L_vis B C], got {vpt.shape}."
        return vpt

    def _forward_vpt(self, x: Tensor) -> Tuple[Tensor]:
        device = x.device
        batch_size, _, height, width = x.shape
        num_h_patches, num_w_patches = height // self.image_encoder.patch_size[0], width // self.image_encoder.patch_size[1]

        image_features = self.image_encoder.conv1(x)
        image_features = image_features.reshape(batch_size, image_features.shape[1], -1)
        image_features = image_features.permute(0, 2, 1)  # (B, num_patches, C)
        image_features = torch.cat([
            self.image_encoder.class_embedding + torch.zeros(batch_size, 1, image_features.shape[-1], dtype=image_features.dtype, device=device),
            image_features,
        ], dim=1)  # (B, num_patches + 1, C)

        pos_embedding = self.image_encoder._interpolate_pos_embed(num_h_patches, num_w_patches)
        image_features = image_features + pos_embedding
        image_features = self.image_encoder.ln_pre(image_features)
        image_features = image_features.permute(1, 0, 2)  # (num_patches + 1, B, C)
        assert image_features.shape[0] == num_h_patches * num_w_patches + 1 and image_features.shape[1] == batch_size, f"Expected image_features to have shape [num_patches + 1, B, C], got {image_features.shape}."

        vpt = self._prepare_vpt(0, batch_size, device)
        for idx in range(self.image_encoder_depth):
            # assemble
            image_features = torch.cat([
                image_features[:1, :, :],  # CLS token
                vpt,
                image_features[1:, :, :],
            ], dim=0)

            # transformer
            image_features = self.image_encoder.transformer.resblocks[idx](image_features)

            # disassemble
            if idx < self.image_encoder_depth - 1:
                if self.deep_vpt:
                    vpt = self._prepare_vpt(idx + 1, batch_size, device)
                else:
                    vpt = image_features[1: (self.num_vpt + 1), :, :]

            image_features = torch.cat([
                image_features[:1, :, :],  # CLS token
                image_features[(self.num_vpt + 1):, :, :],
            ], dim=0)
            
        image_features = image_features.permute(1, 0, 2)  # (B, num_patches + 1, C)
        image_features = self.image_encoder.ln_post(image_features)
        image_features = image_features[:, 1:, :].permute(0, 2, 1)  # (B, C, num_patches)
        image_features = image_features.reshape(batch_size, -1, num_h_patches, num_w_patches)
        return image_features

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        device = x.device

        x = self.image_encoder(x) if self.backbone in resnet_backbones else self._forward_vpt(x)
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


def _clip_ebc(
    backbone: str,
    bins: List[Tuple[float, float]],
    anchor_points: List[float],
    reduction: Optional[int] = None,
    freeze_text_encoder: bool = True,
    prompt_type: str = "number",
    input_size: Optional[int] = None,
    num_vpt: Optional[int] = None,
    deep_vpt: Optional[bool] = None,
    vpt_drop: Optional[float] = None,
    decoder_block: Optional[nn.Module] = None,
    decoder_cfg: Optional[List[Union[str, int]]] = None
) -> CLIP_EBC:
    if backbone in resnet_backbones:
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

    return CLIP_EBC(
        backbone=backbone,
        bins=bins,
        anchor_points=anchor_points,
        reduction=reduction,
        freeze_text_encoder=freeze_text_encoder,
        prompt_type=prompt_type,
        input_size=input_size,
        num_vpt=num_vpt,
        deep_vpt=deep_vpt,
        vpt_drop=vpt_drop,
        decoder_block=decoder_block,
        decoder_cfg=decoder_cfg,
    )
