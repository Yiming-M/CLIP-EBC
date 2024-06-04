import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from einops import rearrange

from ..utils import Conv2dNormActivation, MLP
from ..utils import _log_api_usage_once


weights = {
    "vit_b_16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
    "vit_b_32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    "vit_l_16": "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
    "vit_l_32": "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
    "vit_h_14": "https://download.pytorch.org/models/vit_h_14-6kbcf7eb.pth",
}


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    def __init__(
        self,
        num_h_patches: int,
        num_w_patches: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_h_patches = num_h_patches
        self.num_w_patches = num_w_patches

        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        seq_length = num_h_patches * num_w_patches + 1  # +1 for the class token
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def _get_pos_embedding(self, n_h: int, n_w: int) -> Tensor:
        if n_h == self.num_h_patches and n_w == self.num_w_patches:
            return self.pos_embedding
        else:
            pos_embedding = self.pos_embedding[:, 1:, :]
            pos_embedding = rearrange(pos_embedding, "1 (h w) d -> 1 d h w", h=self.num_h_patches, w=self.num_w_patches)
            pos_embedding = F.interpolate(pos_embedding, size=(n_h, n_w), mode="bicubic")
            pos_embedding = rearrange(pos_embedding, "1 d h w -> 1 (h w) d")
            return torch.cat([self.pos_embedding[:, :1, :], pos_embedding], dim=1)

    def forward(self, input: Tensor, n_h: int, n_w: int) -> Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self._get_pos_embedding(n_h, n_w)
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as a feature extractor."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        # num_classes: int = 1000,  # No need for the classification head as we only need the features
        reduction: Optional[int] = None,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        # self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            image_size // patch_size,
            image_size // patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        # heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        # if representation_size is None:
        #     heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        # else:
        #     heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        #     heads_layers["act"] = nn.Tanh()
        #     heads_layers["head"] = nn.Linear(representation_size, num_classes)

        # self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        # if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
        #     fan_in = self.heads.pre_logits.in_features
        #     nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
        #     nn.init.zeros_(self.heads.pre_logits.bias)

        # if isinstance(self.heads.head, nn.Linear):
        #     nn.init.zeros_(self.heads.head.weight)
        #     nn.init.zeros_(self.heads.head.bias)

        self.encoder_reduction = self.patch_size
        self.reduction = self.encoder_reduction if reduction is None else reduction
        self.channels = hidden_dim

    def _process_input(self, x: Tensor) -> Tuple[Tensor, int, int, int]:
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        n, _, n_h, n_w = x.shape
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x, n, n_h, n_w

    def forward(self, x: Tensor) -> Tensor:
        # Reshape and permute the input tensor
        x, n, n_h, n_w = self._process_input(x)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x, n_h, n_w)  # Allows input image to be of any size.

        # Classifier "token" as used by standard language architectures
        # x = x[:, 0]

        # x = self.heads(x)

        x = x[:, 1:, :]
        x = rearrange(x, "n (h w) d -> n d h w", h=n_h, w=n_w)
        if self.encoder_reduction != self.reduction:
            x = F.interpolate(x, scale_factor=self.encoder_reduction / self.reduction, mode="bilinear")
        return x  # To be consistent with timm models


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: str,
    **kwargs: Any,
) -> VisionTransformer:
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights is not None:
        weights = load_state_dict_from_url(weights, progress=kwargs.get("progress", True))
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

    return model


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    pos_embedding: Tensor,
    interpolation_mode: str = "bicubic",
) -> Tensor:
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        Tensor: The interpolated positional embedding.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        return new_pos_embedding

    return pos_embedding


def vit_b_16(
    image_size: int = 224,
    reduction: int = 16,
    **kwargs: Any,
) -> VisionTransformer:
    vit = _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights["vit_b_16"],
        reduction=reduction,
        **kwargs,
    )
    if image_size != 224:
        vit.image_size = image_size
        new_pos_embedding = interpolate_embeddings(image_size, 16, vit.state_dict()["encoder.pos_embedding"], "bicubic")
        vit.encoder.pos_embedding = nn.Parameter(new_pos_embedding, requires_grad=True)
    return vit


def vit_b_32(
    image_size: int = 224,
    reduction: int = 32,
    **kwargs: Any,
) -> VisionTransformer:
    vit = _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights["vit_b_32"],
        reduction=reduction,
        **kwargs,
    )
    if image_size != 224:
        vit.image_size = image_size
        new_pos_embedding = interpolate_embeddings(image_size, 32, vit.state_dict()["encoder.pos_embedding"], "bicubic")
        vit.encoder.pos_embedding = nn.Parameter(new_pos_embedding, requires_grad=True)
    return vit


def vit_l_16(
    image_size: int = 224,
    reduction: int = 16,
    **kwargs: Any,
) -> VisionTransformer:
    vit = _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights["vit_l_16"],
        reduction=reduction,
        **kwargs,
    )
    if image_size != 224:
        vit.image_size = image_size
        new_pos_embedding = interpolate_embeddings(image_size, 16, vit.state_dict()["encoder.pos_embedding"], "bicubic")
        vit.encoder.pos_embedding = nn.Parameter(new_pos_embedding, requires_grad=True)
    return vit


def vit_l_32(
    image_size: int = 224,
    reduction: int = 32,
    **kwargs: Any,
) -> VisionTransformer:
    vit = _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights["vit_l_32"],
        reduction=reduction,
        **kwargs,
    )
    if image_size != 224:
        vit.image_size = image_size
        new_pos_embedding = interpolate_embeddings(image_size, 32, vit.state_dict()["encoder.pos_embedding"], "bicubic")
        vit.encoder.pos_embedding = nn.Parameter(new_pos_embedding, requires_grad=True)
    return vit


def vit_h_14(
    image_size: int = 224,
    reduction: int = 14,
    **kwargs: Any,
) -> VisionTransformer:
    vit = _vision_transformer(
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        weights=weights["vit_h_14"],
        reduction=reduction,
        **kwargs,
    )
    if image_size != 224:
        vit.image_size = image_size
        new_pos_embedding = interpolate_embeddings(image_size, 14, vit.state_dict()["encoder.pos_embedding"], "bicubic")
        vit.encoder.pos_embedding = nn.Parameter(new_pos_embedding, requires_grad=True)
    return vit

