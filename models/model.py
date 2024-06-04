import torch
from torch import nn, Tensor
import os
from typing import List, Tuple, Union, Callable
from functools import partial

from .utils import _init_weights

from . import encoder
from . import encoder_decoder
from .encoder import _timm_encoder


curr_dir = os.path.abspath(os.path.dirname(__file__))


class Regressor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.reduction = backbone.reduction

        self.regressor = nn.Sequential(
            nn.Conv2d(backbone.channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.regressor.apply(_init_weights)
        self.bins = None
        self.anchor_points = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.regressor(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        bins: List[Tuple[float, float]],
        anchor_points: List[float],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.reduction = backbone.reduction

        assert len(bins) == len(anchor_points), f"Expected bins and anchor_points to have the same length, got {len(bins)} and {len(anchor_points)}"
        assert all(len(b) == 2 for b in bins), f"Expected bins to be a list of tuples of length 2, got {bins}"
        assert all(bin[0] <= p <= bin[1] for bin, p in zip(bins, anchor_points)), f"Expected anchor_points to be within the range of the corresponding bin, got {bins} and {anchor_points}"

        self.bins = bins
        self.anchor_points = torch.tensor(anchor_points, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)

        if backbone.channels > 512:
            self.classifier = nn.Sequential(
                nn.Conv2d(backbone.channels, 512, kernel_size=1),  # serves as a linear layer for feature vectors at each pixel
                nn.ReLU(inplace=True),
                nn.Conv2d(512, len(self.bins), kernel_size=1),
            )
        else:
            self.classifier = nn.Conv2d(backbone.channels, len(self.bins), kernel_size=1)

        self.classifier.apply(_init_weights)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.backbone(x)
        x = self.classifier(x)  # shape (B, C, H, W), where C = len(bins), x is the logits

        probs = x.softmax(dim=1)  # shape (B, C, H, W)
        exp = (probs * self.anchor_points.to(x.device)).sum(dim=1, keepdim=True)  # shape (B, 1, H, W)
        if self.training:
            return x, exp
        else:
            return exp


def _get_backbone(backbone: str, input_size: int, reduction: int) -> Callable:
    assert "clip" not in backbone, f"This function does not support CLIP model, got {backbone}"

    if backbone in ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]:
        return partial(getattr(encoder, backbone), image_size=input_size, reduction=reduction)
    elif backbone in ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]:
        return partial(getattr(encoder, backbone), reduction=reduction)
    elif backbone in ["vgg11_ae", "vgg11_bn_ae", "vgg13_ae", "vgg13_bn_ae", "vgg16_ae", "vgg16_bn_ae", "vgg19_ae", "vgg19_bn_ae"]:
        return partial(getattr(encoder_decoder, backbone), reduction=reduction)
    elif backbone in ["resnet18_ae", "resnet34_ae", "resnet50_ae", "resnet101_ae", "resnet152_ae"]:
        return partial(getattr(encoder_decoder, backbone), reduction=reduction)
    elif backbone in ["cannet", "cannet_bn", "csrnet", "csrnet_bn"]:
        return partial(getattr(encoder_decoder, backbone), reduction=reduction)
    else:
        return partial(_timm_encoder, backbone=backbone, reduction=reduction)


def _regressor(
    backbone: str,
    input_size: int,
    reduction: int,
) -> Regressor:
    backbone = _get_backbone(backbone.lower(), input_size, reduction)
    return Regressor(backbone())


def _classifier(
    backbone: nn.Module,
    input_size: int,
    reduction: int,
    bins: List[Tuple[float, float]],
    anchor_points: List[float],
) -> Classifier:
    backbone = _get_backbone(backbone.lower(), input_size, reduction)
    return Classifier(backbone(), bins, anchor_points)
