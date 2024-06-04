from torch import nn, Tensor
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Optional

from ..utils import make_vgg_layers, vgg_cfgs, vgg_urls


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        reduction: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.features = features
        self.encoder_reduction = 16
        self.reduction = self.encoder_reduction if reduction is None else reduction
        self.channels = 512

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.encoder_reduction != self.reduction:
            x = F.interpolate(x, scale_factor=self.encoder_reduction / self.reduction, mode="bilinear")
        return x


def _load_weights(model: VGG, url: str) -> VGG:
    state_dict = load_state_dict_from_url(url)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Loading pre-trained weights")
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")
    return model


def vgg11(reduction: int = 8) -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["A"]), reduction=reduction)
    return _load_weights(model, vgg_urls["vgg11"])

def vgg11_bn(reduction: int = 8) -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["A"], batch_norm=True), reduction=reduction)
    return _load_weights(model, vgg_urls["vgg11_bn"])

def vgg13(reduction: int = 8) -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["B"]), reduction=reduction)
    return _load_weights(model, vgg_urls["vgg13"])

def vgg13_bn(reduction: int = 8) -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["B"], batch_norm=True), reduction=reduction)
    return _load_weights(model, vgg_urls["vgg13_bn"])

def vgg16(reduction: int = 8) -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["D"]), reduction=reduction)
    return _load_weights(model, vgg_urls["vgg16"])

def vgg16_bn(reduction: int = 8) -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["D"], batch_norm=True), reduction=reduction)
    return _load_weights(model, vgg_urls["vgg16_bn"])

def vgg19(reduction: int = 8) -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["E"]), reduction=reduction)
    return _load_weights(model, vgg_urls["vgg19"])

def vgg19_bn(reduction: int = 8) -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["E"], batch_norm=True), reduction=reduction)
    return _load_weights(model, vgg_urls["vgg19_bn"])
