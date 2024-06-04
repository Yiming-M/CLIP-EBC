from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

from ..utils import _init_weights, make_vgg_layers, vgg_urls
from .vgg import _load_weights

EPS = 1e-6


encoder_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
decoder_cfg = [512, 512, 512, 256, 128, 64]


class CSRNet(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        decoder: nn.Module,
        reduction: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.features = features
        self.features.apply(_init_weights)
        self.decoder = decoder
        self.decoder.apply(_init_weights)

        self.encoder_reduction = 8
        self.reduction = self.encoder_reduction if reduction is None else reduction
        self.channels = 64

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.encoder_reduction != self.reduction:
            x = F.interpolate(x, scale_factor=self.encoder_reduction / self.reduction, mode="bilinear")
        x = self.decoder(x)
        return x


def csrnet(reduction: int = 8) -> CSRNet:
    model = CSRNet(
        make_vgg_layers(encoder_cfg, in_channels=3, batch_norm=False, dilation=1),
        make_vgg_layers(decoder_cfg, in_channels=512, batch_norm=False, dilation=2),
        reduction=reduction
    )
    return _load_weights(model, vgg_urls["vgg16"])

def csrnet_bn(reduction: int = 8) -> CSRNet:
    model = CSRNet(
        make_vgg_layers(encoder_cfg, in_channels=3, batch_norm=True, dilation=1),
        make_vgg_layers(decoder_cfg, in_channels=512, batch_norm=True, dilation=2),
        reduction=reduction
    )
    return _load_weights(model, vgg_urls["vgg16"])
