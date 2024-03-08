import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import List, Optional

from ..utils import _init_weights
from .csrnet import CSRNet, csrnet, csrnet_bn

EPS = 1e-6


class ContextualModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        sizes: List[int] = [1, 2, 3, 6],
    ) -> None:
        super().__init__()
        self.scales = nn.ModuleList([self.__make_scale__(in_channels, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.weight_net = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def __make_weight__(self, feature: Tensor, scale_feature: Tensor) -> Tensor:
        weight_feature = feature - scale_feature
        weight_feature = self.weight_net(weight_feature)
        return F.sigmoid(weight_feature)
    
    def __make_scale__(self, channels: int, size: int) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, feature: Tensor) -> Tensor:
        h, w = feature.shape[-2:]
        multi_scales = [F.interpolate(input=scale(feature), size=(h, w), mode="bilinear") for scale in self.scales]
        weights = [self.__make_weight__(feature, scale_feature) for scale_feature in multi_scales]
        multi_scales = sum([multi_scales[i] * weights[i] for i in range(len(weights))]) / (sum(weights) + EPS)
        overall_features = torch.cat([multi_scales, feature], dim=1)
        overall_features = self.bottleneck(overall_features)
        overall_features = self.relu(overall_features)
        return overall_features


class CANNet(nn.Module):
    def __init__(
        self,
        csrnet: CSRNet,
        sizes: List[int] = [1, 2, 3, 6],
        reduction: Optional[int] = 8,
    ) -> None:
        super().__init__()
        assert isinstance(csrnet, CSRNet), f"csrnet should be an instance of CSRNet, got {type(csrnet)}."
        assert isinstance(sizes, (tuple, list)), f"sizes should be a list or tuple, got {type(sizes)}."
        assert len(sizes) > 0, f"Expected at least one size, got {len(sizes)}."
        assert all([isinstance(size, int) for size in sizes]), f"Expected all size to be int, got {sizes}."
        self.sizes = sizes
        self.encoder_reduction = csrnet.encoder_reduction
        self.reduction = self.encoder_reduction if reduction is None else reduction

        self.features = csrnet.features
        self.decoder = csrnet.decoder
        self.decoder.apply(_init_weights)
        self.context = ContextualModule(512, 512, self.sizes)
        self.context.apply(_init_weights)

        self.channels = csrnet.channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.context(x)
        if self.encoder_reduction != self.reduction:
            x = F.interpolate(x, scale_factor=self.encoder_reduction / self.reduction, mode="bilinear")
        x = self.decoder(x)
        return x


def cannet(sizes=[1, 2, 3, 6], reduction: int = 8) -> CANNet:
    return CANNet(csrnet(), sizes=sizes, reduction=reduction)

def cannet_bn(sizes=[1, 2, 3, 6], reduction: int = 8) -> CANNet:
    return CANNet(csrnet_bn(), sizes=sizes, reduction=reduction)
