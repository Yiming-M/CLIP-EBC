from torch import nn, Tensor
import torch.nn.functional as F
import timm
from typing import Union, Optional

from ..utils import BasicBlock, Bottleneck, make_resnet_layers
from ..utils import _init_weights


model_configs = {
    "resnet18.tv_in1k": {
        "decoder_channels": [512, 256, 128],
    },
    "resnet34.tv_in1k": {
        "decoder_channels": [512, 256, 128],
    },
    "resnet50.tv_in1k": {
        "decoder_channels": [512, 256, 256, 128],
    },
    "resnet101.tv_in1k": {
        "decoder_channels": [512, 512, 256, 256, 128],
    },
    "resnet152.tv_in1k": {
        "decoder_channels": [512, 512, 512, 256, 256, 128],
    },
}


class ResNet(nn.Module):
    def __init__(
        self,
        decoder_block: Union[BasicBlock, Bottleneck],
        backbone: str = "resnet34.tv_in1k",
        reduction: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert backbone in model_configs.keys(), f"Backbone should be in {model_configs.keys()}"
        config = model_configs[backbone]
        encoder = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=(-1,))
        encoder_reduction = encoder.feature_info.reduction()[-1]

        if reduction <= 16:
            if "resnet18" in backbone or "resnet34" in backbone:
                encoder.layer4[0].conv1.stride = (1, 1)
                encoder.layer4[0].downsample[0].stride = (1, 1)
            else:
                encoder.layer4[0].conv2.stride = (1, 1)
                encoder.layer4[0].downsample[0].stride = (1, 1)
            encoder_reduction = encoder_reduction // 2

        self.encoder = encoder
        self.encoder_reduction = encoder_reduction

        encoder_out_channels = self.encoder.feature_info.channels()[-1]

        decoder_channels = config["decoder_channels"]
        self.decoder = make_resnet_layers(
            block=decoder_block,
            cfg=decoder_channels,
            in_channels=encoder_out_channels,
            dilation=1,
            expansion=1,
        )
        self.decoder.apply(_init_weights)

        self.reduction = self.encoder_reduction if reduction is None else reduction
        self.channels = decoder_channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)[-1]
        if self.encoder_reduction != self.reduction:
            x = F.interpolate(x, scale_factor=self.encoder_reduction / self.reduction, mode="bilinear")
        x = self.decoder(x)

        return x


def resnet18(reduction: int = 32) -> ResNet:
    return ResNet(decoder_block=BasicBlock, backbone="resnet18.tv_in1k", reduction=reduction)


def resnet34(reduction: int = 32) -> ResNet:
    return ResNet(decoder_block=BasicBlock, backbone="resnet34.tv_in1k", reduction=reduction)


def resnet50(reduction: int = 32) -> ResNet:
    return ResNet(decoder_block=Bottleneck, backbone="resnet50.tv_in1k", reduction=reduction)


def resnet101(reduction: int = 32) -> ResNet:
    return ResNet(decoder_block=Bottleneck, backbone="resnet101.tv_in1k", reduction=reduction)


def resnet152(reduction: int = 32) -> ResNet:
    return ResNet(decoder_block=Bottleneck, backbone="resnet152.tv_in1k", reduction=reduction)
