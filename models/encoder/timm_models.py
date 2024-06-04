from timm import create_model, list_models
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

from warnings import warn


class TIMMEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        reduction: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert backbone in list_models(), f"Backbone {backbone} not available in timm"
        encoder = create_model(backbone, pretrained=True, features_only=True, out_indices=[-1])
        encoder_reduction = encoder.feature_info.reduction()[-1]

        if reduction <= 16:
            if "resnet" in backbone:
                if "resnet18" in backbone or "resnet34" in backbone:
                    encoder.layer4[0].conv1.stride = (1, 1)
                    encoder.layer4[0].downsample[0].stride = (1, 1)
                else:
                    encoder.layer4[0].conv2.stride = (1, 1)
                    encoder.layer4[0].downsample[0].stride = (1, 1)
                encoder_reduction = encoder_reduction // 2

            elif "mobilenetv2" in backbone:
                encoder.blocks[5][0].conv_dw.stride = (1, 1)
                encoder_reduction = encoder_reduction // 2

            elif "densenet" in backbone:
                encoder.features_transition3.pool = nn.Identity()
                encoder_reduction = encoder_reduction // 2

            else:
                warn(f"Reduction for {backbone} not handled. Using default reduction of {encoder_reduction}")

        self.encoder = encoder
        self.encoder_reduction = encoder_reduction
        self.reduction = self.encoder_reduction if reduction is None else reduction
        self.channels = self.encoder.feature_info.channels()[-1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)[-1]
        if self.encoder_reduction != self.reduction:
            x = F.interpolate(x, scale_factor=self.encoder_reduction / self.reduction, mode="bilinear")
        return x


def _timm_encoder(backbone: str, reduction: Optional[int] = None) -> TIMMEncoder:
    return TIMMEncoder(backbone, reduction)
