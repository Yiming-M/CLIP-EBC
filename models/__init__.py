from typing import List, Tuple, Optional, Any, Union

from .model import _classifier, _regressor, Classifier, Regressor
from .clip import _clip_ebc, CLIP_EBC


clip_names = ["resnet50", "resnet50x4", "resnet50x16", "resnet50x64", "resnet101", "vit_b_16", "vit_b_32", "vit_l_14"]


def get_model(
    backbone: str,
    input_size: int,
    reduction: int,
    bins: Optional[List[Tuple[float, float]]] = None,
    anchor_points: Optional[List[float]] = None,
    **kwargs: Any,
) -> Union[Regressor, Classifier, CLIP_EBC]:
    backbone = backbone.lower()
    if "clip" in backbone:
        backbone = backbone[5:]
        assert backbone in clip_names, f"Expected backbone to be in {clip_names}, got {backbone}"
        return _clip_ebc(
            backbone=backbone,
            input_size=input_size,
            reduction=reduction,
            bins=bins,
            anchor_points=anchor_points,
            **kwargs
        )
    elif bins is None and anchor_points is None:
        return _regressor(
            backbone=backbone,
            input_size=input_size,
            reduction=reduction,
        )
    else:
        assert bins is not None and anchor_points is not None, f"Expected bins and anchor_points to be both None or not None, got {bins} and {anchor_points}"
        return _classifier(
            backbone=backbone,
            input_size=input_size,
            reduction=reduction,
            bins=bins,
            anchor_points=anchor_points,
        )


__all__ = [
    "get_model",
]
