import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from typing import Dict, Tuple, Union, Iterable, List, Optional


def calculate_errors(pred_counts: np.ndarray, target_counts: np.ndarray) -> Dict[str, float]:
    assert isinstance(pred_counts, np.ndarray), f"Expected numpy.ndarray, got {type(pred_counts)}"
    assert isinstance(target_counts, np.ndarray), f"Expected numpy.ndarray, got {type(target_counts)}"
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    errors = {
        "mae": np.abs(pred_counts - target_counts),
        "rmse": (pred_counts - target_counts) ** 2,
    }
    errors["mrae"] = errors["mae"] / np.maximum(target_counts, 1.)  # mean relative absolute error, avoid division by zero
    errors["rmrse"] = errors["rmse"] / np.maximum(target_counts, 1.) ** 2  # root mean relative squared error
    errors = {k: v.mean() for k, v in errors.items()}
    errors["rmse"], errors["rmrse"] = np.sqrt(errors["rmse"]), np.sqrt(errors["rmrse"])
    return errors


def resize_density_map(x: Tensor, size: Tuple[int, int]) -> Tensor:
    x_sum = torch.sum(x, dim=(-1, -2))
    x = F.interpolate(x, size=size, mode="bilinear")
    scale_factor = torch.nan_to_num(torch.sum(x, dim=(-1, -2)) / x_sum, nan=0.0, posinf=0.0, neginf=0.0)
    return x * scale_factor


def sliding_window_predict(
    model: nn.Module,
    image: Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    strategy: str = "mean",
) -> Tensor:
    """
    Use the sliding window strategy to predict the density map of an image.

    Args:
        model (nn.Module): The model to use for prediction.
        image (Tensor): The image to predict.
        window_size (Union[int, Tuple[int, int]]): The size of the window.
        stride (Optional[Union[int, Tuple[int, int]]], optional): The stride of the window. Defaults to None. If None, stride is equal to window_size.
        strategy (str, optional): The strategy to use to aggregate the predictions. Defaults to "mean".

    Returns:
        Tensor: The predicted density map.
    """
    window = (window_size, window_size) if isinstance(window_size, (int, float)) else window_size
    stride = (stride, stride) if isinstance(stride, (int, float)) else stride
    stride = window if stride is None else stride
    assert isinstance(window, Iterable) and len(window) == 2 and window[0] > 0 and window[1] > 0, f"Window size must be a positive integer tuple (h, w), got {window}"
    assert isinstance(stride, Iterable) and len(stride) == 2 and stride[0] > 0 and stride[1] > 0, f"Stride must be a positive integer tuple (h, w), got {stride}"
    assert stride[0] <= window[0] and stride[1] <= window[1], f"Stride must be smaller than window size, got {stride} and {window}"
    assert strategy in ["mean", "max"], f"Strategy must be either 'mean' or 'max', got {strategy}"
    image = image.unsqueeze(0) if len(image.shape) == 3 else image
    assert len(image.shape) == 4, f"Image must be a 3D tensor (h, w, c) or 4D tensor (b, h, w, c), got {image.shape}"

    preds = []
    for x, y, patch in _sliding_window(image, window_size=window, step_size=stride):
        patch_density_map = _process_patch(model, patch)
        preds.append((x, y, patch_density_map))

    return _combine_patches(preds, image.shape[-2:], window, model.reduction, strategy)


def _sliding_window(image: Tensor, window_size: Tuple[int, int], step_size: Tuple[int, int]) -> Iterable[Tuple[int, int, Tensor]]:
    """
    Sliding window generator over the image.

    Args:
        image (Tensor): The image (b, c, h, w) to slide over.
        window_size (Tuple[int, int]): The size (h, w) of the window.
        step_size (Tuple[int, int]): The step size (h, w) of the window.
    """
    assert len(image.shape) == 4, f"Image must be a 4D tensor (b, c, h, w), got {image.shape}"
    img_h, img_w = image.shape[2:]
    p_h, p_w = window_size
    s_h, s_w = step_size
    assert p_h <= img_h and p_w <= img_w, f"Window size must be smaller than image size, got window_size={window_size} and image.shape={image.shape}"
    assert s_h <= p_h and s_w <= p_w, f"Step size must be smaller than window size, got step_size={step_size} and window_size={window_size}"

    for y in range(0, img_h + s_h, s_h):
        for x in range(0, img_w + s_w, s_w):
            if y + p_h <= img_h and x + p_w <= img_w: # yield current window
                yield (x, y, TF.crop(image, y, x, p_h, p_w))
            elif y + p_h > img_h and x + p_w <= img_w: # yield the last window
                yield (x, img_h - p_h, TF.crop(image, img_h - p_h, x, p_h, p_w))
            elif y + p_h <= img_h and x + p_w > img_w: # yield the last window
                yield (img_w - p_w, y, TF.crop(image, y, img_w - p_w, p_h, p_w))
            else: # yield the last window
                yield (img_w - p_w, img_h - p_h, TF.crop(image, img_h - p_h, img_w - p_w, p_h, p_w))


def _process_patch(model: nn.Module, patch: Tensor) -> Tensor:
    """
    Process a patch with the given model.

    Args:
        model (nn.Module): The model to use.
        patch (Tensor): The patch to process.
    """
    assert len(patch.shape) == 4, f"Patch must be a 4D tensor (b, c, h, w), got {patch.shape}"
    return model(patch)


def _combine_patches(
    preds: List[Tuple[int, int, Tensor]],
    image_size: Tuple[int, int],
    window_size: Tuple[int, int],
    reduction: int = 1,
    strategy: str = "mean",
) -> Tensor:
    """
    Combine the density maps of the patches into a single density map.

    Args:
        density_maps (List[Tuple[int, int, Tensor]]): The list of density maps of the patches.
        image_size (Tuple[int, int]): The size of the image.
        window_size (Tuple[int, int]): The size of the window.
        step_size (Tuple[int, int]): The step size of the window.
        strategy (str, optional): The strategy to use to aggregate the predictions. Defaults to "mean".
    """
    assert strategy in ["mean", "max"], f"Strategy must be either 'mean' or 'max', got {strategy}"
    channels, dtype, device = preds[0][2].shape[-3], preds[0][2].dtype, preds[0][2].device
    img_h, img_w = image_size
    p_h, p_w = window_size

    img_h, img_w = img_h // reduction, img_w // reduction
    p_h, p_w = p_h // reduction, p_w // reduction

    full_map = torch.zeros((1, channels, img_h, img_w), dtype=dtype, device=device)
    count_map = torch.zeros(1, channels, img_h, img_w, dtype=dtype, device=device)  # keep track of how many times a pixel has been visited
    max_map = torch.zeros(1, channels, img_h, img_w, dtype=dtype, device=device)

    for x, y, pred in preds:
        x, y = x // reduction, y // reduction
        full_map[:, :, y : y + p_h, x : x + p_w] += pred
        count_map[:, :, y : y + p_h, x : x + p_w] += 1
        max_map[:, :, y : y + p_h, x : x + p_w] = torch.maximum(
            max_map[:, :, y : y + p_h, x : x + p_w], pred
        )

    return full_map / count_map if strategy == "mean" else max_map
