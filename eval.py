import torch
from torch import nn
from torch.utils.data import DataLoader
from einops import rearrange
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from utils import calculate_errors, sliding_window_predict


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    sliding_window: bool = False,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    strategy: str = "mean",
) -> Dict[str, float]:
    model.eval()
    pred_counts, target_counts = [], []
    if sliding_window:
        assert window_size is not None, f"Window size must be provided when sliding_window is True, but got {window_size}"
        assert stride is not None, f"Stride must be provided when sliding_window is True, but got {stride}"

    for image, target_points, _ in tqdm(data_loader):
        image_height, image_width = image.shape[-2:]
        image = image.to(device)
        target_counts.append([len(p) for p in target_points])

        if window_size is not None:
            assert image_height % window_size == 0 and image_width % window_size == 0, f"Image size {image.shape} should be divisible by window size {window_size}."

        with torch.set_grad_enabled(False):
            if sliding_window:
                pred_density = sliding_window_predict(model, image, window_size, stride, strategy=strategy)
            elif window_size is not None:
                image = rearrange(image, "b c (h s1) (w s2) -> (b h w) c s1 s2", s1=window_size, s2=window_size)
                pred_density = model(image)
                pred_density = rearrange(pred_density, "(b h w) 1 s1 s2 -> b 1 (h s1) (w s2)", h=image_height // window_size, w=image_width // window_size)
            else:
                pred_density = model(image)

            pred_counts.append(pred_density.sum(dim=(1, 2, 3)).cpu().numpy().tolist())

    pred_counts = np.array([item for sublist in pred_counts for item in sublist])
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    return calculate_errors(pred_counts, target_counts)
