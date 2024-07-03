import torch
from torch import nn
from torch.utils.data import DataLoader
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
) -> Dict[str, float]:
    model.eval()
    pred_counts, target_counts = [], []
    if sliding_window:
        assert window_size is not None, f"Window size must be provided when sliding_window is True, but got {window_size}"
        assert stride is not None, f"Stride must be provided when sliding_window is True, but got {stride}"

    for image, target_points, _ in tqdm(data_loader):
        image = image.to(device)
        target_counts.append([len(p) for p in target_points])

        with torch.set_grad_enabled(False):
            if sliding_window:
                pred_density = sliding_window_predict(model, image, window_size, stride)
            else:
                pred_density = model(image)

            pred_counts.append(pred_density.sum(dim=(1, 2, 3)).cpu().numpy().tolist())

    pred_counts = np.array([item for sublist in pred_counts for item in sublist])
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    return calculate_errors(pred_counts, target_counts)
