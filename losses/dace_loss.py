import torch
from torch import nn, Tensor
from typing import Any, List, Tuple, Dict

from .dm_loss import DMLoss
from .utils import _reshape_density


class DACELoss(nn.Module):
    def __init__(
        self,
        bins: List[Tuple[float, float]],
        reduction: int,
        weight_count_loss: float = 1.0,
        count_loss: str = "mae",
        **kwargs: Any
    ) -> None:
        super().__init__()
        assert len(bins) > 0, f"Expected at least one bin, got {bins}"
        assert all([len(b) == 2 for b in bins]), f"Expected all bins to be of length 2, got {bins}"
        assert all([b[0] <= b[1] for b in bins]), f"Expected all bins to be in increasing order, got {bins}"
        self.bins = bins
        self.reduction = reduction
        self.cross_entropy_fn = nn.CrossEntropyLoss(reduction="none")

        count_loss = count_loss.lower()
        assert count_loss in ["mae", "mse", "dmcount"], f"Expected count_loss to be one of ['mae', 'mse', 'dmcount'], got {count_loss}"
        self.count_loss = count_loss
        if self.count_loss == "mae":
            self.use_dm_loss = False
            self.count_loss_fn = nn.L1Loss(reduction="none")
        elif self.count_loss == "mse":
            self.use_dm_loss = False
            self.count_loss_fn = nn.MSELoss(reduction="none")
        else:
            self.use_dm_loss = True
            assert "input_size" in kwargs, f"Expected input_size to be in kwargs when count_loss='dmcount', got {kwargs}"
            self.count_loss_fn = DMLoss(reduction=reduction, **kwargs)

        self.weight_count_loss = weight_count_loss

    def _bin_count(self, density_map: Tensor) -> Tensor:
        class_map = torch.zeros_like(density_map, dtype=torch.long)
        for idx, (low, high) in enumerate(self.bins):
            mask = (density_map >= low) & (density_map <= high)
            class_map[mask] = idx
        return class_map.squeeze(1)  # remove channel dimension

    def forward(self, pred_class: Tensor, pred_density: Tensor, target_density: Tensor, target_points: List[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_density = _reshape_density(target_density, reduction=self.reduction) if target_density.shape[-2:] != pred_density.shape[-2:] else target_density
        assert pred_density.shape == target_density.shape, f"Expected pred_density and target_density to have the same shape, got {pred_density.shape} and {target_density.shape}"

        target_class = self._bin_count(target_density)

        cross_entropy_loss = self.cross_entropy_fn(pred_class, target_class).sum(dim=(-1, -2)).mean()

        if self.use_dm_loss:
            count_loss, loss_info = self.count_loss_fn(pred_density, target_density, target_points)
            loss_info["ce_loss"] = cross_entropy_loss.detach()
        else:
            count_loss = self.count_loss_fn(pred_density, target_density).sum(dim=(-1, -2, -3)).mean()
            loss_info = {
                "ce_loss": cross_entropy_loss.detach(),
                f"{self.count_loss}_loss": count_loss.detach(),
            }

        loss = cross_entropy_loss + self.weight_count_loss * count_loss
        loss_info["loss"] = loss.detach()

        return loss, loss_info
