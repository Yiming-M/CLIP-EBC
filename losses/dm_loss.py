import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast
from typing import List, Any, Tuple, Dict

from .bregman_pytorch import sinkhorn
from .utils import _reshape_density

EPS = 1e-8


class OTLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        reduction: int,
        norm_cood: bool,
        num_of_iter_in_ot: int = 100,
        reg: float = 10.0
    ) -> None:
        super().__init__()
        assert input_size % reduction == 0

        self.input_size = input_size
        self.reduction = reduction
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, input_size, step=reduction, dtype=torch.float32) + reduction / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0) # [1, #cood]
        self.cood = self.cood / input_size * 2 - 1 if self.norm_cood else self.cood
        self.output_size = self.cood.size(1)

    @autocast(enabled=True, dtype=torch.float32)  # avoid numerical instability
    def forward(self, pred_density: Tensor, normed_pred_density: Tensor, target_points: List[Tensor]) -> Tuple[Tensor, float, Tensor]:
        batch_size = normed_pred_density.size(0)
        assert len(target_points) == batch_size, f"Expected target_points to have length {batch_size}, but got {len(target_points)}"
        assert self.output_size == normed_pred_density.size(2)
        device = pred_density.device

        loss = torch.zeros([1]).to(device)
        ot_obj_values = torch.zeros([1]).to(device)
        wd = 0 # Wasserstein distance
        cood = self.cood.to(device)
        for idx, points in enumerate(target_points):
            if len(points) > 0:
                # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                points = points / self.input_size * 2 - 1 if self.norm_cood else points
                x = points[:, 0].unsqueeze_(1)  # [#gt, 1]
                y = points[:, 1].unsqueeze_(1)
                x_dist = -2 * torch.matmul(x, cood) + x * x + cood * cood # [#gt, #cood]
                y_dist = -2 * torch.matmul(y, cood) + y * y + cood * cood
                y_dist.unsqueeze_(2)
                x_dist.unsqueeze_(1)
                dist = y_dist + x_dist
                dist = dist.view((dist.size(0), -1)) # size of [#gt, #cood * #cood]

                source_prob = normed_pred_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(points)]) / len(points)).to(device)
                # use sinkhorn to solve OT, compute optimal beta.
                P, log = sinkhorn(target_prob, source_prob, dist, self.reg, maxIter=self.num_of_iter_in_ot, log=True)
                beta = log["beta"] # size is the same as source_prob: [#cood * #cood]
                ot_obj_values += torch.sum(normed_pred_density[idx] * beta.view([1, self.output_size, self.output_size]))
                # compute the gradient of OT loss to predicted density (pred_density).
                # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
                source_density = pred_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                gradient_1 = (source_count) / (source_count * source_count+ EPS) * beta # size of [#cood * #cood]
                gradient_2 = (source_density * beta).sum() / (source_count * source_count + EPS) # size of 1
                gradient = gradient_1 - gradient_2
                gradient = gradient.detach().view([1, self.output_size, self.output_size])
                # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t predicted density is im_grad.
                loss += torch.sum(pred_density[idx] * gradient)
                wd += torch.sum(dist * P).item()

        return loss, wd, ot_obj_values


class DMLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        reduction: int,
        norm_cood: bool = False,
        weight_ot: float = 0.1,
        weight_tv: float = 0.01,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.ot_loss = OTLoss(input_size, reduction, norm_cood, **kwargs)
        self.tv_loss = nn.L1Loss(reduction="none")
        self.count_loss = nn.L1Loss(reduction="mean")
        self.weight_ot = weight_ot
        self.weight_tv = weight_tv

    @autocast(enabled=True, dtype=torch.float32)  # avoid numerical instability
    def forward(self, pred_density: Tensor, target_density: Tensor, target_points: List[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_density = _reshape_density(target_density, reduction=self.ot_loss.reduction) if target_density.shape[-2:] != pred_density.shape[-2:] else target_density
        assert pred_density.shape == target_density.shape, f"Expected pred_density and target_density to have the same shape, got {pred_density.shape} and {target_density.shape}"

        pred_count = pred_density.view(pred_density.shape[0], -1).sum(dim=1)
        normed_pred_density = pred_density / (pred_count.view(-1, 1, 1, 1) + EPS)
        target_count = torch.tensor([len(p) for p in target_points], dtype=torch.float32).to(target_density.device)
        normed_target_density = target_density / (target_count.view(-1, 1, 1, 1) + EPS)

        ot_loss, _, _ = self.ot_loss(pred_density, normed_pred_density, target_points)

        tv_loss = (self.tv_loss(normed_pred_density, normed_target_density).sum(dim=(1, 2, 3)) * target_count).mean()

        count_loss = self.count_loss(pred_count, target_count)

        loss = ot_loss * self.weight_ot + tv_loss * self.weight_tv + count_loss

        loss_info = {
            "loss": loss.detach(),
            "ot_loss": ot_loss.detach(),
            "tv_loss": tv_loss.detach(),
            "count_loss": count_loss.detach(),
        }

        return loss, loss_info
