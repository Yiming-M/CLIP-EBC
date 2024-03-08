import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple


from utils import barrier, reduce_mean, update_loss_info


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    rank: int,
    nprocs: int,
) -> Tuple[nn.Module, Optimizer, Dict[str, float]]:
    model.train()
    info = None
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1
    regression = (model.module.bins is None) if ddp else (model.bins is None)

    for image, target_points, target_density in data_iter:
        image = image.to(device)
        target_points = [p.to(device) for p in target_points]
        target_density = target_density.to(device)
        with torch.set_grad_enabled(True):
            if not regression:
                pred_class, pred_density = model(image)
                loss, loss_info = loss_fn(pred_class, pred_density, target_density, target_points)
            else:
                pred_density = model(image)
                loss, loss_info = loss_fn(pred_density, target_density, target_points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_info = {k: reduce_mean(v.detach(), nprocs).item() if ddp else v.detach().item() for k, v in loss_info.items()}
        # if rank == 0:
            # loss_info = {k: v.item() for k, v in loss_info.items()}
        info = update_loss_info(info, loss_info)

        barrier(ddp)

    return model, optimizer, {k: np.mean(v) for k, v in info.items()}
