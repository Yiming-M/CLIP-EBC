import torch
from torch import nn, Tensor

from torch.optim import Adam
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR

from functools import partial
from argparse import ArgumentParser

import os, sys, math
from typing import Union, Tuple, Dict, List
from collections import OrderedDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import losses


def cosine_annealing_warm_restarts(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
    T_0: int,
    T_mult: int,
    eta_min: float,
) -> float:
    """
    Learning rate scheduler.
    The learning rate will linearly increase from warmup_lr to lr in the first warmup_epochs epochs.
    Then, the learning rate will follow the cosine annealing with warm restarts strategy.
    """
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    assert isinstance(warmup_epochs, int) and warmup_epochs >= 0, f"warmup_epochs must be non-negative, got {warmup_epochs}."
    assert isinstance(warmup_lr, float) and warmup_lr > 0, f"warmup_lr must be positive, got {warmup_lr}."
    assert isinstance(T_0, int) and T_0 >= 1, f"T_0 must be greater than or equal to 1, got {T_0}."
    assert isinstance(T_mult, int) and T_mult >= 1, f"T_mult must be greater than or equal to 1, got {T_mult}."
    assert isinstance(eta_min, float) and eta_min > 0, f"eta_min must be positive, got {eta_min}."
    assert isinstance(base_lr, float) and base_lr > 0, f"base_lr must be positive, got {base_lr}."
    assert base_lr > eta_min, f"base_lr must be greater than eta_min, got base_lr={base_lr} and eta_min={eta_min}."
    assert warmup_lr >= eta_min, f"warmup_lr must be greater than or equal to eta_min, got warmup_lr={warmup_lr} and eta_min={eta_min}."

    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
    else:
        epoch -= warmup_epochs
        if T_mult == 1:
            T_cur = epoch % T_0
            T_i = T_0
        else:
            n = int(math.log((epoch / T_0 * (T_mult - 1) + 1), T_mult))
            T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** (n)
        
        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2

    return lr / base_lr


def get_loss_fn(args: ArgumentParser) -> nn.Module:
    if args.bins is None:
        assert args.weight_ot is not None and args.weight_tv is not None, f"Expected weight_ot and weight_tv to be not None, got {args.weight_ot} and {args.weight_tv}"
        loss_fn = losses.DMLoss(
            input_size=args.input_size,
            reduction=args.reduction,
        )
    else:
        loss_fn = losses.DACELoss(
            bins=args.bins,
            reduction=args.reduction,
            weight_count_loss=args.weight_count_loss,
            count_loss=args.count_loss,
            input_size=args.input_size,
        )
    return loss_fn


def get_optimizer(args: ArgumentParser, model: nn.Module) -> Tuple[Adam, LambdaLR]:
    optimizer = Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=partial(
            cosine_annealing_warm_restarts,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min,
            base_lr=args.lr
        ),
    )

    return optimizer, scheduler


def load_checkpoint(
    args: ArgumentParser,
    model: nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    grad_scaler: GradScaler,
) -> Tuple[nn.Module, Adam, Union[LambdaLR, None], GradScaler, int, Union[Dict[str, float], None], Dict[str, List[float]], Dict[str, float]]:
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        loss_info = ckpt["loss_info"]
        hist_scores = ckpt["hist_scores"]
        best_scores = ckpt["best_scores"]

        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if grad_scaler is not None:
            grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])

        print(f"Loaded checkpoint from {ckpt_path}.")

    else:
        start_epoch = 1
        loss_info, hist_scores = None, {"mae": [], "rmse": []}
        best_scores = {k: [torch.inf] * args.save_best_k for k in hist_scores.keys()}
        print(f"Checkpoint not found at {ckpt_path}.")

    return model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_scores, best_scores


def save_checkpoint(
    epoch: int,
    model_state_dict: OrderedDict[str, Tensor],
    optimizer_state_dict: OrderedDict[str, Tensor],
    scheduler_state_dict: OrderedDict[str, Tensor],
    grad_scaler_state_dict: OrderedDict[str, Tensor],
    loss_info: Dict[str, List[float]],
    hist_scores: Dict[str, List[float]],
    best_scores: Dict[str, float],
    ckpt_dir: str,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
        "loss_info": loss_info,
        "hist_scores": hist_scores,
        "best_scores": best_scores,
    }
    torch.save(ckpt, os.path.join(ckpt_dir, "ckpt.pth"))
