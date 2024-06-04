import torch
from torch import Tensor
from tensorboardX import SummaryWriter
import logging
import os
from typing import Dict, Union, Optional, List, Tuple
from collections import OrderedDict


def get_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_config(config: Dict, mute: bool = False) -> str:
    config = config.copy()
    config = "\n".join([f"{k.ljust(15)}:\t{v}" for k, v in config.items()])
    if not mute:
        print(config)
    return config


def get_writer(ckpt_dir: str) -> SummaryWriter:
    return SummaryWriter(log_dir=os.path.join(ckpt_dir, "logs"))


def print_epoch(epoch: int, total_epochs: int, mute: bool = False) -> Union[str, None]:
    digits = len(str(total_epochs))
    info = f"Epoch: {(epoch):0{digits}d} / {total_epochs:0{digits}d}"
    if mute:
        return info
    print(info)


def print_train_result(loss_info: Dict[str, float], mute: bool = False) -> Union[str, None]:
    loss_info = [f"{k}: {v};" for k, v in loss_info.items()]
    info = "Training: " + " ".join(loss_info)
    if mute:
        return info
    print(info)


def print_eval_result(curr_scores: Dict[str, float], best_scores: Dict[str, float], mute: bool = False) -> Union[str, None]:
    scores = []
    for k in curr_scores.keys():
        info = f"Curr {k}: {curr_scores[k]:.4f}; \t Best {k}: "
        info += " ".join([f"{best_scores[k][i]:.4f};" for i in range(len(best_scores[k]))])
        scores.append(info)

    info = "Evaluation:\n" + "\n".join(scores)
    if mute:
        return info
    print(info)


def update_train_result(epoch: int, loss_info: Dict[str, float], writer: SummaryWriter) -> None:
    for k, v in loss_info.items():
        writer.add_scalar(f"train/{k}", v, epoch)


def update_eval_result(
    epoch: int,
    curr_scores: Dict[str, float],
    hist_scores: Dict[str, List[float]],
    best_scores: Dict[str, List[float]],
    writer: SummaryWriter,
    state_dict: OrderedDict[str, Tensor],
    ckpt_dir: str,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    os.makedirs(ckpt_dir, exist_ok=True)
    for k, v in curr_scores.items():
        hist_scores[k].append(v)
        writer.add_scalar(f"val/{k}", v, epoch)

        # best_scores[k][0] is the best score. Smaller is better.
        # Find the location idx where the new score v should be inserted
        loc = None
        for i in range(len(best_scores[k])):
            if v < best_scores[k][i]:
                best_scores[k].insert(i, v)  # Add the new best score to the location i
                loc = i
                break

        # If the new score is better than the worst best score
        if loc is not None: 
            # Update the best scores
            best_scores[k] = best_scores[k][:len(best_scores[k]) - 1]

            # Rename the best_{k}_{i}.pth to best_{k}_{i+1}.pth, best_{k}_{i+1}.pth to best_{k}_{i+2}.pth ...
            for i in range(len(best_scores[k]) - 1, loc, -1):
                if os.path.exists(os.path.join(ckpt_dir, f"best_{k}_{i-1}.pth")):
                    os.rename(os.path.join(ckpt_dir, f"best_{k}_{i-1}.pth"), os.path.join(ckpt_dir, f"best_{k}_{i}.pth"))

            # Save the best checkpoint
            torch.save(state_dict, os.path.join(ckpt_dir, f"best_{k}_{loc}.pth"))    

    return hist_scores, best_scores


def update_loss_info(hist_scores: Union[Dict[str, List[float]], None], curr_scores: Dict[str, float]) -> Dict[str, List[float]]:
    assert all([isinstance(v, float) for v in curr_scores.values()]), f"Expected all values to be float, got {curr_scores}"
    if hist_scores is None or len(hist_scores) == 0:
        hist_scores = {k: [v] for k, v in curr_scores.items()}
    else:
        for k, v in curr_scores.items():
            hist_scores[k].append(v)
    return hist_scores


def log(
    logger: logging.Logger,
    epoch: int, 
    total_epochs: int,
    loss_info: Optional[Dict[str, float]] = None,
    curr_scores: Optional[Dict[str, float]] = None,
    best_scores: Optional[Dict[str, float]] = None,
    message: Optional[str] = None,
) -> None:
    if epoch is None:
        assert total_epochs is None, f"Expected total_epochs to be None when epoch is None, got {total_epochs}"
        msg = ""
    else:
        assert total_epochs is not None, f"Expected total_epochs to be not None when epoch is not None, got {total_epochs}"
        msg = print_epoch(epoch, total_epochs, mute=True)

    if loss_info is not None:
        msg += "\n" if len(msg) > 0 else ""
        msg += print_train_result(loss_info, mute=True)

    if curr_scores is not None:
        assert best_scores is not None, f"Expected best_scores to be not None when curr_scores is not None, got {best_scores}"
        msg += "\n" if len(msg) > 0 else ""
        msg += print_eval_result(curr_scores, best_scores, mute=True)

    msg += message if message is not None else ""

    logger.info(msg)
