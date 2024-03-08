import torch
from torch import Tensor
import torch.distributed as dist
import numpy as np
import random
import os


def reduce_mean(tensor: Tensor, nprocs: int) -> Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def setup(local_rank: int, nprocs: int) -> None:
    if nprocs > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=local_rank, world_size=nprocs)
    else:
        print("Single process. No need to setup dist.")


def cleanup(ddp: bool = True) -> None:
    if ddp:
        dist.destroy_process_group()


def init_seeds(seed: int, cuda_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, not reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def barrier(ddp: bool = True) -> None:
    if ddp:
        dist.barrier()
