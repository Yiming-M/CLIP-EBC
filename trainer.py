import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from argparse import ArgumentParser
import os, json

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from models import get_model

from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier
from utils import get_dataloader, get_loss_fn, get_optimizer, load_checkpoint, save_checkpoint
from utils import get_writer, update_train_result, update_eval_result, log
from train import train
from eval import evaluate


parser = ArgumentParser(description="Train an EBC model.")

# Parameters for model
parser.add_argument("--model", type=str, default="vgg19_ae", help="The model to train.")
parser.add_argument("--input_size", type=int, default=448, help="The size of the input image.")
parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32], help="The reduction factor of the model.")
parser.add_argument("--regression", action="store_true", help="Use blockwise regression instead of classification.")
parser.add_argument("--truncation", type=int, default=None, help="The truncation of the count.")
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"], help="The representative count values of bins.")
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"], help="The prompt type for CLIP.")
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"], help="The granularity of bins.")
parser.add_argument("--num_vpt", type=int, default=32, help="The number of visual prompt tokens.")
parser.add_argument("--vpt_drop", type=float, default=0.0, help="The dropout rate for visual prompt tokens.")
parser.add_argument("--shallow_vpt", action="store_true", help="Use shallow visual prompt tokens.")

# Parameters for dataset
parser.add_argument("--dataset", type=str, required=True, help="The dataset to train on.")
parser.add_argument("--batch_size", type=int, default=8, help="The training batch size.")
parser.add_argument("--num_crops", type=int, default=1, help="The number of crops for multi-crop training.")
parser.add_argument("--min_scale", type=float, default=1.0, help="The minimum scale for random scale augmentation.")
parser.add_argument("--max_scale", type=float, default=2.0, help="The maximum scale for random scale augmentation.")
parser.add_argument("--brightness", type=float, default=0.1, help="The brightness factor for random color jitter augmentation.")
parser.add_argument("--contrast", type=float, default=0.1, help="The contrast factor for random color jitter augmentation.")
parser.add_argument("--saturation", type=float, default=0.1, help="The saturation factor for random color jitter augmentation.")
parser.add_argument("--hue", type=float, default=0.0, help="The hue factor for random color jitter augmentation.")
parser.add_argument("--kernel_size", type=int, default=5, help="The kernel size for Gaussian blur augmentation.")
parser.add_argument("--saltiness", type=float, default=1e-3, help="The saltiness for pepper salt noise augmentation.")
parser.add_argument("--spiciness", type=float, default=1e-3, help="The spiciness for pepper salt noise augmentation.")
parser.add_argument("--jitter_prob", type=float, default=0.2, help="The probability for random color jitter augmentation.")
parser.add_argument("--blur_prob", type=float, default=0.2, help="The probability for Gaussian blur augmentation.")
parser.add_argument("--noise_prob", type=float, default=0.5, help="The probability for pepper salt noise augmentation.")

# Parameters for evaluation
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to the nearest multiple of the input size.")
parser.add_argument("--zero_pad_to_multiple", action="store_true", help="Zero pad the image to the nearest multiple of the input size.")

# Parameters for loss function
parser.add_argument("--weight_count_loss", type=float, default=1.0, help="The weight for count loss.")
parser.add_argument("--count_loss", type=str, default="mae", choices=["mae", "mse", "dmcount"], help="The loss function for count.")

# Parameters for optimizer (Adam)
parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="The weight decay.")

# Parameters for learning rate scheduler
parser.add_argument("--warmup_epochs", type=int, default=50, help="Number of epochs for warmup. The learning rate will increase from eta_min to lr.")
parser.add_argument("--warmup_lr", type=float, default=1e-6, help="Learning rate for warmup.")
parser.add_argument("--T_0", type=int, default=5, help="Number of epochs for the first restart.")
parser.add_argument("--T_mult", type=int, default=2, help="A factor increases T_0 after a restart.")
parser.add_argument("--eta_min", type=float, default=1e-7, help="Minimum learning rate.")

# Parameters for training
parser.add_argument("--total_epochs", type=int, default=2600, help="Number of epochs to train.")
parser.add_argument("--eval_start", type=int, default=50, help="Start to evaluate after this number of epochs.")
parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate every this number of epochs.")
parser.add_argument("--save_freq", type=int, default=5, help="Save checkpoint every this number of epochs. Could help reduce I/O.")
parser.add_argument("--save_best_k", type=int, default=3, help="Save the best k checkpoints.")
parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision training.")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")


def run(local_rank: int, nprocs: int, args: ArgumentParser) -> None:
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)
    print(f"Initialized successfully. Training with {nprocs} GPUs.")
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    print(f"Using device: {device}.")

    ddp = nprocs > 1

    if args.regression:
        bins, anchor_points = None, None
    else:
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            config = json.load(f)[str(args.truncation)][args.dataset]
        bins = config["bins"][args.granularity]
        anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else config["anchor_points"][args.granularity]["middle"]
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]

    args.bins = bins
    args.anchor_points = anchor_points

    model = get_model(
        backbone=args.model,
        input_size=args.input_size, 
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt
    ).to(device)

    grad_scaler = GradScaler() if args.amp else None

    loss_fn = get_loss_fn(args).to(device)
    optimizer, scheduler = get_optimizer(args, model)

    ckpt_dir_name = f"{args.model}_{args.prompt_type}_" if "clip" in args.model else f"{args.model}_"
    ckpt_dir_name += f"{args.input_size}_{args.reduction}_{args.truncation}_{args.granularity}_"
    ckpt_dir_name += f"{args.weight_count_loss}_{args.count_loss}"

    args.ckpt_dir = os.path.join(current_dir, "checkpoints", args.dataset, ckpt_dir_name)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_val_scores, best_val_scores = load_checkpoint(args, model, optimizer, scheduler, grad_scaler)

    if local_rank == 0:
        model_without_ddp = model
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train.log"))
        logger.info(get_config(vars(args), mute=False))
        val_loader = get_dataloader(args, split="val", ddp=False)

    args.batch_size = int(args.batch_size / nprocs)
    args.num_workers = int(args.num_workers / nprocs)
    train_loader, sampler = get_dataloader(args, split="train", ddp=ddp)

    model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank], output_device=local_rank) if ddp else model

    for epoch in range(start_epoch, args.total_epochs + 1):  # start from 1
        if local_rank == 0:
            message = f"\tlr: {optimizer.param_groups[0]['lr']:.3e}"
            log(logger, epoch, args.total_epochs, message=message)

        if sampler is not None:
            sampler.set_epoch(epoch)

        model, optimizer, grad_scaler, loss_info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, local_rank, nprocs)
        scheduler.step()
        barrier(ddp)

        if local_rank == 0:
            eval = (epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)
            update_train_result(epoch, loss_info, writer)
            log(logger, None, None, loss_info=loss_info, message="\n" * 2 if not eval else None)

            if eval:
                print("Evaluating")
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                model_without_ddp.load_state_dict(state_dict)
                curr_val_scores = evaluate(
                    model_without_ddp,
                    val_loader,
                    device,
                    args.sliding_window,
                    args.input_size,
                    args.stride,
                )
                hist_val_scores, best_val_scores = update_eval_result(epoch, curr_val_scores, hist_val_scores, best_val_scores, writer, state_dict, os.path.join(args.ckpt_dir))
                log(logger, None, None, None, curr_val_scores, best_val_scores, message="\n" * 3)
    
            if (epoch % args.save_freq == 0):
                save_checkpoint(
                    epoch + 1,
                    model.module.state_dict() if ddp else model.state_dict(),
                    optimizer.state_dict(),
                    scheduler.state_dict() if scheduler is not None else None,
                    grad_scaler.state_dict() if grad_scaler is not None else None,
                    loss_info,
                    hist_val_scores,
                    best_val_scores,
                    args.ckpt_dir,
                )

        barrier(ddp)

    if local_rank == 0:
        writer.close()
        print("Training completed. Best scores:")
        for k in best_val_scores.keys():
            scores = " ".join([f"{best_val_scores[k][i]:.4f};" for i in range(len(best_val_scores[k]))])
            print(f"    {k}: {scores}")

    cleanup(ddp)


def main():
    args = parser.parse_args()
    args.model = args.model.lower()
    args.dataset = standardize_dataset_name(args.dataset)

    if args.regression:
        args.truncation = None
        args.anchor_points = None
        args.bins = None
        args.prompt_type = None
        args.granularity = None

    if "clip_vit" not in args.model:
        args.num_vpt = None
        args.vpt_drop = None
        args.shallow_vpt = None
    
    if "clip" not in args.model:
        args.prompt_type = None

    if args.sliding_window:
        args.window_size = args.input_size
        args.stride = args.input_size if args.stride is None else args.stride
        assert not (args.zero_pad_to_multiple and args.resize_to_multiple), "Cannot use both zero pad and resize to multiple."

    else:
        args.window_size = None
        args.stride = None
        args.zero_pad_to_multiple = False
        args.resize_to_multiple = False

    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")
    if args.nprocs > 1:
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)


if __name__ == "__main__":
    main()
