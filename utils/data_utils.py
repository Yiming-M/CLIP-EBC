from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import Compose
import os, sys
from argparse import ArgumentParser
from typing import Union, Tuple

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import datasets


def get_dataloader(args: ArgumentParser, split: str = "train", ddp: bool = False) -> Union[Tuple[DataLoader, Union[DistributedSampler, None]], DataLoader]:
    if split == "train" and args.augment:  # train, strong augmentation
        transforms = Compose([
            datasets.RandomResizedCrop((args.input_size, args.input_size), scale=(args.min_scale, args.max_scale)),
            datasets.RandomHorizontalFlip(),
            datasets.RandomApply([
                datasets.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
                datasets.GaussianBlur(kernel_size=args.kernel_size, sigma=(0.1, 5.0)),
                datasets.PepperSaltNoise(saltiness=args.saltiness, spiciness=args.spiciness),
            ], p=(args.jitter_prob, args.blur_prob, args.noise_prob)),
        ])
    elif split == "train":  # train, weak augmentation
        transforms = datasets.RandomCrop((args.input_size, args.input_size))
    else:  # validation
        if args.resize_to_multiple:
            transforms = datasets.Resize2Multiple(args.window_size)
        elif args.zero_pad_to_multiple:
            transforms = datasets.ZeroPad2Multiple(args.window_size)
        else:
            transforms = None

    dataset = datasets.Crowd(
        dataset=args.dataset,
        split=split,
        transforms=transforms,
        percentage=args.percentage,
        sigma=None,
        return_filename=False,
        num_crops=args.num_crops if split == "train" else 1,
    )

    if ddp and split == "train":  # data_loader for training in DDP
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader, sampler

    elif split == "train":  # data_loader for training
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader, None

    else:  # data_loader for evaluation
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 for evaluation
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader
