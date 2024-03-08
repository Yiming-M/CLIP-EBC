import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize
import os
from glob import glob
from PIL import Image
import numpy as np
from typing import Optional, Callable, Union, Tuple

from .utils import get_id, generate_density_map

curr_dir = os.path.dirname(os.path.abspath(__file__))

available_datasets = [
    "shanghaitech_a", "sha",
    "shanghaitech_b", "shb",
    "ucf_qnrf", "qnrf", "ucf-qnrf",
    "nwpu", "nwpu_crowd", "nwpu-crowd",
    "jhu", "jhu_crowd", "jhu_crowd_v2"
]

available_percentages = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def standardize_dataset_name(dataset: str) -> str:
    assert dataset.lower() in available_datasets, f"Dataset {dataset} is not available."
    if dataset.lower() in ["shanghaitech_a", "sha"]:
        return "sha"
    elif dataset.lower() in ["shanghaitech_b", "shb"]:
        return "shb"
    elif dataset.lower() in ["ucf_qnrf", "qnrf", "ucf-qnrf"]:
        return "qnrf"
    elif dataset.lower() in ["nwpu", "nwpu_crowd", "nwpu-crowd"]:
        return "nwpu"
    else:  # dataset.lower() in ["jhu", "jhu_crowd", "jhu_crowd_v2"]
        return "jhu"


class Crowd(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str,
        transforms: Optional[Callable] = None,
        percentage: int = 100,
        sigma: Optional[float] = None,
        return_filename: bool = False,
        num_crops: int = 1,
    ) -> None:
        """
        Dataset for crowd counting.
        """
        assert dataset.lower() in available_datasets, f"Dataset {dataset} is not available."
        assert split in ["train", "val"], f"Split {split} is not available."
        assert percentage in available_percentages, f"Percentage {percentage} is not available."
        assert num_crops > 0, f"num_crops should be positive, got {num_crops}."

        self.dataset = standardize_dataset_name(dataset)
        self.split = split
        self.percentage = percentage

        self.__find_root__()
        self.__make_dataset__()
        self.__check_sanity__()
        self.__load_labeled_indices__()

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms

        self.sigma = sigma
        self.return_filename = return_filename
        self.num_crops = num_crops

    def __find_root__(self) -> None:
        # if self.dataset == "sha":
        #     self.root = os.path.join(curr_dir, "..", "data", "ShanghaiTech_A")
        # elif self.dataset == "shb":
        #     self.root = os.path.join(curr_dir, "..", "data", "ShanghaiTech_B")
        # elif self.dataset == "qnrf":
        #     self.root = os.path.join(curr_dir, "..", "data", "QNRF")
        # elif self.dataset == "nwpu":
        #     self.root = os.path.join(curr_dir, "..", "data", "NWPU")
        # else:  # self.dataset == "jhu"
        #     self.root = os.path.join(curr_dir, "..", "data", "JHU")
        self.root = os.path.join(curr_dir, "..", "data", self.dataset)

    def __make_dataset__(self) -> None:
        image_npys = glob(os.path.join(self.root, self.split, "images", "*.npy"))
        if len(image_npys) > 0:
            self.image_type = "npy"
            image_names = image_npys
        else:
            self.image_type = "jpg"
            image_names = glob(os.path.join(self.root, self.split, "images", "*.jpg"))

        label_names = glob(os.path.join(self.root, self.split, "labels", "*.npy"))
        image_names = [os.path.basename(image_name) for image_name in image_names]
        label_names = [os.path.basename(label_name) for label_name in label_names]
        image_names.sort(key=get_id)
        label_names.sort(key=get_id)
        image_ids = tuple([get_id(image_name) for image_name in image_names])
        label_ids = tuple([get_id(label_name) for label_name in label_names])
        assert image_ids == label_ids, "image_ids and label_ids do not match."
        self.image_names = tuple(image_names)
        self.label_names = tuple(label_names)

    def __check_sanity__(self) -> None:
        if self.dataset == "sha":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 300, f"ShanghaiTech_A train split should have 300 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 182, f"ShanghaiTech_A val split should have 182 images, but found {len(self.image_names)}."
        elif self.dataset == "shb":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 400, f"ShanghaiTech_B train split should have 400 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 316, f"ShanghaiTech_B val split should have 316 images, but found {len(self.image_names)}."
        elif self.dataset == "nwpu":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 3109, f"NWPU train split should have 3109 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 500, f"NWPU val split should have 500 images, but found {len(self.image_names)}."
        elif self.dataset == "qnrf":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 1201, f"UCF_QNRF train split should have 1201 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 334, f"UCF_QNRF val split should have 334 images, but found {len(self.image_names)}."
        else:  # self.dataset == "jhu"
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 2772, f"JHU train split should have 2772 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 1600, f"JHU val split should have 1600 images, but found {len(self.image_names)}."

    def __load_labeled_indices__(self) -> None:
        if self.percentage < 100 and self.split == "train":
            with open(os.path.join(self.root, self.split, f"{self.percentage}%.txt"), "r") as f:
                indices = f.read().splitlines()
            self.indices = list(map(int, indices))  # indices of labeled images
        else:
            self.indices = list(range(len(self.image_names)))

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, str]]:
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]

        image_path = os.path.join(self.root, self.split, "images", image_name)
        label_path = os.path.join(self.root, self.split, "labels", label_name)

        if self.image_type == "npy":
            with open(image_path, "rb") as f:
                image = np.load(f)
            image = torch.from_numpy(image).float() / 255.  # normalize to [0, 1]
        else:
            with open(image_path, "rb") as f:
                image = Image.open(f).convert("RGB")
            image = self.to_tensor(image)

        with open(label_path, "rb") as f:
            label = np.load(f)

        label = torch.from_numpy(label).float()

        if self.transforms is not None:
            images_labels = [self.transforms(image.clone(), label.clone()) for _ in range(self.num_crops)]
            images, labels = zip(*images_labels)
        else:
            images = [image.clone() for _ in range(self.num_crops)]
            labels = [label.clone() for _ in range(self.num_crops)]

        images = [self.normalize(img) for img in images]
        if idx in self.indices:
            density_maps = torch.stack([generate_density_map(label, image.shape[-2], image.shape[-1], sigma=self.sigma) for image, label in zip(images, labels)], 0)
        else:
            labels = None
            density_maps = None

        image_names = [image_name] * len(images)
        images = torch.stack(images, 0)

        if self.return_filename:
            return images, labels, density_maps, image_names
        else:
            return images, labels, density_maps


class NWPUTest(Dataset):
    def __init__(
        self,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_filename: bool = False,
    ) -> None:
        """
        The test set of NWPU-Crowd dataset. The test set is not labeled, so only images are returned.
        """
        self.root = os.path.join(curr_dir, "..", "data", "nwpu")

        image_npys = glob(os.path.join(self.root, "test", "images", "*.npy"))
        if len(image_npys) > 0:
            self.image_type = "npy"
            image_names = image_npys
        else:
            self.image_type = "jpg"
            image_names = glob(os.path.join(self.root, "test", "images", "*.jpg"))

        image_names = [os.path.basename(image_name) for image_name in image_names]
        assert len(image_names) == 1500, f"NWPU test split should have 1500 images, but found {len(image_names)}."
        image_names.sort(key=get_id)
        self.image_names = tuple(image_names)

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms

        self.sigma = sigma
        self.return_filename = return_filename

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, str]]:
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root, "test", "images", image_name)

        if self.image_type == "npy":
            with open(image_path, "rb") as f:
                image = np.load(f)
            image = torch.from_numpy(image).float() / 255.
        else:
            with open(image_path, "rb") as f:
                image = Image.open(f).convert("RGB")
            image = self.to_tensor(image)
        
        label = torch.tensor([], dtype=torch.float)  # dummy label
        image, _ = self.transforms(image, label) if self.transforms is not None else (image, label)
        image = self.normalize(image)

        if self.return_filename:
            return image, image_name
        else:
            return image
