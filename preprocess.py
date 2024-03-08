import os
from glob import glob
from scipy.io import loadmat
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from typing import Tuple, Union, Optional
from warnings import warn

from datasets import standardize_dataset_name


def _calc_size(
    img_w: int,
    img_h: int,
    min_size: int,
    max_size: int,
    base: int = 32
) -> Union[Tuple[int, int], None]:
    """
    This function generates a new size for an image while keeping the aspect ratio. The new size should be within the given range (min_size, max_size).

    Args:
        img_w (int): The width of the image.
        img_h (int): The height of the image.
        min_size (int): The minimum size of the edges of the image.
        max_size (int): The maximum size of the edges of the image.
    """
    assert min_size % base == 0, f"min_size ({min_size}) must be a multiple of {base}"
    if max_size != float("inf"):
        assert max_size % base == 0, f"max_size ({max_size}) must be a multiple of {base} if provided"

    assert min_size <= max_size, f"min_size ({min_size}) must be less than or equal to max_size ({max_size})"

    aspect_ratios = (img_w / img_h, img_h / img_w)
    if min_size / max_size <= min(aspect_ratios) <= max(aspect_ratios) <= max_size / min_size:  # possible to resize and preserve the aspect ratio
        if min_size <= min(img_w, img_h) <= max(img_w, img_h) <= max_size:  # already within the range, no need to resize
            ratio = 1.
        elif min(img_w, img_h) < min_size:  # smaller than the minimum size, resize to the minimum size
            ratio = min_size / min(img_w, img_h)
        else:  # larger than the maximum size, resize to the maximum size
            ratio = max_size / max(img_w, img_h)

        new_w, new_h = int(round(img_w * ratio / base) * base), int(round(img_h * ratio / base) * base)
        new_w = max(min_size, min(max_size, new_w))
        new_h = max(min_size, min(max_size, new_h))
        return new_w, new_h

    else:  # impossible to resize and preserve the aspect ratio
        msg = f"Impossible to resize {img_w}x{img_h} image while preserving the aspect ratio to a size within the range ({min_size}, {max_size}). Will not limit the maximum size."
        warn(msg)
        return _calc_size(img_w, img_h, min_size, float("inf"), base)


def _generate_random_indices(
    total_size: int,
    out_dir: str,
) -> None:
    """
    Generate randomly selected indices for labelled data in semi-supervised learning.
    """
    rng = np.random.default_rng(42)
    for percent in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        num_select = int(total_size * percent)
        selected = rng.choice(total_size, num_select, replace=False)
        selected.sort()
        selected = selected.tolist()
        with open(os.path.join(out_dir, f"{int(percent * 100)}%.txt"), "w") as f:
            for i in selected:
                f.write(f"{i}\n")


def _resize(image: np.ndarray, label: np.ndarray, min_size: int, max_size: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    image_h, image_w, _ = image.shape
    new_size = _calc_size(image_w, image_h, min_size, max_size)
    if new_size is None:
        return image, label, False
    else:
        new_w, new_h = new_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC) if (new_w, new_h) != (image_w, image_h) else image
        label = label * np.array([[new_w / image_w, new_h / image_h]]) if len(label) > 0 and (new_w, new_h) != (image_w, image_h) else label
        return image, label, True


def _preprocess(
    dataset: str,
    data_src_dir: str,
    data_dst_dir: str,
    min_size: int,
    max_size: int,
    generate_npy: bool = False
) -> None:
    """
    This function organizes the data into the following structure:
    data_dst_dir
    ├── train
    │   ├── images
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   ├── ...
    │   │   images_npy
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── ...
    │   ├── labels
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── ...
    │   ├── 0.01%.txt
    │   ├── 0.05%.txt
    │   ├── ...
    ├── val
    │   ├── images
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   ├── ...
    │   │   images_npy
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── ...
    │   ├── labels
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── ...
    """
    dataset = standardize_dataset_name(dataset)
    assert os.path.isdir(data_src_dir), f"{data_src_dir} does not exist"
    os.makedirs(data_dst_dir, exist_ok=True)
    print(f"Pre-processing {dataset} dataset...")
    if dataset in ["sha", "shb"]:
        _shanghaitech(data_src_dir, data_dst_dir, min_size, max_size, generate_npy)

    elif dataset == "nwpu":
        _nwpu(data_src_dir, data_dst_dir, min_size, max_size, generate_npy)

    elif dataset == "qnrf":
        _qnrf(data_src_dir, data_dst_dir, min_size, max_size, generate_npy)
    
    else:  # dataset == "jhu"
        _jhu(data_src_dir, data_dst_dir, min_size, max_size, generate_npy)


def _resize_and_save(
    image: np.ndarray,
    name: str,
    image_dst_dir: str,
    generate_npy: bool,
    label: Optional[np.ndarray] = None,
    label_dst_dir: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> None:
    os.makedirs(image_dst_dir, exist_ok=True)

    if label is not None:
        assert label_dst_dir is not None, "label_dst_dir must be provided if label is provided"
        os.makedirs(label_dst_dir, exist_ok=True)

    image_dst_path = os.path.join(image_dst_dir, f"{name}.jpg")

    if label is not None:
        label_dst_path = os.path.join(label_dst_dir, f"{name}.npy")
    else:
        label = np.array([])
        label_dst_path = None

    if min_size is not None:
        assert max_size is not None, f"max_size must be provided if min_size is provided, got {max_size}"
        image, label, success = _resize(image, label, min_size, max_size)
        if not success:
            print(f"image: {image_dst_path} is not resized")

    cv2.imwrite(image_dst_path, image)

    if label_dst_path is not None:
        np.save(label_dst_path, label)

    if generate_npy:
        image_npy_dst_path = os.path.join(image_dst_dir, f"{name}.npy")
        image_npy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
        image_npy = np.transpose(image_npy, (2, 0, 1))  # HWC to CHW
        # Don't normalize the image. Keep it as np.uint8 to save space.
        # image_npy = image_npy.astype(np.float32) / 255.  # normalize to [0, 1]
        np.save(image_npy_dst_path, image_npy)


def _shanghaitech(
    data_src_dir: str,
    data_dst_dir: str,
    min_size: int,
    max_size: int,
    generate_npy: bool = False
) -> None:
    for split in ["train", "val"]:
        generate_npy = generate_npy and split == "train"
        print(f"Processing {split}...")
        if split == "train":
            image_src_dir = os.path.join(data_src_dir, "train_data", "images")
            label_src_dir = os.path.join(data_src_dir, "train_data", "ground-truth")
            image_src_paths = glob(os.path.join(image_src_dir, "*.jpg"))
            label_src_paths = glob(os.path.join(label_src_dir, "*.mat"))
            assert len(image_src_paths) == len(label_src_paths) in [300, 400], f"Expected 300 (part_A) or 400 (part_B) images and labels, got {len(image_src_paths)} images and {len(label_src_paths)} labels"
        else:
            image_src_dir = os.path.join(data_src_dir, "test_data", "images")
            label_src_dir = os.path.join(data_src_dir, "test_data", "ground-truth")
            image_src_paths = glob(os.path.join(image_src_dir, "*.jpg"))
            label_src_paths = glob(os.path.join(label_src_dir, "*.mat"))
            assert len(image_src_paths) == len(label_src_paths) in [182, 316], f"Expected 182 (part_A) or 316 (part_B) images and labels, got {len(image_src_paths)} images and {len(label_src_paths)} labels"

        sort_key = lambda x: int((os.path.basename(x).split(".")[0]).split("_")[-1])
        image_src_paths.sort(key=sort_key)
        label_src_paths.sort(key=sort_key)

        image_dst_dir = os.path.join(data_dst_dir, split, "images")
        label_dst_dir = os.path.join(data_dst_dir, split, "labels")
        os.makedirs(image_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)

        size = len(str(len(image_src_paths)))
        for i, (image_src_path, label_src_path) in tqdm(enumerate(zip(image_src_paths, label_src_paths)), total=len(image_src_paths)):
            image_id = int((os.path.basename(image_src_path).split(".")[0]).split("_")[-1])
            label_id = int((os.path.basename(label_src_path).split(".")[0]).split("_")[-1])
            assert image_id == label_id, f"Expected image id {image_id} to match label id {label_id}"
            name = f"{(i + 1):0{size}d}"
            image = cv2.imread(image_src_path)
            label = loadmat(label_src_path)["image_info"][0][0][0][0][0]
            _resize_and_save(
                image=image,
                label=label,
                name=name,
                image_dst_dir=image_dst_dir,
                label_dst_dir=label_dst_dir,
                generate_npy=generate_npy,
                min_size=min_size,
                max_size=max_size
            )

        if split == "train":
            _generate_random_indices(len(image_src_paths), os.path.join(data_dst_dir, split))

def _nwpu(
    data_src_dir: str,
    data_dst_dir: str,
    min_size: int,
    max_size: int,
    generate_npy: bool = False
) -> None:
    for split in ["train", "val"]:
        generate_npy = generate_npy and split == "train"
        print(f"Processing {split}...")
        with open(os.path.join(data_src_dir, f"{split}.txt"), "r") as f:
            indices = f.read().splitlines()
        indices = [idx.split(" ")[0] for idx in indices]
        image_src_paths = [os.path.join(data_src_dir, f"images_part{min(5, (int(idx) - 1) // 1000 + 1)}", f"{idx}.jpg") for idx in indices]
        label_src_paths = [os.path.join(data_src_dir, "mats", f"{idx}.mat") for idx in indices]

        image_dst_dir = os.path.join(data_dst_dir, split, "images")
        label_dst_dir = os.path.join(data_dst_dir, split, "labels")
        os.makedirs(image_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)

        size = len(str(len(image_src_paths)))
        for i, (image_src_path, label_src_path) in tqdm(enumerate(zip(image_src_paths, label_src_paths)), total=len(image_src_paths)):
            image_id = os.path.basename(image_src_path).split(".")[0]
            label_id = os.path.basename(label_src_path).split(".")[0]
            assert image_id == label_id, f"Expected image id {image_id} to match label id {label_id}"
            name = f"{(i + 1):0{size}d}"
            image = cv2.imread(image_src_path)
            label = loadmat(label_src_path)["annPoints"]
            _resize_and_save(
                image=image,
                label=label,
                name=name,
                image_dst_dir=image_dst_dir,
                label_dst_dir=label_dst_dir,
                generate_npy=generate_npy,
                min_size=min_size,
                max_size=max_size
            )

        if split == "train":
            _generate_random_indices(len(image_src_paths), os.path.join(data_dst_dir, split))
    
    # preprocess the test set
    split = "test"
    print(f"Processing {split}...")
    with open(os.path.join(data_src_dir, f"{split}.txt"), "r") as f:
        indices = f.read().splitlines()
    indices = [idx.split(" ")[0] for idx in indices]
    image_src_paths = [os.path.join(data_src_dir, f"images_part{min(5, (int(idx) - 1) // 1000 + 1)}", f"{idx}.jpg") for idx in indices]

    image_dst_dir = os.path.join(data_dst_dir, split, "images")
    os.makedirs(image_dst_dir, exist_ok=True)

    for image_src_path in tqdm(image_src_paths):
        image_id = os.path.basename(image_src_path).split(".")[0]
        image = cv2.imread(image_src_path)
        _resize_and_save(
            image=image,
            label=None,
            name=image_id,
            image_dst_dir=image_dst_dir,
            label_dst_dir=None,
            generate_npy=generate_npy,
            min_size=min_size,
            max_size=max_size
        )


def _qnrf(
    data_src_dir: str,
    data_dst_dir: str,
    min_size: int,
    max_size: int,
    generate_npy: bool = False
) -> None:
    for split in ["train", "val"]:
        generate_npy = generate_npy and split == "train"
        print(f"Processing {split}...")
        if split == "train":
            image_src_dir = os.path.join(data_src_dir, "Train")
            label_src_dir = os.path.join(data_src_dir, "Train")
            image_src_paths = glob(os.path.join(image_src_dir, "*.jpg"))
            label_src_paths = glob(os.path.join(label_src_dir, "*.mat"))
            assert len(image_src_paths) == len(label_src_paths) == 1201, f"Expected 1201 images and labels, got {len(image_src_paths)} images and {len(label_src_paths)} labels"
        else:
            image_src_dir = os.path.join(data_src_dir, "Test")
            label_src_dir = os.path.join(data_src_dir, "Test")
            image_src_paths = glob(os.path.join(image_src_dir, "*.jpg"))
            label_src_paths = glob(os.path.join(label_src_dir, "*.mat"))
            assert len(image_src_paths) == len(label_src_paths) == 334, f"Expected 334 images and labels, got {len(image_src_paths)} images and {len(label_src_paths)} labels"
        
        sort_key = lambda x: int((os.path.basename(x).split(".")[0]).split("_")[1])
        image_src_paths.sort(key=sort_key)
        label_src_paths.sort(key=sort_key)

        image_dst_dir = os.path.join(data_dst_dir, split, "images")
        label_dst_dir = os.path.join(data_dst_dir, split, "labels")
        os.makedirs(image_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)
    
        size = len(str(len(image_src_paths)))
        for i, (image_src_path, label_src_path) in tqdm(enumerate(zip(image_src_paths, label_src_paths)), total=len(image_src_paths)):
            image_id = int((os.path.basename(image_src_path).split(".")[0]).split("_")[1])
            label_id = int((os.path.basename(label_src_path).split(".")[0]).split("_")[1])
            assert image_id == label_id, f"Expected image id {image_id} to match label id {label_id}"
            name = f"{(i + 1):0{size}d}"
            image = cv2.imread(image_src_path)
            label = loadmat(label_src_path)["annPoints"]
            _resize_and_save(
                image=image,
                label=label,
                name=name,
                image_dst_dir=image_dst_dir,
                label_dst_dir=label_dst_dir,
                generate_npy=generate_npy,
                min_size=min_size,
                max_size=max_size
            )

        if split == "train":
            _generate_random_indices(len(image_src_paths), os.path.join(data_dst_dir, split))

def _jhu(
    data_src_dir: str,
    data_dst_dir: str,
    min_size: int,
    max_size: int,
    generate_npy: bool = False
) -> None:
    for split in ["train", "val"]:
        generate_npy = generate_npy and split == "train"
        if split == "train":
            with open(os.path.join(data_src_dir, "train", "image_labels.txt"), "r") as f:
                train_names = f.read().splitlines()
            train_names = [name.split(",")[0] for name in train_names]
            train_image_src_paths = [os.path.join(data_src_dir, "train", "images", f"{name}.jpg") for name in train_names]
            train_label_src_paths = [os.path.join(data_src_dir, "train", "gt", f"{name}.txt") for name in train_names]

            with open(os.path.join(data_src_dir, "val", "image_labels.txt"), "r") as f:
                val_names = f.read().splitlines()
            val_names = [name.split(",")[0] for name in val_names]
            val_image_src_paths = [os.path.join(data_src_dir, "val", "images", f"{name}.jpg") for name in val_names]
            val_label_src_paths = [os.path.join(data_src_dir, "val", "gt", f"{name}.txt") for name in val_names]

            image_src_paths = train_image_src_paths + val_image_src_paths
            label_src_paths = train_label_src_paths + val_label_src_paths

        else:
            with open(os.path.join(data_src_dir, "test", "image_labels.txt"), "r") as f:
                test_names = f.read().splitlines()
            test_names = [name.split(",")[0] for name in test_names]
            image_src_paths = [os.path.join(data_src_dir, "test", "images", f"{name}.jpg") for name in test_names]
            label_src_paths = [os.path.join(data_src_dir, "test", "gt", f"{name}.txt") for name in test_names]

        image_dst_dir = os.path.join(data_dst_dir, split, "images")
        label_dst_dir = os.path.join(data_dst_dir, split, "labels")
        os.makedirs(image_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)

        size = len(str(len(image_src_paths)))
        for i, (image_src_path, label_src_path) in tqdm(enumerate(zip(image_src_paths, label_src_paths)), total=len(image_src_paths)):
            image_id = int(os.path.basename(image_src_path).split(".")[0])
            label_id = int(os.path.basename(label_src_path).split(".")[0])
            assert image_id == label_id, f"Expected image id {image_id} to match label id {label_id}"
            name = f"{(i + 1):0{size}d}"
            image = cv2.imread(image_src_path)
            with open(label_src_path, "r") as f:
                label = f.read().splitlines()
            label = np.array([list(map(float, line.split(" ")[0: 2])) for line in label])
            _resize_and_save(
                image=image,
                label=label,
                name=name,
                image_dst_dir=image_dst_dir,
                label_dst_dir=label_dst_dir,
                generate_npy=generate_npy,
                min_size=min_size,
                max_size=max_size
            )

        if split == "train":
            _generate_random_indices(len(image_src_paths), os.path.join(data_dst_dir, split))


def parse_args():
    parser = ArgumentParser(description="Pre-process datasets to resize images and labeld into a given range.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nwpu", "ucf_qnrf", "jhu", "shanghaitech_a", "shanghaitech_b"],
        required=True,
        help="The dataset to pre-process."
    )
    parser.add_argument("--src_dir", type=str, required=True, help="The root directory of the source dataset.")
    parser.add_argument("--dst_dir", type=str, required=True, help="The root directory of the destination dataset.")
    parser.add_argument("--min_size", type=int, default=256, help="The minimum size of the shorter side of the image.")
    parser.add_argument("--max_size", type=int, default=None, help="The maximum size of the longer side of the image.")
    parser.add_argument("--generate_npy", action="store_true", help="Generate .npy files for images.")

    args = parser.parse_args()
    args.src_dir = os.path.abspath(args.src_dir)
    args.dst_dir = os.path.abspath(args.dst_dir)
    args.max_size = float("inf") if args.max_size is None else args.max_size
    return args


if __name__ == "__main__":
    args = parse_args()
    _preprocess(
        dataset=args.dataset,
        data_src_dir=args.src_dir,
        data_dst_dir=args.dst_dir,
        min_size=args.min_size,
        max_size=args.max_size,
        generate_npy=args.generate_npy
    )
