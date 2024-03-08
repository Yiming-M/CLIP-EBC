import torch
from torch import Tensor
from scipy.ndimage import gaussian_filter
from typing import Optional, List, Tuple


def get_id(x: str) -> int:
    return int(x.split(".")[0])


def generate_density_map(label: Tensor, height: int, width: int, sigma: Optional[float] = None) -> Tensor:
    """
    Generate the density map based on the dot annotations provided by the label.
    """
    density_map = torch.zeros((1, height, width), dtype=torch.float32)

    if len(label) > 0:
        assert len(label.shape) == 2 and label.shape[1] == 2, f"label should be a Nx2 tensor, got {label.shape}."
        label_ = label.long()
        label_[:, 0] = label_[:, 0].clamp(min=0, max=width - 1)
        label_[:, 1] = label_[:, 1].clamp(min=0, max=height - 1)
        density_map[0, label_[:, 1], label_[:, 0]] = 1.0

    if sigma is not None:
        assert sigma > 0, f"sigma should be positive if not None, got {sigma}."
        density_map = torch.from_numpy(gaussian_filter(density_map, sigma=sigma))

    return density_map


def collate_fn(batch: List[Tensor]) -> Tuple[Tensor, List[Tensor], Tensor]:
    batch = list(zip(*batch))
    images = batch[0]
    assert len(images[0].shape) == 4, f"images should be a 4D tensor, got {images[0].shape}."
    if len(batch) == 4:  # image, label, density_map, image_name
        images = torch.cat(images, 0)
        points = batch[1]  # list of lists of tensors, flatten it
        points = [p for points_ in points for p in points_]
        densities = torch.cat(batch[2], 0)
        image_names = batch[3]  # list of lists of strings, flatten it
        image_names = [name for names_ in image_names for name in names_]

        return images, points, densities, image_names

    elif len(batch) == 3:  # image, label, density_map
        images = torch.cat(images, 0)
        points = batch[1]
        points = [p for points_ in points for p in points_]
        densities = torch.cat(batch[2], 0)

        return images, points, densities
    
    elif len(batch) == 2:  # image, image_name. NWPU test dataset
        images = torch.cat(images, 0)
        image_names = batch[1]
        image_names = [name for names_ in image_names for name in names_]

        return images, image_names

    else:
        images = torch.cat(images, 0)

        return images
