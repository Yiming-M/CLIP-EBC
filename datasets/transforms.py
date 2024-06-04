import torch
from torch import Tensor
from torchvision.transforms import ColorJitter as _ColorJitter
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, Union, Optional, Callable


def _crop(
    image: Tensor,
    label: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tuple[Tensor, Tensor]:
    image = TF.crop(image, top, left, height, width)
    if len(label) > 0:
        label[:, 0] -= left
        label[:, 1] -= top
        label_mask = (label[:, 0] >= 0) & (label[:, 0] < width) & (label[:, 1] >= 0) & (label[:, 1] < height)
        label = label[label_mask]

    return image, label


def _resize(
    image: Tensor,
    label: Tensor,
    height: int,
    width: int,
) -> Tuple[Tensor, Tensor]:
    image_height, image_width = image.shape[-2:]
    image = TF.resize(image, (height, width), interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if (image_height != height or image_width != width) else image
    if len(label) > 0 and (image_height != height or image_width != width):
        label[:, 0] = label[:, 0] * width / image_width
        label[:, 1] = label[:, 1] * height / image_height
        label[:, 0] = label[:, 0].clamp(min=0, max=width - 1)
        label[:, 1] = label[:, 1].clamp(min=0, max=height - 1)

    return image, label


class RandomCrop(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        assert len(self.size) == 2, f"size should be a tuple (h, w), got {self.size}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        crop_height, crop_width = self.size
        image_height, image_width = image.shape[-2:]
        assert crop_height <= image_height and crop_width <= image_width, \
            f"crop size should be no larger than image size, got crop size {self.size} and image size {image.shape}."
        
        top = torch.randint(0, image_height - crop_height + 1, (1,)).item()
        left = torch.randint(0, image_width - crop_width + 1, (1,)).item()
        return _crop(image, label, top, left, crop_height, crop_width)


class Resize(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        assert len(self.size) == 2, f"size should be a tuple (h, w), got {self.size}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        return _resize(image, label, self.size[0], self.size[1])


class Resize2Multiple(object):
    """
    Resize the image so that it satisfies:
        img_h = window_h + stride_h * n_h
        img_w = window_w + stride_w * n_w
    """
    def __init__(
        self,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> None:
        window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else window_size
        window_size = tuple(window_size)
        stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else stride
        stride = tuple(stride)
        assert len(window_size) == 2, f"window_size should be a tuple (h, w), got {window_size}."
        assert len(stride) == 2, f"stride should be a tuple (h, w), got {stride}."
        assert all(s > 0 for s in window_size), f"window_size should be positive, got {window_size}."
        assert all(s > 0 for s in stride), f"stride should be positive, got {stride}."
        assert stride[0] <= window_size[0] and stride[1] <= window_size[1], f"stride should be no larger than window_size, got {stride} and {window_size}."
        self.window_size = window_size
        self.stride = stride

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        image_height, image_width = image.shape[-2:]
        window_height, window_width = self.window_size
        stride_height, stride_width = self.stride
        new_height = int(max(round((image_height - window_height) / stride_height), 0) * stride_height + window_height)
        new_width = int(max(round((image_width - window_width) / stride_width), 0) * stride_width + window_width)

        if new_height == image_height and new_width == image_width:
            return image, label
        else:
            return _resize(image, label, new_height, new_width)


class ZeroPad2Multiple(object):
    def __init__(
        self,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> None:
        window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else window_size
        window_size = tuple(window_size)
        stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else stride
        stride = tuple(stride)
        assert len(window_size) == 2, f"window_size should be a tuple (h, w), got {window_size}."
        assert len(stride) == 2, f"stride should be a tuple (h, w), got {stride}."
        assert all(s > 0 for s in window_size), f"window_size should be positive, got {window_size}."
        assert all(s > 0 for s in stride), f"stride should be positive, got {stride}."
        assert stride[0] <= window_size[0] and stride[1] <= window_size[1], f"stride should be no larger than window_size, got {stride} and {window_size}."
        self.window_size = window_size
        self.stride = stride

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        image_height, image_width = image.shape[-2:]
        window_height, window_width = self.window_size
        stride_height, stride_width = self.stride
        new_height = int(max(np.ceil((image_height - window_height) / stride_height), 0) * stride_height + window_height)
        new_width = int(max(np.ceil((image_width - window_width) / stride_width), 0) * stride_width + window_width)

        if new_height == image_height and new_width == image_width:
            return image, label
        else:
            assert new_height >= image_height and new_width >= image_width, f"new size should be no less than the original size, got {new_height} and {new_width}."
            pad_height, pad_width = new_height - image_height, new_width - image_width
            return TF.pad(image, (0, 0, pad_width, pad_height), fill=0), label  # only pad the right and bottom sides so that the label coordinates are not affected


class RandomResizedCrop(object):
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.75, 1.25),
    ) -> None:
        """
        Randomly crop an image and resize it to a given size. The aspect ratio is preserved during this process.
        """
        self.size = size
        self.scale = scale
        assert len(self.size) == 2, f"size should be a tuple (h, w), got {self.size}."
        assert 0 < self.scale[0] <= self.scale[1], f"scale should satisfy 0 < scale[0] <= scale[1], got {self.scale}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        out_height, out_width = self.size
        # out_ratio = out_width / out_height

        scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()  # if scale < 1, then the image will be zoomed in, otherwise zoomed out
        in_height, in_width = image.shape[-2:]

        # if in_width / in_height < out_ratio:  # Image is too tall
        #     crop_width = int(in_width * scale)
        #     crop_height = int(crop_width / out_ratio)
        # else:  # Image is too wide
        #     crop_height = int(in_height * scale)
        #     crop_width = int(crop_height * out_ratio)

        crop_height, crop_width = int(out_height * scale), int(out_width * scale)

        if crop_height <= in_height and crop_width <= in_width:  # directly crop and resize the image
            top = torch.randint(0, in_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, in_width - crop_width + 1, (1,)).item()

        else:  # resize the image and then crop
            ratio = max(crop_height / in_height, crop_width / in_width)  # keep the aspect ratio
            resize_height, resize_width = int(in_height * ratio) + 1, int(in_width * ratio) + 1  # add 1 to make sure the resized image is no less than the crop size
            image, label = _resize(image, label, resize_height, resize_width)
            
            top = torch.randint(0, resize_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, resize_width - crop_width + 1, (1,)).item()

        image, label = _crop(image, label, top, left, crop_height, crop_width)
        return _resize(image, label, out_height, out_width)
        

class RandomHorizontalFlip(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        assert 0 <= self.p <= 1, f"p should be in range [0, 1], got {self.p}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = TF.hflip(image)

            if len(label) > 0:
                label[:, 0] = image.shape[-1] - 1 - label[:, 0]  # if width is 256, then 0 -> 255, 1 -> 254, 2 -> 253, etc.
                label[:, 0] = label[:, 0].clamp(min=0, max=image.shape[-1] - 1)

        return image, label
    

class ColorJitter(object):
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.4,
        contrast: Union[float, Tuple[float, float]] = 0.4,
        saturation: Union[float, Tuple[float, float]] = 0.4,
        hue: Union[float, Tuple[float, float]] = 0.2,
    ) -> None:
        self.color_jitter = _ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        return self.color_jitter(image), label
    

class RandomGrayscale(object):
    def __init__(self, p: float = 0.1) -> None:
        self.p = p
        assert 0 <= self.p <= 1, f"p should be in range [0, 1], got {self.p}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)

        return image, label
    

class GaussianBlur(object):
    def __init__(self, kernel_size: int, sigma: Optional[float] = None) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.gaussian_blur(image, self.kernel_size, self.sigma), label


class RandomApply(object):
    def __init__(self, transforms: Tuple[Callable, ...], p: Union[float, Tuple[float, ...]] = 0.5) -> None:
        self.transforms = transforms
        p = [p] * len(transforms) if isinstance(p, float) else p
        assert all(0 <= p_ <= 1 for p_ in p), f"p should be in range [0, 1], got {p}."
        assert len(p) == len(transforms), f"p should be a float or a tuple of floats with the same length as transforms, got {p}."
        self.p = p

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        for transform, p in zip(self.transforms, self.p):
            if torch.rand(1) < p:
                image, label = transform(image, label)

        return image, label


class PepperSaltNoise(object):
    def __init__(self, saltiness: float = 0.001, spiciness: float = 0.001) -> None:
        self.saltiness = saltiness
        self.spiciness = spiciness
        assert 0 <= self.saltiness <= 1, f"saltiness should be in range [0, 1], got {self.saltiness}."
        assert 0 <= self.spiciness <= 1, f"spiciness should be in range [0, 1], got {self.spiciness}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        noise = torch.rand_like(image)
        image = torch.where(noise < self.saltiness, 1., image)  # Salt
        image = torch.where(noise > 1 - self.spiciness, 0., image)    # Pepper
        return image, label
