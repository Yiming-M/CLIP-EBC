# 🚀 CLIP-EBC 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-nwpu-crowd-val)](https://paperswithcode.com/sota/crowd-counting-on-nwpu-crowd-val?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-ucf-qnrf)](https://paperswithcode.com/sota/crowd-counting-on-ucf-qnrf?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-shanghaitech-b)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-b?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-shanghaitech-a)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-a?p=clip-ebc-clip-can-count-accurately-through)

The official implementation of **CLIP-EBC**, proposed in the paper [*CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification*](https://arxiv.org/abs/2403.09281v1). Pretrained weights are available on Google Drive [here](https://drive.google.com/drive/folders/1hEHRsyOxvtbnq8UR0iXnQ7kcKO7aaYVM?usp=sharing).

## 📣 Announcement

For CLIP-ViT, the current code only supports finetuning the whole image backbone (ViT). We are experimenting using [VPT](https://arxiv.org/abs/2203.12119) to see whether it can provide better results.

## Citation

If you find this work useful, please consider to cite:

- BibTex:
    ```latex
    @article{ma2024clip,
    title={CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification},
    author={Ma, Yiming and Sanchez, Victor and Guha, Tanaya},
    journal={arXiv preprint arXiv:2403.09281},
    year={2024}
    }
    ```
- MLA: Ma, Yiming, Victor Sanchez, and Tanaya Guha. "CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification." arXiv preprint arXiv:2403.09281 (2024).
- APA: Ma, Y., Sanchez, V., & Guha, T. (2024). CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification. arXiv preprint arXiv:2403.09281.


## Usage

### 1. Preprocessing

Download all datasets and unzipped them into the folder `data`.

- ShanghaiTech: https://www.kaggle.com/datasets/tthien/shanghaitech/data
- UCF-QNRF: https://www.crcv.ucf.edu/data/ucf-qnrf/
- NWPU-Crowd: https://www.crowdbenchmark.com/nwpucrowd.html

The `data` folder should look like:
```
data:
├─── ShanghaiTech
│   ├── part_A
│   │   ├── train_data
│   │   │   ├── images
│   │   │   └── ground-truth
│   │   └── test_data
│   │       ├── images
│   │       └── ground-truth
│   └── part_B
│       ├── train_data
│       │   ├── images
│       │   └── ground-truth
│       └── test_data
│           ├── images
│           └── ground-truth
├─── NWPU-Crowd
│   ├── images_part1
│   ├── images_part2
│   ├── images_part3
│   ├── images_part4
│   ├── images_part5
│   ├── mats
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└─── UCF-QNRF
    ├── Train
    └── Test
```

Then, run `bash run.sh` to preprocess the datasets. You can modify the names of the original datasets but do NOT change the names of the processed datasets.

### 2. Training

To train a model, use `trainer.py`. An example `.sh` could be:

```bash
#!/bin/sh
export CUDA_VISIBLE_DEVICES=0  # Set the GPU ID. Comment this line to use all available GPUs.

python trainer.py \
    --model vgg19_ae --input_size 448 --reduction 8 --truncation 4 --anchor_points average \
    --dataset nwpu --augment \
    --count_loss dmcount &&

python trainer.py \
    --model clip_resnet50 --input_size 448 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --dataset nwpu --augment \
    --count_loss dmcount &&

python trainer.py \
    --model clip_vit_b_16 --input_size 256 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --dataset nwpu --augment --num_crops 2 \
    --count_loss dmcount --resize_to_multiple --window_size 256
```

If you don't limit the number of devices, then all GPUs will be used and the code will run in a ddp style.

All available models:
- CLIP-based: `clip_resnet50`, `clip_resnet50x4`, `clip_resnet50x16`, `clip_resnet50x64`, `clip_resnet101`, `clip_vit_b_16`, `clip_vit_b_32`, `vit_l_14`.
- Encoder-Decoder: 
  - `vgg11_ae`, `vgg11_bn_ae`, `vgg13_ae`, `vgg13_bn_ae`, `vgg16_ae`, `vgg16_bn_ae`, `vgg19_ae` (the model used in [DMCount](https://github.com/cvlab-stonybrook/DM-Count) & [BL](https://github.com/ZhihengCV/Bayesian-Crowd-Counting)), `vgg19_bn_ae`;
  - `resnet18_ae`, `resnet34_ae`, `resnet50_ae`, `resnet101_ae`, `resnet152_ae`;
  - `csrnet`, `csrnet_bn`;
  - `cannet`, `cannet_bn`.
- Encoder:
  - `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`, `vit_h_14`;
  - `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`;
  - All `timm` models that support `features_only`, `out_indices` and contain the `feature_info` attribute.

Arguments in `trainer.py`:
- `model`: which model to train. See all available models above.
  - `input_size`: the crop size during training.
  - `reduction`: the reduction factor of the model. This controls the size of the output probability/density map.
  - `truncation`: parameter controlling label correction. Currently supported values:
      - `configs/reduction_8.json`: 2 (all datasets), 4 (all datasets), 11 (only UCF-QNRF).
      - `configs/reduction_16.json`: 16 (only UCF-QNRF).
      - `configs/reduction_19.json`: 19 (only UCF-QNRF).

      To train models in a regression manner, set `truncation` to `None`.
  - `anchor_points`: the representative count values in the paper. Set `average` to use the mean count value of the bin. Set `middle` to use the middle point of the bin.
  - `prompt_type`: how to represent the count value in the prompt (e.g., if `"word"`, then a prompt could be `"There are five people"`). Only supported for CLIP-based models.
  - `granularity`: the granularity of the bins. Choose from `"fine"`, `"dynamic"`, `"coarse"`.
- `dataset`: which dataset to train on. Choose from `"sha"`, `"shb"`, `"nwpu"`, `"qnrf"`.
  - `batch_size`: the batch size for training.
  - `num_crops`: the number of crops generated from each image.
  - `augment`: use the data augmentation or not. Below are the default parameters for augmentation:
    - `min_scale = 1.0`
    - `max_scale = 2.0`
    - `brightness = 0.1`
    - `contrast = 0.1`
    - `saturation = 0.1`
    - `hue = 0.0` (We found setting hue to positive values leads to `NaN` DMCount loss.)
    - `kernel_size = 5` (The kernel size of the Gaussian blur of the cropped image.)
    - `saltiness = 1e-3` (The proportion of pixels to be salted.)
    - `spiciness = 1e-3` (The proportion of pixels to be peppered.)
    - `jitter_prob = 0.2` (The probability of applying the jittering augmentation.)
    - `blur_prob = 0.2` (The probability of applying the Gaussian blur augmentation.)
    - `noise_prob = 0.5` (The probability of applying the salt-and-pepper noise augmentation.)
- `sliding_window`: use the sliding window prediction method or not in evaluation. Could be useful for transformer-based models.
  - `window_size`: the size of the sliding window.
  - `stride`: the stride of the sliding window.
  - `strategy`: how to handle overlapping regions. Choose from `"average"` and `"max"`.
  - If you want to test ViT models, set one of the following:
    - `resize_to_multiple`: resize the image to the nearest multiple of `window_size` before sliding window prediction.
    - `zero_pad_to_multiple`: zero-pad the image to the nearest multiple of `window_size` before sliding window prediction.
- `weight_count_loss`: the weight of the count loss (e.g. DMCount loss) in the total loss.
- `count_loss`: the count loss to use. Choose from `"dmcount"`, `"mae"`, `"mse"`.
- `lr`: the maximum learning rate, default to `1e-4`.
- `weight_decay`: the weight decay, default to `1e-4`.
- `warmup_lr`: the learning rate for the warm-up period, default to `1e-6`.
- `warmup_epochs`: the number of warm-up steps, default to `50`.
- `T_0`, `T_mult`, `eta_min`: the parameters for `CosineAnnealingWarmRestarts` scheduler. The learning rate will increase from `warmup_lr` to `lr` during the first `warmup_epochs` epochs, then adjusted by the cosine annealing schedule.
- `total_epochs`: the total number of epochs to train.
- `eval_start`: the epoch to start evaluation.
- `eval_freq`: the frequency of evaluation.
- `num_workers`: the number of workers for data loading.
- `local_rank`: do not set this argument. It is used for multi-GPU training.
- `seed`: the random seed, default to `42`.

### 3. Testing on NWPU Test

To evaluate get the result on NWPU Test, use the `test_nwpu.py` instead. You can download the pretrained weights [here](https://drive.google.com/drive/folders/1hEHRsyOxvtbnq8UR0iXnQ7kcKO7aaYVM?usp=sharing).

```bash
python test_nwpu.py \
    --model vgg19_ae --input_size 448 --reduction 8 --truncation 4 --anchor_points average \
    --weight_path ./checkpoints/nwpu/vgg19_ae_448_8_4_fine_1.0_dmcount_aug/best_mae.pth
    --device cuda:0 &&

python test_nwpu.py \
    --model clip_resnet50 --input_size 448 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --weight_path ./checkpoints/nwpu/clip_resnet50_word_448_8_4_fine_1.0_dmcount_aug/best_mae.pth
    --device cuda:0
```

#### Results on NWPU Test

| **Methods**                   | **MAE** | **RMSE** |
| ------------------------------|---------|----------|
| DMCount-EBC (based on VGG-19) | 83.7    | 376.0    |
| [CLIP-EBC (based on ResNet50)](https://www.crowdbenchmark.com/resultdetail.html?rid=149)  | 75.8    | 367.3    |

## Visualization

![Visualization](./assets/visualization.jpg)
