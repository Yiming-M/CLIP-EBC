# CLIP-EBC

The official implementation of CLIP-EBC, proposed in the paper *CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification*.

## Usage

To train a model, use `trainer.py`. An example `.sh` could be:

```bash
#!/bin/sh
export CUDA_VISIBLE_DEVICES=0  # Set the GPU ID

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

To evaluate get the result on NWPU Test, use the `test_nwpu.py` instead.

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

## Results on NWPU Test

| **Methods**                   | **MAE** | **RMSE** |
| ------------------------------|---------|----------|
| DMCount-EBC (based on VGG-19) | 83.7    | 376.03   |
| CLIP-EBC (based on ResNet50)  | 78.3    | 358.3    |
