#!/bin/sh
export CUDA_VISIBLE_DEVICES=0  # Set the GPU ID. Comment this line to use all GPUs and DDP.

# Train the commonly used VGG19-based encoder-decoder model on NWPU-Crowd.
# Change `--dataset` to `sha` or `shb` or `qnrf` to train on ShanghaiTech A, or ShanghaiTech B, or UCF-QNRF.
python trainer.py \
    --model vgg19_ae --input_size 448 --reduction 8 --truncation 4 --anchor_points average \
    --dataset nwpu \
    --count_loss dmcount &&

# Train the CLIP-EBC (ResNet50) model on ShanghaiTech A. Use `--dataset shb` if you want to train on ShanghaiTech B.
python trainer.py \
    --model clip_resnet50 --input_size 448 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --dataset sha \
    # --sliding_window --window_size 448 --stride 448 \  # Uncomment this line to enable sliding window prediction with a stride size of 448.
    --count_loss dmcount &&

# Train the CLIP-EBC (ViT-B/16) model on UCF-QNRF, using VPT in training and sliding window prediction in testing.
# By default, 32 tokens for each layer are used in VPT. You can also set `--num_vpt` to change the number of tokens.
# By default, the deep visual prompt tuning is used. You can set `--shallow_vpt` to use the shallow visual prompt tuning.
# `--amp` enables automatic mixed precision training.
python trainer.py \
    --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4 \
    --dataset qnrf --batch_size 16 --amp \
    --num_crops 2 --sliding_window --window_size 224 --stride 224 --warmup_lr 1e-3 \
    --count_loss dmcount

# Generate results on NWPU-Crowd Test.
# python test_nwpu.py \
#     --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
#     --num_vpt 32 --vpt_drop 0.0 --sliding_window --stride 224 \
#     --weight_path ./checkpoints/nwpu/clip_vit_b_16_word_224_8_4_fine_1.0_dmcount/best_rmse_1.pth