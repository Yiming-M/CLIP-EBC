#!/bin/sh
export CUDA_VISIBLE_DEVICES=0  # Set the GPU ID

python trainer.py \
    --model clip_resnet50 --input_size 448 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --dataset nwpu --augment \
    --count_loss dmcount
