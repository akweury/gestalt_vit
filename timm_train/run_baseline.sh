#!/bin/bash

export WANDB_PROJECT=vit-lightweight
export WANDB_NAME=vit-tiny-baseline-tiny224
export WANDB_DIR=/logs  # 把 wandb 日志写到日志文件夹

CUDA_VISIBLE_DEVICES=6 python -m train /tiny-224 \
  --model vit_tiny_patch16_224 \
  --input-size 3 224 224 \
  --epochs 200 \
  --batch-size 128 \
  --workers 2 \
  --lr 5e-4 \
  --opt adamw \
  --weight-decay 0.05 \
  --sched cosine \
  --warmup-epochs 5 \
  --amp \
  --log-wandb \
  --experiment vit-tiny-baseline-tiny224 \
  --output output/vit-tiny-baseline-tiny224

CUDA_VISIBLE_DEVICES=5 python -m train /tiny-224 \
  --model adaptive_vit_tiny_patch16_224 \
  --input-size 3 224 224 \
  --epochs 200 \
  --batch-size 128 \
  --workers 2 \
  --lr 5e-4 \
  --opt adamw \
  --weight-decay 0.05 \
  --sched cosine \
  --warmup-epochs 5 \
  --amp \
  --log-wandb \
  --experiment vit-tiny-baseline-tiny224 \
  --output output/adaptive_vit_tiny_patch16_224