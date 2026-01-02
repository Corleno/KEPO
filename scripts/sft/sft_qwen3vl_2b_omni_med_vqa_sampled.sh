#!/bin/bash

set -e

# 进入 r1-v 子项目目录
cd src/r1-v

export CUDA_VISIBLE_DEVICES=0,1,2,3

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  -m src.open_r1.sft \
  --config configs/qwen3vl/sft_qwen3vl_2b_omni_med_vqa_sampled_config.yaml


