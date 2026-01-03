#!/bin/bash

# Multimodal Gold training script for Qwen3-VL

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port=12346 \
         -m src.gold_multimodal.gold_multimodal \
         --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
         --deepspeed src/training_configs/zero3.json \
         --teacher_model_name_or_path Qwen/Qwen3-VL-32B-Instruct \
         --dataset_name "/mnt/task_runtime/data/omni_med_vqa_processed_sampled/open_access_sft_data_hf_modality_MRI_train_600" \
         --dataset_type vqa \
         --learning_rate 1e-6 \
         --logging_steps 1 \
         --bf16 true \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 1 \
         --output_dir /mnt/task_runtime/output/MM-GOLD/Qwen3-VL-2B-GKD-lmbd0-MRI-600 \
         --num_train_epochs 1 \
         --alpha 1.0 \
         --lmbda 0.0 \
         --attn_implementation flash_attention_2 \
         --run_name Qwen3-VL-2B-GKD-lmbd0-MRI-600 \