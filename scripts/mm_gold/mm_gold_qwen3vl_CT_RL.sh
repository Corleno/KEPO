#!/bin/bash

# Multimodal Gold training script for Qwen3-VL

export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port=12346 \
         -m src.gold_multimodal.gold_multimodal \
         --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
         --deepspeed src/training_configs/zero3.json \
         --teacher_model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
         --dataset_name "/mnt/task_runtime/data/omni_med_vqa_processed/open_access_sft_data_hf_modality_CT(Computed_Tomography)_train" \
         --dataset_type vqa_thinking \
         --learning_rate 1e-6 \
         --logging_steps 1 \
         --bf16 true \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 1 \
         --output_dir gold-model-RL \
         --num_train_epochs 1 \
         --alpha 0.0 \
         --num_generations 8 \
         --attn_implementation flash_attention_2 \
         --reward_funcs accuracy format \
         --run_name Qwen3-VL-2B-MM-GOLD-GRPO-CT-Think \
         --push_to_hub True