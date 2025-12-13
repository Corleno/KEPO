#!/bin/bash

# OPD (On-Policy Distillation) training script for Qwen3-VL
# Based on grpo_qwen2_5vl_nothink_CT.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
# OPD doesn't need DEBUG_MODE and LOG_PATH (no reward functions)
# export DEBUG_MODE="true"
# export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port=12346 \
         -m src.opd.opd_vqa_nothink \
         --output_dir /home/fayang/output/Med-R1/training/OPD/OPD_CT_Qwen3-VL-2B-Instruct_to_Qwen3-VL-4B-Instruct \
         --model_name_or_path /home/fayang/checkpoints/Qwen3-VL-2B-Instruct \
         --teacher_model_name_or_path /home/fayang/checkpoints/Qwen3-VL-4B-Instruct \
         --dataset_name "/data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf_modality_CT(Computed_Tomography)_train" \
         --deepspeed src/r1-v/local_scripts/zero3.json \
         --max_prompt_length 1024 \
         --max_completion_length 1024 \
         --per_device_train_batch_size 6 \
         --gradient_accumulation_steps 2 \
         --logging_steps 1 \
         --bf16 true \
         --report_to wandb \
         --gradient_checkpointing true \
         --attn_implementation flash_attention_2 \
         --max_pixels 401408 \
         --num_train_epochs 1 \
         --run_name Qwen3-VL-2B-OPD-CT \
         --save_steps 100 \
         --save_only_model true \
         --num_generations 4 \
         --dataloader_num_workers 4 \
         --dataloader_pin_memory true

