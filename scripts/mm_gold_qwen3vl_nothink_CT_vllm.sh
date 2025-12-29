#!/bin/bash

# TRL VLLM does not support multi-modal models, thus we postpone this development to a later date.

cd /mnt/task_runtime/Med-R1

teacher_model_name_or_path=Qwen/Qwen3-VL-4B-Instruct
student_model_name_or_path=Qwen/Qwen3-VL-2B-Instruct
vllm_server_port=8001
vllm_server_host=127.0.0.1

# Start vllm server
# Check if trl vllm-server is already running (probe health endpoint first, then by process), start if not
if ! curl -s http://$vllm_server_host:$vllm_server_port/health/; then
  if ! pgrep -f "trl vllm-serve"; then
    echo "Starting vllm server..."
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "$student_model_name_or_path" --port "$vllm_server_port" --host "$vllm_server_host" --dtype bfloat16 --trust_remote_code > logs/vllm_server.log 2>&1 &
  else
    echo "trl vllm-server process is running, but not healthy yet."
    # kill the process
    echo "Killing trl vllm-serve process..."
    # Find all PIDs for the "trl vllm-serve" process and kill them one by one
    pids=$(pgrep -fi "vllm")
    if [ -n "$pids" ]; then
      for pid in $pids; do
        kill -9 $pid || true
      done
    fi
    # check if the process is killed
    if pgrep -fi "vllm"; then
      echo "Failed to kill trl vllm-serve process."
      exit 1
    else
      echo "trl vllm-serve process killed."
      exit 1
    fi
  fi
else
  echo "trl vllm-server is already running and healthy."
fi


max_attempts=20
attempt=1
while ! curl -s http://$vllm_server_host:$vllm_server_port/health/; do
  echo "Waiting for TRL vllm server... (attempt $attempt/$max_attempts)"
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "Failed to connect to TRL vllm server after $max_attempts attempts. Exiting."
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep 10
done

echo "trl vllm-server is ready."

# Multimodal Gold training script for Qwen3-VL

export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port=12346 \
         -m src.gold_multimodal.gold_multimodal \
         --model_name_or_path $student_model_name_or_path \
         --deepspeed src/training_configs/zero3.json \
         --teacher_model_name_or_path $teacher_model_name_or_path \
         --dataset_name "/mnt/task_runtime/data/omni_med_vqa_processed/open_access_sft_data_hf_modality_CT(Computed_Tomography)_train" \
         --learning_rate 2e-5 \
         --logging_steps 1 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 8 \
         --output_dir gold-model \
         --num_train_epochs 1 \
         --use_vllm True \
         --vllm_server_port $vllm_server_port \
         --vllm_server_host $vllm_server_host