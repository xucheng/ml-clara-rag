#!/bin/bash
#
# Training script for CLaRa using Qwen3 (Stage 1: Compression Pretraining)
#

set -ex

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_PROJECT="CLaRa-Qwen3"

# CLaRa Training Stage
# stage1: Compression Pretraining (Learns to compress docs -> thought)
# stage2: Instruction Tuning (Learns to answer from thought)
# e2e: End-to-End Finetuning
STAGE="stage1"

# Automatically select dataset based on stage
if [ "$STAGE" == "stage1" ]; then
    DATASET_PATH="./example/pretrain_data.jsonl"
elif [ "$STAGE" == "stage2" ]; then
    DATASET_PATH="./example/instruction_data.jsonl"
else
    DATASET_PATH="./example/end_to_end_data.jsonl"
fi

echo "Starting training with $MODEL_PATH on dataset $DATASET_PATH (Stage: $STAGE)..."

torchrun --nproc_per_node $NUM_GPUS --master_port 29500 openrlhf/cli/train_sft.py \
   --max_len 2048 \
   --dataset $DATASET_PATH \
   --pretrain $MODEL_PATH \
   --train_batch_size 16 \
   --micro_train_batch_size 1 \
   --ckpt_path $SAVE_PATH \
   --max_samples 100000 \
   --save_path $SAVE_PATH \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --stage $STAGE \
   --generation_top_k 1 \
   --qa_loss \
   --doc_max_length 512 \
   --compress_rate 32 \
   --mse_loss \
   --gradient_checkpointing \
   --quantization int4 \
   --lora_rank 64 \
   --use_wandb "CLaRa-Qwen3"

echo "Training finished. Model saved to $SAVE_PATH"
