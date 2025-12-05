#!/bin/bash
#
# Complete 3-Stage Training Script for CLaRa using Qwen3-4B-Instruct
# Aligned with training_colab_complete.ipynb
#

set -e

# ============================================================================
# GPU Detection - Only run if GPU is available
# ============================================================================
echo "🔍 Checking GPU availability..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. This script requires NVIDIA GPU."
    echo "   Please run this script on a system with CUDA-capable GPU."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "❌ Error: No NVIDIA GPU detected."
    echo "   This script requires at least one CUDA-capable GPU to run."
    exit 1
fi

echo "✅ Found $GPU_COUNT GPU(s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ============================================================================
# Configuration (Aligned with training_colab_complete.ipynb)
# ============================================================================
export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_PROJECT="CLaRa-Qwen3"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
NUM_GPUS="${NUM_GPUS:-$GPU_COUNT}"

# Unified batch configuration (aligned with notebook)
TRAIN_BATCH_SIZE=32
MICRO_BATCH_SIZE=1

# Dataset paths
STAGE1_DATA="./example/pretrain_data.jsonl"
STAGE2_DATA="./example/instruction_data.jsonl"
STAGE3_DATA="./example/end_to_end_data.jsonl"

# Checkpoint paths
STAGE1_SAVE="$CHECKPOINT_ROOT/clara_qwen3_stage1"
STAGE2_SAVE="$CHECKPOINT_ROOT/clara_qwen3_stage2"
STAGE3_SAVE="$CHECKPOINT_ROOT/clara_qwen3_stage3"

echo "📋 Configuration:"
echo "   Model: $MODEL_PATH"
echo "   GPUs: $NUM_GPUS"
echo "   Batch Size: $TRAIN_BATCH_SIZE (micro: $MICRO_BATCH_SIZE)"
echo "   Stage 1 Output: $STAGE1_SAVE"
echo "   Stage 2 Output: $STAGE2_SAVE"
echo "   Stage 3 Output: $STAGE3_SAVE"
echo ""

# ============================================================================
# Stage 1: Compression Pretraining
# ============================================================================
echo "🚀 Stage 1/3: Compression Pretraining"
echo "========================================"
echo ""

mkdir -p $STAGE1_SAVE

torchrun --nproc_per_node $NUM_GPUS --master_port 29500 openrlhf/cli/train_sft.py \
   --max_len 2048 \
   --dataset $STAGE1_DATA \
   --pretrain $MODEL_PATH \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --micro_train_batch_size $MICRO_BATCH_SIZE \
   --ckpt_path $STAGE1_SAVE \
   --max_samples 100000 \
   --save_path $STAGE1_SAVE \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --stage stage1 \
   --qa_loss \
   --doc_max_length 512 \
   --compress_rate 32 \
   --mse_loss \
   --gradient_checkpointing \
   --quantization int4 \
   --lora_rank 64 \
   --use_wandb "CLaRa-Qwen3-Stage1"

echo "✅ Stage 1 complete. Checkpoint saved to $STAGE1_SAVE"
echo ""

# ============================================================================
# Process Cleanup Between Stages
# ============================================================================
echo "🧹 Cleaning up GPU memory..."
sleep 5

# Kill any lingering Python processes
pkill -9 -f "train_sft.py" || true
sleep 2

# Force PyTorch cleanup via Python
python3 -c "
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('✅ GPU memory cleared')
"

sleep 5
nvidia-smi
echo ""

# ============================================================================
# Stage 2: Instruction Tuning (Top-K=5)
# ============================================================================
echo "🚀 Stage 2/3: Instruction Tuning (Top-K=5)"
echo "========================================"
echo ""

mkdir -p $STAGE2_SAVE

torchrun --nproc_per_node $NUM_GPUS --master_port 29500 openrlhf/cli/train_sft.py \
   --max_len 2048 \
   --dataset $STAGE2_DATA \
   --pretrain $MODEL_PATH \
   --pretrain_checkpoint $STAGE1_SAVE \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --micro_train_batch_size $MICRO_BATCH_SIZE \
   --ckpt_path $STAGE2_SAVE \
   --max_samples 100000 \
   --save_path $STAGE2_SAVE \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --stage stage1_2 \
   --generation_top_k 5 \
   --doc_max_length 512 \
   --compress_rate 32 \
   --gradient_checkpointing \
   --quantization int4 \
   --lora_rank 64 \
   --use_wandb "CLaRa-Qwen3-Stage2"

echo "✅ Stage 2 complete. Checkpoint saved to $STAGE2_SAVE"
echo ""

# ============================================================================
# Process Cleanup Between Stages
# ============================================================================
echo "🧹 Cleaning up GPU memory..."
sleep 5

pkill -9 -f "train_sft.py" || true
sleep 2

python3 -c "
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('✅ GPU memory cleared')
"

sleep 5
nvidia-smi
echo ""

# ============================================================================
# Stage 3: End-to-End Training (Top-K=5)
# ============================================================================
echo "🚀 Stage 3/3: End-to-End Training (Top-K=5)"
echo "========================================"
echo ""

mkdir -p $STAGE3_SAVE

torchrun --nproc_per_node $NUM_GPUS --master_port 29500 openrlhf/cli/train_sft.py \
   --max_len 1024 \
   --dataset $STAGE3_DATA \
   --pretrain $MODEL_PATH \
   --pretrain_checkpoint $STAGE2_SAVE \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --micro_train_batch_size $MICRO_BATCH_SIZE \
   --ckpt_path $STAGE3_SAVE \
   --max_samples 100000 \
   --save_path $STAGE3_SAVE \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --stage stage2 \
   --generation_top_k 5 \
   --doc_max_length 512 \
   --compress_rate 32 \
   --gradient_checkpointing \
   --quantization int4 \
   --lora_rank 64 \
   --use_wandb "CLaRa-Qwen3-Stage3"

echo "✅ Stage 3 complete. Checkpoint saved to $STAGE3_SAVE"
echo ""

# ============================================================================
# Training Complete
# ============================================================================
echo "🎉 All 3 stages completed successfully!"
echo "========================================"
echo ""
echo "📊 Summary:"
echo "   Model: $MODEL_PATH"
echo "   Batch Config: ${TRAIN_BATCH_SIZE} (micro: ${MICRO_BATCH_SIZE})"
echo "   Top-K: 5 (Stage 2 & 3)"
echo ""
echo "📁 Checkpoints:"
echo "   Stage 1: $STAGE1_SAVE"
echo "   Stage 2: $STAGE2_SAVE"
echo "   Stage 3: $STAGE3_SAVE"
echo ""
echo "📝 Next Steps:"
echo "   1. Evaluate model: bash scripts/evaluation_end_to_end.sh"
echo "   2. Run inference tests on your data"
echo ""
