#!/bin/bash
#
# Data Pipeline: Extract -> Synthesize -> Ready for Training
#

set -e

# 1. Settings
# Adjust these paths via environment variables or use defaults
RAW_DATA_DIR="${RAW_DATA_DIR:-./raw_data}"
INTERMEDIATE_FILE="./example/raw_knowledge.jsonl"
TRAINING_DATA_DIR="./example"
EXTRACTED_ASSETS_DIR="./example/extracted_assets"

# API Settings (Set these env vars or hardcode them here)
# export OPENAI_API_KEY="sk-..."
# export BASE_URL="https://api.deepseek.com"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set."
    echo "Please export it: export OPENAI_API_KEY='sk-...'"
    exit 1
fi

echo "=== Step 1: Extracting text AND embedded images using Docling (Advanced) ==="
# Clean up previous assets to avoid duplication/corruption
if [ -d "$EXTRACTED_ASSETS_DIR" ]; then
    echo "Cleaning up previous assets in $EXTRACTED_ASSETS_DIR..."
    rm -rf "$EXTRACTED_ASSETS_DIR"
fi

# Using Docling for high-quality PDF/Doc layout analysis and image extraction
# Default to 'dict' mode for precise image positioning (use --extraction_mode markdown for legacy)
EXTRACTION_MODE="${EXTRACTION_MODE:-dict}"
python scripts/extract_with_docling.py \
    --input_dir "$RAW_DATA_DIR" \
    --output_file "$INTERMEDIATE_FILE" \
    --image_output_dir "$EXTRACTED_ASSETS_DIR" \
    --extraction_mode "$EXTRACTION_MODE"

echo "=== Step 1.5: Extracting semantic info from images (Optional) ==="
# Note: Ensure your API_KEY and MODEL support Vision (e.g., gpt-4o, qwen-vl-max)
# If using a text-only model (like deepseek-chat), this step might fail or need a different URL.
# You can comment this out if you don't want to process images.
if [ -n "$VISION_MODEL" ]; then
    echo "--- Pass 1: Processing standalone images in Raw Data Dir ---"
    python scripts/extract_images.py \
        --input_dir "$RAW_DATA_DIR" \
        --output_file "$INTERMEDIATE_FILE" \
        --model "$VISION_MODEL"
        
    echo "--- Pass 2: Processing embedded images extracted from docs ---"
    python scripts/extract_images.py \
        --input_dir "$EXTRACTED_ASSETS_DIR" \
        --output_file "$INTERMEDIATE_FILE" \
        --model "$VISION_MODEL"
else
    echo "Skipping image extraction (VISION_MODEL env var not set)"
fi

echo "=== Step 2: Synthesizing QA pairs using LLM ==="
# Default to qwen-turbo if MODEL env var is not set
MODEL_NAME="${MODEL:-qwen-turbo}"
echo "Using model: $MODEL_NAME"

python scripts/synthesize_data.py \
    --input_file "$INTERMEDIATE_FILE" \
    --output_dir "$TRAINING_DATA_DIR" \
    --chunk_size 1000 \
    --model "$MODEL_NAME"

echo "=== Pipeline Complete ==="
echo "Training data saved to directory: $TRAINING_DATA_DIR"
echo "(Files: pretrain_data.jsonl, instruction_data.jsonl, end_to_end_data.jsonl)"
echo "You can now run: bash scripts/train_qwen3_clara.sh"
