#!/bin/bash
# Complete Data Pipeline for CLaRa with Top-K=5 Support
# Usage: bash scripts/run_data_pipeline_topk5.sh

set -e  # Exit on error

echo "🚀 CLaRa Data Pipeline - Top-K=5 Mode"
echo "======================================"
echo ""

# Configuration
RAW_DATA_DIR="${RAW_DATA_DIR:-./raw_data}"
OUTPUT_DIR="${OUTPUT_DIR:-example}"
EXTRACTED_ASSETS_DIR="${EXTRACTED_ASSETS_DIR:-$OUTPUT_DIR/extracted_assets}"
API_KEY="${OPENAI_API_KEY}"
BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MODEL="${MODEL:-qwen-turbo}"
VISION_MODEL="${VISION_MODEL}"
TARGET_TOP_K="${TARGET_TOP_K:-5}"
USE_EMBEDDINGS="${USE_EMBEDDINGS:-false}"
EXTRACTION_MODE="${EXTRACTION_MODE:-dict}"

# Check API key
if [ -z "$API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo "   Please run: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "📋 Configuration:"
echo "   Input Dir:        $RAW_DATA_DIR"
echo "   Output Dir:       $OUTPUT_DIR"
echo "   Model:            $MODEL"
echo "   Vision Model:     ${VISION_MODEL:-Not set (skip image extraction)}"
echo "   Target Top-K:     $TARGET_TOP_K"
echo "   Use Embeddings:   $USE_EMBEDDINGS"
echo "   Extraction Mode:  $EXTRACTION_MODE"
echo ""

# Step 1: Extract raw knowledge from documents
echo "📄 Step 1/4: Extracting text and images using Docling..."
echo "----------------------------------------"

# Clean up previous assets
if [ -d "$EXTRACTED_ASSETS_DIR" ]; then
    echo "Cleaning up previous assets in $EXTRACTED_ASSETS_DIR..."
    rm -rf "$EXTRACTED_ASSETS_DIR"
fi

if [ -d "$RAW_DATA_DIR" ]; then
    python scripts/extract_with_docling.py \
        --input_dir "$RAW_DATA_DIR" \
        --output_file "$OUTPUT_DIR/raw_knowledge.jsonl" \
        --image_output_dir "$EXTRACTED_ASSETS_DIR" \
        --extraction_mode "$EXTRACTION_MODE"

    # Count extracted documents
    DOC_COUNT=$(grep -c '"content"' "$OUTPUT_DIR/raw_knowledge.jsonl" 2>/dev/null || echo "0")
    echo "✅ Extracted $DOC_COUNT document entries"
else
    echo "⚠️  Input directory not found: $RAW_DATA_DIR"
    echo "   Assuming raw_knowledge.jsonl already exists..."

    if [ ! -f "$OUTPUT_DIR/raw_knowledge.jsonl" ]; then
        echo "❌ Error: $OUTPUT_DIR/raw_knowledge.jsonl not found"
        echo "   Please create input documents or provide raw_knowledge.jsonl"
        exit 1
    fi
fi

echo ""

# Step 1.5: Extract semantic info from images
echo "🖼️  Step 1.5/4: Extracting semantic info from images..."
echo "----------------------------------------"

if [ -n "$VISION_MODEL" ]; then
    echo "--- Pass 1: Processing standalone images in Raw Data Dir ---"
    python scripts/extract_images.py \
        --input_dir "$RAW_DATA_DIR" \
        --output_file "$OUTPUT_DIR/raw_knowledge.jsonl" \
        --model "$VISION_MODEL"

    echo "--- Pass 2: Processing embedded images extracted from docs ---"
    python scripts/extract_images.py \
        --input_dir "$EXTRACTED_ASSETS_DIR" \
        --output_file "$OUTPUT_DIR/raw_knowledge.jsonl" \
        --model "$VISION_MODEL"
else
    echo "⚠️  Skipping image extraction (VISION_MODEL env var not set)"
fi

echo ""

# Step 2: Synthesize training data with top-k support
echo "🤖 Step 2/4: Synthesizing training data (Top-K=$TARGET_TOP_K)..."
echo "----------------------------------------"

EMBED_FLAG=""
if [ "$USE_EMBEDDINGS" = "true" ]; then
    EMBED_FLAG="--use_embeddings"
    echo "🔍 Using embedding-based hard negative mining"
else
    echo "🎲 Using random negative sampling"
fi

python scripts/synthesize_data_topk.py \
    --input_file "$OUTPUT_DIR/raw_knowledge.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --target_top_k "$TARGET_TOP_K" \
    $EMBED_FLAG

echo ""

# Step 3: Validate generated data
echo "✅ Step 3/4: Validating generated data..."
echo "----------------------------------------"

# Count samples
PRETRAIN_COUNT=$(wc -l < "$OUTPUT_DIR/pretrain_data.jsonl" | tr -d ' ')
INSTRUCT_COUNT=$(wc -l < "$OUTPUT_DIR/instruction_data.jsonl" | tr -d ' ')
E2E_COUNT=$(wc -l < "$OUTPUT_DIR/end_to_end_data.jsonl" | tr -d ' ')

echo "📊 Data Statistics:"
echo "   Stage 1 (Pretrain):     $PRETRAIN_COUNT samples"
echo "   Stage 2 (Instruction):  $INSTRUCT_COUNT samples"
echo "   Stage 3 (End-to-End):   $E2E_COUNT samples"
echo ""

# Validate first sample
echo "🔍 Sample from end_to_end_data.jsonl:"
echo "----------------------------------------"
head -1 "$OUTPUT_DIR/end_to_end_data.jsonl" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Question: {data[\"question\"][:80]}...')
print(f'Docs Count: {len(data[\"docs\"])}')
print(f'Answer: {data[\"gold_answer\"][:80]}...')
"

echo ""
echo "✅ Step 4/4: Pipeline complete!"
echo "----------------------------------------"
echo ""
echo "📊 Summary:"
echo "   • Extraction Mode:  $EXTRACTION_MODE"
echo "   • Documents:        $DOC_COUNT entries"
echo "   • Pretrain Samples: $PRETRAIN_COUNT"
echo "   • Instruct Samples: $INSTRUCT_COUNT"
echo "   • E2E Samples:      $E2E_COUNT"
echo "   • Docs per sample:  $TARGET_TOP_K"
echo ""
echo "📝 Next Steps:"
echo "   1. Review generated data in $OUTPUT_DIR/"
echo "   2. Start training with: bash scripts/train_qwen3_clara.sh"
echo "   3. Ensure training config has --generation_top_k $TARGET_TOP_K"
echo ""
echo "📄 Output files:"
echo "   • $OUTPUT_DIR/raw_knowledge.jsonl"
echo "   • $OUTPUT_DIR/pretrain_data.jsonl"
echo "   • $OUTPUT_DIR/instruction_data.jsonl"
echo "   • $OUTPUT_DIR/end_to_end_data.jsonl"
