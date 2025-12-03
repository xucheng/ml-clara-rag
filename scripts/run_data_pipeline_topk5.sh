#!/bin/bash
# Complete Data Pipeline for CLaRa with Top-K=5 Support
# Usage: bash scripts/run_data_pipeline_topk5.sh

set -e  # Exit on error

echo "🚀 CLaRa Data Pipeline - Top-K=5 Mode"
echo "======================================"
echo ""

# Configuration
INPUT_DIR="${INPUT_DIR:-data/raw_documents}"
OUTPUT_DIR="${OUTPUT_DIR:-example}"
API_KEY="${OPENAI_API_KEY}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
MODEL="${MODEL:-gpt-4o-mini}"
TARGET_TOP_K="${TARGET_TOP_K:-5}"
USE_EMBEDDINGS="${USE_EMBEDDINGS:-true}"

# Check API key
if [ -z "$API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo "   Please run: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "📋 Configuration:"
echo "   Input Dir:    $INPUT_DIR"
echo "   Output Dir:   $OUTPUT_DIR"
echo "   Model:        $MODEL"
echo "   Target Top-K: $TARGET_TOP_K"
echo "   Use Embeddings: $USE_EMBEDDINGS"
echo ""

# Step 1: Extract raw knowledge from documents
echo "📄 Step 1/3: Extracting knowledge from documents..."
echo "----------------------------------------"

if [ -d "$INPUT_DIR" ]; then
    python scripts/extract_with_docling.py \
        --input_dir "$INPUT_DIR" \
        --output_file "$OUTPUT_DIR/raw_knowledge.jsonl"

    # Count extracted documents
    DOC_COUNT=$(wc -l < "$OUTPUT_DIR/raw_knowledge.jsonl" | tr -d ' ')
    echo "✅ Extracted $DOC_COUNT documents"
else
    echo "⚠️  Input directory not found: $INPUT_DIR"
    echo "   Assuming raw_knowledge.jsonl already exists..."

    if [ ! -f "$OUTPUT_DIR/raw_knowledge.jsonl" ]; then
        echo "❌ Error: $OUTPUT_DIR/raw_knowledge.jsonl not found"
        echo "   Please create input documents or provide raw_knowledge.jsonl"
        exit 1
    fi
fi

echo ""

# Step 2: Synthesize training data with top-k support
echo "🤖 Step 2/3: Synthesizing training data (Top-K=$TARGET_TOP_K)..."
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
echo "✅ Step 3/3: Validating generated data..."
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
echo "✅ Data pipeline complete!"
echo ""
echo "📝 Next Steps:"
echo "   1. Review generated data in $OUTPUT_DIR/"
echo "   2. Upload to Google Colab for training"
echo "   3. Ensure training config has --generation_top_k $TARGET_TOP_K"
echo ""
echo "🔗 For training instructions, see:"
echo "   - COLAB_TRAINING_GUIDE.md"
echo "   - TOPK_DATA_SYNTHESIS_GUIDE.md"
