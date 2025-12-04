# CLaRa Migration to Qwen3-4B-Instruct-2507

## Overview

This document describes the migration of CLaRa from Mistral-7B-Instruct-v0.2 to Qwen3-4B-Instruct-2507 as the base language model.

**Branch**: `migrate-qwen3-4b-instruct`

**Model**: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

## Motivation

- **Smaller model size**: 4B parameters vs 7B (faster training, lower memory)
- **Better instruction following**: Qwen3 is optimized for instruction tasks
- **Improved multilingual support**: Better Chinese-English bilingual performance
- **Recent training data**: Released in 2025 with updated knowledge cutoff
- **Active development**: Qwen team provides strong community support

## Changes Made

### 1. Training Scripts

All training scripts have been updated to use Qwen3-4B-Instruct-2507:

- `scripts/train_pretraining.sh` - Stage 1 (Compression Pretraining)
- `scripts/train_instruction_tuning.sh` - Stage 2 (Instruction Tuning)
- `scripts/train_stage_end_to_end.sh` - Stage 3 (End-to-End)
- `scripts/train_qwen3_clara.sh` - Simplified Qwen3 training script

**Changed line**:
```bash
# Before
MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"

# After
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
```

### 2. Documentation

- Updated `README.md` with new model references
- Created test script `test_model_loading.py` for validation
- Added this migration guide

### 3. Configuration

The `train_qwen3_clara.sh` script was incomplete and has been fixed with:
- Added `MODEL_PATH` configuration
- Added `CHECKPOINT_ROOT` and `SAVE_PATH` settings
- Added `NUM_GPUS` variable for easy GPU configuration
- Added `mkdir -p $SAVE_PATH` to create checkpoint directory

## Testing

### Quick Validation

Run the model loading test to verify compatibility:

```bash
python test_model_loading.py
```

This will:
1. Load the tokenizer
2. Load the model (CPU mode for quick validation)
3. Test tokenization
4. Test forward pass

Expected output:
```
✅ All tests passed! Model is compatible with CLaRa.
```

### Full Training Test

Start a minimal training run to verify the pipeline:

```bash
# Use the simplified Qwen3 script
export NUM_GPUS=1  # Adjust based on your hardware
bash scripts/train_qwen3_clara.sh
```

The script will use the example data in `example/pretrain_data.jsonl`.

## Model Specifications

### Qwen3-4B-Instruct-2507

| Property | Value |
|----------|-------|
| Parameters | 4.0B |
| Hidden Size | 2560 |
| Layers | 40 |
| Attention Heads | 20 |
| Context Length | 32,768 tokens |
| Vocabulary Size | 151,936 |
| Training Data Cutoff | January 2025 |

### Comparison with Mistral-7B-Instruct-v0.2

| Property | Mistral-7B | Qwen3-4B | Notes |
|----------|------------|----------|-------|
| Parameters | 7.0B | 4.0B | 43% reduction |
| Memory (FP16) | ~14GB | ~8GB | ~40% less VRAM |
| Context Length | 32,768 | 32,768 | Same |
| Multilingual | Good | Excellent | Better CN/EN |
| Training Speed | Baseline | ~1.8x faster | Estimated |

## Training Recommendations

### Hardware Requirements

**Minimum** (Single GPU):
- GPU: 24GB VRAM (e.g., RTX 4090, A5000)
- RAM: 32GB
- Storage: 50GB free space

**Recommended** (Multi-GPU):
- GPUs: 2-4x A100 (40GB) or equivalent
- RAM: 128GB
- Storage: 500GB SSD

### Hyperparameter Adjustments

Since Qwen3-4B is smaller than Mistral-7B, consider these adjustments:

```bash
# Increase batch size (model is smaller)
--train_batch_size 256  # vs 128 for Mistral
--micro_train_batch_size 4  # vs 2 for Mistral

# Learning rate might need tuning
--learning_rate 2e-4  # Start higher, tune down if unstable

# LoRA rank can be reduced
--lora_rank 32  # vs 64 for Mistral
```

### Expected Training Time

Estimated times for full 3-stage training on example data:

| Stage | Mistral-7B (4xA100) | Qwen3-4B (4xA100) |
|-------|---------------------|-------------------|
| Stage 1 | ~6 hours | ~3 hours |
| Stage 2 | ~4 hours | ~2 hours |
| Stage 3 | ~8 hours | ~4 hours |
| **Total** | **~18 hours** | **~9 hours** |

## Compatibility Notes

### Model Loading

CLaRa uses `AutoModelForCausalLM` and `AutoTokenizer`, which are compatible with Qwen3. The model will be automatically downloaded from HuggingFace on first use.

### Trust Remote Code

Qwen3 requires `trust_remote_code=True` in model loading. This is already configured in CLaRa's `modeling_clara.py`.

### Tokenizer Differences

- **Mistral**: Uses SentencePiece tokenizer
- **Qwen3**: Uses tiktoken-based tokenizer

Both are supported by `AutoTokenizer`, no code changes needed.

**Important**: CLaRa has special handling for custom tokenizer attributes:

1. **Slow Tokenizer**: Uses `use_fast=False` to avoid Rust-based tokenizers
2. **Attribute Restoration**: Implements `_ensure_tokenizer_attributes()` to restore custom attributes

This is necessary because:
- CLaRa adds custom attributes to tokenizers (`enc_token`, `mem_tokens`, `ae_token`, etc.)
- Even slow tokenizers lose custom Python attributes when pickled for multiprocessing DataLoader workers
- Without restoration, training fails with: `TypeError: can only concatenate str (not 'NoneType') to str`
- The `_ensure_tokenizer_attributes()` method automatically restores these attributes when needed

### Special Tokens

Qwen3 uses different chat templates. If you encounter formatting issues, check:

```python
# Qwen3 chat template
tokenizer.chat_template
```

## Migration Checklist

- [x] Update all training scripts with new MODEL_PATH
- [x] Fix train_qwen3_clara.sh configuration
- [x] Update README.md documentation
- [x] Create model loading test script
- [x] Create migration guide
- [x] Fix tokenizer initialization (use_fast=False)
- [x] Add tokenizer attribute restoration for multiprocessing
- [ ] Run model loading test
- [ ] Test Stage 1 training (compression pretraining)
- [ ] Test Stage 2 training (instruction tuning)
- [ ] Test Stage 3 training (end-to-end)
- [ ] Benchmark performance vs Mistral baseline
- [ ] Update CLAUDE.md with Qwen3 specifics

## Rollback Instructions

If you need to revert to Mistral-7B:

```bash
# Switch back to main branch
git checkout main

# Or manually change MODEL_PATH
export MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
```

## References

- **Qwen3 Model Card**: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- **Qwen Documentation**: https://qwen.readthedocs.io/
- **CLaRa Paper**: arXiv:2511.18659
- **Original CLaRa Implementation**: Uses Mistral-7B baseline

## Next Steps

1. Run `python test_model_loading.py` to verify model downloads correctly
2. Start with a small training run: `bash scripts/train_qwen3_clara.sh`
3. Monitor WandB metrics to compare with Mistral baseline
4. Adjust hyperparameters based on initial results
5. Run full 3-stage training pipeline

## Common Training Issues

### ZeroDivisionError: num_update_steps_per_epoch

**Error**: `ZeroDivisionError: integer division or modulo by zero`

**Cause**: Dataset size is smaller than batch size (e.g., 10 samples with batch_size=64)

**Status**: ✅ Fixed in commit 35a0580

**Solution**: The code now ensures `num_update_steps_per_epoch >= 1`, allowing training with small verification datasets.

### Tokenizer Custom Attributes Lost

**Error**: `TypeError: can only concatenate str (not 'NoneType') to str`

**Cause**: Tokenizer attributes (`enc_token`, `mem_tokens`) lost during multiprocessing

**Status**: ✅ Fixed in commit 41e5d1f

**Solution**: Added `_ensure_tokenizer_attributes()` method that automatically restores attributes when needed.

### Data Quality: Image Description Enrichment

**Issue**: How to properly use vision LLM-generated image descriptions in training data

**Background**:
- The data pipeline extracts images from documents and uses vision LLM to generate semantic descriptions
- Image descriptions are stored as standalone entries with `source_type: "image"` in `raw_knowledge.jsonl`
- Document text contains `[IMAGE_REF: example/extracted_assets/xxx.png]` markers at image positions

**Status**: ✅ Fixed in commit b6eae74

**Solution** (Image Description Enrichment):
1. Load all image descriptions into memory at synthesis start
2. Replace `[IMAGE_REF: ...]` markers with actual vision LLM descriptions
3. LLM sees enriched documents with full semantic context (text + image descriptions)
4. Can generate questions about image content (architecture diagrams, process flows, UI mockups)
5. Avoids questions about technical artifacts (file paths, folder names)

**Implementation**:
- `load_image_descriptions()`: Reads all image entries from input file
- `replace_image_refs_with_descriptions()`: Merges descriptions into document chunks
- Updated `PROMPT_TEMPLATE`: Allows image content questions, blocks technical artifact questions

**Two Types of Images** (commit 42c4d1d):
1. **Embedded images** (`extracted_assets/`): Extracted from PDF/DOCX, merged into source documents via `[IMAGE_REF: ...]`
2. **Standalone images** (`raw_data/`): Original image files, treated as independent documents

Both types are now correctly processed:
- Embedded images → descriptions merged into parent documents
- Standalone images → vision LLM descriptions become independent training samples

**Usage**:
```bash
# Automatic enrichment in new synthesis (recommended)
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --target_top_k 5

# For legacy data without image descriptions (fallback only)
python scripts/clean_extracted_assets_refs.py \
    --input example/pretrain_data_old.jsonl \
    --output example/pretrain_data_cleaned.jsonl
```

**Example Enrichment**:
```
Before: "架构设计如下 [IMAGE_REF: example/extracted_assets/arch.png]"
After:  "架构设计如下 [IMAGE DESCRIPTION] The diagram shows microservices architecture..."
```

## Questions or Issues?

If you encounter problems during migration:

1. Check model download: `huggingface-cli download Qwen/Qwen3-4B-Instruct-2507`
2. Verify GPU memory: `nvidia-smi`
3. Check logs in `wandb/` directory
4. Review error messages in training output
5. Ensure data files are complete (not truncated)

For Qwen3-specific issues, refer to:
- Qwen GitHub: https://github.com/QwenLM/Qwen
- Qwen Discord: https://discord.gg/qwenlm

---

**Migration Date**: 2025-12-04
**Branch**: `migrate-qwen3-4b-instruct`
**Status**: Ready for testing
