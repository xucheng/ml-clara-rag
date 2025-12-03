# Top-K Training Quick Start

**Goal**: Generate and train with multi-document (top-k > 1) data for better retrieval and ranking performance.

## TL;DR

```bash
# 1. Generate top-k=5 training data
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --target_top_k 5

# 2. Validate the data
python scripts/validate_topk_data.py \
    --input_file example/end_to_end_data.jsonl \
    --expected_top_k 5

# 3. Upload to Colab and train with --generation_top_k 5
```

---

## Why Top-K > 1?

| Feature | Top-K=1 | Top-K=5 |
|---------|---------|---------|
| **Training Signal** | Binary relevance | Document ranking |
| **Model Capability** | Single-doc QA | Multi-doc fusion |
| **Retrieval Quality** | Basic | Advanced reranking |
| **Use Case** | Simple QA | Complex reasoning |

---

## Step-by-Step Guide

### Step 1: Check Your Current Data

```bash
python scripts/validate_topk_data.py --input_file example/end_to_end_data.jsonl
```

**Example output:**
```
📈 Documents per sample distribution:
   1 docs:   10 samples (100.0%)
```

If you see "1 docs", your data is top-k=1. You need to regenerate for top-k > 1.

---

### Step 2: Generate Top-K=5 Data

**Option A: Random Negatives (Fast, Cheap)**

```bash
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --base_url $BASE_URL \
    --model qwen-turbo \
    --target_top_k 5
```

**Pros**: Fast, works with any LLM API
**Cons**: Negative samples may be too easy (random/unrelated)

**Option B: Hard Negatives (Better Quality)**

```bash
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --base_url https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --target_top_k 5 \
    --use_embeddings
```

**Pros**: High-quality semantically similar negatives
**Cons**: Slower, requires OpenAI API for embeddings

---

### Step 3: Validate Generated Data

```bash
python scripts/validate_topk_data.py \
    --input_file example/end_to_end_data.jsonl \
    --expected_top_k 5
```

**Expected output:**
```
📈 Documents per sample distribution:
   5 docs:   10 samples (100.0%)

🎯 Expected top-k: 5
   ✅ All samples match expected top-k

📄 Sample (first entry):
Question: How do I implement gradient descent?...
Docs: 5 documents
  [1] Gradient descent is an optimization algorithm...
  [2] Neural networks consist of layers...
  [3] Learning rate controls the step size...
  [4] Batch normalization can accelerate training...
  [5] Dropout randomly drops neurons...
```

---

### Step 4: Train on Colab

1. **Upload new data to Google Drive**:
   ```bash
   # Compress the data
   tar -czvf clara-data-topk5.tar.gz example/*.jsonl

   # Upload to Google Drive (manual or using gdown)
   ```

2. **Open `training_colab_complete.ipynb` in Colab**

3. **Modify Cell 30 (Stage 3 Training)**:
   ```python
   # Find this line:
   --generation_top_k 1 \

   # Change to:
   --generation_top_k 5 \
   ```

4. **Run all cells**

---

## Data Format Comparison

**Top-K=1 Format:**
```json
{
  "question": "What is gradient descent?",
  "docs": [
    "Gradient descent is an optimization algorithm..."
  ],
  "gold_answer": "Gradient descent is..."
}
```

**Top-K=5 Format:**
```json
{
  "question": "What is gradient descent?",
  "docs": [
    "Neural networks consist of layers...",
    "Gradient descent is an optimization algorithm...",
    "Learning rate controls the step size...",
    "Batch normalization can accelerate training...",
    "Dropout randomly drops neurons..."
  ],
  "gold_answer": "Gradient descent is..."
}
```

**Key difference**: `docs` array has 5 elements instead of 1, with documents in random order (simulating retrieval results).

---

## Troubleshooting

### "RuntimeError: selected index k out of range"

**Cause**: `generation_top_k` exceeds available documents

**Solution**:
1. Check data: `python scripts/validate_topk_data.py --input_file example/end_to_end_data.jsonl`
2. Ensure training parameter matches data: `--generation_top_k N` where N = number of docs in data

### "ValueError: Not enough chunks for negative sampling"

**Cause**: Your `raw_knowledge.jsonl` has < 10 text chunks

**Solution**:
- Add more source documents, OR
- Reduce `--target_top_k` (e.g., use 3 instead of 5)

### Data has inconsistent document counts

**Output**:
```
⚠️  Warning: Inconsistent document counts detected!
   Found 2 different document counts: [1, 5]
```

**Solution**: Regenerate data with consistent `--target_top_k`:
```bash
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --target_top_k 5  # Use consistent value
```

---

## Performance Impact

Based on CLaRa paper benchmarks:

| Setting | Training Time | Retrieval Quality | Best Use Case |
|---------|--------------|-------------------|---------------|
| Top-K=1 | 1x (baseline) | Basic | Simple QA, prototyping |
| Top-K=5 | ~1.5x | +3-5% recall | Production RAG systems |
| Top-K=10 | ~2x | +5-8% recall | Complex multi-hop reasoning |

**Recommendation**: Start with top-k=5 for best balance of quality and speed.

---

## Complete Example

```bash
# 1. Extract knowledge from documents
python scripts/extract_with_docling.py \
    --input_dir data/raw \
    --output_file example/raw_knowledge.jsonl

# 2. Generate top-k=5 training data
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --model qwen-turbo \
    --target_top_k 5 \
    --use_embeddings  # Optional: better quality

# 3. Validate
python scripts/validate_topk_data.py \
    --input_file example/end_to_end_data.jsonl \
    --expected_top_k 5

# 4. Compress and upload
tar -czvf clara-topk5-data.tar.gz example/*.jsonl

# 5. Train on Colab with --generation_top_k 5
```

---

## FAQ

**Q: Can I mix top-k=1 and top-k=5 data?**
A: Not recommended. The model expects consistent document counts. Use separate training runs.

**Q: What if I have a small document corpus (< 20 chunks)?**
A: Use top-k=2 or top-k=3 instead of 5. The rule: `num_chunks >= target_top_k * 3`

**Q: Does top-k affect Stage 1 and Stage 2?**
A: No. Top-k only affects Stage 3 (End-to-End). Stage 1 and Stage 2 always use single documents.

**Q: Can I change top-k during training?**
A: No. The data format is fixed. To change top-k, regenerate data and retrain.

**Q: What's the cost difference?**
A: With `--use_embeddings`, costs increase ~50% (embedding API calls). Without it, cost is the same.

---

## Next Steps

- **Production deployment**: See [TOPK_DATA_SYNTHESIS_GUIDE.md](TOPK_DATA_SYNTHESIS_GUIDE.md)
- **Training optimization**: See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)
- **Data pipeline**: See [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)
