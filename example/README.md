# CLaRa Training Data Examples

This directory contains **anonymized example data** to demonstrate the data format required for CLaRa training.

**⚠️ Important**: These are small-scale examples (10 samples each) for demonstration purposes only. For actual training, you should prepare your own dataset following the same format.

---

## 📁 Data Files

### 1. `pretrain_data.jsonl` (Stage 1: Compression Pretraining)

**Format**: Each line is a JSON object with:
- `data_type`: Type of data (e.g., "qa")
- `question`: List of questions
- `answers`: List of answers
- `docs`: List of supporting documents

**Example**:
```json
{
  "data_type": "qa",
  "question": ["What is machine learning?"],
  "answers": ["Machine learning is a subset of AI..."],
  "docs": ["Machine learning algorithms build models from data..."]
}
```

**Purpose**: Train the model to compress knowledge from documents into latent representations.

**Sample count**: 10 examples (for demo only; actual training needs 1,000+ samples)

---

### 2. `instruction_data.jsonl` (Stage 2: Instruction Tuning)

**Format**: Each line is a JSON object with:
- `data_type`: "instruction"
- `instruction`: The task instruction
- `input`: Optional input context (can be empty)
- `output`: Expected output
- `docs`: Supporting documents

**Example**:
```json
{
  "data_type": "instruction",
  "instruction": "Explain the difference between X and Y.",
  "input": "",
  "output": "X does..., while Y does...",
  "docs": ["Background information about X and Y..."]
}
```

**Purpose**: Fine-tune the model to follow instructions and generate helpful responses.

**Sample count**: 10 examples (for demo only; actual training needs 1,000+ samples)

---

### 3. `end_to_end_data.jsonl` (Stage 3: End-to-End Training)

**Format**: Each line is a JSON object with:
- `data_type`: "end_to_end"
- `question`: List of questions
- `context`: List of context passages
- `answer`: Expected answer

**Example**:
```json
{
  "data_type": "end_to_end",
  "question": ["How do I implement X?"],
  "context": ["X is a technique that..."],
  "answer": "To implement X: 1) First step..., 2) Second step..."
}
```

**Purpose**: Train the complete RAG pipeline end-to-end.

**Sample count**: 10 examples (for demo only; actual training needs 1,000+ samples)

---

## 🔒 Using Your Own Data

### Option 1: Local Development

1. Prepare your data in the same format as the examples
2. Save to `data/internal/` directory (already protected by `.gitignore`):
   ```bash
   data/internal/pretrain_data.jsonl
   data/internal/instruction_data.jsonl
   data/internal/end_to_end_data.jsonl
   ```

### Option 2: Google Colab Training

**Method A: Google Drive**
1. Upload your data to Google Drive
2. In Colab, mount Drive and specify the path:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   PRETRAIN_DATA = '/content/drive/MyDrive/ml-clara-data/pretrain_data.jsonl'
   ```

**Method B: Upload Widget**
1. Use the upload widget in the training notebook (Cell 14)
2. Files are uploaded to the Colab runtime

**Method C: Private Git Repository**
1. Store data in a separate private repository
2. Clone in Colab using a personal access token

---

## 📊 Recommended Dataset Sizes

For effective training:

- **Stage 1 (Pretraining)**: 1,000 - 10,000 samples
  - Each with 1-5 documents
  - Covers your domain knowledge

- **Stage 2 (Instruction Tuning)**: 500 - 5,000 samples
  - Diverse instruction types
  - High-quality responses

- **Stage 3 (End-to-End)**: 500 - 5,000 samples
  - Complete QA pairs with context
  - Representative of real usage

---

## 🔍 Data Quality Guidelines

1. **Diversity**: Cover various topics and question types
2. **Quality**: Ensure accurate, well-written content
3. **Format**: Strictly follow the JSON schema
4. **Validation**: Test with small samples first
5. **Privacy**: Remove sensitive/confidential information

---

## 🛠️ Data Preparation Tools

The `scripts/` directory contains tools to help prepare your data:

- `extract_raw_data.py` - Extract text from documents (PDF, PPTX, DOCX)
- `synthesize_data.py` - Generate training data using LLMs
- `run_data_pipeline.sh` - Complete pipeline from documents to training data

See `DATA_PIPELINE_GUIDE.md` for detailed instructions.

---

## 📚 Additional Resources

- **DATA_SECURITY.md**: Guide for handling sensitive training data
- **COLAB_TRAINING_GUIDE.md**: Complete Colab training instructions
- **README.md**: Project overview and setup

---

**Note**: The example files in this directory are generic, public-domain content suitable for demonstration. They do not contain any proprietary or sensitive information.
