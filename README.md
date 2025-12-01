##  CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning


[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-green)](https://arxiv.org/abs/2511.18659) [![License](https://img.shields.io/badge/License-Apple-blue)](LICENSE) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_Base-FFEB3B)](https://huggingface.co/probejie/CLaRa-Base) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_Instruct-FFEB3B)](https://huggingface.co/probejie/CLaRa-Instruct) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_End_to_end-FFEB3B)](https://huggingface.co/probejie/CLaRa-E2E)

This software project accompanies the research paper, **CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning**.

## Table of Contents

- [Updates](#updates)
- [Motivation](#motivation)
- [Three-Stage Training](#three-stage-training)
- [Quick Start](#quick-start)
- [Getting Started](#getting-started)
  - [1. Prepare code and environment](#1-prepare-code-and-environment)
  - [2. Data preparation](#2-data-preparation)
  - [3. Start training](#3-start-training)
  - [4. Distributed Training](#4-distributed-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Data Pipeline](#data-pipeline)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

### Updates

- Nov 25, 2025. Models are available on Huggingface.

### Motivation

Retrieval-Augmented Generation (RAG) enhances large language models with external knowledge but suffers from **long contexts** and **disjoint retrieval-generation optimization**. Existing soft compression frameworks face two key limitations: (i) reconstruction-based objectives bias compressors toward surface patterns rather than semantic preservation; (ii) retrievers and compressors are trained separately, requiring double encoding despite compressed vectors being inherently retrievable.

In this work, we investigate:

- **How can we improve semantic preservation in compressed representations through better pretraining objectives?**  
- **How can we unify retrieval and generation optimization to avoid redundant encoding and disjoint objectives?**  

<div align="center">

<img src="figs/intro.png" width="100%"/>

</div>

We design a Three-stage training approach and introduce document compression techniques to improve RAG efficiency. The key findings are listed below.

### Findings

- **Efficient Compression**: CLaRa achieves significant compression rates (32x-64x) while preserving essential information for accurate answer generation.

- **Three-Stage Training**: A carefully designed Three-stage training approach (compression pretraining + compression instruction tuning + end-to-end fine-tuning) enables effective learning of both retrieval and generation.

For more interesting findings, please refer to our original paper!

---

### Three-Stage Training

CLaRa uses a carefully designed three-stage training approach:

**Stage 1: Compression Pretraining**
- Train the compressor using KPCP framework with QA pairs and paraphrases
- Retain key semantics through QA-based and paraphrase-guided supervision
- Support compression rates of 1x-256x

**Stage 2: Compression Instruction Tuning**
- Fine-tune the compressor on instruction-following tasks for downstream QA
- Use text-based QA output to ensure compressed representations retain sufficient semantics

**Stage 3: End-to-End Fine-tuning (CLaRa)**
- Jointly train reranker and generator via a single language modeling loss
- Unify retrieval and generation in shared continuous space using differentiable top-k estimator

In this repository, we release our implementation of **CLaRa**, built upon [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

### Quick Start

```bash
# 1. Setup environment
conda create -n clara python=3.10 -y
conda activate clara
pip install -r requirements.txt

# 2. Configure settings
cp .env.example .env
# Edit .env with your API keys and paths

# 3. Process your data (optional - use provided examples to skip)
export RAW_DATA_DIR="./raw_data"
bash scripts/run_data_pipeline.sh

# 4. Train Stage 1 (Compression Pretraining)
export MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
bash scripts/train_pretraining.sh

# 5. Train Stage 2 (Instruction Tuning)
export PRETRAIN_CKPT="./checkpoints/clara_stage1"
bash scripts/train_instruction_tuning.sh

# 6. Train Stage 3 (End-to-End)
export PRETRAIN_CHECKPOINT="./checkpoints/clara_stage2"
bash scripts/train_stage_end_to_end.sh
```

### Project Structure

```
├── scripts/                      # Training, evaluation, and data processing scripts
│   ├── train_pretraining.sh     # Stage 1: Compression pretraining
│   ├── train_instruction_tuning.sh  # Stage 2: Compression instruction tuning
│   ├── train_stage_end_to_end.sh    # Stage 3: End-to-end training
│   ├── run_data_pipeline.sh     # Automated data pipeline
│   ├── extract_with_docling.py  # Document extraction
│   ├── extract_images.py        # Image processing with vision LLMs
│   ├── synthesize_data.py       # QA synthesis
│   └── evaluation_*.sh          # Evaluation scripts
├── openrlhf/                     # Core training framework
│   ├── models/                   # Model implementations
│   │   └── modeling_clara.py   # CLaRa model definition
│   ├── datasets/                 # Dataset handling
│   │   └── sft_dataset.py        # Training dataset
│   ├── trainer/                  # Training utilities
│   │   └── sft_trainer.py        # SFT trainer
│   └── cli/                      # Command line interface
│       └── train_sft.py          # Main training script
├── example/                      # Example training data
│   ├── pretrain_data.jsonl
│   ├── instruction_data.jsonl
│   └── end_to_end_data.jsonl
├── .env.example                  # Environment variables template
├── DATA_PIPELINE_GUIDE.md        # Detailed data pipeline documentation
└── README.md                     # This file
```

#### 1. Prepare code and environment

Clone the repository and set up the environment:

```bash
# Create conda environment
env=clara
conda create -n $env python=3.10 -y
conda activate $env

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your settings (API keys, paths, etc.)
source .env  # Or manually export the variables
```

Key dependencies include:
- PyTorch >= 2.0
- Transformers >= 4.20
- DeepSpeed >= 0.18
- Flash Attention 2
- Accelerate

**Environment Variables:**

Copy `.env.example` to `.env` and configure:

```bash
# Data pipeline settings
RAW_DATA_DIR=./raw_data           # Your raw documents directory
OPENAI_API_KEY=sk-...             # API key for LLM services
BASE_URL=https://...               # API endpoint
VISION_MODEL=qwen-vl-max          # Vision model for image processing
MODEL=qwen-turbo                   # Text model for synthesis

# Training settings
MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.2  # Base model
CHECKPOINT_ROOT=./checkpoints      # Where to save checkpoints
DATA_PATH=./data                   # Training data directory
```

#### 2. Data preparation

**Option A: Use the automated data pipeline (Recommended)**

The repository includes an end-to-end data pipeline that processes raw documents (PDF, DOCX, PPTX, images) into training-ready JSONL files:

```bash
# 1. Place your raw documents in a directory
mkdir -p raw_data
# Copy your PDF, DOCX, PPTX, and image files to raw_data/

# 2. Configure API settings
export RAW_DATA_DIR="./raw_data"
export OPENAI_API_KEY="sk-..."
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VISION_MODEL="qwen-vl-max"
export MODEL="qwen-turbo"

# 3. Run the data pipeline
bash scripts/run_data_pipeline.sh
```

**What happens under the hood:**

1. **Cleanup**: Removes old assets in `example/extracted_assets` to ensure a clean slate
2. **Step 1 - Docling Extraction**:
   - Parses PDF/DOCX/PPTX with layout-aware analysis
   - Extracts text to JSONL format
   - Extracts embedded images to `extracted_assets/`
3. **Step 1.5 - Vision Processing** (if `VISION_MODEL` is set):
   - **Pass 1**: Scans `RAW_DATA_DIR` for standalone JPG/PNG files
   - **Pass 2**: Scans `extracted_assets/` for embedded images from Step 1
   - **Self-Healing**: Validates images with Pillow, converts to JPEG, auto-skips corrupt files
4. **Step 2 - LLM Synthesis**:
   - Uses `$MODEL` to generate dense summaries
   - Creates bilingual QA pairs (3-5 pairs per chunk)
   - Supports cross-lingual retrieval (e.g., EN query → CN doc)

**Output files:**
- `example/clara_training_data.jsonl` - Final training data
- `example/raw_knowledge.jsonl` - Intermediate data (text + image descriptions)
- `example/pretrain_data.jsonl` - Stage 1 training data
- `example/instruction_data.jsonl` - Stage 2 training data
- `example/end_to_end_data.jsonl` - Stage 3 training data
- `example/extracted_assets/` - Extracted image files

**Option B: Manually prepare data**

Prepare training data in JSONL format. For pretraining stage:

```json
{
    "data_type": "qa",
    "question": ["Question 1"],
    "answers": ["Answer 1"],
    "docs": ["Document 1"]
}
```

For instruction tuning and end-to-end training:

```json
{
    "question": "Single question text",
    "docs": ["Document 1", "Document 2", ...],
    "gold_answer": "Reference answer"
}
```

#### 3. Start training

All training scripts now support environment variable configuration. Set the required variables before running:

```bash
# Common settings for all stages
export MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"  # Or your model path
export CHECKPOINT_ROOT="./checkpoints"
export DATA_PATH="./example"  # Where your JSONL files are located
```

**Stage 1: Compression Pretraining (KPCP)**

Train the document compressor using KPCP framework:

```bash
# Set data path (if different from default)
export DATA_PATH="./example"

# Run training
bash scripts/train_pretraining.sh
```

The model will be saved to `$CHECKPOINT_ROOT/clara_cluster2_2m_mix_stage1`

Key parameters (configured in the script):
- `--compress_rate`: Compression rate (default: 32)
- `--doc_max_length`: Maximum document length (default: 256)
- `--stage stage1`: Training stage
- `--mse_loss`: Use MSE loss to align compressed and original representations
- `--qa_loss`: Use QA loss for semantic preservation

**Stage 2: Compression Instruction Tuning**

Fine-tune the compressor on instruction-following tasks:

```bash
# Point to Stage 1 checkpoint
export PRETRAIN_CKPT="./checkpoints/clara_cluster2_2m_mix_stage1"

# Run training
bash scripts/train_instruction_tuning.sh
```

The model will be saved to `$CHECKPOINT_ROOT/clara_cluster1_2_2m_split_data_single_32_mistral`

Key parameters:
- `--pretrain_checkpoint`: Path to stage 1 checkpoint (via `$PRETRAIN_CKPT`)
- `--stage stage1_2`: Training stage
- `--generation_top_k`: Top-k sampling for generation (default: 5)
- `--mse_loss`: Use MSE loss for compression training
- `--do_eval_gen`: Enable generation evaluation

**Stage 3: End-to-End Training**

Jointly train retrieval and generation end-to-end:

```bash
# Point to Stage 2 checkpoint
export PRETRAIN_CHECKPOINT="./checkpoints/clara_cluster1_2_2m_split_data_single_32_mistral"

# Run training
bash scripts/train_stage_end_to_end.sh
```

The model will be saved to `$CHECKPOINT_ROOT/clara_stage2_debug`

Key parameters:
- `--pretrain_checkpoint`: Path to stage 2 checkpoint (via `$PRETRAIN_CHECKPOINT`)
- `--stage stage2`: Training stage
- `--generation_top_k`: Top-k sampling for generation
- `--do_eval_gen`: Enable generation evaluation

#### 4. Distributed Training

The training scripts support distributed training across multiple nodes and GPUs:

- `--max_len`: Maximum sequence length (default: 2048 for stage1/stage2, 1024 for stage3)
- `--train_batch_size`: Training batch size
- `--micro_train_batch_size`: Micro batch size for gradient accumulation
- `--learning_rate`: Learning rate (default: 1e-4 for stage1/stage2, 5e-6 for stage3)
- `--max_epochs`: Maximum training epochs
- `--zero_stage`: ZeRO optimization stage (default: 2)
- `--bf16`: Use bfloat16 precision
- `--flash_attn`: Use Flash Attention 2

### Inference

The CLaRa models can be loaded and used for inference. We provide three models corresponding to different training stages:

<details>
  <summary>Stage 1: Compression Pretraining model (click to expand)</summary>

  ```python
  from transformers import AutoModel

  model_path = "path/to/stage1/model"
  model = AutoModel.from_pretrained(
      model_path, 
      trust_remote_code=True
  ).to('cuda')

  # Example documents
  documents = [
      [
          "Document 1 content...",
          "Document 2 content...",
          "Document 3 content..."
      ]
  ]

  questions = ["" for _ in range(len(documents))]

  # Generate paraphrase from compressed representations
  output = model.generate_from_paraphrase(
      questions=questions, 
      documents=documents, 
      max_new_tokens=64
  )
  
  print('Generated paraphrase:', output[0])
  ```

</details>

<details>
  <summary>Stage 2: Compression Instruction Tuning model (click to expand)</summary>

  ```python
  from transformers import AutoModel

  model_path = "path/to/stage2/model"
  model = AutoModel.from_pretrained(
      model_path, 
      trust_remote_code=True
  ).to('cuda')

  # Example documents and question
  documents = [
      [
          "Document 1 content...",
          "Document 2 content...",
          "Document 3 content..."
      ]
  ]

  questions = ["Your question here"]

  # Generate answer from compressed representations
  output = model.generate_from_text(
      questions=questions, 
      documents=documents, 
      max_new_tokens=64
  )
  
  print('Generated answer:', output[0])
  ```

</details>

<details>
  <summary>Stage 3: End-to-End (CLaRa) model (click to expand)</summary>

  ```python
  from transformers import AutoModel

  model_path = "path/to/stage3/model"
  model = AutoModel.from_pretrained(
      model_path, 
      trust_remote_code=True
  ).to('cuda')

  # Example documents and question
  # Note: Stage 3 supports retrieval with multiple candidate documents
  documents = [
      ["Document 1 content..." for _ in range(20)]  # 20 candidate documents
  ]

  questions = ["Your question here"]

  # Generate answer with retrieval and reranking
  # The top-k is decided by generation_top_k in config.json
  output, topk_indices = model.generate_from_questions(
      questions=questions, 
      documents=documents, 
      max_new_tokens=64
  )
  
  print('Generated answer:', output[0])
  print('Top-k selected document indices:', topk_indices)
  ```

</details>

### Evaluation

The evaluation framework is based on standard RAG benchmarks. Run evaluation:

**End-to-end evaluation:**
```bash
bash scripts/evaluation_end_to_end.sh
```

**Instruction tuning evaluation:**
```bash
bash scripts/evaluation_instruction_tuning.sh
```

Supported datasets:
- **HotpotQA**: Multi-hop question answering
- **MuSiQue**: Multi-hop question answering with diverse reasoning
- **2WikiMultiHopQA**: Multi-hop question answering over Wikipedia
- **Natural Questions**: Open-domain question answering



### Results

#### Compression Performance

We evaluate our document compressor on four QA datasets (NQ, HotpotQA, MuSiQue, 2WikiMultiHopQA) under two settings: **Normal** (retrieving top-5 documents) and **Oracle** (gold document included). CLaRa consistently outperforms all baselines across different compression ratios.

<div align="center">

**Main Results (Mistral-7B, Normal Setting)**

| Model | CR | NQ | HotpotQA | MuSiQue | 2Wiki | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| AutoCompressor | - | 17.24 | 14.61 | 3.81 | 19.89 | 13.89 |
| XRAG | 128 | 32.35 | 25.16 | 3.64 | 28.79 | 22.48 |
| COCOM | 16 | 24.12 | 21.48 | 3.52 | 24.48 | 18.40 |
| PCC | 16 | 31.38 | 22.29 | 3.43 | 19.47 | 19.14 |
| LLMLingua-2 | 4 | 47.53 | 37.05 | 9.02 | 44.35 | 34.49 |
| PISCO | 16 | 54.39 | 41.94 | 10.09 | 44.88 | 37.83 |
| Mistral-7B w/ retrieval | - | 54.58 | 42.94 | 8.94 | 44.24 | 37.67 |
| **CLaRa (CR=4)** | **4** | **57.05** | **45.09** | **10.34** | **46.94** | **39.86** |
| **CLaRa (CR=16)** | **16** | **55.56** | **43.72** | **10.55** | **46.00** | **38.96** |
| **CLaRa (CR=32)** | **32** | **54.64** | **43.52** | **10.55** | **46.58** | **38.82** |

**Oracle Setting Results (Mistral-7B)**

| Model | CR | NQ | HotpotQA | MuSiQue | 2Wiki | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| PISCO | 16 | 73.44 | 66.53 | 33.80 | 60.45 | 58.55 |
| Mistral-7B w/ retrieval | - | 71.64 | 70.77 | 45.72 | 68.83 | 64.24 |
| **CLaRa (CR=4)** | **4** | **76.50** | **73.81** | **46.26** | **70.48** | **66.76** |
| **CLaRa (CR=16)** | **16** | **75.48** | **70.79** | **43.15** | **66.16** | **63.90** |
| **CLaRa (CR=32)** | **32** | **73.77** | **69.51** | **38.31** | **64.54** | **61.53** |

</div>

**Key Findings:**
- ✅ CLaRa outperforms PISCO by **+1.13%** (Normal) and **+5.35%** (Oracle) on average
- ✅ CLaRa outperforms LLMLingua-2 by **+5.37%** (Normal) on average  
- ✅ CLaRa matches/exceeds text-based baseline with **+2.36%** average gain on Mistral-7B

#### Retrieval Performance

<div align="center">

<img src="figs/main_recall.png" width="80%"/>

</div>

For detailed experimental results and analysis, please refer to our paper.

---

## Data Pipeline

The repository includes a comprehensive data pipeline for processing raw documents into training-ready data. The pipeline is designed for **hybrid environments**: data preparation on local machines (macOS/Linux) and optional model training on GPU clouds (Colab/AWS).

### Pipeline Architecture

```
Raw Documents (PDF/DOCX/PPTX/Images)
    ↓
[Step 1: Docling Extraction] → Text + Embedded Images
    ↓
[Step 1.5: Vision Processing] → Image Descriptions
    ↓
[Step 2: LLM Synthesis] → QA Pairs + Training Data
    ↓
Output: pretrain_data.jsonl, instruction_data.jsonl, end_to_end_data.jsonl
```

### Pipeline Components

| Component | Function | Key Technology |
|-----------|----------|----------------|
| **extract_with_docling.py** | Extract text and images from documents | IBM Docling (layout-aware) |
| **extract_images.py** | Generate semantic image descriptions | Vision LLMs (Qwen-VL/GPT-4o) |
| **synthesize_data.py** | Synthesize QA pairs and training data | Text LLMs (Qwen/GPT) |
| **run_data_pipeline.sh** | Orchestrate the entire pipeline | Bash automation |

### Key Features

- ✅ **Smart Extraction**: Layout-aware PDF parsing preserves document structure
- ✅ **Visual Understanding**: Automatic description of charts, diagrams, and flowcharts
- ✅ **Bilingual Support**: Cross-lingual QA pairs (e.g., EN query → CN doc)
- ✅ **Self-Healing**: Auto-validates images, skips corrupt files, supports resume
- ✅ **Flexible APIs**: Works with OpenAI, Aliyun DashScope, and compatible endpoints

### Quick Start

```bash
# 1. Prepare raw documents
mkdir -p raw_data
# Copy your PDF, DOCX, PPTX, and image files to raw_data/

# 2. Configure API settings
export RAW_DATA_DIR="./raw_data"
export OPENAI_API_KEY="sk-..."
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VISION_MODEL="qwen-vl-max"
export MODEL="qwen-turbo"

# 3. Run pipeline
bash scripts/run_data_pipeline.sh
```

### Pipeline Steps Explained

**Step 1 - Cleanup & Docling Extraction**
```bash
# Cleans previous assets
rm -rf example/extracted_assets

# Extracts text and images
python scripts/extract_with_docling.py \
    --input_dir "$RAW_DATA_DIR" \
    --output_file "./example/raw_knowledge.jsonl"
```

**Step 1.5 - Vision Processing** (Optional)
```bash
# Pass 1: Standalone images in RAW_DATA_DIR
# Pass 2: Embedded images from Step 1
python scripts/extract_images.py \
    --input_dir "$RAW_DATA_DIR" \
    --model "$VISION_MODEL"
```

**Step 2 - LLM Synthesis**
```bash
# Generates QA pairs for all 3 training stages
python scripts/synthesize_data.py \
    --input_file "./example/raw_knowledge.jsonl" \
    --output_dir "./example" \
    --model "$MODEL"
```

### Supported APIs

| Provider | BASE_URL | Models |
|----------|----------|--------|
| **Aliyun DashScope** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-turbo, qwen-vl-max |
| **OpenAI** | `https://api.openai.com/v1` | gpt-4o, gpt-4-turbo |
| **DeepSeek** | `https://api.deepseek.com` | deepseek-chat |

### Hybrid Training Workflow

For users without local GPUs (e.g., macOS users):

1. **Local**: Run data pipeline → Generate JSONL files
2. **Package**: `tar -czvf clara-data.tar.gz example/`
3. **Upload**: Transfer to Google Drive
4. **Cloud**: Run training on Colab/AWS with GPU

See [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md) for detailed Colab setup instructions.

---

## Troubleshooting

### Common Issues

**Environment Variables Not Working**
```bash
# Make sure to export variables before running scripts
export MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
export CHECKPOINT_ROOT="./checkpoints"

# Or source .env file
source .env
```

**Data Pipeline Issues**
- **401 Authentication Error**: Check `OPENAI_API_KEY` and `BASE_URL` match your provider
- **404 Model Not Found**: Set `MODEL` to match your provider (e.g., `qwen-turbo` for Aliyun)
- **Image Format Errors**: The pipeline auto-skips corrupt images (self-healing)

**Training Issues**
- **OOM Error**: Reduce `--micro_train_batch_size` in training scripts (default: 2 → 1)
- **NCCL Timeout**: Check `NCCL_DEBUG=INFO` environment variable is set
- **Checkpoint Not Found**: Verify `PRETRAIN_CKPT` or `PRETRAIN_CHECKPOINT` points to correct stage

**Path Issues**
- All paths now use environment variables with sensible defaults
- Use absolute paths for `MODEL_PATH` if loading local models
- Data paths default to `./example` if not specified

### Performance Optimization

- Enable gradient checkpointing: `--gradient_checkpointing`
- Use Flash Attention 2: `--flash_attn` (already enabled in scripts)
- Use mixed precision: `--bf16` (already enabled in scripts)
- Adjust compression rate: Lower values (4-16) = better quality, slower; Higher values (32-64) = faster, less quality

---

## Acknowledgments

We sincerely appreciate the following works for CLaRa:

- Our implementation is built upon the [OpenRLHF framework](https://github.com/OpenRLHF/OpenRLHF).

- Inspired by [PISCO-mistral](https://huggingface.co/naver/pisco-mistral) for document compression techniques

## Citation

```bibtex
@misc{he2025clarabridgingretrievalgeneration,
      title={CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning}, 
      author={Jie He and Richard He Bai and Sinead Williamson and Jeff Z. Pan and Navdeep Jaitly and Yizhe Zhang},
      year={2025},
      eprint={2511.18659},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.18659}, 
}
```
