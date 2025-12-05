# CLaRa Documentation

Complete documentation for CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning.

**Current Version**: Qwen3-4B-Instruct (4B parameters, multilingual support)
**Training**: Unified batch configuration (32/1), Top-K=5 multi-document retrieval

---

## 📚 Documentation Index

### Training & Model

- **[COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)** - Complete guide for training CLaRa on Google Colab
  - Three-stage training pipeline using Qwen3-4B-Instruct
  - Unified batch configuration (TRAIN_BATCH_SIZE=32, MICRO_BATCH_SIZE=1)
  - Top-K=5 multi-document training
  - GPU configuration and optimization tips
  - Troubleshooting common issues

- **[COLAB_QUICK_REFERENCE.md](COLAB_QUICK_REFERENCE.md)** - Quick reference for Colab training
  - Essential commands and configurations
  - Training parameter cheat sheet
  - Common issues and solutions

- **[QWEN3_MIGRATION.md](QWEN3_MIGRATION.md)** - Qwen3-4B-Instruct migration guide
  - ✅ Migration completed (2025-12-05)
  - Performance improvements (43% smaller, ~40% faster)
  - Unified training configuration
  - Troubleshooting tokenizer and GPU memory issues

### Data Processing

- **[DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)** - Complete data processing pipeline
  - Extract raw data from documents (PDF, PPTX, DOCX, images)
  - Process images with vision LLMs
  - Synthesize training data using LLMs
  - Data format specifications
  - End-to-end pipeline scripts

- **[TOPK_DATA_SYNTHESIS_GUIDE.md](TOPK_DATA_SYNTHESIS_GUIDE.md)** - Multi-document data synthesis
  - Generate Top-K=5 training data
  - Hard negative mining with embeddings
  - Multi-document retrieval training
  - Data validation and troubleshooting

- **[IMAGE_EXTRACTION_MODES.md](IMAGE_EXTRACTION_MODES.md)** - Image extraction strategies
  - Hybrid dict/markdown extraction modes
  - Vision LLM integration
  - Image description enrichment
  - Self-healing image validation

- **[DATA_SECURITY.md](DATA_SECURITY.md)** - Data security best practices
  - Handling sensitive training data
  - Separating code and data
  - Pre-publication checklist
  - Emergency response procedures

---

## 🚀 Quick Start

### 1. Training on Google Colab

The fastest way to get started is using Google Colab:

1. Read [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)
2. Open `training_colab_complete.ipynb` in Colab
3. Follow the step-by-step guide
4. Train your model in 3 stages

### 2. Preparing Your Data

To prepare custom training data:

1. Follow [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md) for basic data extraction
2. Use [TOPK_DATA_SYNTHESIS_GUIDE.md](TOPK_DATA_SYNTHESIS_GUIDE.md) for Top-K=5 multi-document data
3. Check [IMAGE_EXTRACTION_MODES.md](IMAGE_EXTRACTION_MODES.md) for image processing strategies
4. Ensure data security with [DATA_SECURITY.md](DATA_SECURITY.md)

### 3. Model Migration & Updates

If you're migrating or updating:

1. Check [QWEN3_MIGRATION.md](QWEN3_MIGRATION.md) for Qwen3 migration details
2. Review unified training configurations
3. Update your training scripts to use Top-K=5
4. Verify GPU memory management settings

---

## 📖 Additional Resources

### Main Documentation
- **[../README.md](../README.md)** - Project overview and quick start
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
- **[../CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)** - Community guidelines

### Academic Resources
- **Paper**: [CLaRa on arXiv](https://arxiv.org/abs/2511.18659)
- **HuggingFace**: [CLaRa Models](https://huggingface.co/probejie)

---

## 🆘 Getting Help

### Common Issues

1. **Training Errors**: Check [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md#常见问题)
2. **Data Format**: See [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)
3. **Top-K Data**: Consult [TOPK_DATA_SYNTHESIS_GUIDE.md](TOPK_DATA_SYNTHESIS_GUIDE.md)
4. **Qwen3 Migration**: Check [QWEN3_MIGRATION.md](QWEN3_MIGRATION.md#common-training-issues)
5. **GPU Memory**: See [QWEN3_MIGRATION.md](QWEN3_MIGRATION.md#gpu-memory-management)

### Support Channels

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Documentation**: Check this folder for detailed guides

---

## 📝 Document Updates

This documentation is continuously updated to reflect the latest features and best practices.

**Last Updated**: 2025-12-05

### Recent Updates (2025-12-05)
- ✅ Completed Qwen3-4B-Instruct migration
- ✅ Added unified batch configuration (32/1)
- ✅ Implemented Top-K=5 multi-document training
- ✅ Added GPU memory management guide
- ✅ Enhanced data pipeline with image extraction modes
- ✅ Updated all training guides for Qwen3

---

## 📊 Available Documentation Files

Quick list of all documentation:
- Training: `COLAB_TRAINING_GUIDE.md`, `COLAB_QUICK_REFERENCE.md`, `QWEN3_MIGRATION.md`
- Data: `DATA_PIPELINE_GUIDE.md`, `TOPK_DATA_SYNTHESIS_GUIDE.md`, `IMAGE_EXTRACTION_MODES.md`, `DATA_SECURITY.md`
- Additional: `DATA_SYNTHESIS_PIPELINE.md`, `TOPK_IMPLEMENTATION_SUMMARY.md`, `TOPK_QUICKSTART.md`
- Core: `getting_started.md`, `training.md`, `inference.md`, `index.md`

---

**Made with ❤️ by the CLaRa Team**
