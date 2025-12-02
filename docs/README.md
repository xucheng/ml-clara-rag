# CLaRa Documentation

Complete documentation for CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning.

---

## 📚 Documentation Index

### Training & Deployment

- **[COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)** - Complete guide for training CLaRa on Google Colab
  - Three-stage training pipeline (Compression → Instruction → End-to-End)
  - GPU configuration and optimization tips
  - Troubleshooting common issues

- **[COLAB_QUICK_REFERENCE.md](COLAB_QUICK_REFERENCE.md)** - Quick reference for Colab training
  - Essential commands and configurations
  - Training parameter cheat sheet
  - Common issues and solutions

- **[DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)** - Deployment guide for production
  - Model serving strategies
  - API endpoint setup
  - Performance optimization

### Data Processing

- **[DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)** - Complete data processing pipeline
  - Extract raw data from documents (PDF, PPTX, DOCX)
  - Synthesize training data using LLMs
  - Data format specifications
  - End-to-end pipeline scripts

- **[DATA_SECURITY.md](DATA_SECURITY.md)** - Data security best practices
  - Handling sensitive training data
  - Separating code and data
  - Pre-publication checklist
  - Emergency response procedures

### Development

- **[PUSH_SCRIPT_GUIDE.md](PUSH_SCRIPT_GUIDE.md)** - Guide for repository management
  - Using push scripts for batch operations
  - Automation best practices
  - Common workflows

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

1. Follow [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)
2. Extract knowledge from your documents
3. Synthesize training data
4. Ensure data security with [DATA_SECURITY.md](DATA_SECURITY.md)

### 3. Deploying Your Model

After training:

1. Check [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)
2. Set up model serving
3. Configure API endpoints
4. Monitor performance

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

1. **Training Errors**: Check [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md#troubleshooting)
2. **Data Format**: See [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md#data-format)
3. **Deployment**: Consult [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md#troubleshooting)

### Support Channels

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Documentation**: Check this folder for detailed guides

---

## 📝 Document Updates

This documentation is continuously updated to reflect the latest features and best practices.

**Last Updated**: 2025-12-02

---

**Made with ❤️ by the CLaRa Team**
