#!/usr/bin/env python3
"""
Test script to verify Qwen3-4B-Instruct-2507 model loading.
This validates that the model can be loaded correctly before starting full training.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading(model_path: str = "Qwen/Qwen3-4B-Instruct-2507"):
    """Test loading the base model and tokenizer."""

    print(f"Testing model loading: {model_path}")
    print("=" * 60)

    try:
        # Test tokenizer loading
        print("\n1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  - Vocab size: {len(tokenizer)}")
        print(f"  - Model max length: {tokenizer.model_max_length}")

        # Test model loading (CPU only for quick validation)
        print("\n2. Loading model (CPU mode for validation)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print(f"✓ Model loaded successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Hidden size: {model.config.hidden_size}")
        print(f"  - Number of layers: {model.config.num_hidden_layers}")
        print(f"  - Number of attention heads: {model.config.num_attention_heads}")
        print(f"  - Vocabulary size: {model.config.vocab_size}")

        # Test tokenization
        print("\n3. Testing tokenization...")
        test_text = "Hello, this is a test for CLaRa with Qwen3."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Tokenization successful")
        print(f"  - Input text: {test_text}")
        print(f"  - Token count: {tokens.input_ids.shape[1]}")

        # Test forward pass (no generation)
        print("\n4. Testing forward pass...")
        with torch.no_grad():
            outputs = model(**tokens)
        print(f"✓ Forward pass successful")
        print(f"  - Logits shape: {outputs.logits.shape}")

        print("\n" + "=" * 60)
        print("✅ All tests passed! Model is compatible with CLaRa.")
        print("\nYou can now proceed with training:")
        print("  bash scripts/train_qwen3_clara.sh")
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Error loading model: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection (model will be downloaded)")
        print("  2. Verify HuggingFace token if model requires authentication")
        print("  3. Check available disk space for model download")
        print("  4. Try running: huggingface-cli login")
        return False

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    success = test_model_loading(model_path)
    sys.exit(0 if success else 1)
