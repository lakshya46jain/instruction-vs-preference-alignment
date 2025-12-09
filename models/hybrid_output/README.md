# Hybrid Fine-Tuning (SFT + DPO)

**Author:** Demi Omoremi  
**Date:** December 2025  
**Status:** ✅ Complete

## Overview

This is the hybrid approach combining Supervised Fine-Tuning (SFT) with Direct Preference Optimization (DPO) through sequential training.

## Training Method
```
Base Model (TinyLlama-1.1B)
    ↓
[SFT Training] (teammate's work)
    ↓
SFT Checkpoint (models/sft_output)
    ↓
[DPO Training] (my work)
    ↓
Hybrid Model ✓
```

## Model Access

### HuggingFace (Primary)
- **Repository:** https://huggingface.co/demi8824/tinyllama-sft-dpo-hybrid
- **Format:** Full merged model (not adapters)
- **Loading:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("demi8824/tinyllama-sft-dpo-hybrid")
tokenizer = AutoTokenizer.from_pretrained("demi8824/tinyllama-sft-dpo-hybrid")
```

### Google Drive (Backup)
- **Location:** MyDrive/hybrid_model_final/
- **Contents:** Model files, logs, evaluation results

## Training Configuration

- **Base Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Starting Point:** models/sft_output (SFT checkpoint)
- **Training Data:** data/processed/prefs_train.jsonl (5,000 examples)
- **Validation Data:** data/processed/prefs_val.jsonl (200 examples)
- **Hardware:** Google Colab Pro A100 GPU
- **Training Time:** ~20 minutes
- **Epochs:** 2
- **Batch Size:** 2 (per device), 4 gradient accumulation = 8 effective
- **Learning Rate:** 5e-6
- **DPO Beta:** 0.3

## Key Differences from Other Models

| Model | Format | Loading Method |
|-------|--------|----------------|
| SFT-only | LoRA adapters | Needs base + adapters |
| DPO-only | LoRA adapters | Needs base + adapters |
| **Hybrid (this)** | **Full merged model** | **Direct load** ✓ |

**Advantage:** Easier to evaluate - no adapter management needed!

## For Evaluation (Erik)

### Quick Load
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "demi8824/tinyllama-sft-dpo-hybrid",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("demi8824/tinyllama-sft-dpo-hybrid")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

# Generate
def generate(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
response = generate("Write a Python function to add two numbers.")
print(response)
```

### Expected Capabilities

1. **Instruction Following** (from SFT):
   - Should understand and follow task instructions
   - Should produce structured outputs
   - Should handle code generation well

2. **Preference Alignment** (from DPO):
   - Should prefer helpful over terse responses
   - Should avoid low-quality outputs
   - Should demonstrate better response quality

## Comparison Framework

The hybrid model should be evaluated against:

1. **SFT-only baseline:**
   - Same instruction-following capability
   - Better preference alignment (hybrid advantage)

2. **DPO-only:**
   - Better instruction-following (hybrid advantage)
   - Similar preference alignment

3. **Expected result:**
   - Best of both worlds
   - Combines strengths of both approaches

## Files in This Directory

- `README.md` - This file
- `train_hybrid.py` - Training script used
- `eval_hybrid.py` - Evaluation script
- `TRAINING_LOG.txt` - Training metrics and duration
- `EVAL_RESULTS.txt` - Sample evaluation outputs

## Training Results

- **Final Loss:** [See TRAINING_LOG.txt]
- **Training Duration:** ~20 minutes on T4 GPU
- **Model Size:** ~1.1B parameters (full merged model)
- **Format:** PyTorch, compatible with transformers library

## Reproducibility

To reproduce this training:
```bash
# Clone repo
git clone https://github.com/lakshya46jain/instruction-vs-preference-alignment.git
cd instruction-vs-preference-alignment

# Install dependencies
pip install transformers datasets peft accelerate trl

# Run training (requires GPU)
python hybrid/train_hybrid.py
```

## Notes

- Model has LoRA improvements permanently integrated (merged)
- Does not require separate base model loading
- Compatible with standard transformers pipeline
- Used official team preference dataset for fair comparison

## Troubleshooting

**If model doesn't load:**
- Check internet connection (downloads from HuggingFace)
- Ensure transformers library is up to date: `pip install --upgrade transformers`
- Try specifying dtype: `torch_dtype=torch.float16`

**For CUDA out of memory:**
- Use CPU: `device_map="cpu"`
- Or use 8-bit loading: `load_in_8bit=True`

## Contact

- Author: Demi Omoremi
- Email: [your email if you want]
- GitHub: @demi8824
- HuggingFace: @demi8824

## Acknowledgments

Part of CS 4804 project at Virginia Tech comparing instruction tuning vs preference alignment methods.

Team members:
- Lakshya Jain (project lead)
- Aditya Choudhary (data)
- Shriram Anand (DPO-only)
- Erik Garcia (evaluation)
- Demi Omoremi (hybrid approach)
