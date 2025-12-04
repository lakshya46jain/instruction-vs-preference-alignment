# Hybrid Fine-Tuning (SFT + DPO)

**Author:** Demi Omoremi  
**Status:** ✅ Complete

## Overview
Hybrid approach combining Supervised Fine-Tuning (SFT) with Direct Preference Optimization (DPO).

## Method
1. Started with SFT checkpoint
2. Applied DPO training with preference data
3. Result: Combined instruction-following + preference alignment

## Training Details
- **Base Model:** TinyLlama-1.1B-Chat-v1.0
- **Hardware:** Google Colab T4 GPU
- **Duration:** 1h 44min
- **Dataset:** 5,000 HH-RLHF preference pairs
- **Final Loss:** 0.8301

## Configuration
- Epochs: 2.0
- Learning Rate: 5e-6
- Batch Size: 8 (effective)
- DPO Beta: 0.3
- LoRA Adapters: Yes (4.3 MB)

## Dataset
Preference data used for training is included:
- **File:** `preference_train.json`
- **Source:** Anthropic HH-RLHF dataset
- **Size:** 5,000 preference pairs
- **Format:** JSON with prompt, chosen, rejected fields

## Files
- `output/` - LoRA adapter weights (~9 MB)
  - `adapter_model.safetensors` (4.3 MB)
  - `tokenizer.json` (3.5 MB)
  - Config files
- `train_dpo_hybrid.py` - Training script
- `preference_train.json` - Training data
- `README.md` - This file
- `TRAINING_LOG.txt` - Detailed results

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "hybrid/output")

tokenizer = AutoTokenizer.from_pretrained("hybrid/output")
```

## Data Format
Each preference example contains:
```json
{
  "prompt": "instruction or question",
  "chosen": "preferred response",
  "rejected": "non-preferred response"
}
```

## Results
Training completed successfully. Model uses LoRA adapters for efficient storage (~9 MB vs ~500 MB for full model).