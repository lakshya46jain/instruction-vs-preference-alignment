# Hybrid Model (SFT + DPO)

## Overview

This model represents the **Hybrid** training approach: combining Supervised Fine-Tuning (SFT) with Direct Preference Optimization (DPO). It starts from the SFT checkpoint and applies preference optimization on top.

## Training Methodology

### Architecture (Following Jain's Design)

**Reference Model (Frozen):**
- Base: TinyLlama-1.1B-Chat-v1.0
- + SFT LoRA adapters (merged)
- Status: Frozen (non-trainable)
- Purpose: Serves as the reference distribution for DPO

**Policy Model (Trainable):**
- Base: TinyLlama-1.1B-Chat-v1.0
- + SFT LoRA adapters (merged)
- + NEW LoRA layer for DPO training
- Status: Trainable
- Purpose: Learns preference alignment while staying close to SFT reference

### Key Design Decisions

1. **Real Frozen Reference Model**: Unlike setting `ref_model=None`, we use a proper frozen copy of the merged SFT model. This is critical for valid DPO training.

2. **Starting from SFT**: Both policy and reference start from the SFT checkpoint, not base TinyLlama. This allows us to answer: "Does preference alignment provide gains beyond SFT?"

3. **Gradient Checkpointing Disabled**: Set to `False` to prevent zero gradients during DPO training (a known compatibility issue).

4. **Different Hyperparameters than DPO Baseline**: Uses extended training schedule to differentiate Hybrid from pure DPO approach.

### Training Pipeline

```
Base TinyLlama → SFT → DPO (with frozen SFT reference) = Hybrid Model
```

## Training Results

- **Duration**: 20 minutes (A100 GPU)
- **Initial Loss**: 0.6937 (log(2), expected when policy = reference)
- **Final Loss**: 0.0695 (90% reduction)
- **Final Accuracy**: 98-100%
- **Epochs**: 2
- **Learning Rate**: 5e-6 (cosine schedule)
- **Beta**: 0.3

## Model Location

### Files in `models/hybrid_output/`
- `adapter_model.safetensors` (49M) - DPO LoRA weights
- `adapter_config.json` - LoRA configuration
- `tokenizer*` files - Tokenization
- `README.md` - This file

## Loading Instructions

The Hybrid model requires loading **both** SFT and DPO LoRA adapters in sequence:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Step 1: Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Step 2: Load SFT LoRA adapters
model = PeftModel.from_pretrained(base_model, "models/sft_output")

# Step 3: Load Hybrid (DPO) LoRA adapters on top
model = PeftModel.from_pretrained(model, "models/hybrid_output")

# Step 4 (Optional): Merge for faster inference
model = model.merge_and_unload()

# Step 5: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/hybrid_output")

print("✅ Hybrid model loaded successfully!")
```

## Comparison with Other Models

| Model | Pipeline | Reference Model |
|-------|----------|-----------------|
| **SFT** | Base → SFT | N/A |
| **DPO** | Base → SFT → DPO | Frozen SFT |
| **Hybrid** | Base → SFT → DPO (extended) | Frozen SFT |

The key difference between DPO and Hybrid is in the **training schedule and hyperparameters**, not the starting point. Both begin from SFT.

## Technical Details

**Trainable Parameters**: 12,615,680 (LoRA only)

**DPO Configuration**:
- Epochs: 2
- Batch Size: 2 (per device)
- Gradient Accumulation: 4
- Learning Rate: 5e-6
- LR Scheduler: Cosine
- Max Length: 512
- Beta: 0.3
- Gradient Checkpointing: False (critical fix)

**LoRA Configuration**:
- r: 16
- alpha: 32
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Dropout: 0.05

## Validation

To verify the model loads correctly:

```python
# Quick generation test
prompt = "Explain the difference between SFT and RLHF."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Credits

Trained as part of the "Instruction vs Preference Alignment" research project comparing SFT-only, DPO-only, and Hybrid (SFT+DPO) approaches.