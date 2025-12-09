# Hybrid Model (SFT + DPO)

## Model Location

### GitHub (LoRA Adapters - 10MB)
- Location: `models/hybrid_output/`
- Format: LoRA adapters (DPO layer only)

## Loading Instructions

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load SFT adapters
model = PeftModel.from_pretrained(base, "models/sft_output")

# Load Hybrid (DPO) adapters on top
model = PeftModel.from_pretrained(model, "models/hybrid_output")

# Optional: Merge for faster inference
model = model.merge_and_unload()
```

## Training Details
- Base: TinyLlama-1.1B-Chat-v1.0
- Method: SFT → DPO (with frozen merged SFT reference)
- Time: ~16 minutes (A100)
- **Reference Model:** Frozen merged SFT model (proper DPO implementation)
- **Policy Model:** SFT (merged) + new LoRA layer for DPO (trainable, ~10MB)
- **Gradient Checkpointing:** Disabled (critical for DPO training)

## Architecture
- Reference: Base + SFT (merged, frozen)
- Policy: Base + SFT (merged) + DPO LoRA (trainable)
- Only the DPO LoRA weights are saved (~10MB)
