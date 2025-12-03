# **README – Supervised Fine-Tuning (SFT)**

This folder contains all scripts and configuration files used for **Supervised Fine-Tuning (SFT)** of our base model.

The final trained SFT LoRA weights are located in:

```
models/sft_output/
```

Once the SFT run finishes, this folder will contain the LoRA adapter used for evaluation.

---

To run evaluation, you will mainly need to:

### **1. Load the same base model used for SFT**

```
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### **2. Load the trained LoRA adapter**

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA = "models/sft_output/checkpoint-5000"   # final SFT checkpoint

tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base_model, LORA)
```

---

## **3. Apply the chat template when generating**

The base model is a **Chat model**, so prompts must be wrapped using:

```python
chat_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False,
    add_generation_prompt=True
)
```

Without the chat template, the model may just echo the prompt.

---

## **4. Compare Base vs SFT Outputs**

For each evaluation prompt:

1. Generate output using the **base model alone**
2. Generate output using **base + LoRA adapter**
3. Compare for:

   - Instruction following
   - Coherence
   - Correctness
   - Formatting
   - Overall quality

This will be used to build the evaluation tables and plots.

---

## **5. Example Script (sft_test.py)**

A minimal working example is provided in:

```
sft/sft_test.py
```

It demonstrates:

- Loading TinyLlama base
- Loading the SFT LoRA adapter
- Applying the chat template
- Generating a response

Use this as a reference when building the full evaluation pipeline.

---

## Files in This Folder

- **train_sft_mac.py** – SFT training script (Mac MPS)
- **sft_config.py** – hyperparameters and model config
- **sft_test.py** – quick test to verify the trained SFT model works

---

## Notes

- The evaluation scripts will take the SFT model and compare it against:
  - Base model
  - DPO model
  - Hybrid model
