# Hybrid SFT + DPO README

This document provides a complete guide for preparing the environment, loading the required adapters, running the Hybrid training script, and evaluating the resulting model.
The Hybrid method extends the SFT pipeline by applying Direct Preference Optimization (DPO) on top of a merged SFT model, with modified training hyperparameters compared to the DPO baseline.

The Hybrid implementation ensures:

- The SFT model defines the reference distribution
- A new LoRA adapter learns preference alignment
- Only the new DPO-stage LoRA adapter is saved
- The SFT and Hybrid adapters can be composed cleanly

This README follows the same structure as the SFT and DPO READMEs for consistency.

---

# 1. Purpose of the Hybrid SFT + DPO Method

The goal of the Hybrid method is to evaluate whether **preference optimization, when applied after SFT, results in improvements over plain SFT**.

The Hybrid model is defined by:

1. Starting from the **SFT-trained model** (TinyLlama + SFT LoRA merged).
2. Creating a **frozen reference model** equal to that SFT model.
3. Adding a **new LoRA adapter** to the SFT-merged backbone.
4. Training this new adapter using **DPO**.
5. Saving only the new adapter so that inference becomes:

```
base model → apply SFT adapter → apply Hybrid adapter → merge
```

This divides responsibilities:

- SFT LoRA: instruction tuning
- Hybrid LoRA: preference alignment

---

# 2. Environment Setup

The Hybrid method uses the same environment as SFT and DPO.

---

## 2.1 Python environment

```bash
pyenv install 3.11.8
pyenv virtualenv 3.11.8 hybrid-env
pyenv local hybrid-env
```

---

## 2.2 Install PyTorch

Mac (MPS):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Linux CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 2.3 Install necessary libraries

```bash
pip install transformers datasets peft accelerate sentencepiece trl
```

No quantization libraries are required for the Hybrid stage.

---

# 3. Preparing the Preference Dataset

Hybrid training uses the same dataset format as the DPO baseline:

```
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

Expected dataset files:

```
data/processed/prefs_train.jsonl
data/processed/prefs_val.jsonl
```

These are identical to the files used in the standalone DPO stage.

---

# 4. Running Hybrid Training

All Hybrid training is performed by:

```
train_hybrid.py
```

This script implements the corrected Hybrid design:

1. Load TinyLlama
2. Load the SFT LoRA adapter and **merge** it
3. Clone the merged model to create the **frozen reference**
4. Attach a new LoRA adapter for preference learning
5. Run DPO with extended training hyperparameters
6. Save only the Hybrid LoRA adapter to:

```
models/hybrid_output/
```

---

## 4.1 Run the script

From the repository root:

```bash
python train_hybrid.py
```

You will see output indicating:

- Reference model created and frozen
- Policy LoRA applied
- Dataset loaded
- DPO trainer initialized
- Training progress logged
- Final adapter saved

---

# 5. What the Hybrid Training Script Does

This section mirrors SFT and DPO READMEs, explaining the conceptual sequence.

---

## 5.1 Construct the frozen reference model

The reference model is created from:

- TinyLlama + SFT LoRA merged into full weights
- A deep copy of this merged model
- All parameters frozen (`requires_grad=False`)

This ensures the DPO comparison is valid and consistent with the alignment pipeline:

```
Pretrain → SFT → DPO
```

The reference model sets the behavioral baseline that the policy model must not drift away from.

---

## 5.2 Construct the policy model

The policy model follows:

1. Load TinyLlama
2. Load SFT LoRA and merge into the backbone
3. Attach a **new** LoRA adapter for preference learning
4. Set this new adapter as trainable

This ensures that:

- SFT behavior is preserved
- Preference alignment is isolated to the new adapter
- SFT and Hybrid adapters remain cleanly composable

---

## 5.3 Load tokenizer and dataset

Tokenizer is loaded from:

```
models/sft_output
```

to preserve the same settings used during SFT and DPO.

The training and validation preference sets are loaded via:

```python
load_dataset("json", ...)
```

---

## 5.4 Configure DPO training

Hybrid uses adjusted hyperparameters compared to pure DPO, giving it a different training signature:

- Epochs: 2
- Learning rate: 5e-6
- Cosine schedule
- Gradient accumulation: 4
- Max length: 512
- Beta: 0.3
- Gradient checkpointing disabled

These choices differentiate Hybrid from the DPO baseline.

---

## 5.5 Initialize the DPO trainer

The trainer consumes:

- `model` as the trainable policy
- `ref_model` as the frozen SFT reference
- The preference datasets
- The tokenizer
- All training parameters

TRL’s trainer handles the forward pass, loss, gradient updates, logging, and evaluation.

---

## 5.6 Training loop

`trainer.train()` runs the complete Hybrid optimization process.

Expected behavior:

- Initial loss near log(2) if policy and reference are identical
- Steady reduction over epochs
- Improved preference alignment in generated outputs

Training concludes with a summary containing:

- Final loss
- Training duration

---

## 5.7 Saving the Hybrid LoRA adapter

The script saves:

```
models/hybrid_output/
    adapter_model.safetensors
    adapter_config.json
    tokenizer files
```

These represent the DPO-stage updates only.
SFT LoRA must still be applied separately during inference.

---

# 6. Evaluating the Hybrid Model

Evaluation is handled by:

```
hybrid_test.py
```

This script:

1. Loads TinyLlama
2. Applies the SFT adapter and merges it
3. Applies the Hybrid adapter and merges it
4. Loads the tokenizer
5. Runs several prompts

Run:

```bash
python hybrid_test.py
```

Hybrid outputs should demonstrate:

- SFT-level instruction understanding
- Additional preference-consistent behavior
- More structured, compliant, or safer responses

---

# 7. Training Outputs

The Hybrid training produces:

```
models/hybrid_output/
    adapter_model.safetensors
    adapter_config.json
    tokenizer.model / tokenizer.json
```

To load the Hybrid model for inference, you must:

1. Load TinyLlama
2. Load SFT LoRA
3. Load Hybrid LoRA

Both adapters may then be merged into a single dense model for faster inference if desired.

---

# 8. Troubleshooting

### Hybrid outputs identical to SFT

Ensure the Hybrid adapter is loaded after SFT and merged correctly.

### DPO loss does not decrease

- Lower learning rate
- Increase or decrease beta
- Ensure reference model is correctly frozen
- Ensure the policy model’s new adapter is the only trainable component

### Errors loading the reference model

Use a deep copy of the merged SFT model, not the LoRA directory.

### Incorrect or truncated generations

Ensure chat formatting matches SFT and DPO training templates.
