# Direct Preference Optimization (DPO) README

This document provides a complete guide for preparing the environment, generating preference datasets, running the DPO training script, and evaluating the resulting policy model using the provided `dpo_test.py`. It is fully aligned with your custom low-VRAM DPO pipeline, which supports:

* 4-bit quantization for the policy model
* A reference model kept on CPU
* Token-level log-probability computation
* The Bradley–Terry preference objective
* A LoRA-adapted SFT model as the initialization checkpoint

---

# 1. Purpose of Direct Preference Optimization (DPO)

DPO is a post-training alignment method that optimizes the model to prefer human-preferred outputs without requiring a learned reward model or reinforcement learning.

The objective:

[
\mathcal{L}*{\text{DPO}} =
-\log \sigma\Big(
\beta[(\pi*\theta(c)-\pi_\theta(r)) - (\pi_{\text{ref}}(c)-\pi_{\text{ref}}(r))]
\Big)
]

encourages the model to:

* Rank the preferred (chosen) answer higher than the rejected one
* Stay close to the reference distribution (the SFT model)
* Update directly on preference pairs rather than reward signals

This removes the complexity of PPO and reward models while remaining theoretically grounded in the Bradley–Terry model.

---

# 2. Environment Setup

DPO uses the same environment as SFT, with one additional dependency if running 4-bit quantization.

---

## 2.1 Python environment

Use Python 3.11.

```bash
pyenv install 3.11.8
pyenv virtualenv 3.11.8 dpo-env
pyenv local dpo-env
```

Verify:

```bash
python3 --version
```

---

## 2.2 Install PyTorch

Mac (MPS):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Linux (CUDA):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 2.3 Install remaining dependencies

```bash
pip install transformers datasets peft accelerate sentencepiece
```

If using 4-bit quantization:

```bash
pip install bitsandbytes
```

Note: bitsandbytes does not support macOS. Set `USE_4BIT = False` when running on a Mac.

---

# 3. Preparing the Preference Dataset

DPO requires a dataset of preference tuples in the format:

```
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

Your project uses:

```
data/processed/prefs_train.jsonl
data/processed/prefs_val.jsonl
```

These files contain curated prompt pairs with human or model-generated preference labels.

---

# 4. Running DPO Training

All training occurs inside the script:

```
train_dpo.py
```

This script implements:

* Loading the SFT model as policy and reference base
* Quantized 4-bit policy model (optional)
* CPU reference model
* Custom data collator
* Log-probability calculations
* The full Bradley–Terry DPO loss
* Gradient accumulation for low-VRAM setups
* Automatic saving of the trained adapter to:

```
models/dpo_output/
```

---

## 4.1 Run the DPO script

From the project root:

```bash
python train_dpo.py
```

You will see:

* Policy model load messages
* Reference model load messages
* Loss values during training
* Epoch summaries

On completion:

```
Saving final DPO model...
DPO training complete!
```

---

# 5. What the Training Script Does

Below is a conceptual summary of the DPO implementation.

---

## 5.1 Load SFT model

The SFT model serves as the base for both:

* The trainable policy model
* The frozen reference model

This ensures that DPO fine-tuning remains stable and anchored to the SFT distribution.

---

## 5.2 Initialize the models

* Policy model: 4-bit quantized (optional), placed on GPU
* Reference model: full precision, placed on CPU, and frozen

This configuration minimizes GPU memory usage and follows the theoretical formulation of DPO where the reference model does not update.

---

## 5.3 Batch construction through `DPOCollator`

The collator:

* Constructs full chosen and rejected sequences
* Applies a consistent system/user/assistant template
* Tokenizes both sequences
* Computes prompt lengths to determine exactly where assistant responses begin

Only assistant tokens contribute to the log-probability calculation.

---

## 5.4 Log-probability computation

The script:

1. Runs a forward pass
2. Computes log-softmax over vocabulary
3. Selects log-probabilities of each ground-truth next token
4. Masks out the prompt portion
5. Sums only assistant-tokens to yield the final sequence log-prob

This matches the standard causal language-modeling likelihood calculation.

---

## 5.5 DPO loss computation

Using:

[
-\log \sigma(\beta[(\pi_\theta(c)-\pi_\theta(r))-(\pi_{\text{ref}}(c)-\pi_{\text{ref}}(r))])
]

The policy is pushed toward the chosen answer while penalizing deviations from the reference model.

---

## 5.6 Optimization

The script uses:

* AdamW optimizer
* Linear warmup
* Gradient accumulation
* VRAM cleanup after each step

This ensures training stability on low-memory hardware.

---

## 5.7 Saving the final model

The trained LoRA adapter and tokenizer files are written to:

```
models/dpo_output/
```

You can load this directory during evaluation.

---

# 6. Evaluating the Trained DPO Model

Use the provided evaluation script:

```
dpo_test.py
```

The script:

* Loads the TinyLlama base model
* Loads the DPO adapter from `models/dpo_output`
* Applies the chat template
* Generates a response for verification

Run:

```bash
python dpo_test.py
```

If DPO has improved alignment, you should observe clearer, more preference-consistent behavior.

---

# 7. Training Outputs

The directory:

```
models/dpo_output/
```

contains:

* `adapter_model.safetensors` (LoRA weights)
* `adapter_config.json`
* Tokenizer files
* Optional intermediate checkpoints
* Final updated policy adapter

---

# 8. Troubleshooting

### Loss exploding or NaN

* Lower learning rate
* Reduce sequence length (`MAX_LENGTH`)
* Increase or decrease beta
* Disable 4-bit quantization for debugging

### Outputs include prompt text

Ensure the evaluation script uses:

```python
tokenizer.apply_chat_template(..., add_generation_prompt=True)
```

### macOS errors with bitsandbytes

Set:

```python
USE_4BIT = False
```

### Slow execution

MPS is slower than CUDA. Reduce:

* Sequence length
* Number of training steps
* Accumulation steps