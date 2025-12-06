Understood.
Below is the **corrected, final SFT README** written for your actual repository, where **there is only one training script**:

```
sft/train_sft.py
```

This script runs on **Mac (MPS)** only, and does **not** support CUDA or QLoRA. The README below removes all references to CUDA, QLoRA, or bitsandbytes.

You can paste this directly into **sft/README.md**.

---

# Supervised Fine-Tuning (SFT) README

This document provides a complete guide for preparing the environment, generating datasets, running the SFT training script, and evaluating the resulting LoRA-adapted model. It is fully aligned with the current project structure, which contains a **single SFT training script** designed for **Apple Silicon (MPS)** systems:

```
sft/train_sft.py
```

This script performs LoRA-based fine-tuning of the TinyLlama model using the processed instruction dataset.

---

# 1. Purpose of Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning adapts a pretrained language model to follow natural-language instructions using supervised learning on `(instruction, input, output)` examples.
The model is trained with cross-entropy loss to reproduce target responses token-by-token.

SFT provides:

* Stronger instruction following
* More structured outputs
* More consistent responses
* A better base model for downstream preference alignment (DPO / hybrid)

LoRA is used to reduce memory requirements by training only low-rank adapter matrices rather than full model weights.

---

# 2. Environment Setup

The SFT pipeline requires a clean, compatible Python environment. Follow these steps carefully.

---

## 2.1 Install pyenv (recommended)

```bash
brew install pyenv pyenv-virtualenv
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
source ~/.zshrc
```

---

## 2.2 Create the Python environment

Use **Python 3.11 only**.
Newer versions (3.12/3.13) break PyTorch, Transformers, and Datasets.

```bash
pyenv install 3.11.8
pyenv virtualenv 3.11.8 sft-env
pyenv local sft-env
```

Verify:

```bash
python3 --version
# Python 3.11.8
```

You should now see `(sft-env)` in your prompt.

---

## 2.3 Install PyTorch with MPS support (Apple Silicon)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

This wheel includes MPS acceleration automatically.

---

## 2.4 Install remaining SFT dependencies

```bash
pip install transformers datasets peft accelerate sentencepiece
```

Notes:

* `bitsandbytes` is **not** installed because it is not supported on macOS.
* This means **QLoRA is not available**, and full-precision LoRA training is used instead.

---

# 3. Preparing the SFT Dataset

The SFT dataset is created from two raw sources:

* `alpaca_data.json`
* `cleaned_data.json`

The preprocessing script:

```
scripts/prepare_sft_data.py
```

normalizes formatting, applies a unified instruction template, and creates the final train/val splits.

---

## 3.1 Run data preparation

From the repository root:

```bash
python scripts/prepare_sft_data.py
```

This generates:

```
data/processed/sft_train.jsonl
data/processed/sft_val.jsonl
```

These files must exist before starting SFT training.

---

# 4. Running SFT Training (Mac MPS Only)

All SFT training is performed by a **single script**:

```
sft/train_sft.py
```

The script:

* Loads TinyLlama in full precision
* Runs on the Apple Silicon `mps` backend
* Applies LoRA adapters
* Trains on the processed dataset
* Saves all checkpoints under:

```
models/sft_output/
```

---

## 4.1 Run the SFT script

From the project root:

```bash
python -m sft.train_sft
```

Training hyperparameters (batch size, epochs, LoRA rank, sequence length, learning rate, etc.) are located in:

```
sft/sft_config.py
```

Tokenization, padding, and label-masking logic is handled in:

```
sft/dataset.py
```

---

# 5. Training Outputs

After training, the directory:

```
models/sft_output/
```

contains:

* `adapter_model.safetensors` – the trained LoRA adapter
* `adapter_config.json` – LoRA configuration
* Tokenizer files (json/model/config)
* Multiple checkpoints (`checkpoint-XXXX`)
* `trainer_state.json` – loss history and metadata
* `training_args.bin` – the full training configuration snapshot

A typical final checkpoint is:

```
models/sft_output/checkpoint-5000
```

---

# 6. Evaluating the Trained SFT Model

Evaluation is performed using:

```
sft/sft_test.py
```

This script loads the base TinyLlama model + LoRA adapter, applies the chat template, and generates responses for verification.

---

# 7. Troubleshooting

### Model outputs its own prompt

You must use `tokenizer.apply_chat_template()`.

### MPS device unavailable

Update macOS and Xcode.
PyTorch must be installed using the CPU/MPS wheel shown above.

### Python 3.12/3.13 errors

Only Python 3.11 is supported.

### Very slow training

This is expected on MPS compared to CUDA.
Reduce:

* max sequence length
* number of steps
* LoRA rank
* batch size