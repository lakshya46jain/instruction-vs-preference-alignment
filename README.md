# Instruction-vs-Preference Alignment

Supervised Fine-Tuning (SFT) + Preference Alignment (DPO/RLHF)

This repository contains the code, datasets, and training pipelines for our CS 4804 group project exploring **instruction tuning** vs. **preference-based alignment** techniques using modern LLMs.

Our workflow follows the real-world alignment stack:

1. **Pretraining** (pretrained open-source model — not done here)
2. **Supervised Fine-Tuning (SFT)**
3. **Preference Optimization (DPO / RLHF)**
4. **Evaluation & Analysis**

This repository includes all assets necessary for the SFT stage and downstream alignment tasks.

---

## Project Datasets

We work with three datasets:

| Filename               | Purpose                                                       | Use in Project |
| ---------------------- | ------------------------------------------------------------- | -------------- |
| `alpaca_data.json`     | Pre-cleaned instruction-tuning dataset (general instructions) | ✔ Used         |
| `code_alpaca_20k.json` | Code instruction dataset (raw)                                | ✘ Not used     |
| `cleaned_data.json`    | Cleaned version of code dataset                               | ✔ Used         |

For this project, **only two datasets are used for fine-tuning**:

- `alpaca_data.json`
- `cleaned_data.json`

These are combined, formatted into a unified prompt style, and split into train/validation sets.

---

## 📁 Repository Structure

```
group_project/
│
├── data/
│   ├── raw/
│   │   ├── alpaca_data.json
│   │   ├── cleaned_data.json
│   │   └── code_alpaca_20k.json (not used)
│   └── processed/
│       ├── sft_train.jsonl
│       └── sft_val.jsonl
│
├── scripts/
│   └── prepare_sft_data.py      # Generates formatted JSONL datasets
│
├── sft/
│   ├── __init__.py
│   ├── dataset.py               # Tokenize + mask labels + dataset loader
│   ├── sft_config.py            # Model + training configuration
│   ├── train_sft_mac.py         # LoRA SFT for Apple Silicon (MPS)
│   ├── train_sft_cuda.py        # QLoRA SFT for CUDA GPUs
│   └── eval_sft.py              # (optional) Compare base vs SFT outputs
│
└── README.md
```

---

# What is Supervised Fine-Tuning (SFT)?

SFT is the first stage of alignment after pretraining.
The goal is to make the pretrained model **follow instructions** by training it on `(instruction, response)` pairs using supervised learning.

For each example:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

The SFT model learns to:

- Incorporate the instruction
- Use optional inputs
- Produce the correct output
- Learn structured formatting
- Reason better than the base model

Later, the preference alignment team will start from the fine-tuned SFT model and apply:

- DPO (Direct Preference Optimization)
- RLHF-style training

Your SFT outputs are the foundation for the rest of the project.

---

## Recommended Python Version

Use **Python 3.11**.
Do **NOT** use Python 3.13 — it breaks Transformers, datasets, and PyTorch.

The simplest way is via **pyenv**:

```bash
brew install pyenv pyenv-virtualenv
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
source ~/.zshrc
```

Install Python:

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

---

## Install Dependencies

Inside the activated environment:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets peft accelerate sentencepiece
```

---

# Step 1 — Prepare the SFT Dataset

Before running any training, generate formatted JSONL files using:

```bash
python scripts/prepare_sft_data.py
```

This produces:

- `data/processed/sft_train.jsonl`
- `data/processed/sft_val.jsonl`

---

# Step 2 — Run SFT on Mac (LoRA + MPS)

Apple Silicon **cannot** run QLoRA (no bitsandbytes support), so LoRA is used.

Run:

```bash
python -m sft.train_sft_mac
```

This will:

- Load model in full precision
- Apply LoRA adapters
- Use Apple’s `mps` backend
- Train on the combined dataset

Outputs are saved in:

```
models/sft_output/
```

---

# Step 3 — Run SFT on CUDA GPU (QLoRA 4-bit)

If you have access to a GPU server or Google Colab with CUDA:

```bash
python -m sft.train_sft_cuda
```

This uses:

- 4-bit quantization via bitsandbytes
- QLoRA
- Much lower memory usage
- Faster training

Outputs are saved in the same directory:

```
models/sft_output/
```

---

# Step 4 — Evaluate SFT Model (Optional)

To compare base vs SFT outputs:

```bash
python -m sft.eval_sft
```

This script loads:

- Base model
- SFT LoRA adapter

and prints side-by-side generations.

---

# Training Notes

- All padding and label masking are handled in `dataset.py`
- All models use a shared configuration in `sft/sft_config.py`
- You can modify LoRA hyperparameters there
- For CUDA machines, `train_sft_cuda.py` mirrors `train_sft_mac.py` with quantization enabled

---

# Credits

This repository was developed as part of the **CS 4804 – Introduction to Artificial Intelligence** group project at Virginia Tech.

The SFT implementation is based on:

- HuggingFace Transformers
- PEFT (LoRA / QLoRA)
- HuggingFace Datasets
- PyTorch (MPS / CUDA)
