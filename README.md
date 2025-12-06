# Instruction-vs-Preference Alignment

Supervised Fine-Tuning (SFT) + Preference Alignment (DPO/RLHF)

This repository contains the code, datasets, and training pipelines for our CS 4804 group project exploring **instruction tuning** vs. **preference-based alignment** techniques using modern LLMs.

Our workflow follows the real-world alignment stack:

1. **Pretraining** (pretrained open-source model — not done here)
2. **Supervised Fine-Tuning (SFT)**
3. **Preference Optimization (DPO / RLHF)**
4. **Evaluation & Analysis**

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

# Credits

This repository was developed as part of the **CS 4804 – Introduction to Artificial Intelligence** group project at Virginia Tech.