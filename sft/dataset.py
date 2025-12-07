# ---------------------------------------------------------
# dataset.py
# ---------------------------------------------------------
# Purpose:
#   - Load the processed JSONL files created by prepare_sft_data.py
#   - Convert each example into tokenized form for SFT training
# ---------------------------------------------------------

import json
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from sft.sft_config import TRAIN_JSON_PATH, VAL_JSON_PATH, MAX_SEQ_LENGTH


# ---------------------------------------------------------
# Load SFT train/validation datasets from JSONL
# ---------------------------------------------------------
def load_sft_datasets():
    """
    Loads the SFT training and validation datasets from their
    absolute JSONL file paths defined in sft_config.py.
    """
    train = load_dataset(
        "json",
        data_files={"train": TRAIN_JSON_PATH},
        split="train",
    )

    val = load_dataset(
        "json",
        data_files={"val": VAL_JSON_PATH},
        split="val",
    )

    return train, val


# ---------------------------------------------------------
# Tokenization Function
# ---------------------------------------------------------
def tokenize_function(example, tokenizer: PreTrainedTokenizerBase):
    """
    Converts a single example into token IDs for training.
    The Trainer expects:
      - input_ids: encoded text
      - attention_mask
      - labels: identical to input_ids but with pad tokens masked to -100

    Parameters:
        example: dict with keys {"prompt", "response"}
        tokenizer: HuggingFace tokenizer

    Returns:
        tokenized: dict formatted for Trainer
    """
    prompt = example["prompt"]
    response = example["response"]

    # Concatenate prompt + output for autoregressive LM training
    full_text = prompt + response

    tokenized = tokenizer(
        full_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",  # consistent batch shape
    )

    # ---------------------------------------------------------
    # Labels: model predicts *every* token except padding.
    # Mask padded positions using -100 (ignored by loss).
    # ---------------------------------------------------------
    labels = tokenized["input_ids"].copy()
    labels = [
        -100 if token == tokenizer.pad_token_id else token
        for token in labels
    ]

    tokenized["labels"] = labels

    return tokenized