from pathlib import Path
from typing import Dict, List

import json
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from .sft_config import TRAIN_JSON_PATH, VAL_JSON_PATH, MAX_SEQ_LENGTH

def load_sft_datasets():
    train = load_dataset(
        "json",
        data_files=TRAIN_JSON_PATH,
        split="train"
    )
    val = load_dataset(
        "json",
        data_files=VAL_JSON_PATH,
        split="train"
    )
    return train, val

def tokenize_function(example: Dict, tokenizer: PreTrainedTokenizerBase):
    # concatenate prompt + response and compute loss only on response tokens later
    prompt = example["prompt"]
    response = example["response"]

    full_text = prompt + response

    tokenized = tokenizer(
        full_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=False,
    )

    # For now, set labels = input_ids; Trainer can handle ignoring certain positions later
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized