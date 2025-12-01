import json
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from sft.sft_config import TRAIN_JSON_PATH, VAL_JSON_PATH, MAX_SEQ_LENGTH

# ---------------------------------------------------------
# Load processed JSONL datasets
# ---------------------------------------------------------

def load_sft_datasets():
    train = load_dataset(
        "json",
        data_files={"train": TRAIN_JSON_PATH},
        split="train"
    )

    val = load_dataset(
        "json",
        data_files={"val": VAL_JSON_PATH},
        split="val"
    )

    return train, val

# ---------------------------------------------------------
# Tokenization function for Trainer
# ---------------------------------------------------------

def tokenize_function(example, tokenizer):
    prompt = example["prompt"]
    response = example["response"]

    full_text = prompt + response

    tokenized = tokenizer(
        full_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Mask pad tokens in labels
    labels = tokenized["input_ids"].copy()
    labels = [
        -100 if token == tokenizer.pad_token_id else token
        for token in labels
    ]
    tokenized["labels"] = labels

    return tokenized