# ---------------------------------------------------------
# train_sft.py
# ---------------------------------------------------------
# Purpose:
#   - Load datasets produced by prepare_sft_data.py
#   - Tokenize using dataset.py
#   - Apply LoRA adapters to TinyLlama
#   - Fine-tune with HuggingFace Trainer
# ---------------------------------------------------------

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# Import configuration & dataset utilities
from sft.sft_config import *
from sft.dataset import load_sft_datasets, tokenize_function


# -----------------------------------------------------
# Device selection
# -----------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS backend")
        return "mps"
    elif torch.cuda.is_available():
        print("Using NVIDIA GPU")
        return "cuda"
    else:
        print("Using CPU")
        return "cpu"


# -----------------------------------------------------
# Tokenizer setup
# -----------------------------------------------------
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


# -----------------------------------------------------
# Load base TinyLlama model
# -----------------------------------------------------
def get_base_model(device):
    """
    Loads the base pretrained model in full precision.
    (Mac cannot run 4-bit quantization.)
    """
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model = model.to(device)
    return model


# -----------------------------------------------------
# Apply LoRA adapters to model
# -----------------------------------------------------
def get_lora_model(model):
    """
    Wraps the model using PEFT LoRA adapters.
    """
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Show trainable % for debugging
    return model


# -----------------------------------------------------
# Main training loop
# -----------------------------------------------------
def main():
    device = get_device()
    tokenizer = get_tokenizer()

    # Base model → LoRA-wrapped model
    model = get_base_model(device)
    model = get_lora_model(model)

    # Load train & validation datasets
    train_ds, val_ds = load_sft_datasets()

    # Apply tokenization with map()
    train_tokenized = train_ds.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tokenized = val_ds.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    # Data collator ensures proper padding & batching
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
    )

    # --------------------------------------------------
    # TrainingArguments controls Trainer behavior
    # --------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=LEARNING_RATE,

        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,

        max_steps=MAX_STEPS,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        report_to=["none"],   # turn off wandb
        seed=SEED,
    )

    # --------------------------------------------------
    # Trainer = HF's training engine
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()