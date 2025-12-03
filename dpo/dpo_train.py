from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import torch
import os

def main():
    # Base model
    base_model = "sshleifer/tiny-gpt2"

    # Paths
    prefs_train = "data/processed/prefs_train.jsonl"
    prefs_val   = "data/processed/prefs_val.jsonl"
    output_dir  = "tiny-gpt2-DPO"

    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files=prefs_train, split="train")
    eval_dataset  = load_dataset("json", data_files=prefs_val,   split="train")

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # important for GPT models

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # -------------------------------
    # Training Arguments (ADDED)
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,                  # increase later
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",               # <--- ensures model gets saved
        save_total_limit=2,
        report_to="none"
    )

    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()

    # ---------------------
    # SAVE MODEL WEIGHTS
    # ---------------------
    print("Saving trained model...")
    trainer.save_model(output_dir)          # generates pytorch_model.bin
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()