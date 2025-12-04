from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import torch
import os

def main():
    # Base model (your trained SFT)
    base_model = "models/sft_output"

    # Paths
    prefs_train = "data/processed/prefs_train.jsonl"
    prefs_val   = "data/processed/prefs_val.jsonl"
    output_dir  = "sft-output-DPO"

    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files=prefs_train, split="train")
    eval_dataset  = load_dataset("json", data_files=prefs_val,   split="train")

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # important for causal LM models

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # -------------------------------
    # Training Arguments (small GPU/CPU friendly)
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,       # smaller batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,      # simulate larger batch size
        fp16=torch.cuda.is_available(),     # use mixed precision on GPU
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none"
    )

    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=512,
        beta=0.1,  # DPO temperature
        preference_columns={
            "query": "prompt",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    )

    print("Starting training...")
    trainer.train()

    # ---------------------
    # SAVE MODEL WEIGHTS
    # ---------------------
    print("Saving trained model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()