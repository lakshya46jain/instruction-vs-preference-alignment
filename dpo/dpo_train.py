from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer
import torch
import os

def main():
    #Hugging Face model for dry run
    base_model = "sshleifer/tiny-gpt2"

    #Paths to preference data
    prefs_train = "data/processed/prefs_train.jsonl"
    prefs_val = "data/processed/prefs_val.jsonl"

    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files=prefs_train, split="train")
    val_dataset = load_dataset("json", data_files=prefs_val, split="train")

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,
        device_map="auto"
    )

    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("Starting training...")
    trainer.train()

    #Saving model weights and tokenizer
    output_dir = "tiny-gpt2-DPO"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving model and tokenizer to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete! Model and tokenizer saved.")

if __name__ == "__main__":
    main()