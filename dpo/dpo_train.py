from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import torch
import os

def main():
    base_model = "../models/sft_output"
    prefs_train = os.path.join(os.path.dirname(__file__), "prefs_train.jsonl")
    prefs_val   = os.path.join(os.path.dirname(__file__), "prefs_val.jsonl")
    output_dir  = "../models/dpo_output"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files=prefs_train, split="train")
    eval_dataset  = load_dataset("json", data_files=prefs_val,   split="train")

    # Rename columns if your dataset uses different names
    train_dataset = train_dataset.rename_column("query", "prompt") if "query" in train_dataset.column_names else train_dataset
    eval_dataset  = eval_dataset.rename_column("query", "prompt") if "query" in eval_dataset.column_names else eval_dataset

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_total_limit=2,
        report_to="none"
    )

    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving DPO model to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("DPO training complete!")

if __name__ == "__main__":
    main()