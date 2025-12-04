from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import torch
import os

def main():
    base_model = "../models/sft_output"               # SFT model directory
    prefs_train = "prefs_train.jsonl"
    prefs_val   = "prefs_val.jsonl"
    output_dir  = "../models/dpo_output"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files=prefs_train, split="train")
    eval_dataset  = load_dataset("json", data_files=prefs_val,   split="train")

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token   # critical for CausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none"
    )

    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        beta=0.1,
        max_length=512,
        preference_columns={
            "query": "prompt",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving DPO model to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("DPO training complete!")

if __name__ == "__main__":
    main()
