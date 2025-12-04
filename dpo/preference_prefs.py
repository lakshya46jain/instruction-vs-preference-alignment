from datasets import load_dataset
import json
import os

#Make sure output directories exist
os.makedirs("data/processed", exist_ok=True)

print("Loading HH-RLHF dataset...")
ds = load_dataset("Anthropic/hh-rlhf")

# Split into train and validation
train_data = ds["train"]  # typically use 'train' split
val_data = ds["test"] if "test" in ds else ds["train"].select(range(100))  # small val set

# Convert to DPO preference format
def convert_to_dpo_format(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }


# Write train dataset
with open("data/processed/prefs_train.jsonl", "w") as f:
    for ex in train_data:
        dpo_ex = convert_to_dpo_format(ex)
        f.write(json.dumps(dpo_ex) + "\n")


# Write validation dataset
with open("data/processed/prefs_val.jsonl", "w") as f:
    for ex in val_data:
        dpo_ex = convert_to_dpo_format(ex)
        f.write(json.dumps(dpo_ex) + "\n")

print("Preference data created!")