"""Download public preference dataset test1"""
from datasets import load_dataset
import json, os

print("Downloading HH-RLHF dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train").select(range(5000))

formatted_data = []
for item in dataset:
    try:
        prompt = item["chosen"].split("\n\nAssistant:")[0].replace("Human:", "").strip()
        chosen = item["chosen"].split("\n\nAssistant:")[-1].strip()
        rejected = item["rejected"].split("\n\nAssistant:")[-1].strip()
        formatted_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    except: continue

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/preference_train.json", 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=2)

print(f"✓ Saved {len(formatted_data)} examples to data/processed/preference_train.json")