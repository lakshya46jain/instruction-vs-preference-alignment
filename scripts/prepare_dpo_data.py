import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Correct relative path (your script is inside scripts/)
model_path = "../models/sft_output"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

os.makedirs("../data/processed", exist_ok=True)

# Load your SFT data
sft_data = [json.loads(line) for line in open("../data/processed/sft_train.jsonl")]

def generate_rejected(prompt, max_new_tokens=150, temperature=1.2):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

prefs = []

for item in sft_data[:1000]:
    prompt = item["prompt"]
    chosen = item["response"]
    rejected = generate_rejected(prompt)
    
    prefs.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

with open("../data/processed/prefs_train.jsonl", "w") as f:
    for p in prefs:
        f.write(json.dumps(p) + "\n")

with open("../data/processed/prefs_val.jsonl", "w") as f:
    for p in prefs[:100]:
        f.write(json.dumps(p) + "\n")

print("Preference dataset created successfully!")