import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Loading SFT model for generating outputs
model_path = "models/sft_output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("data/processed", exist_ok=True)

# Loading SFT dataset
sft_data = [json.loads(line) for line in open("data/processed/sft_train.jsonl", "r")]

def generate_alt_response(prompt, max_new_tokens=200, temperature=1.2):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)

prefs = []

for item in sft_data[:1000]:  
    prompt = item["prompt"]
    chosen = item["response"]  
    rejected = generate_alt_response(prompt) 

    prefs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

# Save for DPO
with open("data/processed/prefs_train.jsonl", "w") as f:
    for ex in prefs:
        f.write(json.dumps(ex) + "\n")