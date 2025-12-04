import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
 
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(REPO_ROOT, "models/sft_output")
 
data_path = os.path.join(REPO_ROOT, "data/processed")
sft_file = os.path.join(data_path, "sft_train_test.jsonl")
 
prefs_train_file = os.path.join(data_path, "prefs_train.jsonl")
prefs_val_file   = os.path.join(data_path, "prefs_val.jsonl")
 
os.makedirs(data_path, exist_ok=True)
 
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
 
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
model.eval()
 
print(f"Loading SFT data from: {sft_file}")
sft_data = [json.loads(line) for line in open(sft_file)]
 
def generate_rejected(prompt, max_new_tokens=80):
    gen_prompt = prompt + "\nProvide an alternative answer:\n"
 
    inputs = tokenizer(gen_prompt, return_tensors="pt").to(device)
 
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=5,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None   # do not stop early
    )
 
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
    # Remove the prompt prefix
    continuation = decoded[len(gen_prompt):].strip()
 
    # Basic cleanup for weird outputs
    if continuation.startswith("```"):
        continuation = continuation.replace("```", "").strip()
 
    # Ensure rejected is not empty
    if continuation == "" or continuation.isspace():
        continuation = "I'm not sure, maybe something else could be considered."
 
    return continuation
 
 
prefs = []
MAX_EXAMPLES = min(1000, len(sft_data))
 
print(f"Generating rejected responses for {MAX_EXAMPLES} examples...")
 
for i, item in enumerate(sft_data[:MAX_EXAMPLES]):
    prompt = item["prompt"]
    chosen = item["response"]
 
    rejected = generate_rejected(prompt)
 
    prefs.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })
 
    if i % 20 == 0:
        print(f"Processed {i}/{MAX_EXAMPLES}")
 
print("Saving DPO datasets...")
 
with open(prefs_train_file, "w") as f:
    for p in prefs:
        f.write(json.dumps(p) + "\n")
 
with open(prefs_val_file, "w") as f:
    for p in prefs[:100]:
        f.write(json.dumps(p) + "\n")
 
print("Preference dataset created successfully!")