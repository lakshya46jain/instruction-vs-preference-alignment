# ------------------------------------------------------------
# prepare_dpo_data.py
# ------------------------------------------------------------
# Purpose:
#   - Load the SFT dataset (already in prompt/response format)
#   - Sample a subset (e.g., 5000 train + 200 val)
#   - For each example:
#         chosen  = ground-truth SFT response
#         rejected = model-generated alternative
#   - Save results as prefs_train.jsonl and prefs_val.jsonl
#
# Used For:
#   - DPO training pipeline
# ------------------------------------------------------------

import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# -----------------------
# Path setup
# -----------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(REPO_ROOT, "models/sft_output")

data_path = os.path.join(REPO_ROOT, "data/processed")
sft_file = os.path.join(data_path, "sft_train.jsonl")
prefs_train_file = os.path.join(data_path, "prefs_train.jsonl")
prefs_val_file   = os.path.join(data_path, "prefs_val.jsonl")

os.makedirs(data_path, exist_ok=True)

# -----------------------
# Tokenizer + Model Load
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Choose device automatically
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Load SFT model for generating rejected responses
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
model.eval()

print(f"Loading SFT data from: {sft_file}")

# -----------------------
# Load SFT dataset
# -----------------------
with open(sft_file, "r", encoding="utf-8") as f:
    sft_data = [json.loads(line) for line in f]

# -----------------------
# Number of preference pairs
# -----------------------
TARGET_TRAIN = 5000
TARGET_VAL = 200
TARGET_TOTAL = TARGET_TRAIN + TARGET_VAL

# Shuffle dataset to randomly pick 5200 examples
random.seed(42)
random.shuffle(sft_data)

sft_subset = sft_data[:TARGET_TOTAL]
print(f"Subset selected: {len(sft_subset)} examples ({TARGET_TRAIN} train + {TARGET_VAL} val)")

# ------------------------------------------------------
# Generate rejected sample using the SFT model
# ------------------------------------------------------
def generate_rejected(prompt, max_new_tokens=80):
    """
    Produces a 'rejected' answer by sampling from the SFT model
    with temperature > 1 to generate a weaker, less preferred output.
    """
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
        eos_token_id=None
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the continuation after the prompt
    continuation = decoded[len(gen_prompt):].strip()

    # Remove accidental Markdown fences
    if continuation.startswith("```"):
        continuation = continuation.replace("```", "").strip()

    # Guarantee a fallback response if blank
    if continuation == "" or continuation.isspace():
        continuation = "I'm not sure, maybe something else could be considered."

    return continuation


# -----------------------
# Create preference pairs
# -----------------------
prefs = []
TOTAL = len(sft_subset)

print(f"Generating rejected responses for {TOTAL} examples...")

for i, item in enumerate(sft_subset):
    prompt = item["prompt"]      # original SFT prompt
    chosen = item["response"]    # ground truth answer

    # Generate model-sampled rejected alternative
    rejected = generate_rejected(prompt)

    prefs.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

    if i % 20 == 0:
        print(f"Processed {i}/{TOTAL}")

# -----------------------
# Save preference datasets
# -----------------------
print("Saving DPO datasets...")

# First N = 5000 → prefs_train.jsonl
with open(prefs_train_file, "w", encoding="utf-8") as f:
    for p in prefs[:TARGET_TRAIN]:
        f.write(json.dumps(p) + "\n")

# Remaining 200 → prefs_val.jsonl
with open(prefs_val_file, "w", encoding="utf-8") as f:
    for p in prefs[TARGET_TRAIN:TARGET_TOTAL]:
        f.write(json.dumps(p) + "\n")

print("Preference dataset created successfully!")