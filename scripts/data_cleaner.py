# ---------------------------------------------
# data-cleaner.py
# ---------------------------------------------
# Purpose:
#   - Load the raw Code Alpaca dataset
#   - Clean whitespace, normalize "<noinput>" tokens
#   - Output a cleaned JSON file compatible with SFT pipeline
#
# NOTE:
#   Anyone else running this repo must also download the raw dataset
#   "code_alpaca_20k.json" into data/raw/ before running this script.
# ---------------------------------------------

from datasets import load_dataset
import json

# Load dataset from local JSON file.
# load_dataset creates a HuggingFace Dataset object; ["train"] extracts the split.
dataset = load_dataset("json", data_files=r"./data/raw/code_alpaca_20k.json")["train"]

# Extract parallel lists of instruction, input, and output fields.
instruction = dataset["instruction"]
input = dataset["input"]
output = dataset["output"]

# Convert into list of triplets for convenience
data = list(zip(instruction, input, output))


# ------------------------------------------------------
# Cleaning function:
#   - Strips whitespace
#   - Normalizes <noinput> markers to empty string
# ------------------------------------------------------
def clear_whiteSpace(data):
    cleaned = []
    for j, i, k in data:

        # Remove leading/trailing whitespace
        j = j.strip()
        i = i.strip()
        k = k.strip()

        # Normalize various versions of "<noinput>"
        if i.lower() in ["<noinput>", "< noinput >", "<noinput >"]:
            i = ""

        cleaned.append([j, i, k])

    return cleaned


# Apply cleaning
final_data = clear_whiteSpace(data)

# Convert cleaned triplets into Alpaca-style dictionary entries
json_data = [{"instruction": j, "input": i, "output": k} for j, i, k in final_data]

# Write the cleaned JSON file to disk
with open("./data/raw/code_alpaca_clean.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)