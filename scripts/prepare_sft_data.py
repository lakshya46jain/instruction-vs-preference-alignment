# ------------------------------------------------------------
# prepare_sft_data.py
# ------------------------------------------------------------
# Purpose:
#   - Load Alpaca + cleaned Code Alpaca dataset
#   - Convert them into the unified "prompt → response" SFT format
#   - Apply 90/10 train/val split
#   - Save as JSONL files used by the SFT training script
# ------------------------------------------------------------

import json
import random
from pathlib import Path

# Directory setup
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Output files
TRAIN_OUT = PROC_DIR / "sft_train.jsonl"
VAL_OUT = PROC_DIR / "sft_val.jsonl"

# 90% of examples used for training
TRAIN_RATIO = 0.9
SEED = 42

# Template for SFT prompt structure (matches Alpaca format)
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


# Helper to load a JSON file
def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def main():
    # Load the base Alpaca dataset
    alpaca = load_json(RAW_DIR / "alpaca_data.json")

    # Load cleaned Code Alpaca dataset
    cleaned = load_json(RAW_DIR / "code_alpaca_clean.json")

    # Combine both datasets
    all_data = alpaca + cleaned

    # Shuffle for randomness and stable train/val separation
    random.seed(SEED)
    random.shuffle(all_data)

    examples = []

    # ---------------------------------------------------------
    # Convert each example into {"prompt": ..., "response": ...}
    # This is EXACTLY what the SFT trainer expects.
    # ---------------------------------------------------------
    for ex in all_data:
        instruction = ex.get("instruction", "").strip()
        input_text = ex.get("input", "").strip() or ""
        output = ex.get("output", "").strip()

        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
        )

        examples.append({"prompt": prompt, "response": output})

    # Split into train/val
    n_train = int(len(examples) * TRAIN_RATIO)
    train = examples[:n_train]
    val = examples[n_train:]

    # Save train set
    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Save validation set
    with VAL_OUT.open("w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Train:", len(train))
    print("Val:", len(val))


if __name__ == "__main__":
    main()