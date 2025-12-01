import json
import random
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = PROC_DIR / "sft_train.jsonl"
VAL_OUT = PROC_DIR / "sft_val.jsonl"

TRAIN_RATIO = 0.9
SEED = 42

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def load_json(path: Path):
    with path.open() as f:
        return json.load(f)

def main():
    alpaca = load_json(RAW_DIR / "alpaca_data.json")
    cleaned = load_json(RAW_DIR / "cleaned_data.json")

    all_data = alpaca + cleaned

    random.seed(SEED)
    random.shuffle(all_data)

    examples = []
    for ex in all_data:
        instruction = ex.get("instruction", "").strip()
        input_text = ex.get("input", "").strip() or ""
        output = ex.get("output", "").strip()

        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
        )

        examples.append({"prompt": prompt, "response": output})

    # Split
    n_train = int(len(examples) * TRAIN_RATIO)
    train = examples[:n_train]
    val = examples[n_train:]

    # Write JSONL
    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with VAL_OUT.open("w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Train:", len(train))
    print("Val:", len(val))

if __name__ == "__main__":
    main()