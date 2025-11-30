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

def make_examples():
    alpaca = load_json(RAW_DIR / "alpaca_data.json")
    cleaned = load_json(RAW_DIR / "cleaned_data.json")

    all_data = alpaca + cleaned

    # shuffle
    random.seed(SEED)
    random.shuffle(all_data)

    examples = []
    for ex in all_data:
        instr = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        out = ex.get("output", "").strip()

        prompt = PROMPT_TEMPLATE.format(
            instruction=instr,
            input=inp if inp != "" else " "
        )

        examples.append({
            "prompt": prompt,
            "response": out
        })

    return examples

def train_val_split(examples):
    n = len(examples)
    n_train = int(TRAIN_RATIO * n)
    train = examples[:n_train]
    val = examples[n_train:]
    return train, val

def write_jsonl(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    examples = make_examples()
    train, val = train_val_split(examples)
    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)
    print(f"Train: {len(train)} examples")
    print(f"Val:   {len(val)} examples")

if __name__ == "__main__":
    main()