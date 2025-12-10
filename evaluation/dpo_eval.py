# -------------------------------------------------------------
# dpo_eval.py
# -------------------------------------------------------------
# Purpose:
#   - Load the pretrained base model and the DPO fine-tuned adapter
#   - Generate outputs from both models on a validation dataset
#   - Score outputs using lightweight heuristic instruction-following
#     rules (similarity, overlap, task-specific cues)
#   - Write detailed comparison results and summary metrics to disk
#
# Notes:
#   - No learning occurs here. This is *offline evaluation* only.
#   - Adapter configs from training may contain extra metadata;
#     prepare_patched_adapter_dir() removes unsupported keys so PEFT
#     can load the adapter cleanly during evaluation.
#   - Heuristic scoring is not a substitute for human evaluation, but
#     enables quick alignment-quality comparisons between SFT vs. DPO.
# -------------------------------------------------------------


import os
import json
import shutil
import inspect
import re
from typing import Iterator, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig

# Base model used for both SFT and DPO experiments.
from sft.sft_config import BASE_MODEL_NAME


# -------------------------------------------------------------
# File paths and evaluation limits
# -------------------------------------------------------------
DATA_PATH = "data/processed/sft_val.jsonl" # evaluation dataset
RESULTS_DIR = "evaluation/results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "dpo_eval_output.txt")

DPO_OUTPUT_DIR = "models/dpo_output"

# Directory containing adapter produced by DPO training
ORIG_ADAPTER_DIR = DPO_OUTPUT_DIR
# Directory where filtered/cleaned adapter config will be stored
PATCHED_ADAPTER_DIR = DPO_OUTPUT_DIR + "_eval"

# Limit number of examples and generation length
MAX_EXAMPLES = 100          
MAX_NEW_TOKENS = 100 


# -------------------------------------------------------------
# Tokenizer loader
# -------------------------------------------------------------
# Ensures pad_token exists and uses right-padding for generation.
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# -------------------------------------------------------------
# Base pretrained model loader
# -------------------------------------------------------------
# Loads the pure base model without adapters.
def load_base_model():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model


# -------------------------------------------------------------
# Adapter directory patching
# -------------------------------------------------------------
# Copies the LoRA adapter directory into a clean evaluation folder,
# removing any config fields unsupported by LoraConfig. This avoids
# PEFT loading errors caused by additional metadata saved during DPO.
def prepare_patched_adapter_dir():
    """Copy adapter weights and strip unknown config keys so Peft can load."""
    if os.path.exists(PATCHED_ADAPTER_DIR):
        return PATCHED_ADAPTER_DIR

    os.makedirs(PATCHED_ADAPTER_DIR, exist_ok=True)

    sig = inspect.signature(LoraConfig.__init__)
    allowed_keys = set(sig.parameters.keys()) - {"self", "kwargs"}

    for fname in os.listdir(ORIG_ADAPTER_DIR):
        src = os.path.join(ORIG_ADAPTER_DIR, fname)
        dst = os.path.join(PATCHED_ADAPTER_DIR, fname)

        # Filter adapter_config.json to allowed LoRA parameters only
        if fname == "adapter_config.json":
            with open(src, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            filtered = {k: v for k, v in cfg.items() if k in allowed_keys}
            dropped = sorted(set(cfg.keys()) - set(filtered.keys()))
            if dropped:
                print("Dropping unsupported config keys in eval copy:", dropped)

            with open(dst, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2)

        # Copy all other files (adapter weights, etc.)
        else:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    return PATCHED_ADAPTER_DIR


# -------------------------------------------------------------
# Load DPO-fine-tuned model (base + LoRA adapter)
# -------------------------------------------------------------
def load_dpo_model():
    base_model = load_base_model()
    patched = prepare_patched_adapter_dir()
    print("Using patched adapter directory:", patched)
    model = PeftModel.from_pretrained(base_model, patched)
    model.eval()
    return model


# -------------------------------------------------------------
# Utility: JSONL reader
# -------------------------------------------------------------
def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -------------------------------------------------------------
# Model generation wrapper
# -------------------------------------------------------------
# Applies chat template, sends prompt through model, and returns
# the decoded output. Deterministic decoding (temperature=0).
def generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = [{"role": "user", "content": prompt}]

    # Convert messages to model chat format
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Generate continuation
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Extract only the newly generated portion
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


# -------------------------------------------------------------
# Prompt parser: Extract instruction + input fields
# -------------------------------------------------------------
# Alpaca-style format:
#   ### Instruction:
#   ### Input:
#   ### Response:
def parse_instruction_and_input(prompt: str) -> Tuple[str, str]:
    """
    Split the Alpaca-style prompt into (instruction, input_text).
    Expected format:
      ### Instruction:\n...\n\n### Input:\n...\n\n### Response:\n
    """
    instr = ""
    inp = ""
    text = prompt

    if "### Instruction:" in text:
        parts = text.split("### Instruction:", 1)[1]
    else:
        parts = text

    if "### Input:" in parts:
        instr_part, rest = parts.split("### Input:", 1)
        instr = instr_part.strip()
    else:
        instr = parts.strip()
        rest = ""

    if "### Response:" in rest:
        inp_part, _ = rest.split("### Response:", 1)
        inp = inp_part.strip()
    else:
        inp = rest.strip()

    return instr, inp


# -------------------------------------------------------------
# Text normalization utilities
# -------------------------------------------------------------
def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text

def _token_set(text: str):
    return set(_normalize(text).split()) if text else set()

def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    return len([p for p in parts if p.strip()])


# -------------------------------------------------------------
# Task-specific heuristic scoring rules
# -------------------------------------------------------------
# Each rule:
#   - Returns a float in [0,1] or None (if rule does not apply)
#   - Targets a specific instruction category (e.g., summarization,
#     rewrite-negative, essay writing, Java/Python/JS code, math, etc.)
# -------------------------------------------------------------
def summary_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "summary" not in instr_l and "summarize" not in instr_l:
        return None
    if not inp:
        return None

    if len(out.strip()) == 0:
        return 0.0
    len_ratio = min(1.0, len(inp) / max(len(out), 1))

    inp_tokens = _token_set(inp)
    out_tokens = _token_set(out)
    if not inp_tokens or not out_tokens:
        overlap = 0.0
    else:
        overlap = len(inp_tokens & out_tokens) / len(inp_tokens)

    return 0.5 * len_ratio + 0.5 * overlap


def negative_rewrite_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "negative connotation" not in instr_l and "negative" not in instr_l:
        return None
    if len(out.strip()) == 0:
        return 0.0

    neg_words = ["not", "no", "never", "bad", "poor", "terrible", "awful", "worse", "worst"]
    has_negative = any(w in out.lower() for w in neg_words)

    inp_tokens = _token_set(inp)
    out_tokens = _token_set(out)
    if not inp_tokens or not out_tokens:
        overlap = 0.0
    else:
        overlap = len(inp_tokens & out_tokens) / len(inp_tokens)

    score = 0.0
    if has_negative:
        score += 0.6
    score += 0.4 * overlap
    return score


def essay_rule(instr: str, inp: str, out: str) -> Optional[float]:
    if "write an essay" not in instr.lower():
        return None
    s_count = sentence_count(out)
    length = len(out)

    if length == 0:
        return 0.0

    score = 0.0
    if s_count >= 3:
        score += 0.6
    score += 0.4 * min(1.0, length / 400.0)
    return score


def java_code_rule(instr: str, inp: str, out: str) -> Optional[float]:
    if "java" not in instr.lower():
        return None
    out_l = out.lower()
    if len(out_l.strip()) == 0:
        return 0.0

    patterns = [
        "class ",
        "public static void main",
        "system.out.println"
    ]
    hits = sum(1 for p in patterns if p in out_l)
    return hits / len(patterns)


def javascript_code_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "javascript" not in instr_l and "java script" not in instr_l:
        return None
    out_l = out.lower()
    if len(out_l.strip()) == 0:
        return 0.0

    patterns = [
        "function ",
        "console.log",
        "let ",
        "const "
    ]
    hits = sum(1 for p in patterns if p in out_l)
    return hits / len(patterns)


def python_class_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "python" not in instr_l:
        return None
    if "class" not in instr_l and "object" not in instr_l:
        return None

    out_l = out.lower()
    if len(out_l.strip()) == 0:
        return 0.0

    patterns = [
        "class ",
        "def __init__",
        "self."
    ]
    hits = sum(1 for p in patterns if p in out_l)
    return hits / len(patterns)


def calculation_rule(instr: str, inp: str, out: str, reference: str) -> Optional[float]:
    instr_l = instr.lower()
    if "calculate" not in instr_l and "compute" not in instr_l and "find the area" not in instr_l:
        return None

    ref_nums = re.findall(r"-?\d+\.?\d*", reference)
    out_nums = re.findall(r"-?\d+\.?\d*", out)
    if not out_nums:
        return 0.0
    if not ref_nums:
        return 0.5

    # Check for matching numeric value
    for rn in ref_nums:
        for on in out_nums:
            try:
                if float(rn) == float(on):
                    return 1.0
            except ValueError:
                continue
    return 0.5


def question_generation_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "question to ask a bot" not in instr_l and "generate an appropriate question" not in instr_l:
        return None
    if len(out.strip()) == 0:
        return 0.0

    score = 0.0
    if "?" in out:
        score += 0.7
    if len(out.split()) <= 25:
        score += 0.3
    return min(score, 1.0)


def title_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "professional title" not in instr_l and "title" not in instr_l:
        return None
    tokens = out.strip().split()
    if not tokens:
        return 0.0

    length_penalty = max(0.0, 1.0 - (len(tokens) - 3) * 0.2)
    capitalized_tokens = sum(1 for t in tokens if t[:1].isupper())
    cap_ratio = capitalized_tokens / max(len(tokens), 1)

    score = 0.5 * length_penalty + 0.5 * cap_ratio
    return max(0.0, min(1.0, score))


# Generic fallback relevance metric (token overlap)
def generic_relevance_rule(instr: str, inp: str, out: str, reference: str) -> float:
    context = " ".join([instr, inp, reference])
    ctx_tokens = _token_set(context)
    out_tokens = _token_set(out)

    if not ctx_tokens or not out_tokens:
        return 0.0

    overlap = len(ctx_tokens & out_tokens)
    return overlap / len(ctx_tokens)


# -------------------------------------------------------------
# Aggregation of all heuristic rules
# -------------------------------------------------------------
# Returns:
#   final_score → weighted score
#   used_special_rule → True if at least one task-specific rule applied
def eval_instruction_following(prompt: str, output: str, reference: str) -> Tuple[float, bool]:
    instr, inp = parse_instruction_and_input(prompt)

    scores = []

    # Evaluate all rule functions
    for rule_fn in [
        summary_rule,
        negative_rewrite_rule,
        essay_rule,
        java_code_rule,
        javascript_code_rule,
        python_class_rule,
        calculation_rule,
        question_generation_rule,
        title_rule,
    ]:
        if rule_fn is calculation_rule:
            val = rule_fn(instr, inp, output, reference)
        else:
            val = rule_fn(instr, inp, output)

        if val is not None:
            scores.append(val)

    used_special = len(scores) > 0

    # If no task-specific rule applied → fallback relevance score
    if not scores:
        score = generic_relevance_rule(instr, inp, output, reference)
        return score, False

    spec_score = sum(scores) / len(scores)
    rel_score = generic_relevance_rule(instr, inp, output, reference)

    # Weighted blend of specialized and generic relevance
    final_score = 0.7 * spec_score + 0.3 * rel_score
    return final_score, True


# -------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check for dataset existence
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find evaluation data at {DATA_PATH}. "
            f"Adjust DATA_PATH in dpo_eval.py if your file lives elsewhere."
        )

    tokenizer = get_tokenizer()
    print("Tokenizer loaded.")

    print("Loading base model...")
    base_model = load_base_model()
    print("Base model loaded.")

    print("Loading DPO model...")
    dpo_model = load_dpo_model()
    print("DPO model loaded.")

    # Tracking metrics
    total = 0
    sum_score_base = 0.0
    sum_score_dpo = 0.0
    special_examples = 0
    sum_spec_base = 0.0
    sum_spec_dpo = 0.0

    with open(RESULTS_PATH, "w", encoding="utf-8") as out_f:
        out_f.write(
            "Heuristic IF-style evaluation on dataset: {}\n\n".format(DATA_PATH)
        )

        # Process each example
        for i, example in enumerate(read_jsonl(DATA_PATH), start=1):
            if MAX_EXAMPLES is not None and i > MAX_EXAMPLES:
                break

            prompt = example.get("prompt", "")
            reference = example.get("response", "")

            print(f"\n=== Starting example {i} ===")
            base_out = generate(base_model, tokenizer, prompt)
            print(f"Finished base generation for example {i}")
            dpo_out = generate(dpo_model, tokenizer, prompt)
            print(f"Finished DPO generation for example {i}")

            total += 1

            # Evaluate outputs
            base_score, base_special = eval_instruction_following(prompt, base_out, reference)
            dpo_score, dpo_special = eval_instruction_following(prompt, dpo_out, reference)

            sum_score_base += base_score
            sum_score_dpo += dpo_score

            if base_special and dpo_special:
                special_examples += 1
                sum_spec_base += base_score
                sum_spec_dpo += dpo_score

            # Write per-example block
            out_f.write("=" * 100 + "\n")
            out_f.write(f"EXAMPLE {i}\n")
            out_f.write("PROMPT:\n{}\n\n".format(prompt))
            out_f.write("REFERENCE RESPONSE:\n{}\n\n".format(reference))
            out_f.write("BASE OUTPUT:\n{}\n\n".format(base_out))
            out_f.write(f"[Base heuristic score: {base_score:.3f}]\n\n")
            out_f.write("DPO OUTPUT:\n{}\n\n".format(dpo_out))
            out_f.write(f"[DPO heuristic score: {dpo_score:.3f}]\n\n")

            if i % 5 == 0:
                print(f"Processed {i} examples...")

        # Summary block
        out_f.write("=" * 100 + "\n")
        out_f.write("SUMMARY\n")
        out_f.write(f"Total examples evaluated: {total}\n")

        if total > 0:
            avg_base = sum_score_base / total
            avg_dpo = sum_score_dpo / total
            out_f.write(f"Base avg heuristic score (all examples): {avg_base:.3f}\n")
            out_f.write(f"DPO avg heuristic score (all examples): {avg_dpo:.3f}\n")

        out_f.write(f"Examples with specialized rules applied to both models: {special_examples}\n")
        if special_examples > 0:
            avg_spec_base = sum_spec_base / special_examples
            avg_spec_dpo = sum_spec_dpo / special_examples
            out_f.write(
                f"Base avg heuristic score (special-rule subset): {avg_spec_base:.3f}\n"
            )
            out_f.write(
                f"DPO avg heuristic score (special-rule subset): {avg_spec_dpo:.3f}\n"
            )

    # Console summary
    print(f"\nDone. Wrote detailed outputs to {RESULTS_PATH}")
    print(f"Total examples evaluated: {total}")
    if total > 0:
        avg_base = sum_score_base / total
        avg_dpo = sum_score_dpo / total
        print(f"Base avg heuristic score (all examples): {avg_base:.3f}")
        print(f"DPO  avg heuristic score (all examples): {avg_dpo:.3f}")
    print(f"Examples with specialized rules applied to both models: {special_examples}")
    if special_examples > 0:
        avg_spec_base = sum_spec_base / special_examples
        avg_spec_dpo = sum_spec_dpo / special_examples
        print(f"Base avg heuristic score (special subset): {avg_spec_base:.3f}")
        print(f"DPO  avg heuristic score (special subset): {avg_spec_dpo:.3f}")


# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    main()