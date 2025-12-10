# -------------------------------------------------------------
# sft_eval.py
# -------------------------------------------------------------
# Purpose:
#   - Evaluate a Supervised Fine-Tuned (SFT) model against the base model.
#   - Generate outputs from both models on a validation dataset.
#   - Score outputs using heuristic instruction-following evaluation rules.
#   - Write detailed per-example logs and overall summary results.
#
# Notes:
#   - Evaluation only; no training occurs here.
#   - The LoRA adapter folder created during SFT training may include keys
#     not recognized by PEFT during loading; prepare_patched_adapter_dir()
#     sanitizes the adapter_config.json to avoid load errors.
#   - The scoring pipeline is consistent with hybrid_eval.py and dpo_eval.py,
#     allowing direct comparison between SFT, DPO, and Hybrid models.
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

# Base TinyLlama model + the SFT output directory
from sft.sft_config import BASE_MODEL_NAME, OUTPUT_DIR


# -------------------------------------------------------------
# Paths and Evaluation Configuration
# -------------------------------------------------------------
DATA_PATH = "data/processed/sft_val.jsonl"  # Evaluation dataset
RESULTS_DIR = "evaluation/results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "sft_eval_output.txt")

# Original LoRA adapter directory from SFT training
ORIG_ADAPTER_DIR = OUTPUT_DIR
# Cleaned/filtered version to fix unsupported LoRA config keys
PATCHED_ADAPTER_DIR = OUTPUT_DIR + "_eval"

# Control how many samples to evaluate and how long generations are
MAX_EXAMPLES = 100                     # Set to None for full evaluation
MAX_NEW_TOKENS = 100                   # Max tokens to generate per answer


# -------------------------------------------------------------
# Tokenizer Loader
# -------------------------------------------------------------
# Ensures the chat tokenizer is configured with padding tokens and right padding.
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# -------------------------------------------------------------
# Load the Base (Unfine-tuned) Model
# -------------------------------------------------------------
# Used to establish a performance baseline against the SFT model.
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
# Patch and Sanitize the Adapter Directory
# -------------------------------------------------------------
# Some LoRA config keys in SFT outputs are not compatible with PEFT.
# Unsupported keys are removed so the evaluation adapter loads safely.
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

        # Only adapter_config.json requires filtering.
        if fname == "adapter_config.json":
            with open(src, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            # Keep only supported LoRA parameters.
            filtered = {k: v for k, v in cfg.items() if k in allowed_keys}
            dropped = sorted(set(cfg.keys()) - set(filtered.keys()))
            if dropped:
                print("Dropping unsupported config keys in eval copy:", dropped)

            with open(dst, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2)

        else:
            # Copy folder contents or single files directly
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    return PATCHED_ADAPTER_DIR


# -------------------------------------------------------------
# Load SFT Model: Base + Patched LoRA Adapter
# -------------------------------------------------------------
def load_sft_model():
    base_model = load_base_model()
    patched = prepare_patched_adapter_dir()
    print("Using patched adapter directory:", patched)
    model = PeftModel.from_pretrained(base_model, patched)
    model.eval()
    return model


# -------------------------------------------------------------
# Utility: Read JSONL Dataset
# -------------------------------------------------------------
def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -------------------------------------------------------------
# Generation Wrapper
# -------------------------------------------------------------
# Applies HF chat template, runs deterministic decoding (temp=0),
# strips prompt portion, returns clean generated text.
def generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = [{"role": "user", "content": prompt}]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the input token portion, keeping generated output only
    generated = output[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Ensure empty model output is handled visibly
    if text == "":
        text = ""

    return text


# -------------------------------------------------------------
# Parse Alpaca-Style Prompt: Extract Instruction & Input
# -------------------------------------------------------------
# Used to determine which heuristic rules apply to a given prompt.
def parse_instruction_and_input(prompt: str) -> Tuple[str, str]:
    """
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
# Token / Text Utilities for Scoring Rules
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
# Heuristic Scoring Rules
# -------------------------------------------------------------
# Each rule determines whether it applies to the instruction + output.
# If applicable, returns score ∈ [0,1]; otherwise returns None.
# These rules measure correctness for: summarization, rewriting,
# essays, code generation, math questions, titles, etc.
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
    overlap = 0.0 if not inp_tokens or not out_tokens else len(inp_tokens & out_tokens) / len(inp_tokens)

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
    overlap = 0.0 if not inp_tokens or not out_tokens else len(inp_tokens & out_tokens) / len(inp_tokens)

    score = 0.6 * has_negative + 0.4 * overlap
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

    patterns = ["class ", "public static void main", "system.out.println"]
    hits = sum(1 for p in patterns if p in out_l)
    return hits / len(patterns)


def javascript_code_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "javascript" not in instr_l and "java script" not in instr_l:
        return None

    out_l = out.lower()
    if len(out_l.strip()) == 0:
        return 0.0

    patterns = ["function ", "console.log", "let ", "const "]
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

    patterns = ["class ", "def __init__", "self."]
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

    # Exact numeric match yields full credit
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

    score = 0.7 * ("?" in out) + 0.3 * (len(out.split()) <= 25)
    return min(score, 1.0)


def title_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "professional title" not in instr_l and "title" not in instr_l:
        return None

    tokens = out.strip().split()
    if not tokens:
        return 0.0

    length_penalty = max(0.0, 1.0 - (len(tokens) - 3) * 0.2)
    cap_ratio = sum(1 for t in tokens if t[:1].isupper()) / max(len(tokens), 1)

    return max(0.0, min(1.0, 0.5 * length_penalty + 0.5 * cap_ratio))


# -------------------------------------------------------------
# Generic Fallback Rule
# -------------------------------------------------------------
# Applies lexical overlap relevance regardless of task type.
def generic_relevance_rule(instr: str, inp: str, out: str, reference: str) -> float:
    """
    Fallback: measure lexical similarity between output and
    (instruction + input + reference).
    """
    context = " ".join([instr, inp, reference])
    ctx_tokens = _token_set(context)
    out_tokens = _token_set(out)

    if not ctx_tokens or not out_tokens:
        return 0.0

    return len(ctx_tokens & out_tokens) / len(ctx_tokens)


# -------------------------------------------------------------
# Combined Instruction-Following Evaluation
# -------------------------------------------------------------
# Returns:
#   - final score in [0,1]
#   - used_special_rules (bool)
def eval_instruction_following(prompt: str, output: str, reference: str) -> Tuple[float, bool]:
    instr, inp = parse_instruction_and_input(prompt)

    scores = []

    # Apply specialized rules in order
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

    if not scores:
        # No specialized rule applies → fallback
        return generic_relevance_rule(instr, inp, output, reference), False

    spec_score = sum(scores) / len(scores)
    rel_score = generic_relevance_rule(instr, inp, output, reference)

    return 0.7 * spec_score + 0.3 * rel_score, True


# -------------------------------------------------------------
# Main Evaluation Loop
# -------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Ensure dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find evaluation data at {DATA_PATH}. "
            f"Adjust DATA_PATH in sft_eval.py if your file lives elsewhere."
        )

    tokenizer = get_tokenizer()
    print("Tokenizer loaded.")

    print("Loading base model...")
    base_model = load_base_model()
    print("Base model loaded.")

    print("Loading SFT model...")
    sft_model = load_sft_model()
    print("SFT model loaded.")

    # Metrics
    total = 0
    sum_score_base = 0.0
    sum_score_sft = 0.0
    special_examples = 0
    sum_spec_base = 0.0
    sum_spec_sft = 0.0

    # Results file
    with open(RESULTS_PATH, "w", encoding="utf-8") as out_f:
        out_f.write("Heuristic IF-style evaluation on dataset: {}\n\n".format(DATA_PATH))

        for i, example in enumerate(read_jsonl(DATA_PATH), start=1):
            if MAX_EXAMPLES is not None and i > MAX_EXAMPLES:
                break

            prompt = example.get("prompt", "")
            reference = example.get("response", "")

            print(f"\n=== Starting example {i} ===")
            base_out = generate(base_model, tokenizer, prompt)
            print(f"Finished base generation for example {i}")
            sft_out = generate(sft_model, tokenizer, prompt)
            print(f"Finished SFT generation for example {i}")

            total += 1

            base_score, base_special = eval_instruction_following(prompt, base_out, reference)
            sft_score, sft_special = eval_instruction_following(prompt, sft_out, reference)

            sum_score_base += base_score
            sum_score_sft += sft_score

            if base_special and sft_special:
                special_examples += 1
                sum_spec_base += base_score
                sum_spec_sft += sft_score

            # Write detailed example output
            out_f.write("=" * 100 + "\n")
            out_f.write(f"EXAMPLE {i}\n")
            out_f.write("PROMPT:\n{}\n\n".format(prompt))
            out_f.write("REFERENCE RESPONSE:\n{}\n\n".format(reference))
            out_f.write("BASE OUTPUT:\n{}\n\n".format(base_out))
            out_f.write(f"[Base heuristic score: {base_score:.3f}]\n\n")
            out_f.write("SFT OUTPUT:\n{}\n\n".format(sft_out))
            out_f.write(f"[SFT heuristic score: {sft_score:.3f}]\n\n")

            if i % 5 == 0:
                print(f"Processed {i} examples...")

        # Summary block
        out_f.write("=" * 100 + "\n")
        out_f.write("SUMMARY\n")
        out_f.write(f"Total examples evaluated: {total}\n")

        if total > 0:
            avg_base = sum_score_base / total
            avg_sft = sum_score_sft / total
            out_f.write(f"Base avg heuristic score (all examples): {avg_base:.3f}\n")
            out_f.write(f"SFT avg heuristic score (all examples): {avg_sft:.3f}\n")

        out_f.write(f"Examples with specialized rules applied to both models: {special_examples}\n")

        if special_examples > 0:
            avg_spec_base = sum_spec_base / special_examples
            avg_spec_sft = sum_spec_sft / special_examples
            out_f.write(f"Base avg heuristic score (special-rule subset): {avg_spec_base:.3f}\n")
            out_f.write(f"SFT avg heuristic score (special-rule subset): {avg_spec_sft:.3f}\n")

    # Console summary
    print(f"\nDone. Wrote detailed outputs to {RESULTS_PATH}")
    print(f"Total examples evaluated: {total}")

    if total > 0:
        avg_base = sum_score_base / total
        avg_sft = sum_score_sft / total
        print(f"Base avg heuristic score (all examples): {avg_base:.3f}")
        print(f"SFT avg heuristic score (all examples): {avg_sft:.3f}")

    print(f"Examples with specialized rules applied to both models: {special_examples}")

    if special_examples > 0:
        avg_spec_base = sum_spec_base / special_examples
        avg_spec_sft = sum_spec_sft / special_examples
        print(f"Base avg heuristic score (special subset): {avg_spec_base:.3f}")
        print(f"SFT avg heuristic score (special subset): {avg_spec_sft:.3f}")


# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    main()