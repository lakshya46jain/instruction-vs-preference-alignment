import os
import json
import shutil
import inspect
import re
from typing import Iterator, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig

from sft.sft_config import BASE_MODEL_NAME, OUTPUT_DIR


DATA_PATH = "data/processed/sft_val.jsonl" # evaluation dataset
RESULTS_DIR = "evaluation/results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "sft_eval_output.txt")

ORIG_ADAPTER_DIR = OUTPUT_DIR           
PATCHED_ADAPTER_DIR = OUTPUT_DIR + "_eval" 

MAX_EXAMPLES = 100 # set to None to eval full file
MAX_NEW_TOKENS = 100 # how long each answer can be

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model


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

        if fname == "adapter_config.json":
            with open(src, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            filtered = {k: v for k, v in cfg.items() if k in allowed_keys}
            dropped = sorted(set(cfg.keys()) - set(filtered.keys()))
            if dropped:
                print("Dropping unsupported config keys in eval copy:", dropped)

            with open(dst, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2)
        else:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    return PATCHED_ADAPTER_DIR


def load_sft_model():
    base_model = load_base_model()
    patched = prepare_patched_adapter_dir()
    print("Using patched adapter directory:", patched)
    model = PeftModel.from_pretrained(base_model, patched)
    model.eval()
    return model

def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    # Wrap as a single user message
    messages = [{"role": "user", "content": prompt}]

    # Convert to chat-format text (adds BOS, role tokens, etc.)
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

    # Strip off the prompt portion
    generated = output[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # if the model produced nothing, at least let us see that.
    if text == "":
        text = ""

    return text



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


def summary_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "summary" not in instr_l and "summarize" not in instr_l:
        return None
    if not inp:
        return None

    # output should be shorter than input and share key words
    if len(out.strip()) == 0:
        return 0.0
    len_ratio = min(1.0, len(inp) / max(len(out), 1))

    inp_tokens = _token_set(inp)
    out_tokens = _token_set(out)
    if not inp_tokens or not out_tokens:
        overlap = 0.0
    else:
        overlap = len(inp_tokens & out_tokens) / len(inp_tokens)

    # average of length score and content overlap
    return 0.5 * len_ratio + 0.5 * overlap


def negative_rewrite_rule(instr: str, inp: str, out: str) -> Optional[float]:
    instr_l = instr.lower()
    if "negative connotation" not in instr_l and "negative" not in instr_l:
        return None
    if len(out.strip()) == 0:
        return 0.0

    neg_words = ["not", "no", "never", "bad", "poor", "terrible", "awful", "worse", "worst"]
    has_negative = any(w in out.lower() for w in neg_words)

    # require some lexical overlap with input too
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

    # want at least 3 sentences, reward more length up to a point
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

    # parse numbers from reference and output, give credit if one matches
    ref_nums = re.findall(r"-?\d+\.?\d*", reference)
    out_nums = re.findall(r"-?\d+\.?\d*", out)
    if not out_nums:
        return 0.0
    if not ref_nums:
        # cannot align with reference, but at least has a number
        return 0.5

    # check if any number matches exactly
    for rn in ref_nums:
        for on in out_nums:
            try:
                if float(rn) == float(on):
                    return 1.0
            except ValueError:
                continue
    # partial credit for having some number at all
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
    # short-ish, single question preferred
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

    # prefer short, capitalized phrases
    length_penalty = max(0.0, 1.0 - (len(tokens) - 3) * 0.2)  # ideal ~3 words
    capitalized_tokens = sum(1 for t in tokens if t[:1].isupper())
    cap_ratio = capitalized_tokens / max(len(tokens), 1)

    score = 0.5 * length_penalty + 0.5 * cap_ratio
    return max(0.0, min(1.0, score))


def generic_relevance_rule(instr: str, inp: str, out: str, reference: str) -> float:
    """
    Fallback: measure simple lexical relevance between output
    and (instruction + input + reference).
    Always returns a score in [0,1].
    """
    context = " ".join([instr, inp, reference])
    ctx_tokens = _token_set(context)
    out_tokens = _token_set(out)

    if not ctx_tokens or not out_tokens:
        return 0.0

    overlap = len(ctx_tokens & out_tokens)
    return overlap / len(ctx_tokens)


def eval_instruction_following(prompt: str, output: str, reference: str) -> Tuple[float, bool]:
    """
    Returns (score in [0,1], used_special_rules: bool).
    Combines several heuristic rules; if none apply, falls back to generic relevance.
    """
    instr, inp = parse_instruction_and_input(prompt)

    scores = []

    # specialized rules
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
        try:
            val = rule_fn(instr, inp, output) if rule_fn is not calculation_rule else rule_fn(instr, inp, output, reference)
        except TypeError:
            # for safety if signature mismatch
            val = None

        if val is not None:
            scores.append(val)

    used_special = len(scores) > 0

    if not scores:
        # fallback relevance-only
        score = generic_relevance_rule(instr, inp, output, reference)
        return score, False

    # average of all applicable specialized scores, but mixed with relevance for robustness
    spec_score = sum(scores) / len(scores)
    rel_score = generic_relevance_rule(instr, inp, output, reference)
    final_score = 0.7 * spec_score + 0.3 * rel_score
    return final_score, True

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

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

    total = 0
    sum_score_base = 0.0
    sum_score_sft = 0.0
    special_examples = 0
    sum_spec_base = 0.0
    sum_spec_sft = 0.0

    with open(RESULTS_PATH, "w", encoding="utf-8") as out_f:
        out_f.write(
            "Heuristic IF-style evaluation on dataset: {}\n\n".format(DATA_PATH)
        )

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

            # write detailed outputs
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

        # summary
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
            out_f.write(
                f"Base avg heuristic score (special-rule subset): {avg_spec_base:.3f}\n"
            )
            out_f.write(
                f"SFT avg heuristic score (special-rule subset): {avg_spec_sft:.3f}\n"
            )

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


if __name__ == "__main__":
    main()
