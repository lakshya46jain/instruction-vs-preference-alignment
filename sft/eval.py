import os
import json
import shutil
import inspect
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from sft.sft_config import BASE_MODEL_NAME, OUTPUT_DIR

ORIG_ADAPTER_DIR = OUTPUT_DIR
PATCHED_ADAPTER_DIR = OUTPUT_DIR + "_eval"


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
        device_map="auto"
    )
    model.eval()
    return model


def prepare_patched_adapter_dir():
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
                print("Dropping unsupported config keys:", dropped)

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


def format_prompt(tokenizer, content):
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate(model, tokenizer, prompt, max_new_tokens=200):
    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    tokenizer = get_tokenizer()
    print("Tokenizer loaded.")

    base_model = load_base_model()
    print("Base model loaded.")

    sft_model = load_sft_model()
    print("SFT model loaded.")

    prompts = [
        "Explain the difference between instruction tuning and preference-based alignment in 3 bullet points.",
        "Summarize reinforcement learning from human feedback (RLHF) in two concise sentences.",
        "List three benefits of supervised fine-tuning (SFT) for small language models."
    ]

    for i, p in enumerate(prompts, 1):
        print("\n" + "=" * 100)
        print("PROMPT {}: {}".format(i, p))
        print()

        base_out = generate(base_model, tokenizer, p)
        sft_out = generate(sft_model, tokenizer, p)

        print("---- BASE MODEL OUTPUT ----")
        print(base_out if base_out else "[EMPTY]")
        print()

        print("---- SFT MODEL OUTPUT ----")
        print(sft_out if sft_out else "[EMPTY]")
        print()


if __name__ == "__main__":
    main()
