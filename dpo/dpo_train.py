import os
import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

# ==============================
# ==== CONFIG (VRAM SAFE) ======
# ==============================
BASE_MODEL_DIR = "models/sft_output"
PREFS_TRAIN = "data/processed/prefs_train.jsonl"
PREFS_VAL   = "data/processed/prefs_val.jsonl"
OUTPUT_DIR  = "models/dpo_output"

MAX_LENGTH = 384          # ↓↓↓ reduce context to save VRAM
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 1
LR = 1e-5
WEIGHT_DECAY = 0.01
BETA = 0.1
LOG_EVERY = 20
USE_4BIT = True      # <-- enable 4-bit quantization
# ==============================


# ------------------------------
# Helper to free memory
# ------------------------------
def clean_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------
# Collator
# ------------------------------
class DPOCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        prompts = [ex["prompt"] for ex in batch]
        chosen  = [ex["chosen"] for ex in batch]
        rejected = [ex["rejected"] for ex in batch]

        sys = "<|system|>\nYou are a helpful assistant.\n"

        chosen_texts = [
            f"{sys}<|user|>\n{p}\n<|assistant|>\n{c}"
            for p, c in zip(prompts, chosen)
        ]
        rejected_texts = [
            f"{sys}<|user|>\n{p}\n<|assistant|>\n{r}"
            for p, r in zip(prompts, rejected)
        ]
        prompt_only = [
            f"{sys}<|user|>\n{p}\n<|assistant|>\n"
            for p in prompts
        ]

        prompt_enc = self.tokenizer(prompt_only, truncation=True, padding=True,
                                    max_length=self.max_length, return_tensors="pt")
        chosen_enc = self.tokenizer(chosen_texts, truncation=True, padding=True,
                                    max_length=self.max_length, return_tensors="pt")
        rejected_enc = self.tokenizer(rejected_texts, truncation=True, padding=True,
                                      max_length=self.max_length, return_tensors="pt")

        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)

        return {
            "chosen_ids": chosen_enc["input_ids"],
            "chosen_mask": chosen_enc["attention_mask"],
            "reject_ids": rejected_enc["input_ids"],
            "reject_mask": rejected_enc["attention_mask"],
            "prompt_lens": prompt_lens,
        }


# ------------------------------
# Logprob computation
# ------------------------------
def compute_logprobs(model, ids, mask, prompt_lens, device=None):
    """
    Computes log-probabilities of the response tokens for given model.
    If device is provided, inputs are moved to that device before the forward pass.
    """
    if device is not None:
        ids = ids.to(device)
        mask = mask.to(device)
        prompt_lens = prompt_lens.to(device)

    # Forward pass (no grad when used for reference)
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1]
    labels = ids[:, 1:]
    attn_shift = mask[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    B, T = token_lp.shape
    response_mask = torch.zeros_like(token_lp)

    for i in range(B):
        start = prompt_lens[i].item() - 1
        if start < T:
            response_mask[i, start:] = 1

    token_lp = token_lp * response_mask * attn_shift
    return token_lp.sum(dim=-1)


# ------------------------------
# DPO LOSS
# ------------------------------
def dpo_loss(policy, ref, batch, beta, device):
    # batch tensors are placed on training device by default (device)
    c_ids = batch["chosen_ids"]
    c_mask = batch["chosen_mask"]
    r_ids = batch["reject_ids"]
    r_mask = batch["reject_mask"]
    p_lens = batch["prompt_lens"]

    # Device of policy (trainable) and reference (could be CPU)
    policy_device = device
    ref_device = next(ref.parameters()).device if any(p is not None for p in ref.parameters()) else torch.device("cpu")

    # Ensure policy use_cache disabled if gradient checkpointing enabled
    try:
        if getattr(policy.config, "use_cache", None) and getattr(policy, "gradient_checkpointing", False):
            policy.config.use_cache = False
    except Exception:
        pass

    # Policy forward pass on GPU (or device)
    pi_c = compute_logprobs(policy, c_ids, c_mask, p_lens, device=policy_device)
    pi_r = compute_logprobs(policy, r_ids, r_mask, p_lens, device=policy_device)

    # Reference forward pass on its device (no grad)
    with torch.no_grad():
        ref_c = compute_logprobs(ref, c_ids, c_mask, p_lens, device=ref_device)
        ref_r = compute_logprobs(ref, r_ids, r_mask, p_lens, device=ref_device)

    # Move ref values to policy device for arithmetic (small tensors)
    if ref_c.device != pi_c.device:
        ref_c = ref_c.to(pi_c.device)
        ref_r = ref_r.to(pi_r.device)

    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
    return -torch.nn.functional.logsigmoid(logits).mean()


# ------------------------------
# MAIN
# ------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_mem()

    # DEVICE
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device)
    print(f"Using device: {device}")

    # DATA
    train_set = load_dataset("json", data_files=PREFS_TRAIN, split="train")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    collator = DPOCollator(tokenizer, MAX_LENGTH)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=collator)

    # QUANTIZATION CONFIG
    quant_cfg = None
    dtype = torch.float32
    if USE_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        dtype = torch.float16
        print(">>> Using 4-bit quantization for policy model")

    # POLICY MODEL
    policy = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map="auto" if USE_4BIT else None
    )

    # If gradient checkpointing is enabled, disable use_cache to avoid warnings
    try:
        policy.gradient_checkpointing_enable()
        if hasattr(policy.config, "use_cache"):
            policy.config.use_cache = False
    except Exception:
        pass

    policy.train()

    # REFERENCE MODEL — keep on CPU for VRAM savings
    print(">>> Loading reference model on CPU to save VRAM")
    ref = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float32,
        device_map=None
    )
    # explicitly ensure ref is on CPU
    ref.to(torch.device("cpu"))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    # OPTIMIZER + SCHEDULER
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup = max(10, total_steps // 10)

    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    # TRAIN LOOP
    from tqdm import tqdm
    step = 0
    running = 0.0

    print(">>> Starting DPO training...")

    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(loader):
            loss = dpo_loss(policy, ref, batch, BETA, device)

            loss.backward()
            running += loss.item()
            step += 1

            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                clean_mem()

        print(f"Epoch {epoch+1} done | avg loss = {running/step:.4f}")

    # SAVE MODEL
    print("Saving model...")
    policy.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("DPO complete!")


if __name__ == "__main__":
    main()