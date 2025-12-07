# ---------------------------------------------------------
# train_dpo.py
# ---------------------------------------------------------
# Purpose:
#   - Load preference pairs (prompt, chosen, rejected)
#   - Apply a compact DPO implementation that works on low VRAM
#   - Support 4-bit quantization for the POLICY model only
#   - Keep REFERENCE model on CPU for maximum VRAM savings
#   - Compute token-level log-probabilities for policy vs. reference
#   - Optimize using the DPO loss described in class (Bradley–Terry)
#
# High-level idea:
#   POLICY MODEL (trainable) should assign higher probability
#   to the preferred “chosen” answer than the “rejected” answer.
#
#   DPO works by comparing:
#       πθ(chosen | prompt) − πθ(rejected | prompt)
#   against the same difference from a frozen REFERENCE model.
#
#   The objective pushes POLICY to increase preference for the
#   chosen response while staying close to the REFERENCE distribution.
#
# ---------------------------------------------------------


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


# =========================================================
# ================== CONFIGURATION =========================
# =========================================================

BASE_MODEL_DIR = "models/sft_output"   # SFT model becomes policy + reference base
PREFS_TRAIN = "data/processed/prefs_train.jsonl"
PREFS_VAL   = "data/processed/prefs_val.jsonl"
OUTPUT_DIR  = "models/dpo_output"

# VRAM-friendly hyperparameters
MAX_LENGTH = 384
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 1
LR = 1e-5
WEIGHT_DECAY = 0.01
BETA = 0.1               # DPO "temperature" hyperparameter
LOG_EVERY = 20
USE_4BIT = True          # Enable NF4 quantization for POLICY model

# ----------------------------------------------------------
# Utility to free GPU cache between steps
# ----------------------------------------------------------
def clean_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# ================== COLLATOR =============================
# =========================================================
# Converts one JSON preference example into tensors.
#
# Each record has:
#   prompt, chosen, rejected
#
# We format into model-chat template:
#
#   <|system|> ...
#   <|user|> prompt
#   <|assistant|> chosen
#
# Prompt-only → needed to locate where the assistant response begins.
# =========================================================

class DPOCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        prompts  = [ex["prompt"] for ex in batch]
        chosen   = [ex["chosen"] for ex in batch]
        rejected = [ex["rejected"] for ex in batch]

        # System instruction used during training — stable & simple.
        sys = "<|system|>\nYou are a helpful assistant.\n"

        # Construct full sequences
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

        # Tokenize each group
        prompt_enc = self.tokenizer(
            prompt_only,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        chosen_enc = self.tokenizer(
            chosen_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Compute prompt lengths to isolate the RESPONSE tokens
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)

        return {
            "chosen_ids":  chosen_enc["input_ids"],
            "chosen_mask": chosen_enc["attention_mask"],
            "reject_ids":  rejected_enc["input_ids"],
            "reject_mask": rejected_enc["attention_mask"],
            "prompt_lens": prompt_lens,
        }


# =========================================================
# ========== TOKEN-LEVEL LOGPROB COMPUTATION =============
# =========================================================
# Given:
#    ids, mask, prompt_lens
#
# We:
#   1. Forward pass model → logits
#   2. Convert logits → log-softmax
#   3. Gather log-probabilities for the labels (next tokens)
#   4. Mask out the prompt portion so that only assistant
#      response tokens contribute.
#
# This matches language modeling log-likelihood.
# =========================================================

def compute_logprobs(model, ids, mask, prompt_lens, device=None):
    if device is not None:
        ids = ids.to(device)
        mask = mask.to(device)
        prompt_lens = prompt_lens.to(device)

    # Forward pass
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1]    # shift for LM targets
    labels = ids[:, 1:]            # next-token labels
    attn_shift = mask[:, 1:]       # ensure masked tokens don't contribute

    # log-softmax over vocabulary
    log_probs = torch.log_softmax(logits, dim=-1)

    # Select log-prob of the correct next token
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Mask out the prompt portion — only evaluate assistant response
    B, T = token_lp.shape
    response_mask = torch.zeros_like(token_lp)

    for i in range(B):
        start = prompt_lens[i].item() - 1
        if start < T:
            response_mask[i, start:] = 1  # keep only assistant tokens

    # Final log-prob over response tokens
    token_lp = token_lp * response_mask * attn_shift
    return token_lp.sum(dim=-1)   # shape: (batch,)


# =========================================================
# ====================== DPO LOSS =========================
# =========================================================
# The DPO objective:
#
#   L = -log σ( β[(πθ(c)-πθ(r)) - (πref(c)-πref(r))] )
#
# where:
#   πθ = policy model log-probabilities
#   πref = reference model log-probabilities (frozen)
#   β = scaling constant controlling "sharpness"
#
# This pushes policy to increase preference for “chosen”
# while remaining close to reference behavior.
# =========================================================

def dpo_loss(policy, ref, batch, beta, device):
    c_ids = batch["chosen_ids"]
    c_mask = batch["chosen_mask"]
    r_ids = batch["reject_ids"]
    r_mask = batch["reject_mask"]
    p_lens = batch["prompt_lens"]

    # POLICY logprobs — computed on GPU (trainable)
    pi_c = compute_logprobs(policy, c_ids, c_mask, p_lens, device=device)
    pi_r = compute_logprobs(policy, r_ids, r_mask, p_lens, device=device)

    # REFERENCE logprobs — computed on CPU (frozen)
    with torch.no_grad():
        ref_c = compute_logprobs(ref, c_ids, c_mask, p_lens, device=ref.device)
        ref_r = compute_logprobs(ref, r_ids, r_mask, p_lens, device=ref.device)

    # Move tiny tensors to GPU for computation
    ref_c = ref_c.to(device)
    ref_r = ref_r.to(device)

    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))

    # Standard DPO final loss
    return -torch.nn.functional.logsigmoid(logits).mean()


# =========================================================
# ====================== MAIN LOOP ========================
# =========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_mem()

    # ----------------------------
    # DEVICE SELECTION
    # ----------------------------
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device)
    print(f"Using device: {device}")

    # ----------------------------
    # DATA LOADING
    # ----------------------------
    train_set = load_dataset("json", data_files=PREFS_TRAIN, split="train")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    collator = DPOCollator(tokenizer, MAX_LENGTH)
    loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    # ----------------------------
    # QUANTIZATION (POLICY ONLY)
    # ----------------------------
    quant_cfg = None
    dtype = torch.float32

    if USE_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        dtype = torch.float16
        print(">>> Using 4-bit quantization for policy model")

    # ----------------------------
    # LOAD POLICY MODEL (TRAINABLE)
    # ----------------------------
    policy = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map="auto" if USE_4BIT else None
    )

    # Enable gradient checkpointing for VRAM savings
    try:
        policy.gradient_checkpointing_enable()
        if hasattr(policy.config, "use_cache"):
            policy.config.use_cache = False
    except Exception:
        pass

    policy.train()

    # ----------------------------
    # LOAD REFERENCE MODEL (FROZEN ON CPU)
    # ----------------------------
    print(">>> Loading reference model on CPU to save VRAM")
    ref = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float32,
        device_map=None,
    )
    ref.to(torch.device("cpu"))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    # ----------------------------
    # OPTIMIZER + LR SCHEDULER
    # ----------------------------
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup = max(10, total_steps // 10)

    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------
    from tqdm import tqdm
    step = 0
    total_loss = 0.0

    print(">>> Starting DPO training...")

    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(loader):
            loss = dpo_loss(policy, ref, batch, BETA, device)

            loss.backward()
            total_loss += loss.item()
            step += 1

            # Gradient accumulation
            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                clean_mem()

        avg_loss = total_loss / step
        print(f"Epoch {epoch+1} complete | avg loss = {avg_loss:.4f}")

    # ----------------------------
    # SAVE FINAL POLICY
    # ----------------------------
    print("Saving final DPO model...")
    policy.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("DPO training complete!")


if __name__ == "__main__":
    main()