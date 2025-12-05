import os
import math
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

BASE_MODEL_DIR = "models/sft_output"
PREFS_TRAIN_PATH = "data/processed/prefs_train.jsonl"
PREFS_VAL_PATH = "data/processed/prefs_val.jsonl"
OUTPUT_DIR = "models/dpo_output"

MAX_LENGTH = 512
BATCH_SIZE = 1           # per step
GRAD_ACCUM_STEPS = 4     # effective batch = 4
NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
LOG_EVERY = 20
BETA = 0.1               # DPO beta


def build_text(prompt: str, response: str) -> str:
    # TinyLlama-Chat official formatting (newline-separated messages)
    return (
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\n" + prompt + "\n"
        "<|assistant|>\n" + response
    )

class DPOCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        prompts = [ex["prompt"] for ex in batch]
        chosen_responses = [ex["chosen"] for ex in batch]
        rejected_responses = [ex["rejected"] for ex in batch]

        # Build full texts
        chosen_texts = [
            "<|system|>\nYou are a helpful assistant.\n"
            "<|user|>\n" + p + "\n"
            "<|assistant|>\n" + c
            for p, c in zip(prompts, chosen_responses)
        ]

        rejected_texts = [
            "<|system|>\nYou are a helpful assistant.\n"
            "<|user|>\n" + p + "\n"
            "<|assistant|>\n" + r
            for p, r in zip(prompts, rejected_responses)
        ]

        # Also tokenize prompt-only to get correct prompt lengths AFTER padding
        prompt_only_texts = [
            "<|system|>\nYou are a helpful assistant.\n"
            "<|user|>\n" + p + "\n"
            "<|assistant|>\n"
            for p in prompts
        ]

        prompt_enc = self.tokenizer(
            prompt_only_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        chosen_enc = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        rejected_enc = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # TRUE prompt lengths (count non-pad tokens)
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "prompt_lens": prompt_lens,
        }

def compute_logprobs(model, input_ids, attention_mask, prompt_lens):
    """
    Computes logprobs ONLY for response tokens,
    properly correcting for left padding.
    """

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    # Shift for next-token prediction
    logits = logits[:, :-1]
    labels = input_ids[:, 1:]
    mask_shifted = attention_mask[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    B, T = token_log_probs.shape
    response_mask = torch.zeros_like(token_log_probs)

    for i in range(B):
        pl = prompt_lens[i].item()

        # Because the shifted labels start at position 1,
        # the response begins exactly at index (pl - 1)
        start = pl - 1

        if start < T:
            response_mask[i, start:] = 1

    # Only keep logprobs of response tokens
    token_log_probs = token_log_probs * response_mask * mask_shifted

    return token_log_probs.sum(dim=-1)

def dpo_loss(
    policy_model,
    ref_model,
    batch: Dict[str, torch.Tensor],
    beta: float,
    device: torch.device,
):
    # Move tensors
    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_mask = batch["chosen_attention_mask"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_mask = batch["rejected_attention_mask"].to(device)

    prompt_lens = batch["prompt_lens"].to(device)

    pi_chosen = compute_logprobs(policy_model, chosen_ids, chosen_mask, prompt_lens)
    pi_rejected = compute_logprobs(policy_model, rejected_ids, rejected_mask, prompt_lens)

    with torch.no_grad():
        ref_chosen = compute_logprobs(ref_model, chosen_ids, chosen_mask, prompt_lens)
        ref_rejected = compute_logprobs(ref_model, rejected_ids, rejected_mask, prompt_lens)

    # Advantages
    pi_diff = pi_chosen - pi_rejected        # [B]
    ref_diff = ref_chosen - ref_rejected     # [B]

    # DPO loss: -log σ( β[(π(y⁺)-π(y⁻)) - (π_ref(y⁺)-π_ref(y⁻))] )
    logits = beta * (pi_diff - ref_diff)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # Sanity check (debug-style, but cheap)
    if not loss.requires_grad:
        raise RuntimeError("DPO loss ended up without grad; check policy_model parameters.")

    return loss


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- Device -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- Data -----
    print("Loading preference datasets...")
    train_dataset = load_dataset("json", data_files=PREFS_TRAIN_PATH, split="train")
    eval_dataset = load_dataset("json", data_files=PREFS_VAL_PATH, split="train")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DPOCollator(tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    # ----- Models -----
    print("Loading policy model from SFT checkpoint...")
    # float32 everywhere (safer on MPS/CPU)
    dtype = torch.float32
    policy_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=dtype,
    ).to(device)
    policy_model.gradient_checkpointing_enable()

    print("Creating frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=dtype,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Make absolutely sure policy model is trainable
    policy_model.train()
    for p in policy_model.parameters():
        p.requires_grad = True

    # ----- Optimizer / Scheduler -----
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in policy_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in policy_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = max(10, max_train_steps // 10)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # ----- Training -----
    from tqdm import tqdm
    import time
    import json

    print("Starting custom DPO training loop...")

    global_step = 0
    optimizer_step = 0
    running_loss = 0.0
    smoothed_loss = None
    loss_log = []  # store losses for export

    start_time = time.time()

    # Progress bar for batches
    progress_bar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

    for batch in progress_bar:
        # Use collated batch from DataLoader → compute proper DPO loss
        loss = dpo_loss(
            policy_model,
            ref_model,
            batch,
            beta=BETA,
            device=device,
        )

        # Backprop
        loss.backward()
        running_loss += loss.item()
        global_step += 1

        # Update smoothed loss (EMA)
        if smoothed_loss is None:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss.item()

        # Update tqdm progress bar info
        progress_bar.set_postfix({
            "loss": f"{smoothed_loss:.4f}",
            "opt_step": optimizer_step,
        })

        # Optimizer step every GRAD_ACCUM_STEPS
        if global_step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1

            # Log the averaged loss
            avg_loss = running_loss / GRAD_ACCUM_STEPS
            running_loss = 0.0

            loss_log.append({
                "optimizer_step": optimizer_step,
                "loss": avg_loss,
                "time": time.time() - start_time
            })

            # Explicit print (for reassurance)
            print(
                f"\nOptimizer Step {optimizer_step} "
                f" | Avg Loss: {avg_loss:.4f} "
                f" | Smoothed Loss: {smoothed_loss:.4f}"
            )

    # END OF TRAINING
    total_time = time.time() - start_time

    print("\n===== TRAINING COMPLETE =====")
    print(f"Total optimizer steps: {optimizer_step}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Final smoothed loss: {smoothed_loss:.4f}")

    # Save loss log
    with open("models/dpo_output/loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    print("Saved loss log to models/dpo_output/loss_log.json")

    # ----- Save -----
    print(f"Saving DPO model to {OUTPUT_DIR} ...")
    policy_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("DPO training complete!")


if __name__ == "__main__":
    main()