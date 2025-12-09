# ---------------------------------------------------------
# train_hybrid.py
# ---------------------------------------------------------
# Purpose:
#   - Load TinyLlama + SFT LoRA
#   - Merge SFT weights to form strong initialization
#   - Add a new LoRA adapter for preference optimization
#   - Load a frozen reference model for DPO comparison
#   - Run DPO training to produce the hybrid model
#   - Save only the new LoRA adapter
# ---------------------------------------------------------

import torch
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model

print("=" * 70)
print("HYBRID TRAINING (SFT + DPO)")
print("=" * 70)

start = datetime.now()

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SFT_CHECKPOINT = "models/sft_output"
TRAIN_DATA = "data/processed/prefs_train.jsonl"
VAL_DATA = "data/processed/prefs_val.jsonl"
OUTPUT_DIR = "models/hybrid_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load frozen reference model (merged SFT)
# ---------------------------------------------------------
print("\nLoading reference model (frozen SFT)...")

ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
ref_model = PeftModel.from_pretrained(ref_model, SFT_CHECKPOINT, torch_dtype=torch.bfloat16)
ref_model = ref_model.merge_and_unload()
ref_model.eval()

for p in ref_model.parameters():
    p.requires_grad = False

print("Reference model loaded and frozen.")


# ---------------------------------------------------------
# Load policy model (SFT merged) + attach NEW LoRA for DPO
# ---------------------------------------------------------
print("\nLoading policy model and applying new LoRA adapter...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

# Merge SFT LoRA into base model
model = PeftModel.from_pretrained(model, SFT_CHECKPOINT, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

# Add new LoRA adapter for hybrid/DPO training
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.train()

print("Policy model ready with new LoRA layer.")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ---------------------------------------------------------
# Load tokenizer and datasets
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "validation": VAL_DATA})
print(f"Loaded dataset: {len(dataset['train'])} train samples, {len(dataset['validation'])} validation samples.")


# ---------------------------------------------------------
# DPO Configuration
# ---------------------------------------------------------
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    max_length=512,
    max_prompt_length=256,
    logging_steps=10,
    eval_steps=100,
    save_steps=250,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=False,  # Prevents zero-gradient issues in TRL
    max_grad_norm=1.0,
    beta=0.3,
    remove_unused_columns=False,
    report_to="none",
    seed=42,
    optim="adamw_torch",
)


# ---------------------------------------------------------
# Initialize DPO Trainer
# ---------------------------------------------------------
print("\nInitializing DPO trainer...")

trainer = DPOTrainer(
    model=model,             # Policy model (trainable)
    ref_model=ref_model,     # Reference model (frozen)
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

print("Trainer initialized with policy and reference models.")


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("Starting hybrid DPO training")
print("=" * 70)

result = trainer.train()
end = datetime.now()

print("\nTraining complete.")
print(f"Duration: {end - start}")
print(f"Final training loss: {result.training_loss:.4f}")


# ---------------------------------------------------------
# Save LoRA adapters only
# ---------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved hybrid LoRA adapter to: {OUTPUT_DIR}")