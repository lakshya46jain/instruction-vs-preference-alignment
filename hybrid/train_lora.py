
import torch
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import PeftModel

print("HYBRID TRAINING (LoRA Format)")
start = datetime.now()

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SFT_CHECKPOINT = "models/sft_output"
TRAIN_DATA = "data/processed/prefs_train.jsonl"
VAL_DATA = "data/processed/prefs_val.jsonl"
OUTPUT_DIR = "models/hybrid_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model (NO MERGE - keep as LoRA)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, SFT_CHECKPOINT, torch_dtype=torch.bfloat16)
model.train()
print("Model loaded (LoRA format)")

tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "validation": VAL_DATA})
print(f"Data: {len(dataset['train'])} train")

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR, num_train_epochs=2, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, learning_rate=5e-6, lr_scheduler_type="cosine",
    warmup_steps=100, max_length=512, max_prompt_length=256, logging_steps=10,
    eval_steps=100, save_steps=250, save_total_limit=3, bf16=True,
    gradient_checkpointing=True, max_grad_norm=1.0, beta=0.3,
    remove_unused_columns=False, report_to="none", seed=42, optim="adamw_torch",
)

trainer = DPOTrainer(model=model, ref_model=None, args=dpo_config,
                     train_dataset=dataset["train"], eval_dataset=dataset["validation"],
                     processing_class=tokenizer)

print("Training...")
result = trainer.train()
end = datetime.now()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done! Duration: {end-start}, Loss: {result.training_loss:.4f}")
