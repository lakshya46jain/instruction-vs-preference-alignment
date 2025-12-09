# ---------------------------------------------------------
# hybrid_test.py - Test the Hybrid model
# ---------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SFT_DIR = "models/sft_output"
HYBRID_DIR = "models/hybrid_output"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading SFT adapters...")
model = PeftModel.from_pretrained(model, SFT_DIR)

print("Loading Hybrid (DPO) adapters...")
model = PeftModel.from_pretrained(model, HYBRID_DIR)

print("Merging for inference...")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(HYBRID_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print("✅ Model loaded successfully!\n")

# Test function
def generate(prompt: str) -> str:
    formatted = (
        "<|system|>\nYou are helpful.\n"
        "<|user|>\n" + prompt + "\n"
        "<|assistant|>\n"
    )
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )
    
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# Run tests
tests = [
    "Explain machine learning.",
    "Write Python code to reverse a string.",
    "What are the benefits of exercise?"
]

print("HYBRID MODEL TEST RESULTS")
print("=" * 70)

for i, t in enumerate(tests, 1):
    print(f"\nTest {i}: {t}")
    response = generate(t)
    print(response)
    print("-" * 70)
