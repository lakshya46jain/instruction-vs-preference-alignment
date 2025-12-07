# ---------------------------------------------------------
# dpo_test.py
# ---------------------------------------------------------
# Purpose:
#   - Load base TinyLlama model + DPO LoRA adapter
#   - Generate output for simple sanity-check
#   - Confirm that DPO training actually changed behavior
# ---------------------------------------------------------
 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
 
# Base pretrained checkpoint
BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
 
# Correct path to your final DPO adapter weights
DPO_ADAPTER = "models/dpo_output"
 
 
def main():
    # -----------------------------------------------
    # Choose device (MPS → CUDA → CPU)
    # -----------------------------------------------
    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"\nUsing device: {device}\n")
 
    # -----------------------------------------------
    # Load tokenizer from base model
    # -----------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE)
 
    # -----------------------------------------------
    # Load base model
    # -----------------------------------------------
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float32,  # Safe for CPU/MPS
    ).to(device)
 
    # -----------------------------------------------
    # Load your DPO LoRA adapter weights ON TOP
    # -----------------------------------------------
    print("Loading DPO adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        DPO_ADAPTER
    ).to(device)
 
    model.eval()
 
    # -----------------------------------------------
    # Create a test prompt using TinyLlama chat template
    # -----------------------------------------------
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain the concept of model alignment."}],
        tokenize=False,
        add_generation_prompt=True
    )
 
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
 
    # -----------------------------------------------
    # Generate output
    # -----------------------------------------------
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
 
    print("\nGenerated DPO Response:\n")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
 
 
if __name__ == "__main__":
    main()