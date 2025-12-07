# ---------------------------------------------------------
# sft_test.py
# ---------------------------------------------------------
# Purpose:
#   - Load a base TinyLlama model + your trained LoRA checkpoint
#   - Apply the HF chat template
#   - Generate a sample response for manual sanity-check
# ---------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Base pretrained weights
BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Location of LoRA fine-tuned checkpoint
LORA = "models/sft_output"


def main():
    # -----------------------------------------------
    # Choose device (prefers MPS → CUDA → CPU)
    # -----------------------------------------------
    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(BASE)

    # Load full base model
    base_model = AutoModelForCausalLM.from_pretrained(BASE)

    # Load your LoRA adapter weights ON TOP of the base
    model = PeftModel.from_pretrained(base_model, LORA).to(device)

    # ------------------------------------------------
    # Generate a test prompt using the official
    # chat template for TinyLlama Chat
    # ------------------------------------------------
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain supervised fine-tuning in simple terms."}],
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize text for model input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ------------------------------------------------
    # Generate a sample output
    # ------------------------------------------------
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

    print("\nGenerated Response:\n")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()