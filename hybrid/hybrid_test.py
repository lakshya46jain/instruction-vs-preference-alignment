# ---------------------------------------------------------
# hybrid_test.py
# ---------------------------------------------------------
# Purpose:
#   - Load the TinyLlama base model
#   - Apply the SFT LoRA adapter (merged during SFT phase)
#   - Apply the Hybrid (SFT + DPO) LoRA adapter
#   - Merge adapters for inference
#   - Run a few test prompts to confirm Hybrid model behavior
# ---------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------
# Paths for required model components
# ---------------------------------------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # Base pretrained weights
SFT_DIR    = "models/sft_output"                   # SFT LoRA checkpoint
HYBRID_DIR = "models/hybrid_output"                # Hybrid (DPO-trained) LoRA checkpoint


def main():
    # ---------------------------------------------------------
    # Load the base TinyLlama model (full weights)
    # ---------------------------------------------------------
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # ---------------------------------------------------------
    # Load and merge the SFT adapter onto the base model
    # ---------------------------------------------------------
    print("Loading SFT adapter...")
    model = PeftModel.from_pretrained(model, SFT_DIR)
    model = model.merge_and_unload()  # Merges SFT LoRA into the base weights

    # Remove stale PEFT metadata to avoid "multiple adapters" warning
    if hasattr(model, "peft_config"):
        del model.peft_config

    # ---------------------------------------------------------
    # Load the Hybrid (SFT + DPO) adapter
    # ---------------------------------------------------------
    print("Loading Hybrid (DPO) adapter...")
    model = PeftModel.from_pretrained(model, HYBRID_DIR)

    # After loading both SFT and Hybrid LoRA, merge for inference
    print("Merging adapters for inference...")
    model = model.merge_and_unload()

    # ---------------------------------------------------------
    # Load tokenizer (Hybrid folder contains correct special tokens)
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(HYBRID_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded successfully.\n")

    # ---------------------------------------------------------
    # Helper function: Generate a model response for a given prompt
    # ---------------------------------------------------------
    def generate(prompt: str) -> str:
        """
        Format the prompt using TinyLlama-style chat format,
        feed to the model, and decode the generated assistant output.
        """
        # Chat formatting consistent with training
        formatted = (
            "<|system|>\nYou are helpful.\n"
            "<|user|>\n" + prompt + "\n"
            "<|assistant|>\n"
        )

        # Tokenize and move to model device
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        # Generate continuation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9
            )

        # Extract generated tokens (exclude the prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # ---------------------------------------------------------
    # Run a few manual correctness tests
    # ---------------------------------------------------------
    tests = [
        "Explain machine learning.",
        "Write Python code to reverse a string.",
        "What are the benefits of exercise?"
    ]

    print("Hybrid Model Test Results")
    print("=" * 70)

    for i, t in enumerate(tests, 1):
        print(f"\nTest {i}: {t}")
        print(generate(t))
        print("-" * 70)


if __name__ == "__main__":
    main()