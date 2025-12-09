# ---------------------------------------------------------
# hybrid_test.py
# ---------------------------------------------------------
# Purpose:
#   - Load the final Hybrid (SFT + DPO) merged model
#   - Run a few sample prompts through it
#   - Confirm that the hybrid model generates responses correctly
# ---------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Path to the fully merged hybrid model checkpoint
HYBRID_DIR = "models/hybrid_output"


def main():
    # -----------------------------------------------
    # Load model and tokenizer
    # -----------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        HYBRID_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(HYBRID_DIR)

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # -----------------------------------------------
    # Helper generation function
    # -----------------------------------------------
    def generate(prompt: str) -> str:
        """
        Generate a reply from the hybrid model using a simple 
        TinyLlama-style chat wrapper.
        """
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

        # Extract generated continuation (exclude prompt tokens)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # -----------------------------------------------
    # Test prompts for basic functional verification
    # -----------------------------------------------
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


if __name__ == "__main__":
    main()