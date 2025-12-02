import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA = "models/sft_output/checkpoint-5000"

def main():
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    base_model = AutoModelForCausalLM.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base_model, LORA).to(device)

    # PROPER CHAT PROMPT
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain supervised fine-tuning in simple terms."}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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