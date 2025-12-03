from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model_path = "tiny-gpt2-DPO/checkpoint-3"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

gen_cfg = GenerationConfig(
    max_new_tokens=120,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.4,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        generation_config=gen_cfg,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate("Explain why pets should be allowed in dorms."))