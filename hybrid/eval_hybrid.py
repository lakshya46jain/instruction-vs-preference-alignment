
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/hybrid_output/final", 
                                              torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("models/hybrid_output/final")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

def generate(prompt):
    formatted = f"<|system|>\nYou are helpful.\n<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

tests = ["Explain machine learning.", "Write Python code to reverse a string.", "Benefits of exercise?"]
print("EVALUATION RESULTS\n" + "="*70)
results = []
for i, t in enumerate(tests, 1):
    print(f"\n[{i}] {t}")
    r = generate(t)
    print(f"→ {r[:150]}...")
    results.append(r)

with open("EVAL_RESULTS.txt", "w") as f:
    for t, r in zip(tests, results):
        f.write(f"Q: {t}\nA: {r}\n\n")
print("\n✅ Evaluation saved to EVAL_RESULTS.txt")
