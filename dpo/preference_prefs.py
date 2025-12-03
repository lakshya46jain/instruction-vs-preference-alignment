import json

samples = [
    {
        "prompt": "Explain gravity simply.",
        "chosen": "Gravity is a force that pulls objects together.",
        "rejected": "Gravity is when magnets pull things."
    },
    {
        "prompt": "Why is the sky blue?",
        "chosen": "Air scatters sunlight and blue light scatters the most.",
        "rejected": "Because the ocean reflects its color into the sky."
    },
]

with open("data/processed/prefs_train.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

with open("data/processed/prefs_val.jsonl", "w") as f:
    for s in samples[:1]:
        f.write(json.dumps(s) + "\n")

print("Preference data created!")