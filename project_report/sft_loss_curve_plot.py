import json
import matplotlib.pyplot as plt
import os

trainer_state_path = "../models/sft_output/checkpoint-5000/trainer_state.json"

if not os.path.exists(trainer_state_path):
    raise FileNotFoundError(f"File not found: {trainer_state_path}")

with open(trainer_state_path, "r") as f:
    state = json.load(f)

log_history = state.get("log_history", [])

steps = []
losses = []

for entry in log_history:
    # We only want entries that contain both "step" and "loss"
    if "step" in entry and "loss" in entry:
        steps.append(entry["step"])
        losses.append(entry["loss"])

if len(steps) == 0:
    raise ValueError("No loss entries found in trainer_state.json")

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, marker="o", linestyle="-", linewidth=1.5)

plt.xlabel("Training Step", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("SFT Training Loss Curve (5,000 Steps)", fontsize=14)
plt.grid(True)
plt.tight_layout()

save_dir = "./figs"
os.makedirs(save_dir, exist_ok=True)   # Create folder if missing

output_path = os.path.join(save_dir, "sft_loss_curve.png")

plt.savefig(output_path, dpi=300)
print(f"Saved training loss plot as: {output_path}")

# Show plot in interactive environments
plt.show()