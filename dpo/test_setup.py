# test_setup.py
import torch
import transformers
import datasets
import trl
import peft
import accelerate
import bitsandbytes as bnb

print("All libraries imported successfully!")

# Check torch and GPU
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

# Optional: Check bitsandbytes
try:
    import bitsandbytes as bnb
    print("bitsandbytes version:", bnb.__version__)
except Exception as e:
    print("bitsandbytes not working:", e)
