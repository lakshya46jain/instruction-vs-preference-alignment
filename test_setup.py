"""
Test script to verify setup
"""

print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers error: {e}")

try:
    import trl
    print(f"✓ TRL {trl.__version__}")
except Exception as e:
    print(f"✗ TRL error: {e}")

try:
    import peft
    print(f"✓ PEFT {peft.__version__}")
except Exception as e:
    print(f"✗ PEFT error: {e}")

try:
    from datasets import load_dataset
    print(f"✓ Datasets library working")
except Exception as e:
    print(f"✗ Datasets error: {e}")

print("\n" + "="*50)
print("Setup verification complete!")
print("="*50)