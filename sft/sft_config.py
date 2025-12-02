# ================================
# Supervised Fine-Tuning CONFIG
# ================================

import os

# Compute absolute project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute dataset paths (fix for spaces + brackets in folder names)
TRAIN_JSON_PATH = os.path.join(PROJECT_ROOT, "data/processed/sft_train.jsonl")
VAL_JSON_PATH   = os.path.join(PROJECT_ROOT, "data/processed/sft_val.jsonl")

# Choose your base model
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Training sequence length
MAX_SEQ_LENGTH = 512

# LoRA config — used by BOTH Mac (LoRA) AND GPU (QLoRA) scripts
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training hyperparameters
BATCH_SIZE = 2               # Mac-friendly; increase on GPU if needed
GRADIENT_ACCUM = 4
LEARNING_RATE = 2e-4
MAX_STEPS = 5000
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

LOGGING_STEPS = 50
EVAL_STEPS = 300

# Output & logging directories
OUTPUT_DIR = "models/sft_output"
LOG_DIR = "logs/sft_logs"

# Random seed for reproducibility
SEED = 42