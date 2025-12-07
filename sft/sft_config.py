# ======================================================
# sft_config.py
# ======================================================
# Configuration file for SFT training.
# Stores paths, hyperparameters, and LoRA settings.
# Imported by train_sft.py and dataset.py
# ======================================================

import os

# ------------------------------------------------------
# Compute absolute project root, regardless of cwd
# ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------
# Dataset paths
# ------------------------------------------------------
TRAIN_JSON_PATH = os.path.join(PROJECT_ROOT, "data/processed/sft_train.jsonl")
VAL_JSON_PATH   = os.path.join(PROJECT_ROOT, "data/processed/sft_val.jsonl")

# ------------------------------------------------------
# Base model name for SFT
# ------------------------------------------------------
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Maximum sequence length for tokenization
MAX_SEQ_LENGTH = 512

# ------------------------------------------------------
# LoRA configuration
# These adapters are inserted into attention projection layers.
# ------------------------------------------------------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------
BATCH_SIZE = 2            # Mac-friendly small batch size
GRADIENT_ACCUM = 4        # Simulates larger batch size
LEARNING_RATE = 2e-4
MAX_STEPS = 5000
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

LOGGING_STEPS = 50
EVAL_STEPS = 300

# ------------------------------------------------------
# Output & logging directories
# ------------------------------------------------------
OUTPUT_DIR = "models/sft_output"
LOG_DIR = "logs/sft_logs"

# Random seed for reproducibility
SEED = 42