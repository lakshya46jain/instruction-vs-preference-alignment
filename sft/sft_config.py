BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# data
TRAIN_JSON_PATH = "data/processed/sft_train.jsonl"
VAL_JSON_PATH = "data/processed/sft_val.jsonl"
MAX_SEQ_LENGTH = 1024

# LoRA / QLoRA configuration
USE_QLORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# training hyperparameters
BATCH_SIZE = 4           # per device
GRADIENT_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EVAL_STEPS = 500

OUTPUT_DIR = "models/sft-lora-tinyllama"
LOG_DIR = "logs/sft"
SEED = 42