"""
Hybrid Fine-Tuning: DPO on top of SFT checkpoint
Author: Demi Omoremi
Course: CS 4804 - Introduction to Artificial Intelligence
Date: December 2025

This script applies Direct Preference Optimization (DPO) to the 
SFT checkpoint to create a hybrid model combining instruction
following with preference alignment.

Test 3
"""

import torch
import os
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import PeftModel

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")

def print_section(text):
    """Print a formatted section header."""
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)

def check_file_exists(filepath, file_description):
    """Check if a required file exists."""
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {file_description} not found at: {filepath}")
        print(f"\nPlease ensure this file exists before running training.")
        sys.exit(1)
    print(f"✓ Found {file_description}: {filepath}")

def load_preference_dataset(data_path):
    """Load and validate preference dataset."""
    print_section("Loading Preference Dataset")
    
    print(f"Loading from: {data_path}")
    
    try:
        dataset = load_dataset("json", data_files=data_path)
        
        # Validate format
        sample = dataset['train'][0]
        required_keys = ['prompt', 'chosen', 'rejected']
        
        missing_keys = [k for k in required_keys if k not in sample]
        if missing_keys:
            print(f"❌ ERROR: Dataset missing required keys: {missing_keys}")
            print(f"Required format: {required_keys}")
            print(f"Found keys: {list(sample.keys())}")
            sys.exit(1)
        
        print(f"✓ Loaded {len(dataset['train'])} preference examples")
        print(f"✓ Dataset format validated")
        print(f"\nSample entry:")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Chosen: {sample['chosen'][:100]}...")
        print(f"  Rejected: {sample['rejected'][:100]}...")
        
        return dataset
        
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        sys.exit(1)

def load_sft_model(base_model_name, sft_checkpoint_path):
    """Load the SFT checkpoint with LoRA adapters merged."""
    print_section("Loading SFT Model")
    
    print(f"Base model: {base_model_name}")
    print(f"Loading base model...")
    
    try:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
            
        )
        
        print(f"✓ Base model loaded")
        print(f"Loading LoRA adapters from: {sft_checkpoint_path}")
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, sft_checkpoint_path)
        
        print(f"Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        
        print(f"✓ SFT model loaded and merged successfully")
        
        return model
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        sys.exit(1)

def main():
    """Main training function."""
    
    print_header("HYBRID FINE-TUNING: SFT + DPO")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    print_section("Configuration")
    
    BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # TODO: Confirm with team
    SFT_CHECKPOINT = "models/sft_output"
    PREFERENCE_DATA = "data/processed/preference_train.json"  # TODO: Get from Aditya
    OUTPUT_DIR = "models/hybrid_sft_dpo_output"
    
    print(f"Base Model:        {BASE_MODEL_NAME}")
    print(f"SFT Checkpoint:    {SFT_CHECKPOINT}")
    print(f"Preference Data:   {PREFERENCE_DATA}")
    print(f"Output Directory:  {OUTPUT_DIR}")
    
    # Validate all required files exist
    print_section("Validating Files")
    check_file_exists(SFT_CHECKPOINT, "SFT checkpoint")
    check_file_exists(PREFERENCE_DATA, "Preference dataset")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")
    
    # ========================================
    # LOAD MODEL
    # ========================================
    
    model = load_sft_model(BASE_MODEL_NAME, SFT_CHECKPOINT)
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # ========================================
    # LOAD DATASET
    # ========================================
    
    dataset = load_preference_dataset(PREFERENCE_DATA)
    
    # ========================================
    # CONFIGURE TRAINING
    # ========================================
    
    print_section("DPO Training Configuration")
    
    training_args = DPOConfig(
        # Output
        output_dir=OUTPUT_DIR,
        
        # Training epochs and batch size
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        
        # Optimization
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        
        # Sequence lengths
        max_length=512,
        max_prompt_length=256,
        
        # Logging and saving
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        logging_dir=f"{OUTPUT_DIR}/logs",
        
        # Performance optimizations
        bf16=torch.cuda.is_available(),  # Use bf16 on CUDA
        fp16=False,
        gradient_checkpointing=True,
        
        # DPO-specific parameters
        beta=0.1,  # Temperature for DPO
        
        # Other settings
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    
    print(f"Training Configuration:")
    print(f"  Epochs:                {training_args.num_train_epochs}")
    print(f"  Batch size:            {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size:  {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate:         {training_args.learning_rate}")
    print(f"  DPO Beta:              {training_args.beta}")
    print(f"  Max length:            {training_args.max_length}")
    print(f"  BF16:                  {training_args.bf16}")
    
    # ========================================
    # INITIALIZE TRAINER
    # ========================================
    
    print_section("Initializing DPO Trainer")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL will create frozen reference model automatically
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )
    
    print("✓ Trainer initialized successfully")
    
    # ========================================
    # TRAIN
    # ========================================
    
    print_header("STARTING DPO TRAINING")
    print("This will take several hours (6-12 hours typical)")
    print("You can safely close this window - training will continue")
    print("Check logs in:", f"{OUTPUT_DIR}/logs")
    print()
    
    try:
        # Start training
        train_result = trainer.train()
        
        print_header("TRAINING COMPLETE!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTraining metrics:")
        print(f"  Final loss: {train_result.training_loss:.4f}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_model(f"{OUTPUT_DIR}/interrupted_checkpoint")
        print("✓ Checkpoint saved")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"{e}")
        sys.exit(1)
    
    # ========================================
    # SAVE MODEL
    # ========================================
    
    print_section("Saving Final Model")
    
    final_output = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_output)
    
    print(f"✓ Model saved to: {final_output}")
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print_header("TRAINING SUMMARY")
    print(f"✓ Hybrid SFT+DPO model training complete!")
    print(f"✓ Model location: {final_output}")
    print(f"✓ Logs location: {OUTPUT_DIR}/logs")
    print()
    print("Next steps:")
    print("  1. Review training logs")
    print("  2. Test model with sample prompts")
    print("  3. Share checkpoint path with Erik for evaluation")
    print("  4. Document results in HYBRID_README.md")
    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()