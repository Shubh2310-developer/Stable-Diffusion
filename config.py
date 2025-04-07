import torch

# General configuration
CONFIG = {
    # Model paths
    "model_path": "./models",
    "sd_model_path": "./models/stable-diffusion",
    "clip_model_path": "./models/clip",
    "lora_path": "./models/lora_weights",
    
    # Device configuration
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    
    # Training configuration
    "train_data_dir": "./data/train",
    "val_data_dir": "./data/val",
    "output_dir": "./output",
    "logging_dir": "./logs",
    
    # LoRA configuration
    "lora_r": 16,            # LoRA rank
    "lora_alpha": 32,        # LoRA alpha scaling
    "lora_dropout": 0.05,    # LoRA dropout
    
    # Training parameters
    "batch_size": 1,         # Small batch size for 6GB VRAM
    "gradient_accumulation_steps": 4,  # Increase effective batch size
    "learning_rate": 1e-4,
    "epochs": 10,
    "save_steps": 500,
    "eval_steps": 100,
    
    # Optimization for low VRAM
    "enable_xformers_memory_efficient_attention": True,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",  # Or "bf16" if supported
    
    # Generation parameters
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "image_size": 512,  # Can be reduced to 384 or 256 for lower VRAM usage
    
    # Web app configuration
    "upload_folder": "./app/static/uploads",
    "output_folder": "./app/static/outputs",
    "allowed_extensions": {"png", "jpg", "jpeg"},
    "port": 5000,
    
    # CLIP image conditioning weight
    "image_conditioning_scale": 0.5,  # Weight of image conditioning vs text
}