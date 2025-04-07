import torch
import gc
import os
import numpy as np

def get_gpu_memory_info():
    """
    Get GPU memory usage information
    
    Returns:
        Dictionary with memory usage information
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    # Get memory information
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
    max_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_gb": allocated,
        "memory_reserved_gb": reserved,
        "max_memory_gb": max_mem,
        "memory_available_gb": max_mem - allocated
    }

def print_gpu_memory_usage():
    """Print GPU memory usage information"""
    info = get_gpu_memory_info()
    
    if not info["cuda_available"]:
        print("CUDA is not available")
        return
    
    print(f"GPU: {info['device_name']}")
    print(f"Memory allocated: {info['memory_allocated_gb']:.2f} GB")
    print(f"Memory reserved: {info['memory_reserved_gb']:.2f} GB")
    print(f"Max memory: {info['max_memory_gb']:.2f} GB")
    print(f"Memory available: {info['memory_available_gb']:.2f} GB")

def clear_gpu_memory():
    """
    Clear GPU memory to avoid OOM errors
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def optimize_model_memory(model, use_8bit=False, use_4bit=False):
    """
    Optimize model memory usage
    
    Args:
        model: PyTorch model
        use_8bit: Whether to use 8-bit quantization
        use_4bit: Whether to use 4-bit quantization
    
    Returns:
        Optimized model
    """
    from accelerate import init_empty_weights
    from torch.nn import init
    
    # Function to initialize weights to zeros
    def init_to_zeros(m):
        if hasattr(m, 'weight') and m.weight is not None:
            init.zeros_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)
    
    if use_8bit:
        try:
            import bitsandbytes as bnb
            # Convert linear layers to 8-bit
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    new_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None
                    )
                    # Copy weights (approximately)
                    with torch.no_grad():
                        new_module.weight.data = module.weight.data.to(torch.float16)
                        if module.bias is not None:
                            new_module.bias.data = module.bias.data
                    
                    # Replace the old module
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        child_name = name.rsplit(".", 1)[1]
                        setattr(parent, child_name, new_module)
                    else:
                        setattr(model, name, new_module)
        except ImportError:
            print("bitsandbytes not available, skipping 8-bit quantization")
    
    # Enable gradient checkpointing
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    
    # Set float16 for CUDA
    if torch.cuda.is_available():
        model = model.half()
    
    return model

def set_vram_optimization_config(config):
    """
    Set VRAM optimization settings based on available GPU memory
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated config
    """
    if not torch.cuda.is_available():
        return config
    
    # Get GPU memory
    info = get_gpu_memory_info()
    max_mem = info["max_memory_gb"]
    
    # 6GB VRAM (RTX 4050) optimizations
    if max_mem <= 6:
        print("Optimizing for 6GB VRAM (RTX 4050)")
        config["image_size"] = 384  # Reduce from 512
        config["batch_size"] = 1
        config["gradient_accumulation_steps"] = 4
        config["enable_xformers_memory_efficient_attention"] = True
        config["gradient_checkpointing"] = True
        config["mixed_precision"] = "fp16"
    
    # For limited but more than 6GB
    elif max_mem <= 8:
        print("Optimizing for 8GB VRAM")
        config["batch_size"] = 1
        config["gradient_accumulation_steps"] = 2
        config["enable_xformers_memory_efficient_attention"] = True
        config["gradient_checkpointing"] = True
    
    return config