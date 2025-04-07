import torch
import torch.nn as nn
from diffusers.loaders import LoraLoaderMixin, AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from peft import LoraConfig, get_peft_model

def inject_lora_to_unet(unet, config):
    """
    Inject LoRA adapters into the UNet model's attention layers
    
    Args:
        unet: The UNet model
        config: Config dict with LoRA parameters
    
    Returns:
        UNet with LoRA adapters
    """
    # Get LoRA configuration
    lora_r = config["lora_r"]
    lora_alpha = config["lora_alpha"] 
    lora_dropout = config["lora_dropout"]
    
    # Extract attention processors
    attn_procs = {}
    
    # Add LoRA to cross-attention and self-attention layers
    attention_module_types = [
        "CrossAttention",
        "Attention",  # Self-attention
    ]
    
    for name, module in unet.named_modules():
        if any(attn_type in module.__class__.__name__ for attn_type in attention_module_types):
            # Get existing query, key, value projection layers
            if hasattr(module, "to_q"):
                q_proj, k_proj, v_proj = module.to_q, module.to_k, module.to_v
                
                # Replace with LoRA attention processors
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=q_proj.in_features,
                    cross_attention_dim=None if k_proj.in_features == q_proj.in_features else k_proj.in_features,
                    rank=lora_r,
                    network_alpha=lora_alpha,
                    dropout=lora_dropout,
                )
                
    # Set attention processors
    unet.set_attn_processor(attn_procs)
    
    # Mark only LoRA parameters as trainable
    for param in unet.parameters():
        param.requires_grad = False
    
    for processor in attn_procs.values():
        for param in processor.parameters():
            param.requires_grad = True
            
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    
    return unet

def inject_lora_to_clip_projection(dual_encoder, config):
    """
    Apply LoRA to the CLIP projection layer
    
    Args:
        dual_encoder: The dual encoder model
        config: Config dict with LoRA parameters
    """
    # Get image projection layer
    img_projection = dual_encoder.img_projection
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["img_projection"],
        lora_dropout=config["lora_dropout"],
        bias="none",
    )
    
    # Apply LoRA to projection
    peft_img_projection = get_peft_model(img_projection, lora_config)
    dual_encoder.img_projection = peft_img_projection
    
    # Ensure only LoRA parameters are trainable
    for name, param in dual_encoder.img_projection.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

def save_lora_weights(unet, dual_encoder, output_dir):
    """
    Save LoRA weights for both UNet and CLIP projection
    
    Args:
        unet: UNet model with LoRA adapters
        dual_encoder: Dual encoder with LoRA on projection layer
        output_dir: Directory to save weights
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save UNet LoRA weights
    unet.save_attn_procs(output_dir)
    
    # Save projection layer weights
    torch.save(
        dual_encoder.img_projection.state_dict(),
        os.path.join(output_dir, "img_projection.pt")
    )