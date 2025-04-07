import os
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

def download_models(model_path="./models"):
    """
    Download the necessary models for the system.
    
    Args:
        model_path: Directory to save models
    """
    os.makedirs(model_path, exist_ok=True)
    
    print("Downloading Stable Diffusion model...")
    
    # Using SD 1.5 which has lower VRAM requirements than SD 2.1
    # Can load directly from the CompVis model
    sd_model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,  # Use half precision to save VRAM
        safety_checker=None,  # Disable safety checker to save VRAM
        requires_safety_checker=False
    )
    
    # Save model to disk
    sd_model.save_pretrained(os.path.join(model_path, "stable-diffusion"))
    
    print("Downloading CLIP model...")
    
    # Download CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Save CLIP model
    clip_model.save_pretrained(os.path.join(model_path, "clip"))
    clip_processor.save_pretrained(os.path.join(model_path, "clip"))
    
    print(f"Models downloaded and saved to {model_path}")

if __name__ == "__main__":
    download_models()