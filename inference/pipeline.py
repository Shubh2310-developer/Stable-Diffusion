import torch
import PIL
from PIL import Image
import numpy as np
from diffusers import DDIMScheduler
from tqdm import tqdm
import gc

class DualConditioningPipeline:
    """
    Inference pipeline for image generation with dual text and image conditioning
    """
    def __init__(self, sd_model, config):
        self.sd_model = sd_model
        self.config = config
        self.device = config["device"]
        
        # Set models to eval mode
        self.sd_model.unet.eval()
        self.sd_model.vae.eval()
        self.sd_model.text_encoder.eval()
        self.sd_model.clip_encoder.model.eval()
        
        # Use faster scheduler for inference
        self.scheduler = DDIMScheduler.from_pretrained(
            config["sd_model_path"], 
            subfolder="scheduler"
        )
        self.scheduler.set_timesteps(config["num_inference_steps"])

    def prepare_inputs(self, prompt, reference_image, guidance_scale=7.5):
        """
        Prepare inputs for inference
        
        Args:
            prompt: Text prompt
            reference_image: Reference image or path
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Combined text and image embeddings
        """
        if isinstance(reference_image, str):
            reference_image = Image.open(reference_image).convert("RGB")
        
        # Create text embeddings
        text_embeddings = self.sd_model.dual_encoder.encode_text([prompt])
        
        # For classifier-free guidance, we also need unconditional embeddings
        uncond_embeddings = self.sd_model.dual_encoder.encode_text([""])
        
        # Create image embeddings
        image_embeddings = self.sd_model.dual_encoder.encode_image(reference_image)
        
        # Apply conditioning scale to image
        image_conditioning_scale = self.config["image_conditioning_scale"]
        
        # Combine text and image embeddings for conditional input
        combined_embeddings = self.sd_model.dual_encoder.combine_embeddings(
            text_embeddings, 
            image_embeddings,
            image_conditioning_scale
        )
        
        # Create combined embeddings for unconditional input (no image influence)
        uncond_combined = self.sd_model.dual_encoder.combine_embeddings(
            uncond_embeddings,
            torch.zeros_like(image_embeddings),
            0.0  # No image conditioning for unconditional
        )
        
        # Concatenate for classifier-free guidance [unconditional, conditional]
        embeddings = torch.cat([uncond_combined, combined_embeddings])
        
        return embeddings, guidance_scale
        
    def generate(self, prompt, reference_image, num_inference_steps=None, guidance_scale=None,
                 height=None, width=None, seed=None):
        """
        Generate an image from a text prompt and reference image
        
        Args:
            prompt: Text prompt
            reference_image: Reference image or path
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height: Height of output image
            width: Width of output image
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        # Use config values if not specified
        if num_inference_steps is None:
            num_inference_steps = self.config["num_inference_steps"]
        
        if guidance_scale is None:
            guidance_scale = self.config["guidance_scale"]
            
        if height is None or width is None:
            height = width = self.config["image_size"]
            
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Prepare inputs
        embeddings, effective_guidance_scale = self.prepare_inputs(
            prompt, reference_image, guidance_scale
        )
        
        # Set number of inference steps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Initialize latents
        latents_shape = (1, 4, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=torch.float16)
        
        # Denoise latents
        with torch.no_grad():
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                # Duplicate latents for guidance
                latent_model_input = torch.cat([latents] * 2)
                
                # Predict noise
                noise_pred = self.sd_model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=embeddings
                ).sample
                
                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + effective_guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                # Clear memory
                if self.device == "cuda" and i % 5 == 0:
                    torch.cuda.empty_cache()
        
        # Decode latents to image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents  # Scale latents
            image = self.sd_model.vae.decode(latents).sample
        
        # Convert to PIL image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        # Clear memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        return image