import os
import re
import sys
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
import time
import uuid
import io
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure environment variables first
# os.environ["DIFFUSERS_NO_XFORMERS"] = "1"  # Remove this - let diffusers decide or enable explicitly for CUDA
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online model downloads
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "torch")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "hf_cache")
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Remove this - only needed for specific debugging
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120.0" # Set timeout to 120 seconds

# --- Detect CUDA and Set Device/dtype ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
    print(f"CUDA detected. Using device: {DEVICE} with dtype: {DTYPE}")
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    print(f"CUDA not available. Using device: {DEVICE} with dtype: {DTYPE}")
# --------------------------------------

# Create Flask app
app = Flask(__name__)

# Default configuration in case config import fails
DEFAULT_CONFIG = {
    "upload_folder": "uploads",
    "output_folder": "app/static/outputs",
    "allowed_extensions": {"png", "jpg", "jpeg"},
    "image_size": 512,
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "image_conditioning_weight": 0.3,
    "stable_diffusion_model": "runwayml/stable-diffusion-v1-5",
    "clip_vision_model": "openai/clip-vit-base-patch32",
    "lora_path": "models/lora/style_lora.safetensors"
}

# Try to import config, use default if fails
try:
    from config import CONFIG as USER_CONFIG # Import user config with a different name
    # Merge default and user configs, user config takes precedence
    merged_config = DEFAULT_CONFIG.copy()
    merged_config.update(USER_CONFIG)
    CONFIG = merged_config
    print("Loaded configuration from config.py and merged with defaults.")
except ImportError:
    print("Warning: Could not import config.py, using default configuration")
    CONFIG = DEFAULT_CONFIG
except Exception as e:
    print(f"Warning: Error loading or merging config.py: {e}. Using default configuration.")
    CONFIG = DEFAULT_CONFIG

# Configure upload folders
app.config['UPLOAD_FOLDER'] = CONFIG["upload_folder"]
app.config['OUTPUT_FOLDER'] = CONFIG["output_folder"]
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables for models
generation_pipeline = None
txt2img_pipeline = None

class StableDiffusionWrapper:
    def __init__(self, config):
        """Initialize Stable Diffusion model"""
        self.config = config
        # Use the globally determined device and dtype
        self.device = DEVICE 
        self.dtype = DTYPE 
        print(f"StableDiffusionWrapper using device: {self.device}, dtype: {self.dtype}")
        
        # Set memory optimization before loading models
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.9)  # Allow up to 90% for 6GB VRAM
            torch.backends.cudnn.benchmark = True # Usually faster for fixed input sizes
            torch.backends.cudnn.deterministic = False
        
        # Define model paths
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        sd_model_path = os.path.join(model_dir, "stable-diffusion")
        clip_model_path = os.path.join(model_dir, "clip-vision")
        os.makedirs(os.path.join(model_dir, "cache"), exist_ok=True)
        
        print("Loading required libraries...")
        try:
            # Import required libraries
            from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
            from diffusers.schedulers import DDIMScheduler
            from transformers import CLIPVisionModelWithProjection, CLIPProcessor
            from transformers import CLIPTextModel, CLIPTokenizer

            
            
            # Load tokenizer and text encoder
            print("Loading text encoder and tokenizer...")
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    # No dtype needed for tokenizer
                    cache_dir=os.path.join(model_dir, "cache")
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    torch_dtype=self.dtype, # Use determined dtype
                    cache_dir=os.path.join(model_dir, "cache")
                ).to(self.device) # Move to device
                print("Text encoder and tokenizer loaded")
            except Exception as e:
                print(f"Error loading text encoder: {e}")
                raise
            
            # Load CLIP vision model
            print("Loading CLIP vision model...")
            try:
                self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
                    config["clip_vision_model"],
                    torch_dtype=self.dtype, # Use determined dtype
                    cache_dir=os.path.join(model_dir, "cache")
                ).to(self.device) # Move to device
                self.clip_processor = CLIPProcessor.from_pretrained(
                    config["clip_vision_model"],
                    # No dtype needed for processor
                    cache_dir=os.path.join(model_dir, "cache")
                )
                print("CLIP vision model loaded")
            except Exception as e:
                print(f"Error loading CLIP vision model: {e}")
                raise
            
            # Load SD pipeline
            print("Loading Stable Diffusion pipeline...")
            try:
                # Try loading from local first (Assume local model is also full precision)
                if os.path.exists(sd_model_path) and os.path.isdir(sd_model_path):
                    self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                        sd_model_path,
                        torch_dtype=self.dtype, # Use determined dtype
                        use_safetensors=True,
                        safety_checker=None
                    ).to(self.device) # Move to device
                else:
                    # Download from Hugging Face
                    self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                        config["stable_diffusion_model"],
                        torch_dtype=self.dtype, # Use determined dtype
                        use_safetensors=True,
                        safety_checker=None,
                        cache_dir=os.path.join(model_dir, "cache"),
                        local_files_only=False
                    ).to(self.device) # Move to device
                
                # Use the DDIM scheduler for better stability
                self.sd_pipeline.scheduler = DDIMScheduler.from_config(
                    self.sd_pipeline.scheduler.config
                )
                
                print("Stable Diffusion pipeline loaded")
            except Exception as e:
                print(f"Error loading Stable Diffusion pipeline: {e}")
                raise
            
            # Optimize for memory
            self.optimize_memory()
            
        except ImportError as e:
            print(f"Import error: {e}")
            raise ImportError(f"Required libraries not installed: {e}")
    
    def optimize_memory(self):
        """Optimize memory usage for lower VRAM"""
        if not hasattr(self, 'sd_pipeline') or self.sd_pipeline is None:
            print("Warning: No SD pipeline to optimize")
            return
            
        if self.device.type == 'cuda':
            try:
                # Enable memory efficient attention (xformers if available)
                try:
                    self.sd_pipeline.enable_xformers_memory_efficient_attention()
                    print("Enabled xformers memory efficient attention.")
                except Exception as e:
                    print(f"Could not enable xformers: {e}. Enabling sliced attention as fallback.")
                    # Fallback to sliced attention if xformers is not available or fails
                    self.sd_pipeline.enable_attention_slicing(slice_size="auto") # or "max"
                
                # Enable VAE slicing for memory efficiency
                if hasattr(self.sd_pipeline, 'enable_vae_slicing'):
                    self.sd_pipeline.enable_vae_slicing()
                    print("Enabled VAE slicing.")

                # VAE Tiling (optional, might help further on very low VRAM)
                # if hasattr(self.sd_pipeline, 'enable_vae_tiling'):
                #     self.sd_pipeline.enable_vae_tiling()
                #     print("Enabled VAE tiling.")
                
                print("GPU memory optimization applied")
            except Exception as e:
                print(f"Error applying GPU memory optimizations: {e}")
        else:
            # Apply CPU optimizations if any (e.g., attention slicing can still help)
            try:
                self.sd_pipeline.enable_attention_slicing(slice_size=1)
                print("Enabled attention slicing for CPU.")
            except Exception as e:
                print(f"Error applying CPU memory optimizations: {e}")
    
    def load_lora_weights(self, lora_path):
        """Load LoRA weights for fine-tuned generation"""
        if not hasattr(self, 'sd_pipeline') or self.sd_pipeline is None:
            print("Warning: No SD pipeline to load LoRA weights into")
            return
            
        try:
            if os.path.exists(lora_path):
                print(f"Loading LoRA weights from {lora_path}")
                from diffusers.loaders import AttnProcsLayers
                
                # Load the LoRA weights
                self.sd_pipeline.unet.load_attn_procs(lora_path)
                print("LoRA weights loaded successfully")
            else:
                print(f"LoRA weights not found at {lora_path}")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
    
    def encode_prompt(self, prompt):
        """Encode text prompt to embeddings"""
        try:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            text_inputs = text_inputs.to(self.device)
            
            with torch.no_grad():
                prompt_embeds = self.text_encoder(
                    text_inputs.input_ids.to(self.device), # Ensure input IDs are on correct device
                    attention_mask=text_inputs.attention_mask.to(self.device) # Ensure attention mask is on correct device
                )[0]
            
            return prompt_embeds
            
        except Exception as e:
            print(f"Error encoding prompt: {e}")
            raise
    
    def encode_image(self, image):
        """Encode image to embeddings using CLIP vision model"""
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            
            # Process image with CLIP processor
            inputs = self.clip_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype) # Move to device and set dtype
            
            with torch.no_grad():
                image_features = self.clip_vision_model(pixel_values).image_embeds
            
            return image_features
            
        except Exception as e:
            print(f"Error encoding image: {e}")
            raise

    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        tensor = (tensor * 255).round().astype("uint8")
        return Image.fromarray(tensor)

# Add a new class for text-to-image generation after the StableDiffusionWrapper class
class TextToImagePipeline:
    def __init__(self, sd_model, config):
        """Initialize the text-to-image pipeline"""
        self.sd_model = sd_model
        self.config = config
        self.device = sd_model.device
        self.dtype = sd_model.dtype  # Add this line to inherit dtype from parent
        
        # Import text-to-image pipeline
        try:
            from diffusers import StableDiffusionPipeline
            
            # Load the pipeline
            print("Loading text-to-image pipeline...")
            try:
                # Try loading from the same model as img2img
                model_path = config["stable_diffusion_model"]
                print(f"Attempting to load model from: {model_path}")
                
                # Check if we can use the existing pipeline from sd_model
                if hasattr(self.sd_model, 'sd_pipeline') and self.sd_model.sd_pipeline is not None:
                    print("Reusing existing pipeline from sd_model")
                    # We need to create a new pipeline for text-to-image, not reuse the img2img one
                    self.txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self.dtype,
                        use_safetensors=False,
                        safety_checker=None,
                        cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                             "models", "cache"),
                        local_files_only=False
                    ).to(self.device)
                else:
                    # Try loading with use_safetensors=False to allow .bin files
                    self.txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self.dtype,
                        use_safetensors=False,  # Changed to False to allow .bin files
                        safety_checker=None,
                        cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                             "models", "cache"),
                        local_files_only=False
                    ).to(self.device)
                
                # Use the same scheduler as the img2img pipeline
                if hasattr(self.sd_model, 'sd_pipeline') and hasattr(self.sd_model.sd_pipeline, 'scheduler'):
                    from diffusers.schedulers import DDIMScheduler
                    self.txt2img_pipeline.scheduler = DDIMScheduler.from_config(
                        self.sd_model.sd_pipeline.scheduler.config
                    )
                
                # Apply memory optimizations based on device
                if self.device.type == 'cuda':
                    try:
                        self.txt2img_pipeline.enable_xformers_memory_efficient_attention()
                        print("Txt2Img: Enabled xformers memory efficient attention.")
                    except Exception as e:
                        print(f"Txt2Img: Could not enable xformers: {e}. Enabling sliced attention.")
                        self.txt2img_pipeline.enable_attention_slicing(slice_size="auto")
                    if hasattr(self.txt2img_pipeline, 'enable_vae_slicing'):
                        self.txt2img_pipeline.enable_vae_slicing()
                        print("Txt2Img: Enabled VAE slicing.")
                else:
                     self.txt2img_pipeline.enable_attention_slicing(slice_size=1)
                     print("Txt2Img: Enabled attention slicing for CPU.")

                # If on CPU, ensure float32 anyway (should be handled by dtype, but double check)
                # if self.device.type == "cpu":
                #    self.txt2img_pipeline.to(torch_dtype=torch.float32)
                
                print("Text-to-image pipeline loaded successfully")
            except Exception as e:
                print(f"Error loading text-to-image pipeline: {e}")
                raise
                
        except ImportError as e:
            print(f"Import error: {e}")
            raise ImportError(f"Required libraries not installed: {e}")
    
    def generate(self, prompt, negative_prompt=None, width=512, height=512, 
                 guidance_scale=7.5, num_inference_steps=50, seed=None):
        """Generate an image from text prompt only"""
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Default negative prompt if none provided
        if negative_prompt is None:
            negative_prompt = "blurry, bad quality, distorted, low resolution, ugly, unrealistic, deformed"
        
        # Enhanced prompt engineering
        enhanced_prompt = prompt
        if not enhanced_prompt.startswith("High quality"):
            enhanced_prompt = f"High quality, detailed {prompt}"
        
        # Reduce steps for CPU inference
        actual_steps = num_inference_steps # Start with requested steps
        if self.device.type == "cpu" and num_inference_steps > 30:
            print(f"Reducing steps from {num_inference_steps} to 30 for CPU inference")
            actual_steps = 30
        
        try:
            # Generate image
            with torch.no_grad():
                # Check if we're using a text-to-image pipeline or an img2img pipeline
                if hasattr(self.txt2img_pipeline, '__class__') and 'Img2Img' in self.txt2img_pipeline.__class__.__name__:
                    print("Warning: Using img2img pipeline for text-to-image generation")

                    
                    # Create a blank image for img2img
                    blank_image = Image.new('RGB', (width, height), color=(0, 0, 0))
                    output = self.txt2img_pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=blank_image,
                        num_inference_steps=actual_steps,
                        guidance_scale=guidance_scale,
                        strength=0.99  # High strength to generate from scratch
                    )
                else:
                    # Use the text-to-image pipeline
                    output = self.txt2img_pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=actual_steps,
                        guidance_scale=guidance_scale
                    )
            
            if hasattr(output, 'images') and len(output.images) > 0:
                output_image = output.images[0]
                # Ensure the output is a proper PIL Image
                if not isinstance(output_image, Image.Image):
                    if isinstance(output_image, torch.Tensor):
                        # Convert tensor to PIL Image
                        output_image = self._tensor_to_pil(output_image)
                    elif isinstance(output_image, bytes):
                        # Convert bytes to PIL Image
                        output_image = Image.open(io.BytesIO(output_image))
            else:
                raise ValueError("Pipeline didn't return images")
            
            return output_image
            
        except Exception as e:
            print(f"Error in text-to-image generation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a fallback image with an error message
            fallback_image = Image.new('RGB', (512, 512), color=(0, 0, 0))
            return fallback_image
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        tensor = (tensor * 255).round().astype("uint8")
        return Image.fromarray(tensor)
    
def generate_and_rank_images(self, prompt, num_images=3, **kwargs):
    """Generate multiple images and rank them using CLIP scores"""
    results = []
    
    # Generate multiple images
    for i in range(num_images):
        # Use different seeds for variety
        seed = int(time.time()) + i
        image = self.generate(prompt=prompt, seed=seed, **kwargs)
        results.append(image)
    
    # Score images using CLIP
    clip_scores = []
    for img in results:
        # Process image with CLIP
        inputs = self.sd_model.clip_processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.sd_model.device)
        
        # Process text with tokenizer
        text_inputs = self.sd_model.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=self.sd_model.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.sd_model.device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = self.sd_model.clip_vision_model(pixel_values).image_embeds
            text_features = self.sd_model.text_encoder(
                text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )[0]
            
            # Calculate similarity score
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).mean()
            clip_scores.append(similarity.item())
    
    # Return best image based on CLIP score
    best_index = clip_scores.index(max(clip_scores))
    return results[best_index], results, clip_scores

def optimize_prompt(self, base_prompt, variations, **kwargs):
    """Try different prompt variations and return the one with highest CLIP score"""
    scores = []
    images = []
    
    # Test each prompt variation
    for prompt_variation in variations:
        image = self.generate(prompt=prompt_variation, **kwargs)
        images.append(image)
        
        # Calculate CLIP score
        inputs = self.sd_model.clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.sd_model.device)
        
        text_inputs = self.sd_model.tokenizer(
            prompt_variation, 
            padding="max_length", 
            max_length=self.sd_model.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.sd_model.device)
        
        with torch.no_grad():
            image_features = self.sd_model.clip_vision_model(pixel_values).image_embeds
            text_features = self.sd_model.text_encoder(
                text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )[0]
            
            # Calculate similarity score
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).mean()
            scores.append(similarity.item())
    
    # Return best prompt, image, and score
    best_index = scores.index(max(scores))
    return variations[best_index], images[best_index], scores[best_index]

def guided_inpainting(self, image, mask, prompt, **kwargs):
    """Inpaint parts of an image using text guidance"""
    from diffusers import StableDiffusionInpaintPipeline
    
    # Create inpainting pipeline if needed
    if not hasattr(self, 'inpaint_pipe'):
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.config["stable_diffusion_model"],
            torch_dtype=self.sd_model.dtype,
            safety_checker=None
        ).to(self.sd_model.device)
        
        # Apply memory optimizations
        if self.sd_model.device.type == 'cuda':
            try:
                self.inpaint_pipe.enable_xformers_memory_efficient_attention()
            except:
                self.inpaint_pipe.enable_attention_slicing()
    
    # Ensure mask is proper format
    if isinstance(mask, Image.Image):
        mask_image = mask
    else:
        # Create mask from user input or segmentation
        mask_image = self.create_mask_from_segmentation(image)
    
    # Run inpainting with enhanced prompt
    enhanced_prompt = f"High quality, detailed {prompt}"
    output = self.inpaint_pipe(
        prompt=enhanced_prompt,
        image=process_image(image),
        mask_image=mask_image,
        num_inference_steps=kwargs.get('num_inference_steps', 30),
        guidance_scale=kwargs.get('guidance_scale', 7.5)
    )
    
    return output.images[0]

def interrogate_image(self, image):
    """Generate a description of an image using CLIP"""
    # List of concepts to test against the image
    concepts = [
        "photograph", "painting", "digital art", "illustration", "3D render",
        "landscape", "portrait", "abstract", "realistic", "surreal",
        "anime", "cartoon", "sketch", "oil painting", "watercolor",
        "indoor", "outdoor", "daytime", "nighttime", "colorful", "monochrome"
    ]
    
    # Additional objects and subjects to detect
    objects = ["person", "animal", "building", "vehicle", "nature", "technology", 
               "food", "cat", "dog", "bird", "city", "mountains", "ocean", "forest"]
    
    # Process image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    inputs = self.sd_model.clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(self.sd_model.device)
    
    # Get image embedding
    with torch.no_grad():
        image_features = self.sd_model.clip_vision_model(pixel_values).image_embeds
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Score concepts
    results = []
    for concept_list in [concepts, objects]:
        text_inputs = self.sd_model.tokenizer(
            concept_list,
            padding="max_length", 
            max_length=self.sd_model.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.sd_model.device)
        
        with torch.no_grad():
            text_features = self.sd_model.text_encoder(
                text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )[0]
            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            values, indices = similarity[0].topk(5)
            for value, idx in zip(values, indices):
                results.append((concept_list[idx], value.item()))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Generate caption from top concepts
    top_concepts = [item[0] for item in results[:8]]
    caption = f"This image appears to be a {', '.join(top_concepts[0:2])}"
    if len(top_concepts) > 2:
        caption += f" featuring {', '.join(top_concepts[2:5])}"
    if len(top_concepts) > 5:
        caption += f" with {', '.join(top_concepts[5:])}"
    
    return {
        'caption': caption,
        'concepts': results
    }
class DualConditioningPipeline:
    def __init__(self, sd_model, config):
        """Initialize the dual conditioning pipeline"""
        self.sd_model = sd_model
        self.config = config
        self.device = sd_model.device
        self.dtype = sd_model.dtype  # Add this line to inherit dtype from parent
        
        # Import segmentation model for subject/background separation
        try:
            from transformers import AutoFeatureExtractor, AutoModelForImageSegmentation
            import numpy as np
            from torchvision import transforms
            
            # Load segmentation model (lightweight model for performance)
            print("Loading segmentation model...")
            try:
                self.seg_extractor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
                self.seg_model = AutoModelForImageSegmentation.from_pretrained(
                    "facebook/detr-resnet-50-panoptic", 
                    torch_dtype=self.dtype # Use inherited dtype
                ).to(self.device) # Move to device
                
                self.has_segmentation = True
                print("Segmentation model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load segmentation model: {e}")
                print("Falling back to regular image-to-image without segmentation")
                self.has_segmentation = False
        except Exception as e:
            print(f"Warning: Could not load segmentation model: {e}")
            print("Falling back to regular image-to-image without segmentation")
            self.has_segmentation = False
    
    def create_mask_from_segmentation(self, image, target_classes=None):
        """Create binary mask separating foreground subject from background"""
        if not self.has_segmentation:
            # Return blank mask if segmentation not available
            return None
            
        if target_classes is None:
            # Default classes to consider foreground (including animals)
            target_classes = [
                "person", "animal", "horse", "dog", "cat", "bird", 
                "elephant", "zebra", "giraffe", "cow", "sheep"
            ]
            
        try:
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            # Extract features
            inputs = self.seg_extractor(images=image, return_tensors="pt").to(self.device)
            
            # Get segmentation predictions
            with torch.no_grad():
                outputs = self.seg_model(**inputs)
            
            # Process segmentation output
            from transformers.models.detr.modeling_detr import post_process_panoptic
            processed_sizes = torch.as_tensor([image.size[::-1]]).to(self.device)
            result = post_process_panoptic(outputs, processed_sizes, threshold=0.85)[0]
            
            # Create mask where 1=foreground (target), 0=background
            mask = torch.zeros(image.size[::-1], dtype=torch.uint8)
            
            # Process segmentation
            for segment_info in result["segments_info"]:
                label_id = segment_info["label_id"]
                label = self.seg_model.config.id2label[label_id]
                
                # Check if this segment is in target classes
                if any(tclass.lower() in label.lower() for tclass in target_classes):
                    # This is a foreground segment
                    segment_mask = result["segmentation"] == segment_info["id"]
                    mask = torch.logical_or(mask, segment_mask)
            
            # Convert to PIL Image with 0 and 255 values
            mask_pil = Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255)
            
            # Apply slight smoothing to mask edges
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))
            
            return mask_pil
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate(self, prompt, reference_image=None, guidance_scale=7.5, num_inference_steps=30, seed=None):

        """Generate an image using both text and image conditioning with background replacement"""
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Process reference image
        if reference_image is not None:
           if isinstance(reference_image, str):
               reference_image = Image.open(reference_image).convert("RGB")
        else:
            # If no reference image, this function shouldn't be called
            # The routing in /generate should handle this.
            print("Error: DualConditioningPipeline called without a reference image.")
            return Image.new('RGB', (512, 512), color=(255, 0, 0)) # Red error image

        
        try:
            # Check if prompt contains background change keywords
            background_change = any(keyword in prompt.lower() for keyword in 
                ["change background", "replace background", "new background", "different background"])
            
            # Enhanced prompt engineering
            if background_change:
                # Extract original subject from prompt (if possible)
                subject_match = re.search(r"image of ([^,]+)", prompt.lower())
                subject = subject_match.group(1).strip() if subject_match else "subject"
                
                # Extract desired background from prompt (if specified)
                background_match = re.search(r"background (?:with|of) ([^,\.]+)", prompt.lower())
                desired_background = background_match.group(1).strip() if background_match else ""
                
                # Create a more specific prompt that preserves the subject
                enhanced_prompt = f"High quality detailed photo of {subject} with {desired_background} in the background, realistic, professional photography"
                
                # Create a negative prompt
                negative_prompt = "blurry, bad quality, distorted, low resolution, ugly, unrealistic, deformed, cartoon"
                
                print(f"Enhanced prompt: {enhanced_prompt}")
            else:
                enhanced_prompt = prompt
                negative_prompt = "blurry, bad quality, distorted, low resolution, ugly, unrealistic"
            
            # Get or create segmentation mask for subject/background
            mask = None
            if background_change and self.has_segmentation:
                print("Creating segmentation mask...")
                mask = self.create_mask_from_segmentation(reference_image)
                if mask:
                    print("Segmentation mask created successfully")
                    
                    # Save mask for debugging if needed
                    # mask_path = f"mask_{int(time.time())}.png"
                    # mask.save(mask_path)
                    # print(f"Mask saved to {mask_path}")
            
            # Reduce steps for CPU inference
            actual_steps = num_inference_steps # Start with requested steps
            if self.device.type == "cpu" and actual_steps > 30:
                print(f"Reducing steps from {num_inference_steps} to 30 for CPU inference")
                actual_steps = 30
            
            # Set strength based on whether we're doing background replacement
            # Lower strength preserves more of the original image
            if background_change and mask:
                # Use inpainting for precise background replacement
                from diffusers import StableDiffusionInpaintPipeline
                
                try:
                    # Create inpainting pipeline on demand
                    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        self.config["stable_diffusion_model"],
                        torch_dtype=self.dtype, # Use inherited dtype
                        safety_checker=None
                    ).to(self.device) # Move to device
                    
                    # Set scheduler to same as main pipeline for consistency
                    inpaint_pipe.scheduler = self.sd_model.sd_pipeline.scheduler.from_config(
                        self.sd_model.sd_pipeline.scheduler.config
                    )
                    
                    # Enable memory optimizations based on device
                    if self.device.type == 'cuda':
                        try:
                            inpaint_pipe.enable_xformers_memory_efficient_attention()
                            print("Inpaint: Enabled xformers memory efficient attention.")
                        except Exception as e:
                            print(f"Inpaint: Could not enable xformers: {e}. Enabling sliced attention.")
                            inpaint_pipe.enable_attention_slicing(slice_size="auto")
                        if hasattr(inpaint_pipe, 'enable_vae_slicing'):
                            inpaint_pipe.enable_vae_slicing()
                            print("Inpaint: Enabled VAE slicing.")
                    else:
                        inpaint_pipe.enable_attention_slicing(slice_size=1)
                        print("Inpaint: Enabled attention slicing for CPU.")
                    
                    # Invert mask - in inpainting 1=area to inpaint, 0=area to keep
                    # Our mask has 1=foreground (keep), 0=background (replace)
                    inpaint_mask = ImageOps.invert(mask)
                    
                    print("Using inpainting pipeline for background replacement...")
                    output = inpaint_pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=reference_image,
                        mask_image=inpaint_mask,
                        num_inference_steps=actual_steps,
                        guidance_scale=guidance_scale
                    )
                    
                except Exception as e:
                    print(f"Inpainting error: {e}, falling back to img2img")
                    # Fall back to img2img with custom strength
                    strength = 0.65  # Balance between preserving subject and changing background
                    
                    with torch.no_grad():
                        output = self.sd_model.sd_pipeline(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            image=reference_image,
                            num_inference_steps=actual_steps,
                            guidance_scale=guidance_scale,
                            strength=strength
                        )
            else:
                # For regular image-to-image generation (no background change or mask failed/not available)
                # Adjust strength based on context
                strength = 0.75 if background_change else 0.65 # If background change was intended but mask failed, use higher strength
                
                print(f"Using img2img with strength {strength}...")
                
                # Handle errors that might occur during pipeline execution
                try:
                    with torch.no_grad():
                        output = self.sd_model.sd_pipeline(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            image=reference_image,
                            num_inference_steps=actual_steps,
                            guidance_scale=guidance_scale,
                            strength=strength
                        )
                except RuntimeError as e:
                    if "CUDA" in str(e) or "cuDNN" in str(e) or "engine" in str(e):
                        print(f"CUDA error detected: {e}")
                        print("Falling back to safer generation method")
                        
                        # Try with more conservative parameters
                        actual_steps = min(actual_steps, 20)
                        strength = 0.6  # Lower strength for better stability
                        
                        # Try with even simpler scheduler
                        from diffusers.schedulers import DDIMScheduler
                        original_scheduler = self.sd_model.sd_pipeline.scheduler
                        self.sd_model.sd_pipeline.scheduler = DDIMScheduler.from_config(
                            self.sd_model.sd_pipeline.scheduler.config
                        )
                        
                        with torch.no_grad():
                            output = self.sd_model.sd_pipeline(
                                prompt=enhanced_prompt,
                                negative_prompt=negative_prompt,
                                image=reference_image,
                                num_inference_steps=actual_steps,
                                guidance_scale=guidance_scale,
                                strength=strength
                            )
                        
                        # Restore original scheduler
                        self.sd_model.sd_pipeline.scheduler = original_scheduler
                    else:
                        raise
            
            
            if hasattr(output, 'images') and len(output.images) > 0:
                output_image = output.images[0]
                # Ensure the output is a proper PIL Image
                if not isinstance(output_image, Image.Image):
                    if isinstance(output_image, torch.Tensor):
                        # Convert tensor to PIL Image
                        output_image = self._tensor_to_pil(output_image)
                    elif isinstance(output_image, bytes):
                        # Convert bytes to PIL Image
                        output_image = Image.open(io.BytesIO(output_image))
            else:
                raise ValueError("Pipeline didn't return images")
            
            return output_image
        
            
        except Exception as e:
            print(f"Error in generation pipeline: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a fallback image with an error message
            fallback_image = Image.new('RGB', (512, 512), color=(0, 0, 0))
            return fallback_image
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        tensor = (tensor * 255).round().astype("uint8")
        return Image.fromarray(tensor)

def process_image(image_path, target_size=512):
    """Process image for model input"""
    try:
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        # Resize image maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create square image with padding
        squared_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        offset = ((target_size - new_width) // 2, (target_size - new_height) // 2)
        squared_image.paste(image, offset)
        
        return squared_image
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def save_image(image, output_folder, filename=None):
    """Save generated image to disk and return its URL path"""
    try:
        if filename is None:
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            filename = f"output_{timestamp}_{unique_id}.png"
        
        # Ensure we're using the actual folder path, not the relative one
        if output_folder.startswith('app/'):
            # Strip app/ prefix if it exists
            output_folder = output_folder[4:]
        
        # Make sure we're saving to static/outputs
        if not output_folder.startswith('static/'):
            output_folder = os.path.join('static', 'outputs')
        
        # Create the directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create full output path including filename
        output_path = os.path.join(output_folder, filename)
        
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL Image
                image = tensor_to_pil(image)
            elif isinstance(image, bytes):
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image))
        
        # Save with explicit format
        image.save(output_path, format="PNG")
        
        # Return the URL path relative to the static folder
        url_path = f"/{output_path}"  # Make sure the path starts with /
        if not url_path.startswith('/static/'):
            url_path = f"/static/outputs/{filename}"
            
        print(f"Image saved to {output_path}, URL path: {url_path}")
        return url_path
    except Exception as e:
        print(f"Error saving image: {e}")
        raise

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).round().astype("uint8")
    return Image.fromarray(tensor)

def print_gpu_memory_usage():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        try:
            # Get current device
            device = torch.cuda.current_device()
            
            # Get memory usage in bytes
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
            max_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
            
            print(f"GPU Memory Usage:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Total:     {max_memory:.2f} GB")
            print(f"  Available: {max_memory - reserved:.2f} GB")
        except Exception as e:
            print(f"Error getting GPU memory usage: {e}")
    else:
        print("CUDA not available")

def clear_gpu_memory():
    """Clear unused GPU memory"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("GPU memory cleared")
        except Exception as e:
            print(f"Error clearing GPU memory: {e}")

def get_pipeline():
    """Get or initialize pipeline"""
    global generation_pipeline
    global txt2img_pipeline
    
    if generation_pipeline is None or txt2img_pipeline is None:
        print("Initializing model pipelines...")
        
        try:
            # Create SD model wrapper (contains base models)
            sd_model = StableDiffusionWrapper(CONFIG)
            
            # Load LoRA weights if they exist
            if "lora_path" in CONFIG and os.path.exists(CONFIG["lora_path"]):
                sd_model.load_lora_weights(CONFIG["lora_path"])
            
            # Create image-to-image/conditioning pipeline
            try:
                generation_pipeline = DualConditioningPipeline(sd_model, CONFIG)
                print("DualConditioningPipeline initialized successfully")
            except Exception as e:
                print(f"Error initializing DualConditioningPipeline: {e}")
                print("Falling back to basic image-to-image pipeline")
                # Create a simplified pipeline that just uses the base model
                generation_pipeline = sd_model
            
            # Create text-to-image pipeline
            try:
                txt2img_pipeline = TextToImagePipeline(sd_model, CONFIG)
                print("TextToImagePipeline initialized successfully")
            except Exception as e:
                print(f"Error initializing TextToImagePipeline: {e}")
                print("Falling back to using the same pipeline for text-to-image")
                # Use the same pipeline for text-to-image
                txt2img_pipeline = sd_model
            
            print("Model pipelines initialized")
            if torch.cuda.is_available():
                print_gpu_memory_usage()
            else:
                print("Running on CPU")
        except Exception as e:
            print(f"Error initializing pipelines: {e}")
            # Don't raise the exception, just log it and continue
            # This allows the app to start even if model loading fails
            print("Application will start but may have limited functionality")
    
    # Return both pipelines (or rely on globals) - relying on globals for now
    # return generation_pipeline, txt2img_pipeline 

def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in CONFIG["allowed_extensions"]

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    """Serve output files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    """Download output files with proper mimetype and disposition"""
    # Get the full path to verify the file exists
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({
            'success': False,
            'message': 'File not found'
        }), 404
    
    # Set the appropriate mimetype for image files
    mimetype = None
    if filename.lower().endswith('.png'):
        mimetype = 'image/png'
    elif filename.lower().endswith(('.jpg', '.jpeg')):
        mimetype = 'image/jpeg'
    
    # Send the file with proper attachment disposition
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        filename, 
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename  # Ensures browser uses this filename
    )

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image endpoint with support for both text-to-image and image-to-image"""
    if 'prompt' not in request.form:
        return jsonify({'error': 'Missing prompt'}), 400
    
    raw_prompt = request.form['prompt']
    prompt = raw_prompt.strip()
    
    has_reference_image = 'file' in request.files and request.files['file'].filename != ''
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        guidance_scale = float(request.form.get('guidance_scale', CONFIG["guidance_scale"]))
        steps = int(request.form.get('steps', CONFIG["num_inference_steps"]))
        
        if torch.cuda.is_available() == False and steps > 30:
            print(f"Reducing steps from {steps} to 30 for CPU inference")
            steps = 30
            
        seed = request.form.get('seed')
        if seed:
            seed = int(seed)
        else:
            seed = int(time.time())
        
        # Initialize the pipelines if needed
        try:
            get_pipeline() 
        except Exception as e:
            print(f"Error initializing pipelines: {e}")
            return jsonify({
                'success': False,
                'message': 'Error initializing model pipelines. Please check the server logs.',
                'error': str(e)
            }), 500
        
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        
        if has_reference_image:
            # IMAGE-TO-IMAGE (Use DualConditioningPipeline)
            file = request.files['file']
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            filename = f"upload_{timestamp}_{unique_id}.png"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"File saved to {file_path}")
            
            reference_image = process_image(file_path, CONFIG["image_size"])
            print(f"Image processed to size {reference_image.size}")
            
            is_background_change = any(keyword in prompt.lower() for keyword in 
                                      ["change background", "replace background", "new background", "different background"])
            
            if is_background_change:
                enhanced_prompt = f"High quality, detailed {prompt}" if not prompt.startswith("High quality") else prompt
                guidance_scale = max(guidance_scale, 8.0)
            else:
                enhanced_prompt = f"High quality, detailed {prompt}" if not prompt.startswith("High quality") else prompt
                guidance_scale = max(guidance_scale, 7.5)
                
            print(f"Starting image-to-image generation with prompt: '{enhanced_prompt}'")
            print(f"Parameters: guidance_scale={guidance_scale}, steps={steps}, seed={seed}")
            
            try:
                # Check if generation_pipeline is available
                if generation_pipeline is None:
                    return jsonify({
                        'success': False,
                        'message': 'Image-to-image pipeline not available. Please check the server logs.',
                    }), 500
                
                # Use the image conditioning pipeline
                generated_image = generation_pipeline.generate(
                    prompt=enhanced_prompt,
                    reference_image=reference_image,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    seed=seed
                )
                
                output_filename = f"output_{timestamp}_{unique_id}.png"
                output_path = save_image(generated_image, 'static/outputs', output_filename)

                
                elapsed_time = time.time() - start_time
                
                print(f"[DEBUG] Returning output_image path (img2img): {output_path}")
                
                return jsonify({
                    'success': True,
                    'message': 'Image generated successfully',
                    'input_image': f"/uploads/{filename}",
                    'output_image': f"/outputs/{output_filename}",
                    'parameters': {
                        'prompt': enhanced_prompt,
                        'guidance_scale': guidance_scale,
                        'steps': steps,
                        'seed': seed
                    },
                    'statistics': {
                        'generation_time_seconds': round(elapsed_time, 2),
                        'device': str(DEVICE),
                        'model': CONFIG.get("stable_diffusion_model", "unknown")
                    }
                })
                
            except Exception as e:
                print(f"Error during image-to-image generation: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'message': f'Generation error: {str(e)}',
                    'input_image': f"/uploads/{filename}",
                    'error': str(e)
                }), 500
                
        else:
            # TEXT-TO-IMAGE (Use TextToImagePipeline)
            enhanced_prompt = f"High quality, detailed {prompt}" if not prompt.startswith("High quality") else prompt
            
            print(f"Starting text-to-image generation with prompt: '{enhanced_prompt}'")
            print(f"Parameters: guidance_scale={guidance_scale}, steps={steps}, seed={seed}")
            
            
            try:
                # Check if txt2img_pipeline is available
                if txt2img_pipeline is None:
                    return jsonify({
                        'success': False,
                        'message': 'Text-to-image pipeline not available. Please check the server logs.',
                    }), 500
                
                # Use the text-to-image pipeline
                generated_image = txt2img_pipeline.generate(
                    prompt=enhanced_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    seed=seed,
                    width=CONFIG.get("image_size", 512), # Pass width/height
                    height=CONFIG.get("image_size", 512)
                )
                
                output_filename = f"output_{timestamp}_{unique_id}.png"
                output_path = save_image(generated_image, 'static/outputs', output_filename)
                
                elapsed_time = time.time() - start_time
                
                print(f"[DEBUG] Returning output_image path (txt2img): {output_path}")
                
                return jsonify({
                   'success': True,
                   'message': 'Image generated from text successfully',
                   'output_image':'outputs',  # Use the path returned by save_image
                   'parameters': {
                   'prompt': enhanced_prompt,
                   'guidance_scale': guidance_scale,
                   'steps': steps,
                   'seed': seed
                 },
                   'statistics': {
                    'generation_time_seconds': round(elapsed_time, 2),
                    'device': str(DEVICE),
                    'model': CONFIG.get("stable_diffusion_model", "unknown")
                 }
             })
                
            except Exception as e:
                print(f"Error during text-to-image generation: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'message': f'Generation error: {str(e)}',
                    'error': str(e)
                }), 500
    
    except Exception as e:
        print(f"Error in API endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/generate_ranked', methods=['POST'])
def generate_ranked_images():
    """Generate multiple images and return the best one based on CLIP score"""
    if 'prompt' not in request.form:
        return jsonify({'error': 'Missing prompt'}), 400
    
    prompt = request.form['prompt'].strip()
    num_images = int(request.form.get('num_images', 3))
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        guidance_scale = float(request.form.get('guidance_scale', CONFIG["guidance_scale"]))
        steps = int(request.form.get('steps', CONFIG["num_inference_steps"]))
        
        # Initialize pipelines
        get_pipeline()
        
        # Generate and rank images
        best_image, all_images, scores = txt2img_pipeline.generate_and_rank_images(
            prompt=prompt,
            num_images=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )
        
        # Save best image
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{timestamp}_{unique_id}.png"
        output_path = save_image(best_image, 'static/outputs', output_filename)
        
        # Save all images for reference
        all_paths = []
        for i, img in enumerate(all_images):
            all_filename = f"output_{timestamp}_{unique_id}_variant_{i}.png"
            all_path = save_image(img, 'static/outputs', all_filename)
            all_paths.append(all_path)
        
        return jsonify({
            'success': True,
            'message': 'Ranked images generated successfully',
            'best_image': output_path,
            'all_images': all_paths,
            'clip_scores': scores
        })
        
    except Exception as e:
        print(f"Error in ranked generation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimize_prompt', methods=['POST'])
def optimize_prompt_endpoint():
    """Try different prompt variations and return the best one"""
    if 'prompts' not in request.form:
        return jsonify({'error': 'Missing prompt variations'}), 400
    
    # Get prompt variations as JSON list
    import json
    try:
        variations = json.loads(request.form['prompts'])
    except:
        return jsonify({'error': 'Invalid prompt variations format'}), 400
    
    if not variations or not isinstance(variations, list):
        return jsonify({'error': 'Prompt variations must be a non-empty list'}), 400
    
    try:
        guidance_scale = float(request.form.get('guidance_scale', CONFIG["guidance_scale"]))
        steps = int(request.form.get('steps', CONFIG["num_inference_steps"]))
        
        # Initialize pipelines
        get_pipeline()
        
        # Find best prompt
        best_prompt, best_image, score = txt2img_pipeline.optimize_prompt(
            base_prompt="",  # Not used but kept for compatibility
            variations=variations,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )
        
        # Save best image
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{timestamp}_{unique_id}.png"
        output_path = save_image(best_image, 'static/outputs', output_filename)
        
        return jsonify({
            'success': True,
            'message': 'Prompt optimization completed',
            'best_prompt': best_prompt,
            'best_image': output_path,
            'clip_score': score
        })
        
    except Exception as e:
        print(f"Error in prompt optimization: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/style_transfer', methods=['POST'])
def style_transfer_endpoint():
    """Apply style transfer to an image"""
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No image uploaded'}), 400
    
    if 'style_prompt' not in request.form:
        return jsonify({'error': 'Missing style prompt'}), 400
    
    file = request.files['file']
    style_prompt = request.form['style_prompt'].strip()
    content_weight = float(request.form.get('content_weight', 0.7))
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"upload_{timestamp}_{unique_id}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Initialize pipelines
        get_pipeline()
        
        # Apply style transfer
        styled_image = generation_pipeline.style_transfer(
            image=file_path,
            style_prompt=style_prompt,
            content_weight=content_weight,
            guidance_scale=float(request.form.get('guidance_scale', 7.5)),
            num_inference_steps=int(request.form.get('steps', 30))
        )
        
        # Save result
        output_filename = f"output_{timestamp}_{unique_id}.png"
        output_path = save_image(styled_image, 'static/outputs', output_filename)
        
        return jsonify({
            'success': True,
            'message': 'Style transfer completed',
            'input_image': f"/uploads/{filename}",
            'output_image': output_path
        })
        
    except Exception as e:
        print(f"Error in style transfer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/interrogate', methods=['POST'])
def interrogate_endpoint():
    """Analyze image and generate caption"""
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['file']
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"upload_{timestamp}_{unique_id}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Initialize pipelines
        get_pipeline()
        
        # Interrogate image
        result = generation_pipeline.interrogate_image(file_path)
        
        return jsonify({
            'success': True,
            'input_image': f"/uploads/{filename}",
            'caption': result['caption'],
            'concepts': result['concepts']
        })
        
    except Exception as e:
        print(f"Error in image interrogation: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # Set PyTorch to use deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
        # Get GPU memory info
        device_props = torch.cuda.get_device_properties(0)
        total_memory_gb = device_props.total_memory / (1024**3)
        print(f"VRAM: {total_memory_gb:.2f} GB")
    else:
        print("Running on CPU - inference will be slow but stable")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)