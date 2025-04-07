import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from model.clip_encoder import CLIPEncoder
from model.dual_encoder import DualConditioningEncoder
import gc
import os

class StableDiffusionWrapper:
    """
    Wrapper for the Stable Diffusion model with CLIP integration
    """
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.dtype = config["dtype"]
        
        # Load models with memory optimization
        self.load_models()
        
    def load_models(self):
        """Load all necessary models with memory optimizations"""
        print("Loading models...")
        
        # Load the VAE component
        self.vae = AutoencoderKL.from_pretrained(
            self.config["sd_model_path"],
            subfolder="vae",
            torch_dtype=self.dtype
        ).to(self.device)
        
        # Load the text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config["sd_model_path"],
            subfolder="text_encoder",
            torch_dtype=self.dtype
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config["sd_model_path"],
            subfolder="tokenizer"
        )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config["sd_model_path"],
            subfolder="unet",
            torch_dtype=self.dtype
        ).to(self.device)
        
        # Enable memory optimizations
        if self.config["enable_xformers_memory_efficient_attention"] and self.device == "cuda":
            try:
                import xformers
                self.unet.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except ImportError:
                print("xformers not available, using default attention")
        
        # Enable gradient checkpointing for VRAM efficiency during training
        if self.config["gradient_checkpointing"]:
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
            
        # Load CLIP encoder
        self.clip_encoder = CLIPEncoder(
            model_path=self.config["clip_model_path"],
            device=self.device
        )
        
        # Create dual encoder
        self.dual_encoder = DualConditioningEncoder(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            clip_encoder=self.clip_encoder,
            device=self.device
        )
        
        # Setup scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config["sd_model_path"],
            subfolder="scheduler"
        )
        
        # Set up pipeline components
        self.components = {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
            "unet": self.unet,
            "scheduler": self.scheduler
        }
        
        print("Models loaded successfully")
    
    def load_lora_weights(self, lora_path):
        """
        Load trained LoRA weights
        
        Args:
            lora_path: Path to saved LoRA weights
        """
        if os.path.exists(lora_path):
            print(f"Loading LoRA weights from {lora_path}")
            self.unet.load_attn_procs(lora_path)
            
            # Also load projection layer weights if they exist
            projection_path = os.path.join(lora_path, "img_projection.pt")
            if os.path.exists(projection_path):
                self.dual_encoder.img_projection.load_state_dict(torch.load(
                    projection_path, 
                    map_location=self.device
                ))
        else:
            print(f"LoRA weights not found at {lora_path}")
    
    def create_pipeline(self):
        """
        Create a StableDiffusionPipeline using our components
        """
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        return pipeline
    
    def free_memory(self):
        """
        Free CUDA memory to avoid OOM errors
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()