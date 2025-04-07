import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from diffusers.optimization import get_scheduler
from model.lora_adapter import inject_lora_to_unet, inject_lora_to_clip_projection, save_lora_weights
import gc

class DualConditioningTrainer:
    """
    Trainer class for fine-tuning with both text and image conditioning
    """
    def __init__(self, sd_model, config):
        self.sd_model = sd_model
        self.config = config
        self.device = config["device"]
        
        # Apply LoRA to UNet
        self.sd_model.unet = inject_lora_to_unet(
            self.sd_model.unet, 
            config
        )
        
        # Apply LoRA to CLIP projection layer
        inject_lora_to_clip_projection(
            self.sd_model.dual_encoder,
            config
        )
        
        # Put models in train mode
        self.sd_model.unet.train()
        self.sd_model.dual_encoder.img_projection.train()
        
        # Keep other components in eval mode
        self.sd_model.vae.eval()
        self.sd_model.text_encoder.eval()
        self.sd_model.clip_encoder.model.eval()
    
    def prepare_latents(self, batch, weight_dtype):
        """
        Encode input images into latent space using VAE
        
        Args:
            batch: Dictionary containing "target" images
            weight_dtype: Data type for weights
            
        Returns:
            Latent representations of input images
        """
        with torch.no_grad():
            latents = self.sd_model.vae.encode(
                batch["target"].to(weight_dtype).to(self.device)
            ).latent_dist.sample()
            
            # Scale latents
            latents = latents * 0.18215
            
            # Add noise
            noise = torch.randn_like(latents)
            
            # Create random timesteps
            timesteps = torch.randint(
                0,
                self.sd_model.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=self.device
            ).long()
            
            # Add noise to latents using scheduler
            noisy_latents = self.sd_model.scheduler.add_noise(latents, noise, timesteps)
            
            return noisy_latents, noise, timesteps
    
    def train_step(self, batch, optimizer):
        """
        Single training step
        
        Args:
            batch: Training batch with target images, prompts, reference images
            optimizer: Optimizer for updating weights
            
        Returns:
            Loss value
        """
        # Move to device and convert to half precision if using GPU
        weight_dtype = torch.float32
        if self.device == "cuda":
            weight_dtype = torch.float16
            
        # Prepare latents (VAE encoding + noise)
        noisy_latents, noise, timesteps = self.prepare_latents(batch, weight_dtype)
        
        # Get text + image conditioned embeddings
        with torch.no_grad():
            # Get combined embeddings from text and reference image
            combined_embeddings = self.sd_model.dual_encoder(
                batch["prompt"],
                batch["reference_image"],
                self.config["image_conditioning_scale"]
            ).to(weight_dtype)
        
        # Predict noise with UNet
        noise_pred = self.sd_model.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=combined_embeddings
        ).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        # Backward pass
        loss.backward()
        
        return loss.detach().item()
    
    def train(self, train_dataloader, num_epochs=None, learning_rate=None):
        """
        Train the model
        
        Args:
            train_dataloader: DataLoader with training data
            num_epochs: Number of epochs to train (default: from config)
            learning_rate: Learning rate (default: from config)
        """
        # Use config values if not specified
        if num_epochs is None:
            num_epochs = self.config["epochs"]
            
        if learning_rate is None:
            learning_rate = self.config["learning_rate"]
        
        # Create optimizer
        trainable_params = []
        
        # Add UNet LoRA parameters
        for _, module in self.sd_model.unet.named_attn_processors.items():
            trainable_params.extend(module.parameters())
        
        # Add CLIP projection parameters
        trainable_params.extend(
            [p for p in self.sd_model.dual_encoder.img_projection.parameters() if p.requires_grad]
        )
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate
        )
        
        # Create scheduler
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_dataloader) * num_epochs
        )
        
        # Training loop
        global_step = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch+1}")
            
            epoch_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                # Reset gradients
                optimizer.zero_grad()
                
                # Perform training step
                loss = self.train_step(batch, optimizer)
                epoch_loss += loss
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss})
                
                global_step += 1
                
                # Save checkpoint
                if global_step % self.config["save_steps"] == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")
                    
                # Clear GPU memory
                if self.device == "cuda" and step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Average loss for epoch {epoch+1}: {avg_loss:.4f}")
            
            # Save after each epoch
            self.save_checkpoint(f"epoch-{epoch+1}")
            
            # Clear GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
    
    def save_checkpoint(self, name):
        """
        Save model checkpoint
        
        Args:
            name: Name for the checkpoint
        """
        output_dir = os.path.join(self.config["output_dir"], name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA weights
        save_lora_weights(
            self.sd_model.unet,
            self.sd_model.dual_encoder,
            output_dir
        )
        
        print(f"Saved checkpoint: {output_dir}")
        
    def load_checkpoint(self, checkpoint_dir):
        """
        Load model checkpoint
        
        Args:
            checkpoint_dir: Directory with checkpoint files
        """
        self.sd_model.load_lora_weights(checkpoint_dir)
        print(f"Loaded checkpoint from {checkpoint_dir}")