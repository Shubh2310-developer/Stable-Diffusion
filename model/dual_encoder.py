import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class DualConditioningEncoder(nn.Module):
    """
    Combined encoder for both text and image conditioning
    """
    def __init__(self, text_encoder, tokenizer, clip_encoder, img_embedding_dim=768, device="cuda"):
        super().__init__()
        self.device = device
        
        # Stable Diffusion's text encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
        # CLIP encoder for images
        self.clip_encoder = clip_encoder
        
        # Projection layer to match CLIP image embeddings to text embedding space
        self.img_projection = nn.Linear(
            in_features=clip_encoder.model.projection_dim,
            out_features=img_embedding_dim
        ).to(device)
        
    def encode_text(self, prompt, max_length=77):
        """Encode text prompt using SD's text encoder"""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
            
        return text_embeddings
    
    def encode_image(self, image):
        """Encode image using CLIP and project to text embedding space"""
        # Get image features from CLIP
        with torch.no_grad():
            image_features = self.clip_encoder.encode_image(image)
        
        # Project to text embedding dimension
        projected_image_features = self.img_projection(image_features)
        
        return projected_image_features
    
    def combine_embeddings(self, text_embeddings, image_embeddings, image_conditioning_scale=0.5):
        """
        Combine text and image embeddings
        
        Args:
            text_embeddings: Text embeddings from SD text encoder
            image_embeddings: Projected image embeddings from CLIP
            image_conditioning_scale: Weight of image embeddings (0-1)
            
        Returns:
            Combined embeddings
        """
        # Expand image embeddings to match text embeddings shape if needed
        if image_embeddings.shape[1] != text_embeddings.shape[1]:
            image_embeddings = image_embeddings.unsqueeze(1).expand(-1, text_embeddings.shape[1], -1)
        
        # Combine embeddings with weighting
        combined = (1 - image_conditioning_scale) * text_embeddings + image_conditioning_scale * image_embeddings
        
        return combined
    
    def get_embeddings(self, prompt, image, image_conditioning_scale=0.5):
        """
        Get combined embeddings from both text and image
        
        Args:
            prompt: Text prompt
            image: Image for conditioning
            image_conditioning_scale: Weight of image conditioning
        
        Returns:
            Combined embeddings
        """
        text_embeddings = self.encode_text(prompt)
        image_embeddings = self.encode_image(image)
        combined_embeddings = self.combine_embeddings(
            text_embeddings, 
            image_embeddings, 
            image_conditioning_scale
        )
        
        return combined_embeddings
    
    def forward(self, prompt, image, image_conditioning_scale=0.5):
        """Forward pass combining text and image conditioning"""
        return self.get_embeddings(prompt, image, image_conditioning_scale)