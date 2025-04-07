import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPEncoder(nn.Module):
    """
    CLIP-based image encoder to extract image features
    """
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.device = device
        
        # Load CLIP model and processor
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        
        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Set model to eval mode
        self.model.eval()
        
    def encode_image(self, image):
        """
        Encode an image using CLIP's vision encoder
        
        Args:
            image: PIL image or path to image
            
        Returns:
            Image features from CLIP
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        # Process image for CLIP
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        return image_features
        
    def encode_text(self, text):
        """
        Encode text using CLIP's text encoder
        
        Args:
            text: String or list of strings
            
        Returns:
            Text features from CLIP
        """
        if isinstance(text, str):
            text = [text]
            
        # Process text for CLIP
        inputs = self.processor(text=text, padding=True, return_tensors="pt").to(self.device)
        
        # Get text features
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        return text_features