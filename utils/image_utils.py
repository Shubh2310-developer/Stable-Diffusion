import os
import cv2
import numpy as np
from PIL import Image
import torch
import uuid
import time

def resize_image(image, target_size):
    """
    Resize image while preserving aspect ratio
    
    Args:
        image: PIL Image or path to image
        target_size: Target size (single dimension)
        
    Returns:
        Resized PIL Image
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Get original size
    width, height = image.size
    
    # Calculate new size
    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    
    # Resize image
    return image.resize((new_width, new_height), Image.LANCZOS)

def center_crop(image, crop_size):
    """
    Center crop image
    
    Args:
        image: PIL Image
        crop_size: Crop size (single dimension)
        
    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    
    # Calculate crop box
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    # Crop image
    return image.crop((left, top, right, bottom))

def process_image(image_path, target_size=512):
    """
    Process image for model input (resize and center crop)
    
    Args:
        image_path: Path to image
        target_size: Target size
        
    Returns:
        Processed PIL Image
    """
    # Load and resize image
    image = resize_image(image_path, target_size)
    
    # Center crop if needed
    if image.width != image.height:
        crop_size = min(image.width, image.height)
        image = center_crop(image, crop_size)
        
        # Resize to target size
        image = image.resize((target_size, target_size), Image.LANCZOS)
    
    return image

def save_image(image, output_dir, filename=None):
    """
    Save image to disk
    
    Args:
        image: PIL Image
        output_dir: Output directory
        filename: Optional filename
        
    Returns:
        Path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"generated_{timestamp}_{unique_id}.png"
    
    # Save image
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    
    return output_path

def image_grid(images, rows=None, cols=None):
    """
    Create a grid of images
    
    Args:
        images: List of PIL Images
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        PIL Image with grid
    """
    if rows is None and cols is None:
        # Calculate rows and columns
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    elif rows is None:
        rows = int(np.ceil(len(images) / cols))
    elif cols is None:
        cols = int(np.ceil(len(images) / rows))
    
    # Get image size
    width, height = images[0].size
    
    # Create grid
    grid = Image.new('RGB', size=(cols * width, rows * height))
    
    # Paste images
    for i, img in enumerate(images):
        grid.paste(img, box=((i % cols) * width, (i // cols) * height))
    
    return grid