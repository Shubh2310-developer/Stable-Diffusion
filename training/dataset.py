import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from torchvision import transforms

class DualConditioningDataset(Dataset):
    """
    Dataset for training with both text and image conditioning
    """
    def __init__(self, data_dir, transform=None, image_size=512):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Default transformation if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform
            
        # Load dataset
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        """
        Load dataset from directory structure:
        - data_dir/
          - metadata.json (contains text prompts)
          - images/ (contains target images)
          - reference_images/ (contains reference images)
        """
        samples = []
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            for item in metadata:
                target_image_path = os.path.join(self.data_dir, "images", item["image_file"])
                reference_image_path = os.path.join(self.data_dir, "reference_images", item["reference_file"])
                
                if os.path.exists(target_image_path) and os.path.exists(reference_image_path):
                    samples.append({
                        "prompt": item["prompt"],
                        "target_image": target_image_path,
                        "reference_image": reference_image_path
                    })
        else:
            # If no metadata.json, try to infer from directory structure
            images_dir = os.path.join(self.data_dir, "images")
            ref_images_dir = os.path.join(self.data_dir, "reference_images")
            
            if os.path.exists(images_dir) and os.path.exists(ref_images_dir):
                for img_file in os.listdir(images_dir):
                    if img_file.endswith((".png", ".jpg", ".jpeg")):
                        # Check if there's a matching reference image (by name)
                        ref_file = img_file
                        ref_path = os.path.join(ref_images_dir, ref_file)
                        
                        if os.path.exists(ref_path):
                            # Create a simple prompt from filename
                            prompt = img_file.split(".")[0].replace("_", " ")
                            
                            samples.append({
                                "prompt": prompt,
                                "target_image": os.path.join(images_dir, img_file),
                                "reference_image": ref_path
                            })
        
        print(f"Loaded {len(samples)} samples from {self.data_dir}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load target image (the one we want to generate)
        target_image = Image.open(sample["target_image"]).convert("RGB")
        target_tensor = self.transform(target_image)
        
        # Load reference image (for conditioning)
        reference_image = Image.open(sample["reference_image"]).convert("RGB")
        
        return {
            "prompt": sample["prompt"],
            "target": target_tensor,
            "reference_image": reference_image
        }

def create_dataset_directories(root_dir):
    """
    Create directories for a dual conditioning dataset
    
    Args:
        root_dir: Root directory for the dataset
    """
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.join(root_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "reference_images"), exist_ok=True)
    
    # Create empty metadata file if it doesn't exist
    metadata_file = os.path.join(root_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        with open(metadata_file, "w") as f:
            json.dump([], f)
    
    print(f"Created dataset directories in {root_dir}")

def add_sample_to_dataset(
    root_dir, 
    target_image_path, 
    reference_image_path, 
    prompt, 
    target_filename=None, 
    reference_filename=None
):
    """
    Add a new sample to the dataset
    
    Args:
        root_dir: Root directory of the dataset
        target_image_path: Path to the target image
        reference_image_path: Path to the reference image
        prompt: Text prompt
        target_filename: Custom filename for target image (optional)
        reference_filename: Custom filename for reference image (optional)
    """
    # Create directories if they don't exist
    create_dataset_directories(root_dir)
    
    # Generate filenames if not provided
    if target_filename is None:
        target_filename = f"{len(os.listdir(os.path.join(root_dir, 'images')))}.png"
    
    if reference_filename is None:
        reference_filename = target_filename
    
    # Copy images to dataset
    target_dest = os.path.join(root_dir, "images", target_filename)
    reference_dest = os.path.join(root_dir, "reference_images", reference_filename)
    
    # Copy target image
    target_img = Image.open(target_image_path).convert("RGB")
    target_img.save(target_dest)
    
    # Copy reference image
    ref_img = Image.open(reference_image_path).convert("RGB")
    ref_img.save(reference_dest)
    
    # Update metadata
    metadata_file = os.path.join(root_dir, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    metadata.append({
        "prompt": prompt,
        "image_file": target_filename,
        "reference_file": reference_filename
    })
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Added new sample to dataset: {prompt}")