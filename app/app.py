import os
import sys
import torch
from PIL import Image
import time
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from config import CONFIG
from model.sd_model import StableDiffusionWrapper
from inference.pipeline import DualConditioningPipeline
from utils.image_utils import process_image, save_image
from utils.memory_utils import print_gpu_memory_usage, clear_gpu_memory

# Create Flask app
app = Flask(__name__)

# Configure upload folders
app.config['UPLOAD_FOLDER'] = CONFIG["upload_folder"]
app.config['OUTPUT_FOLDER'] = CONFIG["output_folder"]
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variable for model
generation_pipeline = None

def get_pipeline():
    """Get or initialize pipeline"""
    global generation_pipeline
    
    if generation_pipeline is None:
        print("Initializing model...")
        
        # Create SD model
        sd_model = StableDiffusionWrapper(CONFIG)
        
        # Load LoRA weights if they exist
        if os.path.exists(CONFIG["lora_path"]):
            sd_model.load_lora_weights(CONFIG["lora_path"])
        
        # Create inference pipeline
        generation_pipeline = DualConditioningPipeline(sd_model, CONFIG)
        
        print("Model initialized")
        print_gpu_memory_usage()
    
    return generation_pipeline

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
    """Serve generated files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image endpoint"""
    # Check if file and prompt are included
    if 'file' not in request.files or 'prompt' not in request.form:
        return jsonify({'error': 'Missing file or prompt'}), 400
    
    file = request.files['file']
    prompt = request.form['prompt']
    
    # Handle empty submissions
    if file.filename == '' or not prompt:
        return jsonify({'error': 'No file or prompt provided'}), 400
    
    # Check file type
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Create unique filename
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"upload_{timestamp}_{unique_id}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save uploaded file
        file.save(file_path)
        
        # Process image
        reference_image = process_image(file_path, CONFIG["image_size"])
        
        # Initialize pipeline
        pipeline = get_pipeline()
        
        # Extract additional parameters
        guidance_scale = float(request.form.get('guidance_scale', CONFIG["guidance_scale"]))
        steps = int(request.form.get('steps', CONFIG["num_inference_steps"]))
        seed = request.form.get('seed')
        if seed:
            seed = int(seed)
        
        # Generate image
        generated_image = pipeline.generate(
            prompt=prompt,
            reference_image=reference_image,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed
        )
        
        # Save generated image
        output_filename = f"output_{timestamp}_{unique_id}.png"
        output_path = save_image(generated_image, app.config['OUTPUT_FOLDER'], output_filename)
        
        # Clear CUDA memory
        clear_gpu_memory()
        
        # Return response
        return jsonify({
            'success': True,
            'message': 'Image generated successfully',
            'input_image': f"/uploads/{filename}",
            'output_image': f"/outputs/{output_filename}",
            'parameters': {
                'prompt': prompt,
                'guidance_scale': guidance_scale,
                'steps': steps,
                'seed': seed if seed else 'random'
            }
        })
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'error': str(e)}), 500

# Now let's implement the missing modules

# model/sd_model.py
class StableDiffusionWrapper:
    def __init__(self, config):
        """Initialize Stable Diffusion model"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Import diffusers inside the class to avoid loading everything at startup
        from diffusers import StableDiffusionImg2ImgPipeline
        from transformers import CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
        
        # Load SD pipeline
        self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            config["stable_diffusion_model"],
            torch_dtype=torch.float16,
            revision="fp16",
            use_safetensors=True
        ).to(self.device)
        
        # Load CLIP vision model
        self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
            config["clip_vision_model"],
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Store CLIP text encoder and tokenizer for later use
        self.text_encoder = self.sd_pipeline.text_encoder
        self.tokenizer = self.sd_pipeline.tokenizer
        
        # Optimize for RTX 4050 with 4GB VRAM
        self.optimize_memory()
    
    def optimize_memory(self):
        """Optimize memory usage for lower VRAM"""
        # Enable memory efficient attention
        self.sd_pipeline.enable_attention_slicing(slice_size="auto")
        
        # Use deterministic algorithms for better memory efficiency
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Enable xformers if available
        try:
            import xformers
            self.sd_pipeline.enable_xformers_memory_efficient_attention()
            print("XFormers memory efficient attention enabled")
        except ImportError:
            print("XFormers not available, using default attention")
    
    def load_lora_weights(self, lora_path):
        """Load LoRA weights for fine-tuned generation"""
        try:
            print(f"Loading LoRA weights from {lora_path}")
            self.sd_pipeline.unet.load_attn_procs(lora_path)
            print("LoRA weights loaded successfully")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
    
    def encode_prompt(self, prompt):
        """Encode text prompt to embeddings"""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        
        return text_embeddings
    
    def encode_image(self, image):
        """Encode image to embeddings using CLIP vision model"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Preprocess image for CLIP
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        image_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_vision_model(image_tensor).image_embeds
        
        return image_features

# inference/pipeline.py
class DualConditioningPipeline:
    def __init__(self, sd_model, config):
        """Initialize the dual conditioning pipeline"""
        self.sd_model = sd_model
        self.config = config
        self.device = sd_model.device
    
    def generate(self, prompt, reference_image, guidance_scale=7.5, num_inference_steps=50, seed=None):
        """Generate an image using both text and image conditioning"""
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Process reference image for img2img pipeline
        if isinstance(reference_image, str):
            reference_image = Image.open(reference_image).convert("RGB")
        
        # Get text embeddings
        text_embeddings = self.sd_model.encode_prompt(prompt)
        
        # Get image embeddings
        image_embeddings = self.sd_model.encode_image(reference_image)
        
        # Combine text and image embeddings
        # There are multiple approaches to combine them:
        # 1. Weighted sum
        # 2. Concatenation and projection
        # 3. Cross-attention between the two
        
        # For simplicity, we'll use a weighted sum
        image_weight = self.config.get("image_conditioning_weight", 0.3)
        text_weight = 1.0 - image_weight
        
        # Normalize embeddings
        text_norm = torch.norm(text_embeddings, dim=-1, keepdim=True)
        image_norm = torch.norm(image_embeddings, dim=-1, keepdim=True)
        
        # Adjust dimensions if needed
        if text_embeddings.shape[-1] != image_embeddings.shape[-1]:
            # Simple projection if dimensions don't match
            projection = torch.nn.Linear(
                image_embeddings.shape[-1], 
                text_embeddings.shape[-1],
                device=self.device
            )
            image_embeddings = projection(image_embeddings)
        
        # Combine embeddings
        combined_embeddings = (
            text_weight * (text_embeddings / text_norm) + 
            image_weight * (image_embeddings / image_norm)
        )
        
        # Generate image using combined conditioning
        with torch.no_grad():
            output_image = self.sd_model.sd_pipeline(
                image=reference_image,
                prompt_embeds=combined_embeddings,
                negative_prompt="",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=0.75  # How much to transform the reference image
            ).images[0]
        
        return output_image

# utils/image_utils.py
def process_image(image_path, target_size=512):
    """Process image for model input"""
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

def save_image(image, output_folder, filename=None):
    """Save generated image to disk"""
    if filename is None:
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"output_{timestamp}_{unique_id}.png"
    
    output_path = os.path.join(output_folder, filename)
    image.save(output_path)
    
    return output_path

# utils/memory_utils.py
def print_gpu_memory_usage():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        # Get current device
        device = torch.cuda.current_device()
        
        # Get memory usage in bytes
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
        
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Available: {4.0 - reserved:.2f} GB (assuming 4GB VRAM on RTX 4050)")
    else:
        print("CUDA not available")

def clear_gpu_memory():
    """Clear unused GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# config.py
CONFIG = {
    # Model paths
    "stable_diffusion_model": "runwayml/stable-diffusion-v1-5",  # Or "stabilityai/stable-diffusion-2-1"
    "clip_vision_model": "openai/clip-vit-large-patch14",
    "lora_path": "models/lora/dual_conditioning_lora.safetensors",  # Path to LoRA weights
    
    # Folders
    "upload_folder": "static/uploads",
    "output_folder": "static/outputs",
    
    # Image parameters
    "image_size": 512,
    "allowed_extensions": {"png", "jpg", "jpeg"},
    
    # Inference parameters
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "image_conditioning_weight": 0.3,  # Weight for image conditioning vs text
    
    # Resource limits
    "max_batch_size": 1,  # For low VRAM, only process one image at a time
}

# Add main execution code
if __name__ == "__main__":
    # Configure for RTX 4050 with 4GB VRAM
    torch.cuda.set_per_process_memory_fraction(0.9)  # Limit to 90% of VRAM
    
    # Print CUDA information
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: ~4GB (RTX 4050)")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)