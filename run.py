import torch
import numpy as np
import yaml
from omegaconf import OmegaConf
from torchvision.utils import save_image
import sys

# This is the new, important line.
# It tells Python to look for the 'taming' folder in the current directory (which will be /app).
sys.path.append('.')

# Helper functions from the repo to load the model
from taming.models.vqgan import VQModel

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path):
    # The config file already contains the model parameters
    model = VQModel(**config.model.params)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"], strict=False)
    return model

def preprocess_image(image_tensor):
    image_tensor = image_tensor.clamp(-1, 1)
    image_tensor = (image_tensor + 1) / 2
    return image_tensor

# --- 1. CONFIGURE YOUR MODEL ---
# These paths are now the paths *inside* the Docker container.

# We will mount your checkpoints to /checkpoints
CONFIG_PATH = "/checkpoints/vqgan_imagenet_f16_16384/model.yaml"

# We will mount your checkpoints to /checkpoints
MODEL_PATH = "/checkpoints/vqgan_imagenet_f16_16384/last.ckpt" 

# Set device (use 'cuda' if you have a GPU, 'cpu' if not)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. LOAD THE VQGAN (DECODER + CODEBOOK) ---
print(f"Loading config from: {CONFIG_PATH}")
config = load_config(CONFIG_PATH)

print(f"Loading model from: {MODEL_PATH}")
model = load_vqgan(config, MODEL_PATH).to(device)
model.eval() # Put model in evaluation mode
print("Model loaded successfully.")

# --- 3. CREATE YOUR MANUAL BLUEPRINT ---
print("Creating manual blueprint...")
# Let's fill the canvas with index 1000.
indices = torch.ones(1, 16, 16, dtype=torch.long).to(device) * 1000

# --- EXPERIMENT HERE! ---
# 1. "Paint" a square in the middle with index 500
# indices[:, 4:12, 4:12] = 500

# 2. Use random indices
# indices = torch.randint(0, 16384, (1, 16, 16), dtype=torch.long).to(device)

# --- 4. RUN THE DECODER ---
with torch.no_grad(): # We don't need to track gradients
    
    z_q = model.quantize.embedding(indices) 
    z_q = z_q.permute(0, 3, 1, 2)
    
    # --- 5. DECODE! ---
    print("Decoding blueprint into image...")
    decoded_image = model.decode(z_q)
    
    # --- 6. SAVE YOUR CREATION ---
    # We save to the current directory (/app), which will be mapped
    # back to your taming-transformers folder on the host.
    processed_image = preprocess_image(decoded_image)
    save_image(processed_image, "my_manual_creation.png")
    
    print("Done! Check 'my_manual_creation.png' in your ~/vq-gan/taming-transformers folder.")