import os
import glob
import random
import time
import json
import math
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
sys.path.append('.')
from taming.modules.losses.lpips import LPIPS
from taming.models.vqgan import VQModel
from skimage.metrics import structural_similarity as ssim

CONFIG_PATH = "/checkpoints/vqgan_imagenet_f16_16384/model.yaml"
MODEL_PATH = "/checkpoints/vqgan_imagenet_f16_16384/last.ckpt"

# Robustness Configuration
# noise in model space 0.1 [-1,1] → 0.05 in [0,1]
NOISE_STD_LOW = 0.1 
NOISE_STD_MID = 0.2
NOISE_STD_HIGH = 0.5
MAX_SAMPLES = None # Set to None to run on all samples
BATCH_SIZE = 32

STAMP = int(time.time())
OUTDIR = f"{STAMP}_robustness_dataset_vqgan"
SIZE = 256 # size to resize smallest side to, then center crop
IMAGENET_VAL_ROOT = "/datasets/imagenet/val"

def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

def preprocess(path, size):
    img = Image.open(path).convert("RGB")
    s = min(img.size)
    r = size / s
    new_size = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, new_size, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=[size, size])
    x = T.ToTensor()(img)
    return x

def preprocess_vqgan(x):
    return 2.0 * x - 1.0

# convert [-1,1] tensors to [0,1] for metrics calculation, reverse of preprocess_vqgan(x)
def to_01(x):
    x = x.clamp(-1,1) # fix out-of-range values
    return (x + 1) / 2 # linear shift

def list_imagenet_val_dirs(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def gather_val_paths_and_labels(root):
    exts = ("*.JPEG","*.JPG","*.jpg")
    classes = list_imagenet_val_dirs(root)
    paths = []
    labels = []
    # Just used to reconstruct path structure, not class label logic
    for li, wnid in enumerate(classes):
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(root, wnid, e)))
        files = sorted(files)
        # We don't limit per class here, we limit globally via MAX_SAMPLES
        paths.extend(files)
        labels.extend([li]*len(files))
    return paths, labels

def batch_to_uint8_hwc(x):
    y = to_01(x).mul(255.0).round().clamp(0,255).to(torch.uint8)
    # [B, C, H, W] -> [B, H, W, C]
    #  .contiguous() to guarantee memory layout before the numpy export
    y = y.permute(0,2,3,1).contiguous()
    # detach().cpu().numpy() – the safe exit from PyTorch to NumPy -> .numpu() only works on a CPU tensor
    return y.detach().cpu().numpy()

class ImageNetValidationDataset(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return preprocess(path, self.size), path

class RobustnessDatasetGenerator:
    def __init__(self, model, device, output_dir):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.metadata_path = os.path.join(output_dir, "metadata.jsonl")
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Open metadata file in append mode
        self.metadata_file = open(self.metadata_path, "w")

    def close(self):
        self.metadata_file.close()

    def process_batch(self, batch, paths):
        # batch: [B, 3, H, W] in [0, 1]
        x_clean = preprocess_vqgan(batch).to(self.device)
        
        # Generate noisy inputs
        base_noise = torch.randn_like(x_clean)
        x_low  = x_clean + base_noise * NOISE_STD_LOW
        x_mid  = x_clean + base_noise * NOISE_STD_MID
        x_high = x_clean + base_noise * NOISE_STD_HIGH

        
        x_low  = torch.clamp(x_clean + base_noise * NOISE_STD_LOW,  -1.0, 1.0)
        x_mid  = torch.clamp(x_clean + base_noise * NOISE_STD_MID,  -1.0, 1.0)
        x_high = torch.clamp(x_clean + base_noise * NOISE_STD_HIGH, -1.0, 1.0)

        
        # Inference (Encode & Decode)
        with torch.no_grad():
            # Clean
            quant_clean, _, (_, _, ind_clean) = self.model.encode(x_clean)
            rec_clean = self.model.decode(quant_clean)
            
            # Low
            quant_low, _, (_, _, ind_low) = self.model.encode(x_low)
            rec_low = self.model.decode(quant_low)
            
            # Mid
            quant_mid, _, (_, _, ind_mid) = self.model.encode(x_mid)
            rec_mid = self.model.decode(quant_mid)
            
            # High
            quant_high, _, (_, _, ind_high) = self.model.encode(x_high)
            rec_high = self.model.decode(quant_high)

        # Convert to uint8 for saving
        img_clean_uint8 = batch_to_uint8_hwc(x_clean)
        rec_clean_uint8 = batch_to_uint8_hwc(rec_clean)
        
        img_low_uint8 = batch_to_uint8_hwc(x_low)
        rec_low_uint8 = batch_to_uint8_hwc(rec_low)
        
        img_mid_uint8 = batch_to_uint8_hwc(x_mid)
        rec_mid_uint8 = batch_to_uint8_hwc(rec_mid)
        
        img_high_uint8 = batch_to_uint8_hwc(x_high)
        rec_high_uint8 = batch_to_uint8_hwc(rec_high)

        # Iterate over batch to save files and write metadata
        for i, original_path in enumerate(paths):
            # Parse path to get class and filename
            # /datasets/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG
            parts = original_path.split("/")
            class_id = parts[-2]
            filename = parts[-1].split(".")[0]
            
            # Create directory structure
            # images/n01440764/ILSVRC2012_val_00000293/
            sample_dir = os.path.join(self.images_dir, class_id, filename)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save images
            Image.fromarray(img_clean_uint8[i]).save(os.path.join(sample_dir, "0_original.png"))
            Image.fromarray(rec_clean_uint8[i]).save(os.path.join(sample_dir, "1_recon_clean.png"))
            
            Image.fromarray(img_low_uint8[i]).save(os.path.join(sample_dir, "2_input_noise_low.png"))
            Image.fromarray(rec_low_uint8[i]).save(os.path.join(sample_dir, "3_recon_noise_low.png"))
            
            Image.fromarray(img_mid_uint8[i]).save(os.path.join(sample_dir, "4_input_noise_mid.png"))
            Image.fromarray(rec_mid_uint8[i]).save(os.path.join(sample_dir, "5_recon_noise_mid.png"))
            
            Image.fromarray(img_high_uint8[i]).save(os.path.join(sample_dir, "6_input_noise_high.png"))
            Image.fromarray(rec_high_uint8[i]).save(os.path.join(sample_dir, "7_recon_noise_high.png"))
            
            # Write metadata
            metadata = {
                "image_id": f"{class_id}/{filename}",
                "original_path": original_path,
                "indices_clean": ind_clean[i].cpu().numpy().tolist(),
                "indices_low": ind_low[i].cpu().numpy().tolist(),
                "indices_mid": ind_mid[i].cpu().numpy().tolist(),
                "indices_high": ind_high[i].cpu().numpy().tolist(),
                "noise_std": [NOISE_STD_LOW, NOISE_STD_MID, NOISE_STD_HIGH]
            }
            self.metadata_file.write(json.dumps(metadata) + "\n")

def main():
    print(f"Starting Robustness Dataset Generation to {OUTDIR}")
    print(f"Noise Levels: Low={NOISE_STD_LOW}, Mid={NOISE_STD_MID}, High={NOISE_STD_HIGH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    model = load_model(CONFIG_PATH, MODEL_PATH, device)
    
    paths, _ = gather_val_paths_and_labels(IMAGENET_VAL_ROOT)
    
    # Shuffle paths to get a random distribution if we stop early
    # But for reproducibility we might want to sort or seed. 
    # Let's rely on MAX_SAMPLES and just take the first N (sorted by gather_val function usually)
    # Actually, gather_val sorts them. So we get them in order.
    
    dataset = ImageNetValidationDataset(paths, SIZE)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    generator = RobustnessDatasetGenerator(model, device, OUTDIR)
    
    total_processed = 0
    
    try:
        for batch, batch_paths in tqdm(loader, desc="Processing Batches"):
            current_batch_size = len(batch_paths)
            
            # Check if we need to trim the batch to fit MAX_SAMPLES
            if MAX_SAMPLES is not None and (total_processed + current_batch_size) > MAX_SAMPLES:
                remaining = MAX_SAMPLES - total_processed
                if remaining <= 0:
                    break
                batch = batch[:remaining]
                batch_paths = batch_paths[:remaining]
                current_batch_size = remaining
            
            generator.process_batch(batch, batch_paths)
            total_processed += current_batch_size
            
            if MAX_SAMPLES is not None and total_processed >= MAX_SAMPLES:
                print(f"Reached MAX_SAMPLES limit ({MAX_SAMPLES}). Stopping.")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Closing gracefully...")
    finally:
        generator.close()
        print(f"Done. Processed {total_processed} samples.")
        print(f"Output saved to {OUTDIR}")

if __name__ == "__main__":
    main()
