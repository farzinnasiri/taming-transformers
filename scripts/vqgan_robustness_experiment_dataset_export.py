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
import multiprocessing as mp
import sys
sys.path.append('.')
from taming.models.vqgan import VQModel

CONFIG_PATH = "/checkpoints/vqgan_imagenet_f16_16384/model.yaml"
MODEL_PATH = "/checkpoints/vqgan_imagenet_f16_16384/last.ckpt"

# Robustness Configuration
# noise in model space 0.1 [-1,1] → 0.05 in [0,1]
NOISE_STD_LOW = 0.1 
NOISE_STD_MID = 0.2
NOISE_STD_HIGH = 0.5
MAX_SAMPLES = None # Set to None to run on all samples
BATCH_SIZE = 32
NUM_SAVE_WORKERS = 8  # Workers for saving images and metadata

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

def save_worker(worker_id, queue, output_dir):
    """
    Worker process to save images and write metadata.
    Reads tasks from queue and writes to its own metadata file.
    """
    images_dir = os.path.join(output_dir, "images")
    metadata_path = os.path.join(output_dir, f"metadata_part_{worker_id}.jsonl")
    
    # Open unique metadata file for this worker
    with open(metadata_path, "w") as f:
        while True:
            task = queue.get()
            if task is None:
                break
            
            try:
                original_path = task['original_path']
                
                # Parse path to get class and filename
                # /datasets/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG
                parts = original_path.split("/")
                class_id = parts[-2]
                filename = parts[-1].split(".")[0]
                
                # Create directory structure
                # images/n01440764/ILSVRC2012_val_00000293/
                sample_dir = os.path.join(images_dir, class_id, filename)
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save images
                Image.fromarray(task['img_clean']).save(os.path.join(sample_dir, "0_original.png"))
                Image.fromarray(task['rec_clean']).save(os.path.join(sample_dir, "1_recon_clean.png"))
                
                Image.fromarray(task['img_low']).save(os.path.join(sample_dir, "2_input_noise_low.png"))
                Image.fromarray(task['rec_low']).save(os.path.join(sample_dir, "3_recon_noise_low.png"))
                
                Image.fromarray(task['img_mid']).save(os.path.join(sample_dir, "4_input_noise_mid.png"))
                Image.fromarray(task['rec_mid']).save(os.path.join(sample_dir, "5_recon_noise_mid.png"))
                
                Image.fromarray(task['img_high']).save(os.path.join(sample_dir, "6_input_noise_high.png"))
                Image.fromarray(task['rec_high']).save(os.path.join(sample_dir, "7_recon_noise_high.png"))
                
                # Write metadata
                metadata = {
                    "image_id": f"{class_id}/{filename}",
                    "original_path": original_path,
                    "indices_clean": task['indices_clean'],
                    "indices_low": task['indices_low'],
                    "indices_mid": task['indices_mid'],
                    "indices_high": task['indices_high'],
                    "noise_std": task['noise_std']
                }
                f.write(json.dumps(metadata) + "\n")
                
            except Exception as e:
                print(f"Worker {worker_id} error processing {original_path}: {e}")

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
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize Workers
        self.queue = mp.Queue(maxsize=NUM_SAVE_WORKERS * 10) # Buffer size
        self.workers = []
        print(f"Starting {NUM_SAVE_WORKERS} save workers...")
        for i in range(NUM_SAVE_WORKERS):
            p = mp.Process(target=save_worker, args=(i, self.queue, output_dir))
            p.start()
            self.workers.append(p)

    def close(self):
        print("Waiting for workers to finish...")
        for _ in self.workers:
            self.queue.put(None)
        for p in self.workers:
            p.join()

    def process_batch(self, batch, paths):
        # batch: [B, 3, H, W] in [0, 1]
        x_clean = preprocess_vqgan(batch).to(self.device)
        # x_clean is the clean image in model space, in range [-1,1]

        # Generate noisy inputs
         # raw noise from standard normal distribution (mean 0, std 1), same shape as x_clean
        base_noise = torch.randn_like(x_clean)

        # three leves of noisy images from the same initial noise  
        # we use a base_noise to ensure the noise direction is consistent across levels (same distribution, scaled at each level)     
        x_low  = torch.clamp(x_clean + base_noise * NOISE_STD_LOW,  -1.0, 1.0)
        x_mid  = torch.clamp(x_clean + base_noise * NOISE_STD_MID,  -1.0, 1.0)
        x_high = torch.clamp(x_clean + base_noise * NOISE_STD_HIGH, -1.0, 1.0)

        
        # Inference (Encode & Decode)
        with torch.no_grad():
            # Clean
            quant_clean, _, (_, _, ind_clean) = self.model.encode(x_clean)
            ind_clean = ind_clean.reshape(x_clean.shape[0], -1)
            rec_clean = self.model.decode(quant_clean)
            
            # Low
            quant_low, _, (_, _, ind_low) = self.model.encode(x_low)
            ind_low = ind_low.reshape(x_low.shape[0], -1)
            rec_low = self.model.decode(quant_low)
            
            # Mid
            quant_mid, _, (_, _, ind_mid) = self.model.encode(x_mid)
            ind_mid = ind_mid.reshape(x_mid.shape[0], -1)
            rec_mid = self.model.decode(quant_mid)
            
            # High
            quant_high, _, (_, _, ind_high) = self.model.encode(x_high)
            ind_high = ind_high.reshape(x_high.shape[0], -1)
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

        # Iterate over batch to enqueue tasks
        for i, original_path in enumerate(paths):
            task = {
                'original_path': original_path,
                'img_clean': img_clean_uint8[i],
                'rec_clean': rec_clean_uint8[i],
                'img_low': img_low_uint8[i],
                'rec_low': rec_low_uint8[i],
                'img_mid': img_mid_uint8[i],
                'rec_mid': rec_mid_uint8[i],
                'img_high': img_high_uint8[i],
                'rec_high': rec_high_uint8[i],
                'indices_clean': ind_clean[i].cpu().numpy().tolist(),
                'indices_low': ind_low[i].cpu().numpy().tolist(),
                'indices_mid': ind_mid[i].cpu().numpy().tolist(),
                'indices_high': ind_high[i].cpu().numpy().tolist(),
                'noise_std': [NOISE_STD_LOW, NOISE_STD_MID, NOISE_STD_HIGH]
            }
            self.queue.put(task)

def main():
    print(f"Starting Robustness Dataset Generation to {OUTDIR}")
    print(f"Noise Levels: Low={NOISE_STD_LOW}, Mid={NOISE_STD_MID}, High={NOISE_STD_HIGH}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    model = load_model(CONFIG_PATH, MODEL_PATH, device)
    
    paths, _ = gather_val_paths_and_labels(IMAGENET_VAL_ROOT)
    
    
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
