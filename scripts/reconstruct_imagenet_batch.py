import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.utils import save_image
import sys

sys.path.append('.')

from taming.models.vqgan import VQModel

CONFIG_PATH = "/checkpoints/vqgan_imagenet_f16_16384/model.yaml"
MODEL_PATH = "/checkpoints/vqgan_imagenet_f16_16384/last.ckpt"
IMAGENET_ROOT = "/datasets/imagenet/val/n02480495"
OUTDIR = "recon_imagenet_val"
SIZE = 256
LIMIT = 96
BATCH_SIZE = 12

def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

def resize_smallest_max(img, size):
    s = min(img.size)
    scale = size / s
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC)

def center_crop(img, size):
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))

def preprocess(path, size):
    img = Image.open(path).convert("RGB")
    img = resize_smallest_max(img, size)
    img = center_crop(img, size)
    arr = np.array(img).astype(np.uint8)
    arr = (arr / 127.5 - 1.0).astype(np.float32)
    x = torch.from_numpy(arr).permute(2,0,1)
    return x

def to_01(x):
    x = x.clamp(-1,1)
    return (x + 1) / 2

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CONFIG_PATH, MODEL_PATH, device)

    exts = ("*.JPEG","*.JPG","*.jpg","*.png")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(IMAGENET_ROOT, "**", e), recursive=True))
    random.shuffle(files)
    if LIMIT > 0:
        files = files[:LIMIT]

    for i in range(0, len(files), BATCH_SIZE):
        batch_paths = files[i:i+BATCH_SIZE]
        xs = [preprocess(p, SIZE) for p in batch_paths]
        x = torch.stack(xs, 0).to(device)
        with torch.no_grad():
            quant, _, _ = model.encode(x)
            recon = model.decode(quant)
        orig = to_01(x)
        rec = to_01(recon)
        save_image(orig, os.path.join(OUTDIR, f"orig_{i:06}.png"), nrow=len(batch_paths))
        save_image(rec, os.path.join(OUTDIR, f"recon_{i:06}.png"), nrow=len(batch_paths))

if __name__ == "__main__":
    main()