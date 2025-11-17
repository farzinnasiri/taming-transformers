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
OUTDIR = "recon_imagenet_single"
SIZE = 256
LIMIT = 16
CODEBOOK_PRINT_LIMIT = 16
CODEBOOK_SAVE_PATH = os.path.join(OUTDIR, "codebook.pt")
CODEBOOK_NPY_SAVE_PATH = os.path.join(OUTDIR, "codebook.npy")

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

def print_codebook(model, limit=None):
    q = model.quantize
    emb = getattr(q, "embedding", None)
    if emb is None:
        emb = getattr(q, "embed", None)
    w = emb.weight.detach().cpu()
    print("codebook shape:", tuple(w.shape))
    if limit is None:
        print(w)
    else:
        print(w[:limit])

def save_codebook_npy(model, path):
    q = model.quantize
    emb = getattr(q, "embedding", None)
    if emb is None:
        emb = getattr(q, "embed", None)
    w = emb.weight.detach().cpu().numpy()
    np.save(path, w)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CONFIG_PATH, MODEL_PATH, device)
    print_codebook(model, CODEBOOK_PRINT_LIMIT)
    save_codebook_npy(model, CODEBOOK_NPY_SAVE_PATH)

    exts = ("*.JPEG","*.JPG","*.jpg")
    pictures = []
    for e in exts:
        pictures.extend(glob.glob(os.path.join(IMAGENET_ROOT, e)))
    random.shuffle(pictures)

    pictures = pictures[:LIMIT]

    for picture in pictures:
        x = preprocess(picture, SIZE).unsqueeze(0).to(device)
        with torch.no_grad():
            quant, _, _ = model.encode(x)
            recon = model.decode(quant)
        orig = to_01(x)[0]
        rec = to_01(recon)[0]
        base = os.path.splitext(os.path.basename(picture))[0]
        save_image(orig, os.path.join(OUTDIR, f"orig_{base}.png"))
        save_image(rec, os.path.join(OUTDIR, f"recon_{base}.png"))

if __name__ == "__main__":
    main()