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
IMAGENET_ROOTS = [
    "/datasets/imagenet/val/n01440764",  # tench (fish)
    "/datasets/imagenet/val/n02074367",  # dugong
    "/datasets/imagenet/val/n02124075",  # Egyptian cat
    "/datasets/imagenet/val/n02123394",  # Persian cat
    "/datasets/imagenet/val/n02123045",  # tabby cat
    "/datasets/imagenet/val/n02123159",  # tiger cat
    "/datasets/imagenet/val/n02123597",  # Siamese cat
    "/datasets/imagenet/val/n02129165",  # lion
    "/datasets/imagenet/val/n02129604",  # tiger
    "/datasets/imagenet/val/n02130308",  # cheetah
    "/datasets/imagenet/val/n02480495",  # orangutan
    "/datasets/imagenet/val/n02481823",  # chimpanzee
    "/datasets/imagenet/val/n02480855",  # gorilla
    "/datasets/imagenet/val/n02489166",  # proboscis monkey
    "/datasets/imagenet/val/n02486410",  # baboon
    "/datasets/imagenet/val/n02109961",  # husky
    "/datasets/imagenet/val/n02099601",  # golden retriever
    "/datasets/imagenet/val/n02106166",  # Border collie
    "/datasets/imagenet/val/n02108089",  # boxer
    "/datasets/imagenet/val/n02107142",  # Doberman pinscher
    "/datasets/imagenet/val/n02120079",  # Arctic fox
    "/datasets/imagenet/val/n02119022",  # red fox
    "/datasets/imagenet/val/n02326432",  # hare
    "/datasets/imagenet/val/n02690373",  # airliner
    "/datasets/imagenet/val/n03642806",  # laptop
    "/datasets/imagenet/val/n04254680",  # soccer ball
    "/datasets/imagenet/val/n04507155"   # umbrella
]
STAMP = int(time.time())
OUTDIR = f"{STAMP}_recon_imagenet_single"
SIZE = 256 # size to resize smallest side to, then center crop
LIMIT = 50 # limit for number of images to process, only if RUN_METRICS_ON_CLASS_SUBSET is True
CODEBOOK_PRINT_LIMIT = 16
CODEBOOK_NPY_SAVE_PATH = os.path.join(OUTDIR, "codebook.npy")
IMAGENET_VAL_ROOT = "/datasets/imagenet/val"
EXPORT_BATCH_SIZE = 16
EXPORT_NPZ_PATH = os.path.join(OUTDIR, f"{STAMP}_imagenet256_recon.npz")
EXPORT_IMAGENET_NPZ = True
RUN_METRICS_ON_VAL = False
RUN_METRICS_ON_CLASS_SUBSET = False

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

# MSE: mean of squared differences in [0,1]
# MSE(x, y) = mean((x - y)^2)
def compute_mse(orig01, rec01):
    return torch.mean((orig01 - rec01) ** 2).item()

# LPIPS: perceptual distance on [-1,1] tensors using pretrained VGG
# Lower is more similar
def compute_lpips(lpips_model, x_m11, recon_m11):
    return lpips_model(x_m11, recon_m11).mean().item()

# PSNR: peak signal-to-noise ratio in dB on [0,1]; PSNR = 10*log10(1/MSE)
# Returns inf if MSE == 0
def compute_psnr(mse):
    if mse <= 0.0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)

def compute_ssim(orig01, rec01):
    # [C, H, W] -> [H, W, C]
    x = orig01.permute(1, 2, 0).detach().cpu().numpy()
    y = rec01.permute(1, 2, 0).detach().cpu().numpy()
    return float(ssim(x, y, multichannel=True, data_range=x.max() - x.min()))

def gather_uniform_samples(roots, limit):
    exts = ("*.JPEG","*.JPG","*.jpg")
    n = len(roots)
    if n == 0 or limit <= 0:
        return []
    base = limit // n
    rem = limit % n
    selected = []
    used = set()
    all_files = []
    per_root_files = []
    for r in roots:
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(r, e)))
        random.shuffle(files)
        per_root_files.append(files)
        all_files.extend(files)
    for i, files in enumerate(per_root_files):
        need = base + (1 if i < rem else 0)
        take = min(need, len(files))
        chosen = files[:take]
        selected.extend(chosen)
        used.update(chosen)
    if len(selected) < limit:
        remaining = [f for f in all_files if f not in used]
        random.shuffle(remaining)
        fill = remaining[:(limit - len(selected))]
        selected.extend(fill)
    random.shuffle(selected)
    return selected[:limit]

def list_imagenet_val_dirs(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def gather_val_paths_and_labels(root):
    exts = ("*.JPEG","*.JPG","*.jpg")
    classes = list_imagenet_val_dirs(root)
    paths = []
    labels = []
    for li, wnid in enumerate(classes):
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(root, wnid, e)))
        files = sorted(files)
        if len(files) < 50:
            raise RuntimeError(f"class {wnid} has {len(files)} images, expected 50")
        files = files[:50]
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
        return preprocess(path, self.size)

def export_imagenet_val_npz(model, val_root, out_npz_path, batch_size):
    device = next(model.parameters()).device
    paths, labels = gather_val_paths_and_labels(val_root)
    
    dataset = ImageNetValidationDataset(paths, SIZE)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,  # Parallelize preprocessing
        pin_memory=True # ages the tensor to a fixed RAM buffer so the transfer to GPU is non-blocking
    )
    
    print(f"Exporting ImageNet256 recon to {out_npz_path}")
    
    all_recons = []
    
    with torch.no_grad():
        for x01 in tqdm(loader, desc="Encoding/Decoding", unit="batch"):
            x01 = x01.to(device) # a batched tensor produced by the workers
            x = preprocess_vqgan(x01)
            quant, _, _ = model.encode(x)
            recon = model.decode(quant)

            arr = batch_to_uint8_hwc(recon)
            all_recons.append(arr)
            
    # Concatenate all batches
    full_arr = np.concatenate(all_recons, axis=0)
    
    # Save compressed
    print(f"Saving compressed npz (shape: {full_arr.shape})...")
    np.savez(out_npz_path, arr_0=full_arr, arr_1=np.array(labels, dtype=np.int64))
    print(f"saved npz: {out_npz_path}")

    # quick sanity checks
    print("[val-export] sanity check ---")
    print(f"  reconstructions dtype: {full_arr.dtype}  range: {full_arr.min()} … {full_arr.max()}")
    print(f"  labels shape: {np.array(labels).shape}  dtype: {np.array(labels).dtype}")
    print(f"  first 5 labels: {labels[:5]}")
    print(f"  last 5 labels:  {labels[-5:]}")
    print(f"  total images: {len(full_arr)}  unique labels: {len(np.unique(labels))}")

def print_codebook(model, limit=None):
    q = model.quantize
    emb = getattr(q, "embedding", None)
    w = emb.weight.detach().cpu()
    print("codebook shape:", tuple(w.shape))
    if limit is None:
        print(w)
    else:
        print(w[:limit])

def save_codebook_npy(model, path):
    q = model.quantize
    emb = getattr(q, "embedding", None)
    w = emb.weight.detach().cpu().numpy()
    np.save(path, w)

def run_metrics_on_paths(model, paths, device, lpips_model):
    total_mse = 0.0
    total_lpips = 0.0
    total_psnr = 0.0
    psnr_count = 0
    total_ssim = 0.0
    n_images = 0
    per_image = []
    for p in tqdm(paths, desc="Metrics", unit="img"):
        x01 = preprocess(p, SIZE).unsqueeze(0).to(device)
        x = preprocess_vqgan(x01)
        with torch.no_grad():
            quant, _, info = model.encode(x)
            recon = model.decode(quant)
        orig = x01[0]
        rec = to_01(recon)[0]

        mse = compute_mse(orig, rec)
        lp = compute_lpips(lpips_model, x, recon)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(orig, rec)
        total_mse += mse
        total_lpips += lp
        if math.isfinite(psnr):
            total_psnr += psnr
            psnr_count += 1
        total_ssim += ssim
        n_images += 1
        per_image.append({"path": p, "mse": mse, "lpips": lp, "psnr": (psnr if math.isfinite(psnr) else None), "ssim": ssim})
    if n_images == 0:
        return None
    avg_mse = total_mse / n_images
    avg_lpips = total_lpips / n_images
    avg_psnr = (total_psnr / psnr_count) if psnr_count > 0 else float('inf')
    avg_ssim = total_ssim / n_images
    return {
        "timestamp": STAMP,
        "num_images": n_images,
        "avg_mse": avg_mse,
        "avg_lpips": avg_lpips,
        "avg_psnr": (avg_psnr if math.isfinite(avg_psnr) else None),
        "avg_ssim": avg_ssim,
        # "per_image": per_image,
    }

# This function is for testing purposes 
def run_metrics_class_subset(model, device, lpips_model):
    pictures = gather_uniform_samples(IMAGENET_ROOTS, LIMIT)
    return run_metrics_on_paths(model, pictures, device, lpips_model)

def run_metrics_val_all(model, val_root, device, lpips_model):
    paths, _ = gather_val_paths_and_labels(val_root)
    return run_metrics_on_paths(model, paths, device, lpips_model)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CONFIG_PATH, MODEL_PATH, device)
    print_codebook(model, CODEBOOK_PRINT_LIMIT)

    # save_codebook_npy(model, CODEBOOK_NPY_SAVE_PATH)

    if EXPORT_IMAGENET_NPZ:
        export_imagenet_val_npz(model, IMAGENET_VAL_ROOT, EXPORT_NPZ_PATH, EXPORT_BATCH_SIZE)

    lpips = LPIPS().to(device).eval()

    if RUN_METRICS_ON_VAL:
        summary_val = run_metrics_val_all(model, IMAGENET_VAL_ROOT, device, lpips)
        if summary_val is not None:
            avg_psnr_val = summary_val["avg_psnr"]
            psnr_str_val = f"{avg_psnr_val:.2f}" if (avg_psnr_val is not None and math.isfinite(avg_psnr_val)) else "inf"
            print(f"summary n={summary_val['num_images']} mse={summary_val['avg_mse']:.6f} lpips={summary_val['avg_lpips']:.6f} psnr={psnr_str_val} ssim={summary_val['avg_ssim']:.4f}")
            with open(os.path.join(OUTDIR, f"{STAMP}_summary_val.json"), "w") as f:
                json.dump(summary_val, f, indent=2)

    if RUN_METRICS_ON_CLASS_SUBSET:
        summary_subset = run_metrics_class_subset(model, device, lpips)
        if summary_subset is not None:
            summary_subset["limit"] = LIMIT
            avg_psnr_subset = summary_subset["avg_psnr"]
            psnr_str_subset = f"{avg_psnr_subset:.2f}" if (avg_psnr_subset is not None and math.isfinite(avg_psnr_subset)) else "inf"
            print(f"summary n={summary_subset['num_images']} mse={summary_subset['avg_mse']:.6f} lpips={summary_subset['avg_lpips']:.6f} psnr={psnr_str_subset} ssim={summary_subset['avg_ssim']:.4f}")
            with open(os.path.join(OUTDIR, f"{STAMP}_summary_subset.json"), "w") as f:
                json.dump(summary_subset, f, indent=2)

if __name__ == "__main__":
    main()
