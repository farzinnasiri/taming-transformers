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
from torchvision.utils import save_image
from tqdm import tqdm
import sys
sys.path.append('.')
from taming.modules.losses.lpips import LPIPS
from taming.models.vqgan import VQModel

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
LIMIT = 500 # limit for number of images to process, only if RUN_METRICS_ON_CLASS_SUBSET is True
CODEBOOK_PRINT_LIMIT = 16
CODEBOOK_NPY_SAVE_PATH = os.path.join(OUTDIR, "codebook.npy")
IMAGENET_VAL_ROOT = "/datasets/imagenet/val"
EXPORT_BATCH_SIZE = 32
EXPORT_NPZ_PATH = os.path.join(OUTDIR, f"{STAMP}_imagenet256_recon.npz")
EXPORT_IMAGENET_NPZ = False
SAVE_CODEBOOK_NPY = False
RUN_METRICS_ON_VAL = True
RUN_METRICS_ON_CLASS_SUBSET = False

def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

def resize_smallest_max(img, size):
    # Match training preprocessing: resize smallest side to `size` while preserving aspect ratio.
    # Training uses `SmallestMaxSize` followed by a crop at the same `size` (see taming/data/imagenet.py:244-270).
    s = min(img.size)
    scale = size / s
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC)

def center_crop(img, size):
    # Center crop to `size×size` to align with the training pipeline.
    # Produces fixed 256×256 inputs for VQGAN so the latent grid is 16×16 (stride 16).
    # Reference for training crop: taming/data/imagenet.py:244-270 (CenterCrop vs RandomCrop).
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))

def preprocess(path, size):
    # Preprocessing mirrors training: resize smallest side to `size` then center crop.
    # This ensures evaluation uses the same distribution of inputs as the model was trained on.
    # See training preprocessor in taming/data/imagenet.py:244-270.
    img = Image.open(path).convert("RGB")
    img = resize_smallest_max(img, size)
    img = center_crop(img, size)
    arr = np.array(img).astype(np.uint8)
    arr = (arr / 127.5 - 1.0).astype(np.float32)
    x = torch.from_numpy(arr).permute(2,0,1)
    return x

# convert [-1,1] tensors to [0,1] for metrics calculation
def to_01(x):
    x = x.clamp(-1,1) # fix out of range values
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

# Calculaing Global SSIM (not patch based)
# SSIM: structural similarity index averaged over channels (global variant)
# SSIM(x, y) = ((2μxμy + C1)(2σxy + C2)) / ((μx^2 + μy^2 + C1)(σx^2 + σy^2 + C2))
# Inputs are [0,1] C×H×W tensors 
def compute_ssim(orig01, rec01, c1=0.01**2, c2=0.03**2):
    x = orig01
    y = rec01
    mu_x = x.mean(dim=(1,2), keepdim=True)
    mu_y = y.mean(dim=(1,2), keepdim=True)
    var_x = ((x - mu_x) ** 2).mean(dim=(1,2), keepdim=True)
    var_y = ((y - mu_y) ** 2).mean(dim=(1,2), keepdim=True)
    cov_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(1,2), keepdim=True)
    ssim_map = ((2*mu_x*mu_y + c1) * (2*cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return ssim_map.mean().item()

# Perplexity: exp(-sum p_i log p_i) over usage probs p_i
def compute_perplexity_from_counts(counts):
    total = int(counts.sum().item())
    if total == 0:
        return 0.0
    p = counts.float() / total
    return float(torch.exp(-(p * torch.log(p + 1e-10)).sum()).item())

# Flatten indices to a 1D long tensor
# Handles indices with shape B×H×W or already flattened
def flatten_indices(indices):
    if hasattr(indices, 'ndim') and indices.ndim == 3:
        return indices.reshape(-1).cpu().long()
    return indices.flatten().cpu().long()

# Code usage counts (histogram)
# counts[k] = number of assignments to code k
def compute_code_usage_counts(inds, n_codes):
    return torch.bincount(inds, minlength=n_codes)

# Used/dead code counts
# used = |{k : counts[k] > 0}|, dead = n_codes - used
def compute_used_dead_codes(counts, n_codes):
    used = int((counts > 0).sum().item())
    dead = int(n_codes - used)
    return used, dead

# Total tokens assigned
# total = sum_k counts[k]
def compute_total_tokens(counts):
    return int(counts.sum().item())

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
    y = y.permute(0,2,3,1).contiguous()
    return y.detach().cpu().numpy()

def export_imagenet_val_npz(model, val_root, out_npz_path, batch_size):
    device = next(model.parameters()).device
    paths, labels = gather_val_paths_and_labels(val_root)
    n = len(paths)
    tmp_memmap_path = os.path.join(OUTDIR, "tmp_arr0_uint8_memmap.npy")
    mm = np.memmap(tmp_memmap_path, dtype=np.uint8, mode='w+', shape=(n, SIZE, SIZE, 3))
    with tqdm(total=n, desc="Exporting ImageNet256 recon", unit="img") as pbar:
        idx = 0
        while idx < n:
            bs = min(batch_size, n - idx)
            batch_paths = paths[idx:idx+bs]
            xs = []
            for p in batch_paths:
                xs.append(preprocess(p, SIZE).unsqueeze(0))
            x = torch.cat(xs, 0).to(device)
            with torch.no_grad():
                quant, _, _ = model.encode(x)
                recon = model.decode(quant)
            arr = batch_to_uint8_hwc(recon)
            mm[idx:idx+bs] = arr
            idx += bs
            pbar.update(bs)
    mm.flush()
    np.savez_compressed(out_npz_path, arr_0=mm, arr_1=np.array(labels, dtype=np.int64))
    print(f"saved npz: {out_npz_path}")

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
    n_codes = getattr(model.quantize, "n_e", getattr(model.quantize, "n_embed", None))
    counts = torch.zeros(n_codes, dtype=torch.long)
    total_mse = 0.0
    total_lpips = 0.0
    total_psnr = 0.0
    psnr_count = 0
    total_ssim = 0.0
    n_images = 0
    per_image = []
    for p in tqdm(paths, desc="Metrics", unit="img"):
        x = preprocess(p, SIZE).unsqueeze(0).to(device)
        with torch.no_grad():
            quant, _, info = model.encode(x)
            recon = model.decode(quant)
        orig = to_01(x)[0]
        rec = to_01(recon)[0]
        mse = compute_mse(orig, rec)
        lp = compute_lpips(lpips_model, x, recon)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(orig, rec)
        inds = flatten_indices(info[2])
        counts += compute_code_usage_counts(inds, n_codes)
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
    # total_tokens = compute_total_tokens(counts)
    # used, dead = compute_used_dead_codes(counts, n_codes)
    # perplexity = compute_perplexity_from_counts(counts)
    return {
        "timestamp": STAMP,
        "num_images": n_images,
        "avg_mse": avg_mse,
        "avg_lpips": avg_lpips,
        "avg_psnr": (avg_psnr if math.isfinite(avg_psnr) else None),
        "avg_ssim": avg_ssim,
        "n_codes": int(n_codes),
        # "used_codes": used,
        # "dead_codes": dead,
        # "total_tokens": total_tokens,
        # "perplexity": perplexity,
        # "per_image": per_image,
    }

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
    if SAVE_CODEBOOK_NPY:
        save_codebook_npy(model, CODEBOOK_NPY_SAVE_PATH)

    if EXPORT_IMAGENET_NPZ:
        export_imagenet_val_npz(model, IMAGENET_VAL_ROOT, EXPORT_NPZ_PATH, EXPORT_BATCH_SIZE)

    lpips = LPIPS().to(device).eval()

    if RUN_METRICS_ON_VAL:
        summary_val = run_metrics_val_all(model, IMAGENET_VAL_ROOT, device, lpips)
        if summary_val is not None:
            avg_psnr_val = summary_val["avg_psnr"]
            psnr_str_val = f"{avg_psnr_val:.2f}" if (avg_psnr_val is not None and math.isfinite(avg_psnr_val)) else "inf"
            print(f"summary n={summary_val['num_images']} mse={summary_val['avg_mse']:.6f} lpips={summary_val['avg_lpips']:.6f} psnr={psnr_str_val} ssim={summary_val['avg_ssim']:.4f}")
            print(f"codes used={summary_val['used_codes']}/{summary_val['n_codes']} dead={summary_val['dead_codes']} tokens={summary_val['total_tokens']} perplexity={summary_val['perplexity']:.2f}")
            with open(os.path.join(OUTDIR, f"{STAMP}_summary_val.json"), "w") as f:
                json.dump(summary_val, f, indent=2)

    if RUN_METRICS_ON_CLASS_SUBSET:
        summary_subset = run_metrics_class_subset(model, device, lpips)
        if summary_subset is not None:
            summary_subset["limit"] = LIMIT
            avg_psnr_subset = summary_subset["avg_psnr"]
            psnr_str_subset = f"{avg_psnr_subset:.2f}" if (avg_psnr_subset is not None and math.isfinite(avg_psnr_subset)) else "inf"
            print(f"summary n={summary_subset['num_images']} mse={summary_subset['avg_mse']:.6f} lpips={summary_subset['avg_lpips']:.6f} psnr={psnr_str_subset} ssim={summary_subset['avg_ssim']:.4f}")
            print(f"codes used={summary_subset['used_codes']}/{summary_subset['n_codes']} dead={summary_subset['dead_codes']} tokens={summary_subset['total_tokens']} perplexity={summary_subset['perplexity']:.2f}")
            with open(os.path.join(OUTDIR, f"{STAMP}_summary_subset.json"), "w") as f:
                json.dump(summary_subset, f, indent=2)

if __name__ == "__main__":
    main()