import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(".")

from taming.models.vqgan import VQModel
from taming.modules.losses.lpips import LPIPS


VALID_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct an ImageNet-style dataset with VQGAN.")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing one subdirectory per class.")
    parser.add_argument("--config-path", type=str, default="/checkpoints/vqgan_imagenet_f16_16384/model.yaml")
    parser.add_argument("--model-path", type=str, default="/checkpoints/vqgan_imagenet_f16_16384/last.ckpt")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory. Defaults to a timestamped directory.")
    parser.add_argument("--size", type=int, default=256, help="Resize shortest side to this value, then center crop.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--codebook-print-limit", type=int, default=16)
    parser.add_argument("--skip-export-npz", action="store_true", help="Do not export reconstructed images as NPZ.")
    parser.add_argument("--skip-save-summary", action="store_true", help="Do not write the JSON/TXT metric summary files.")
    return parser.parse_args()


def class_sort_key(path: Path):
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def discover_dataset(root: str):
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    class_dirs = sorted([path for path in root_path.iterdir() if path.is_dir()], key=class_sort_key)
    if not class_dirs:
        raise RuntimeError(f"No class directories found under {root}")

    paths = []
    labels = []
    class_names = []
    next_label = 0
    for class_dir in class_dirs:
        image_paths = sorted(
            [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS],
            key=lambda path: path.name,
        )
        if not image_paths:
            continue
        class_names.append(class_dir.name)
        paths.extend(str(path) for path in image_paths)
        labels.extend([next_label] * len(image_paths))
        next_label += 1

    if not paths:
        raise RuntimeError(f"No supported images found under {root}")

    return paths, labels, class_names


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def preprocess(path, size):
    img = Image.open(path).convert("RGB")
    shortest_side = min(img.size)
    resize_ratio = size / shortest_side
    new_size = (round(resize_ratio * img.size[1]), round(resize_ratio * img.size[0]))
    img = TF.resize(img, new_size, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=[size, size])
    return T.ToTensor()(img)


def preprocess_vqgan(x):
    return 2.0 * x - 1.0


def to_01(x):
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def compute_batch_mse(orig01, rec01):
    return ((orig01 - rec01) ** 2).mean(dim=(1, 2, 3))


def compute_batch_psnr(mse_values):
    safe_mse = torch.clamp(mse_values, min=torch.finfo(mse_values.dtype).tiny)
    return 10.0 * torch.log10(1.0 / safe_mse)


def compute_ssim_per_image(orig01, rec01):
    values = []
    for orig_image, rec_image in zip(orig01, rec01):
        x = orig_image.permute(1, 2, 0).detach().cpu().numpy()
        y = rec_image.permute(1, 2, 0).detach().cpu().numpy()
        values.append(float(ssim(x, y, data_range=1.0, channel_axis=-1)))
    return values


def metric_summary(values):
    if not values:
        return {"mean": None, "std": None, "count": 0}

    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": None, "std": None, "count": int(arr.size), "finite_count": 0}

    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "count": int(arr.size),
        "finite_count": int(finite.size),
    }


def format_metric_line(name, summary):
    mean = summary["mean"]
    std = summary["std"]
    if mean is None or std is None:
        return f"{name}: unavailable"
    return f"{name}: mean={mean:.6f} std={std:.6f}"


def batch_to_uint8_hwc(x):
    y = to_01(x).mul(255.0).round().clamp(0, 255).to(torch.uint8)
    return y.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()


class ImageFolderDataset(Dataset):
    def __init__(self, paths, labels, size):
        self.paths = paths
        self.labels = labels
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return preprocess(self.paths[idx], self.size), self.labels[idx], self.paths[idx]


def print_codebook(model, limit=None):
    embedding = getattr(model.quantize, "embedding", None)
    weights = embedding.weight.detach().cpu()
    print("codebook shape:", tuple(weights.shape))
    print(weights if limit is None else weights[:limit])


def default_outdir(data_root):
    dataset_name = Path(data_root).name
    stamp = int(time.time())
    return f"{stamp}_recon_{dataset_name}"


def main():
    args = parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")
    if not os.path.isfile(args.config_path):
        raise FileNotFoundError(f"Config path does not exist: {args.config_path}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    outdir = args.outdir or default_outdir(args.data_root)
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is not available: {args.device}")

    model = load_model(args.config_path, args.model_path, device)
    print_codebook(model, args.codebook_print_limit)

    paths, labels, class_names = discover_dataset(args.data_root)
    print(f"Discovered {len(paths)} images across {len(class_names)} classes.")

    dataset = ImageFolderDataset(paths, labels, args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    lpips_model = LPIPS().to(device).eval()

    all_recons = []
    all_export_labels = []
    mse_values = []
    lpips_values = []
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for x01, batch_labels, _ in tqdm(loader, desc="Encoding/Decoding", unit="batch"):
            x01 = x01.to(device, non_blocking=True)
            x = preprocess_vqgan(x01)
            quant, _, _ = model.encode(x)
            recon = model.decode(quant)

            rec01 = to_01(recon)
            batch_mse = compute_batch_mse(x01, rec01)
            batch_lpips = lpips_model(x, recon).view(-1)
            batch_psnr = compute_batch_psnr(batch_mse)
            batch_ssim = compute_ssim_per_image(x01, rec01)

            mse_values.extend(float(value) for value in batch_mse.detach().cpu().numpy())
            lpips_values.extend(float(value) for value in batch_lpips.detach().cpu().numpy())
            psnr_values.extend(float(value) for value in batch_psnr.detach().cpu().numpy())
            ssim_values.extend(batch_ssim)

            if not args.skip_export_npz:
                all_recons.append(batch_to_uint8_hwc(recon))
                if isinstance(batch_labels, torch.Tensor):
                    all_export_labels.extend(int(label) for label in batch_labels.detach().cpu().tolist())
                else:
                    all_export_labels.extend(int(label) for label in batch_labels)

    summary = {
        "timestamp": int(time.time()),
        "data_root": args.data_root,
        "config_path": args.config_path,
        "model_path": args.model_path,
        "num_images": len(paths),
        "num_classes": len(class_names),
        "class_names": class_names,
        "metrics": {
            "mse": metric_summary(mse_values),
            "psnr": metric_summary(psnr_values),
            "ssim": metric_summary(ssim_values),
            "lpips": metric_summary(lpips_values),
        },
    }

    print(format_metric_line("MSE", summary["metrics"]["mse"]))
    print(format_metric_line("PSNR", summary["metrics"]["psnr"]))
    print(format_metric_line("SSIM", summary["metrics"]["ssim"]))
    print(format_metric_line("LPIPS", summary["metrics"]["lpips"]))

    if not args.skip_export_npz:
        out_npz_path = os.path.join(outdir, f"{Path(args.data_root).name}_recon.npz")
        full_arr = np.concatenate(all_recons, axis=0)
        np.savez(
            out_npz_path,
            arr_0=full_arr,
            arr_1=np.array(all_export_labels, dtype=np.int64),
            class_names=np.array(class_names),
        )
        print(f"saved npz: {out_npz_path}")
        print("[val-export] sanity check ---")
        print(f"  reconstructions dtype: {full_arr.dtype}  range: {full_arr.min()} … {full_arr.max()}")
        print(f"  labels shape: {np.array(all_export_labels).shape}  dtype: {np.array(all_export_labels).dtype}")
        print(f"  total images: {len(full_arr)}  unique labels: {len(np.unique(all_export_labels))}")

    if not args.skip_save_summary:
        summary_json_path = os.path.join(outdir, "summary.json")
        summary_txt_path = os.path.join(outdir, "summary.txt")
        with open(summary_json_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        with open(summary_txt_path, "w") as handle:
            handle.write(f"Processed {summary['num_images']} samples.\n")
            handle.write(f"{format_metric_line('MSE', summary['metrics']['mse'])}\n")
            handle.write(f"{format_metric_line('PSNR', summary['metrics']['psnr'])}\n")
            handle.write(f"{format_metric_line('SSIM', summary['metrics']['ssim'])}\n")
            handle.write(f"{format_metric_line('LPIPS', summary['metrics']['lpips'])}\n")
        print(f"saved summary: {summary_json_path}")


if __name__ == "__main__":
    main()
