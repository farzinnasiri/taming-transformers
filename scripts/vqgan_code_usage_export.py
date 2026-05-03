import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(".")

from taming.models.vqgan import VQModel


VALID_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Export VQGAN tokenizer code usage for an image-folder dataset.")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing one subdirectory per class.")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True, help="Directory where usage outputs are written.")
    parser.add_argument("--config-path", type=str, default="/checkpoints/vqgan_imagenet_f16_16384/model.yaml")
    parser.add_argument("--model-path", type=str, default="/checkpoints/vqgan_imagenet_f16_16384/last.ckpt")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-indices", action="store_true", help="Also save per-image token grids to indices.npz.")
    return parser.parse_args()


def class_sort_key(path: Path):
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def discover_dataset(root: str):
    root_path = Path(root)
    class_dirs = sorted([path for path in root_path.iterdir() if path.is_dir()], key=class_sort_key)

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


class ImageFolderDataset(Dataset):
    def __init__(self, paths, labels, size):
        self.paths = paths
        self.labels = labels
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return preprocess(path, self.size), self.labels[idx], path


def token_grid_from_indices(indices, batch_size):
    ind_flat = indices.reshape(batch_size, -1)
    side = int(round(ind_flat.shape[1] ** 0.5))
    return ind_flat.reshape(batch_size, side, side)


def update_counts(ind_grid, global_counts, position_counts):
    ind_np = ind_grid.detach().cpu().numpy().astype(np.int64)
    global_counts += np.bincount(ind_np.ravel(), minlength=global_counts.shape[0])

    height, width = ind_np.shape[1], ind_np.shape[2]
    for row in range(height):
        for col in range(width):
            position_counts[:, row, col] += np.bincount(ind_np[:, row, col], minlength=global_counts.shape[0])

    return ind_np


def entropy_and_perplexity(counts):
    total = int(counts.sum())
    probs = counts[counts > 0].astype(np.float64) / float(total)
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy, float(np.exp(entropy))


def top_mass(counts, k):
    total = int(counts.sum())
    return float(np.sort(counts)[-k:].sum() / float(total))


def write_usage_csv(path, counts):
    order = np.argsort(-counts)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(counts) + 1)
    total = int(counts.sum())

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code_id", "count", "frequency", "is_active", "rank"])
        for code_id, count in enumerate(counts.tolist()):
            writer.writerow([code_id, count, count / total, int(count > 0), int(ranks[code_id])])


def write_top_codes_csv(path, counts, top_k=500):
    order = np.argsort(-counts)[:top_k]
    total = int(counts.sum())

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "code_id", "count", "frequency"])
        for rank, code_id in enumerate(order.tolist(), start=1):
            count = int(counts[code_id])
            writer.writerow([rank, code_id, count, count / total])


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths, labels, class_names = discover_dataset(args.data_root)
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
        labels = labels[: args.max_samples]

    print(f"Dataset: {args.dataset_name}")
    print(f"Images: {len(paths)} across {len(class_names)} classes")
    print(f"Output: {outdir}")

    dataset = ImageFolderDataset(paths, labels, args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = load_model(args.config_path, args.model_path, device)

    global_counts = np.zeros(args.codebook_size, dtype=np.int64)
    position_counts = None
    all_indices = []
    saved_paths = []

    for batch, _, batch_paths in tqdm(loader, desc="Encoding"):
        x = preprocess_vqgan(batch).to(device)
        quant, _, (_, _, indices) = model.encode(x)
        ind_grid = token_grid_from_indices(indices, x.shape[0])

        if position_counts is None:
            position_counts = np.zeros((args.codebook_size, ind_grid.shape[1], ind_grid.shape[2]), dtype=np.int64)

        ind_np = update_counts(ind_grid, global_counts, position_counts)
        if args.save_indices:
            all_indices.append(ind_np.astype(np.int32))
            saved_paths.extend(list(batch_paths))

    active_codes = int((global_counts > 0).sum())
    entropy, perplexity = entropy_and_perplexity(global_counts)

    np.save(outdir / "global_counts.npy", global_counts)
    np.save(outdir / "position_counts.npy", position_counts)
    write_usage_csv(outdir / "usage.csv", global_counts)
    write_top_codes_csv(outdir / "top_codes.csv", global_counts)

    if args.save_indices:
        np.savez_compressed(
            outdir / "indices.npz",
            indices=np.concatenate(all_indices, axis=0),
            paths=np.asarray(saved_paths),
        )

    summary = {
        "model": "VQGAN",
        "dataset": args.dataset_name,
        "data_path": args.data_root,
        "n_images": len(paths),
        "n_classes": len(class_names),
        "codebook_size": args.codebook_size,
        "token_grid_hw": [int(position_counts.shape[1]), int(position_counts.shape[2])],
        "total_tokens": int(global_counts.sum()),
        "active_codes": active_codes,
        "dead_codes": int(args.codebook_size - active_codes),
        "active_fraction_full": float(active_codes / args.codebook_size),
        "entropy": entropy,
        "perplexity": perplexity,
        "top_10_mass": top_mass(global_counts, 10),
        "top_100_mass": top_mass(global_counts, 100),
        "top_500_mass": top_mass(global_counts, 500),
        "class_names": class_names,
    }

    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
