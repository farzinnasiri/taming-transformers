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
# Noise values are specified in VQGAN's internal pixel space (i.e., after `preprocess_vqgan`, so values live in [-1, 1]).
# Example: std=0.1 in [-1,1] corresponds to ~0.05 in [0,1].
NOISE_STD_LOW = 0.1 
NOISE_STD_MID = 0.2
NOISE_STD_HIGH = 0.5
MAX_SAMPLES = None # Set to None to run on all samples
BATCH_SIZE = 32
NUM_SAVE_WORKERS = 8  # Workers for saving images and metadata

EXPERIMENT_MODE = "h2_patch_token_edit_decoder" # can be "global_noise", "h1_patch_noise_encoder", "h2_patch_token_edit_decoder"
# Fraction of image area to be covered by the patch (0.25 = 25% area) - Used in H1 and H2 experiments
PATCH_FRACTION = 0.25 
# Strategy for patch placement: "random" (anywhere) or "center" (fixed center) - Used in H1 and H2 experiments
PATCH_PLACEMENT = "random" 
# Strategy for replacing tokens in H2 experiment: "random_uniform" (sample from codebook uniformly)
TOKEN_EDIT_MODE = "random_uniform"
SEED = 0

# Experiment definitions:
#
# Let x be an input image after `preprocess_vqgan` (so x ∈ [-1, 1]^{3×H×W}).
# Let E be the VQGAN encoder producing discrete code indices z = E(x) on a token grid (e.g., 16×16 for 256×256 with f=16).
# Let D be the VQGAN decoder that maps quantized codebook vectors back to pixels.
#
# 1) global_noise
#    - Math: x' = clip(x + σ·ε, -1, 1), where ε ~ N(0, I) over all pixels.
#    - Plain language: add Gaussian noise everywhere in the image, then encode/decode.
#
# 2) h1_patch_noise_encoder (Encoder locality: We want to verify that the encoder is local)
#    - Math: sample a pixel patch mask M ∈ {0,1}^{1×H×W}; x' = clip(x + σ·(ε ⊙ M), -1, 1).
#           Compare z = E(x) vs z' = E(x') and check how often indices outside the patch change.
#    - Plain language: perturb pixels only inside a patch, then measure whether token indices outside the patch stay stable.
#    - If tokens far away change, it means the encoder has "global sensitivity" or "leakage"
#
# 3) h2_patch_token_edit_decoder (Decoder locality: We want to verify that the decoder is spatial)
#    - Math: z = E(x). Sample a token-grid patch P; replace z_P with random indices (e.g., Uniform over codebook).
#           Decode x̂' = D(codebook(z')). Measure how much reconstructed pixels outside the corresponding spatial region change.
#    - Plain language: edit a small patch of tokens, decode, and see whether changes stay localized in the reconstruction.



STAMP = int(time.time())
OUTDIR = f"{STAMP}_robustness_dataset_vqgan_{EXPERIMENT_MODE}"
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

def infer_square_grid_hw(num_tokens):
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f"Expected square token grid, got {num_tokens} tokens")
    return side, side

def indices_to_flat_and_grid(indices, batch_size):
    """Convert raw VQ indices into both flat and (H_tok, W_tok) grid forms.

    VQGAN's `encode` returns indices in a shape that depends on implementation details.
    This helper standardizes them into:
    - `ind_flat`: [B, H_tok*W_tok]
    - `ind_grid`: [B, H_tok, W_tok]

    Assumes a square token grid for simplicity (true for typical 256×256 with f=16 -> 16×16).
    """
    ind_flat = indices.reshape(batch_size, -1)
    h, w = infer_square_grid_hw(ind_flat.shape[1])
    ind_grid = ind_flat.reshape(batch_size, h, w)
    return ind_flat, ind_grid


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
                
                for entry in task["images"]:
                    Image.fromarray(entry["array"]).save(os.path.join(sample_dir, entry["filename"]))

                metadata = dict(task["metadata"])
                metadata.update({
                    "image_id": f"{class_id}/{filename}",
                    "original_path": original_path,
                })
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

    def patch_side_from_fraction(self, height, width, fraction):
        side = int(round(math.sqrt(max(0.0, min(1.0, fraction))) * min(height, width)))
        return max(1, min(side, height, width))

    def sample_patch_bboxes_px(self, batch_size, height, width, fraction, placement):
        side = self.patch_side_from_fraction(height, width, fraction)
        if placement == "center":
            x0 = (width - side) // 2
            y0 = (height - side) // 2
            x1 = x0 + side
            y1 = y0 + side
            return [(int(x0), int(y0), int(x1), int(y1)) for _ in range(batch_size)]
        elif placement == "random":
            max_x0 = max(0, width - side)
            max_y0 = max(0, height - side)
            bboxes = []
            for _ in range(batch_size):
                x0 = random.randint(0, max_x0) if max_x0 > 0 else 0
                y0 = random.randint(0, max_y0) if max_y0 > 0 else 0
                x1 = x0 + side
                y1 = y0 + side
                bboxes.append((int(x0), int(y0), int(x1), int(y1)))
            return bboxes

    def make_mask_from_bboxes_px(self, bboxes, height, width, device):
        mask = torch.zeros((len(bboxes), 1, height, width), device=device)
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            mask[i, :, y0:y1, x0:x1] = 1.0
        return mask

    def bbox_px_to_bbox_tok(self, bbox_px, height_px, width_px, height_tok, width_tok):
        """Map a pixel-space bounding box into a token-grid bounding box.

        VQGAN's token grid corresponds to a downsampled spatial lattice.
        We approximate the mapping using integer strides (width_px//width_tok, height_px//height_tok),
        and use a ceil-like conversion for the end coordinates to keep coverage conservative.
        """
        x0, y0, x1, y1 = bbox_px
        stride_x = max(1, width_px // width_tok)
        stride_y = max(1, height_px // height_tok)
        j0 = max(0, min(width_tok, x0 // stride_x))
        i0 = max(0, min(height_tok, y0 // stride_y))
        j1 = max(0, min(width_tok, (x1 + stride_x - 1) // stride_x))
        i1 = max(0, min(height_tok, (y1 + stride_y - 1) // stride_y))
        return (int(j0), int(i0), int(j1), int(i1))

    def run_global_noise_experiment(self, x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, height_tok, width_tok):
        """Global pixel noise baseline.

        Math: x' = clip(x + σ·ε, -1, 1), with ε ~ N(0, I) over all pixels.
        Plain language: add Gaussian noise everywhere in the image, then encode/decode.
        """
        base_noise = torch.randn_like(x_clean)
        x_low = torch.clamp(x_clean + base_noise * NOISE_STD_LOW, -1.0, 1.0)
        x_mid = torch.clamp(x_clean + base_noise * NOISE_STD_MID, -1.0, 1.0)
        x_high = torch.clamp(x_clean + base_noise * NOISE_STD_HIGH, -1.0, 1.0)

        with torch.no_grad():
            rec_low, ind_low_raw = self.encode_decode(x_low)
            rec_mid, ind_mid_raw = self.encode_decode(x_mid)
            rec_high, ind_high_raw = self.encode_decode(x_high)

        ind_low_flat, _ = indices_to_flat_and_grid(ind_low_raw, batch_size)
        ind_mid_flat, _ = indices_to_flat_and_grid(ind_mid_raw, batch_size)
        ind_high_flat, _ = indices_to_flat_and_grid(ind_high_raw, batch_size)

        img_low_uint8 = batch_to_uint8_hwc(x_low)
        rec_low_uint8 = batch_to_uint8_hwc(rec_low)
        img_mid_uint8 = batch_to_uint8_hwc(x_mid)
        rec_mid_uint8 = batch_to_uint8_hwc(rec_mid)
        img_high_uint8 = batch_to_uint8_hwc(x_high)
        rec_high_uint8 = batch_to_uint8_hwc(rec_high)

        for i, original_path in enumerate(paths):
            images = [
                {"filename": "0_original.png", "array": img_clean_uint8[i]},
                {"filename": "1_recon_clean.png", "array": rec_clean_uint8[i]},
                {"filename": "2_input_noise_low.png", "array": img_low_uint8[i]},
                {"filename": "3_recon_noise_low.png", "array": rec_low_uint8[i]},
                {"filename": "4_input_noise_mid.png", "array": img_mid_uint8[i]},
                {"filename": "5_recon_noise_mid.png", "array": rec_mid_uint8[i]},
                {"filename": "6_input_noise_high.png", "array": img_high_uint8[i]},
                {"filename": "7_recon_noise_high.png", "array": rec_high_uint8[i]},
            ]
            metadata = {
                "experiment_mode": EXPERIMENT_MODE,
                "noise_std": [NOISE_STD_LOW, NOISE_STD_MID, NOISE_STD_HIGH],
                "token_grid_hw": [int(height_tok), int(width_tok)],
                "indices_clean": ind_clean_flat[i].detach().cpu().numpy().tolist(),
                "indices_low": ind_low_flat[i].detach().cpu().numpy().tolist(),
                "indices_mid": ind_mid_flat[i].detach().cpu().numpy().tolist(),
                "indices_high": ind_high_flat[i].detach().cpu().numpy().tolist(),
            }
            task = {"original_path": original_path, "images": images, "metadata": metadata}
            self.queue.put(task)

    def run_h1_patch_noise_encoder_experiment(self, x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, height_tok, width_tok):
        """Encoder locality under pixel-space patch noise.

        Math:
          - Sample a pixel mask M (1 inside the patch, 0 outside).
          - x' = clip(x + σ·(ε ⊙ M), -1, 1), ε ~ N(0, I).
          - Compare z = E(x) vs z' = E(x'). Locality hypothesis: z outside the patch mostly stays unchanged.

        Plain language: add Gaussian noise only inside a patch, then check whether token IDs outside the patch stay stable.
        """
        height_px, width_px = x_clean.shape[2], x_clean.shape[3]
        bboxes_px = self.sample_patch_bboxes_px(batch_size, height_px, width_px, PATCH_FRACTION, PATCH_PLACEMENT)
        bboxes_tok = [self.bbox_px_to_bbox_tok(b, height_px, width_px, height_tok, width_tok) for b in bboxes_px]

        # Generate random noise for the whole image
        base_noise = torch.randn_like(x_clean)

        # Create a mask (1 inside the patch, 0 outside)
        mask_px = self.make_mask_from_bboxes_px(bboxes_px, height_px, width_px, device=self.device)

        # Add noise ONLY inside the patch (where mask is 1.0)
        x_low = torch.clamp(x_clean + base_noise * NOISE_STD_LOW * mask_px, -1.0, 1.0)
        x_mid = torch.clamp(x_clean + base_noise * NOISE_STD_MID * mask_px, -1.0, 1.0)
        x_high = torch.clamp(x_clean + base_noise * NOISE_STD_HIGH * mask_px, -1.0, 1.0)

        with torch.no_grad():
            rec_low, ind_low_raw = self.encode_decode(x_low)
            rec_mid, ind_mid_raw = self.encode_decode(x_mid)
            rec_high, ind_high_raw = self.encode_decode(x_high)

        ind_low_flat, _ = indices_to_flat_and_grid(ind_low_raw, batch_size)
        ind_mid_flat, _ = indices_to_flat_and_grid(ind_mid_raw, batch_size)
        ind_high_flat, _ = indices_to_flat_and_grid(ind_high_raw, batch_size)

        img_low_uint8 = batch_to_uint8_hwc(x_low)
        rec_low_uint8 = batch_to_uint8_hwc(rec_low)
        img_mid_uint8 = batch_to_uint8_hwc(x_mid)
        rec_mid_uint8 = batch_to_uint8_hwc(rec_mid)
        img_high_uint8 = batch_to_uint8_hwc(x_high)
        rec_high_uint8 = batch_to_uint8_hwc(rec_high)

        for i, original_path in enumerate(paths):
            images = [
                {"filename": "0_original.png", "array": img_clean_uint8[i]},
                {"filename": "1_recon_clean.png", "array": rec_clean_uint8[i]},
                {"filename": "2_input_patch_noise_low.png", "array": img_low_uint8[i]},
                {"filename": "3_recon_patch_noise_low.png", "array": rec_low_uint8[i]},
                {"filename": "4_input_patch_noise_mid.png", "array": img_mid_uint8[i]},
                {"filename": "5_recon_patch_noise_mid.png", "array": rec_mid_uint8[i]},
                {"filename": "6_input_patch_noise_high.png", "array": img_high_uint8[i]},
                {"filename": "7_recon_patch_noise_high.png", "array": rec_high_uint8[i]},
            ]
            metadata = {
                "experiment_mode": EXPERIMENT_MODE,
                "noise_std": [NOISE_STD_LOW, NOISE_STD_MID, NOISE_STD_HIGH],
                "token_grid_hw": [int(height_tok), int(width_tok)],
                "patch_bbox_px": list(map(int, bboxes_px[i])),
                "patch_bbox_tok": list(map(int, bboxes_tok[i])),
                "indices_clean": ind_clean_flat[i].detach().cpu().numpy().tolist(),
                "indices_low": ind_low_flat[i].detach().cpu().numpy().tolist(),
                "indices_mid": ind_mid_flat[i].detach().cpu().numpy().tolist(),
                "indices_high": ind_high_flat[i].detach().cpu().numpy().tolist(),
            }
            task = {"original_path": original_path, "images": images, "metadata": metadata}
            self.queue.put(task)

    def run_h2_patch_token_edit_decoder_experiment(self, x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, ind_clean_grid, height_tok, width_tok):
        """Decoder locality under token-space patch edits.

        Math:
          - Encode z = E(x) on an H_tok×W_tok grid.
          - Choose a token patch P and replace indices inside P with random code IDs.
          - Decode using codebook lookup + decoder: x̂' = D(codebook(z')).
          - Locality hypothesis: pixels outside the corresponding region change much less.

        Plain language: change a small contiguous block of token IDs, decode, and see if effects stay local.
        """
        height_px, width_px = x_clean.shape[2], x_clean.shape[3]
        bboxes_px = self.sample_patch_bboxes_px(batch_size, height_px, width_px, PATCH_FRACTION, PATCH_PLACEMENT)
        bboxes_tok = [self.bbox_px_to_bbox_tok(b, height_px, width_px, height_tok, width_tok) for b in bboxes_px]
        
        ind_edit_grid = ind_clean_grid.clone()

        if TOKEN_EDIT_MODE != "random_uniform":
            raise ValueError(f"Unknown TOKEN_EDIT_MODE: {TOKEN_EDIT_MODE}")
        n_e = getattr(self.model.quantize, "n_e", None)
        if n_e is None:
            raise ValueError("Could not infer codebook size from model.quantize.n_e")

        for i, (j0, i0, j1, i1) in enumerate(bboxes_tok):
            rand_patch = torch.randint(
                low=0,
                high=int(n_e),
                size=(i1 - i0, j1 - j0),
                device=ind_edit_grid.device,
            )
            ind_edit_grid[i, i0:i1, j0:j1] = rand_patch

        ind_edit_flat = ind_edit_grid.reshape(batch_size, -1)
        with torch.no_grad():
            rec_edit = self.decode_from_indices_flat(ind_edit_flat, height_tok, width_tok)
        rec_edit_uint8 = batch_to_uint8_hwc(rec_edit)

        for i, original_path in enumerate(paths):
            images = [
                {"filename": "0_original.png", "array": img_clean_uint8[i]},
                {"filename": "1_recon_clean.png", "array": rec_clean_uint8[i]},
                {"filename": "2_recon_token_edit.png", "array": rec_edit_uint8[i]},
            ]
            metadata = {
                "experiment_mode": EXPERIMENT_MODE,
                "token_edit_mode": TOKEN_EDIT_MODE,
                "token_grid_hw": [int(height_tok), int(width_tok)],
                "patch_bbox_px": list(map(int, bboxes_px[i])),
                "patch_bbox_tok": list(map(int, bboxes_tok[i])),
                "indices_clean": ind_clean_flat[i].detach().cpu().numpy().tolist(),
                "indices_edit": ind_edit_flat[i].detach().cpu().numpy().tolist(),
            }
            task = {"original_path": original_path, "images": images, "metadata": metadata}
            self.queue.put(task)

    def encode_decode(self, x):
        """Encode to discrete code indices and decode back to pixel space.

        Returns:
        - rec: reconstruction in [-1, 1]
        - indices: discrete codebook indices (token IDs)
        """
        quant, _, (_, _, indices) = self.model.encode(x)
        rec = self.model.decode(quant)
        return rec, indices

    def decode_from_indices_flat(self, ind_flat, height_tok, width_tok):
        """Decode from a batch of flattened token indices.

        `get_codebook_entry` expects a 1D index vector and an output shape
        of the quantized latent tensor: [B, H_tok, W_tok, e_dim].
        """
        indices_1d = ind_flat.reshape(-1).to(dtype=torch.long, device=self.device)
        shape = (ind_flat.shape[0], height_tok, width_tok, self.model.quantize.e_dim)
        quant = self.model.quantize.get_codebook_entry(indices_1d, shape=shape)
        rec = self.model.decode(quant)
        return rec

    def process_batch(self, batch, paths):
        # batch: [B, 3, H, W] in [0, 1]
        x_clean = preprocess_vqgan(batch).to(self.device)

        batch_size = x_clean.shape[0]

        with torch.no_grad():
            rec_clean, ind_clean_raw = self.encode_decode(x_clean)
        ind_clean_flat, ind_clean_grid = indices_to_flat_and_grid(ind_clean_raw, batch_size)
        height_tok, width_tok = ind_clean_grid.shape[1], ind_clean_grid.shape[2]
        
        img_clean_uint8 = batch_to_uint8_hwc(x_clean)
        rec_clean_uint8 = batch_to_uint8_hwc(rec_clean)

        if EXPERIMENT_MODE == "global_noise":
            self.run_global_noise_experiment(x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, height_tok, width_tok)
            return

        if EXPERIMENT_MODE == "h1_patch_noise_encoder":
            self.run_h1_patch_noise_encoder_experiment(x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, height_tok, width_tok)
            return

        if EXPERIMENT_MODE == "h2_patch_token_edit_decoder":
            self.run_h2_patch_token_edit_decoder_experiment(x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, ind_clean_grid, height_tok, width_tok)
            return

        raise ValueError(f"Unknown EXPERIMENT_MODE: {EXPERIMENT_MODE}")

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Starting Robustness Dataset Generation to {OUTDIR}")
    print(f"Experiment Mode: {EXPERIMENT_MODE}")
    print(f"Noise Levels: Low={NOISE_STD_LOW}, Mid={NOISE_STD_MID}, High={NOISE_STD_HIGH}")
    print(f"Patch Fraction: {PATCH_FRACTION}, Patch Placement: {PATCH_PLACEMENT}")

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
