import sys
import argparse
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch_fidelity

class NpzDataset(Dataset):
    def __init__(self, file_path, key='arr_0'):
        # mmap_mode='r' keeps the file on disk, saving RAM
        self.data = np.load(file_path, mmap_mode='r')[key]
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        # Read image: [H, W, C] -> [C, H, W]
        # torch-fidelity expects tensors in [0, 255] (byte) or [0, 1] (float)
        img = self.data[idx]
        # Ensure copy if mmap to avoid issues with some torch versions/operations
        img = np.array(img, copy=True)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

def main():
    parser = argparse.ArgumentParser(description="Evaluate using torch-fidelity on NPZ files")
    parser.add_argument("ref_batch", help="Path to reference batch npz file")
    parser.add_argument("sample_batch", help="Path to sample batch npz file")
    parser.add_argument("--output", "-o", help="Path to save metrics as JSON", default=None)
    parser.add_argument("--gpu", type=str, default=None, help="GPU index to use (e.g., '0'). If not set, uses default CUDA device if available.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    
    args = parser.parse_args()

    print(f"Reference: {args.ref_batch}")
    print(f"Samples:   {args.sample_batch}")

    # Handle GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # If we set env var inside python, we might need to do it before importing torch or at least before initializing cuda.
        # However, usually it's better to set it before running the script.
        # Let's assume the user might want to use torch.device logic, but torch-fidelity takes 'cuda' boolean.
        # We will stick to the 'cuda=True' flag and let the environment/torch handle the device.
        pass 

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")

    # Register datasets so torch-fidelity can find them by name
    # We use unique names based on file paths to avoid collisions if running multiple times in same process (unlikely here but good practice)
    torch_fidelity.register_dataset("ref_dataset", lambda: NpzDataset(args.ref_batch))
    torch_fidelity.register_dataset("sample_dataset", lambda: NpzDataset(args.sample_batch))

    print("Calculating metrics...")
    metrics = torch_fidelity.calculate_metrics(
        input1="sample_dataset", 
        input2="ref_dataset", 
        cuda=use_cuda,
        isc=True, 
        fid=True, 
        kid=True, 
        prc=True,
        verbose=True,
        batch_size=args.batch_size,
    )
    
    print("\nResults:")
    print(json.dumps(metrics, indent=4))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nMetrics saved to {args.output}")

if __name__ == "__main__":
    import os
    main()
