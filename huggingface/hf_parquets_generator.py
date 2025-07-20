import os
import glob
import torch
import pandas as pd
from tqdm import tqdm

# Output directory
PARQUET_DIR = "huggingface"
os.makedirs(PARQUET_DIR, exist_ok=True)

# Mapping of source folders to output prefixes
DATASETS = [
    # ("datasets/test/real", "test-real-datasets"),
    # ("datasets/test/semi-synthetic", "test-semi-datasets"),
    # ("datasets/test/synthetic", "test-synth-datasets"),
    # ("datasets/train/real", "train-real-datasets"),
    # ("datasets/train/semi-synthetic", "train-semi-datasets"),
    ("datasets/train/synthetic", "train-synth-datasets"),
]

def pt_to_parquet(pt_path, parquet_path):
    images = torch.load(pt_path, map_location="cpu")
    # Flatten each image to a 1D array
    if len(images.shape) == 4:
        N = images.shape[0]
        flat_images = images.view(N, -1).numpy()
    else:
        flat_images = images.numpy()
    # Store each image as a list in the 'image' column
    df = pd.DataFrame({"image": [img.tolist() for img in flat_images]})
    df.to_parquet(parquet_path, index=False)

if __name__ == "__main__":
    for src_dir, prefix in DATASETS:
        pt_files = sorted(glob.glob(os.path.join(src_dir, "*.pt")))
        for pt_file in tqdm(pt_files, desc=f"Converting {src_dir}"):
            base = os.path.basename(pt_file)
            base_noext = os.path.splitext(base)[0]
            parquet_file = f"{prefix}-{base_noext}.parquet"
            parquet_path = os.path.join(PARQUET_DIR, parquet_file)
            print(f"Converting {pt_file} -> {parquet_path}")
            pt_to_parquet(pt_file, parquet_path)
