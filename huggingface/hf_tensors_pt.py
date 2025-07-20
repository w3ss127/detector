import os
import glob
import torch
import pandas as pd
from tqdm import tqdm

# Input directory (where parquet files are located)
PARQUET_DIR = "huggingface"
# Output directory for pt files
PT_DIR = "huggingface/pt_files"
os.makedirs(PT_DIR, exist_ok=True)

def parquet_to_pt(parquet_path, pt_path):
    """Convert images from parquet file back to tensor format and save as .pt file"""
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    # Convert the 'image' column back to tensors
    images_list = df['image'].tolist()
    
    # Convert each image list back to tensor
    # Assuming the original shape was preserved in the list format
    tensors = []
    for img_list in images_list:
        # Convert list to tensor
        img_tensor = torch.tensor(img_list, dtype=torch.float32)
        tensors.append(img_tensor)
    
    # Stack all tensors into a single tensor
    if tensors:
        # Assuming all images have the same shape, we can stack them
        stacked_tensor = torch.stack(tensors)
        torch.save(stacked_tensor, pt_path)
        print(f"Saved tensor with shape: {stacked_tensor.shape}")
    else:
        print(f"Warning: No images found in {parquet_path}")

if __name__ == "__main__":
    # Find all parquet files in the huggingface directory
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    
    for parquet_file in tqdm(parquet_files, desc="Converting parquet to pt"):
        base = os.path.basename(parquet_file)
        base_noext = os.path.splitext(base)[0]
        pt_file = f"{base_noext}.pt"
        pt_path = os.path.join(PT_DIR, pt_file)
        
        print(f"Converting {parquet_file} -> {pt_path}")
        parquet_to_pt(parquet_file, pt_path)
