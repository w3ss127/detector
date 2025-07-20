import os
import glob
import torch

# Define dataset root
dataset_root = os.path.abspath("datasets/train")
print(f"\n🔍 Checking dataset root: {dataset_root}")

# Expected class folders
class_names = ["real", "synthetic", "semi-synthetic"]

for class_name in class_names:
    full_path = os.path.join(dataset_root, class_name)
    print(f"\n📁 Checking class: {class_name}")
    print(f"→ Looking in: {full_path}")

    # Check if folder exists
    if not os.path.isdir(full_path):
        print(f"❌ Folder not found: {full_path}")
        continue

    # Glob all .pt files
    pt_files = glob.glob(os.path.join(full_path, "*.pt"))
    print(f"→ Found {len(pt_files)} .pt files")

    if not pt_files:
        print(f"⚠️ No .pt files found in {full_path}")
        continue

    # Validate each .pt file
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location="cpu")
            if not isinstance(data, torch.Tensor):
                print(f"❌ Not a tensor: {pt_file}")
            elif len(data.shape) != 4 or data.shape[1:] != (3, 256, 256):
                print(f"⚠️ Invalid shape: {pt_file} → shape={data.shape}")
            else:
                print(f"✅ {os.path.basename(pt_file)}: shape = {data.shape}")
        except Exception as e:
            print(f"❌ Error loading {pt_file}: {e}")
