"""
Simple script to view and work with saved tensor files (.pt)
"""

import torch
import numpy as np
from pathlib import Path

# ==== Static configuration ====
TENSOR_FILE = 'datasets/test/semi-synthetic/seg_inpainting_0.pt'  # Path to .pt tensor file
START_INDEX = 0  # Start index of tensor to extract
COUNT = 50       # Number of tensors to extract
OUTPUT_DIR = 'datasets/extracted_images'  # Directory to save images

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ matplotlib not available. Install with: pip install matplotlib")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠ PIL not available. Install with: pip install pillow")

def load_and_view_tensors(tensor_path: str, num_images: int = 5):
    """
    Load tensor file and display some images.
    
    Args:
        tensor_path: Path to the .pt file
        num_images: Number of images to display
    """
    print(f"Loading tensors from: {tensor_path}")
    
    # Load the tensors
    tensors = torch.load(tensor_path, weights_only=False)
    
    print(f"✅ Loaded tensors successfully!")
    print(f"📊 Tensor shape: {tensors.shape}")
    print(f"📊 Tensor dtype: {tensors.dtype}")
    print(f"📊 Number of images: {tensors.shape[0]}")
    
    # Get tensor statistics
    print(f"📈 Statistics:")
    print(f"   Min value: {torch.min(tensors):.4f}")
    print(f"   Max value: {torch.max(tensors):.4f}")
    tensors_float = tensors.float()
    print(f"   Mean value: {torch.mean(tensors_float):.4f}")
    print(f"   Std value: {torch.std(tensors_float):.4f}")
    
    # Display some images
    if num_images > 0 and MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, min(num_images, tensors.shape[0]), figsize=(15, 3))
        if num_images == 1:
            axes = [axes]
        
        for i in range(min(num_images, tensors.shape[0])):
            # Get image tensor (C, H, W) -> (H, W, C)
            img = tensors[i].permute(1, 2, 0)
            
            # Convert to numpy and ensure values are in [0, 1]
            img_np = img.numpy()
            if img_np.max() <= 1.0:
                img_np = img_np
            else:
                img_np = img_np / 255.0
            
            axes[i].imshow(img_np)
            axes[i].set_title(f'Image {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    elif num_images > 0:
        print("⚠ matplotlib not available - skipping image display")
        print("Install with: pip install matplotlib")
    
    return tensors

def save_individual_images(tensors, output_dir: str = "extracted_images"):
    """
    Save individual images from tensor file.
    
    Args:
        tensors: Loaded tensor file
        output_dir: Directory to save individual images
    """
    if not PIL_AVAILABLE:
        print("❌ PIL not available - cannot save images")
        print("Install with: pip install pillow")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Saving individual images to: {output_path}")
    
    for i in range(min(10, tensors.shape[0])):  # Save first 10 images
        # Get image tensor
        img = tensors[i].permute(1, 2, 0)
        
        # Convert to PIL Image
        if img.max() <= 1.0:
            img = (img * 255).byte()
        
        pil_img = Image.fromarray(img.numpy().astype(np.uint8))
        
        # Save image
        img_path = output_path / f"image_{i+1:03d}.png"
        pil_img.save(img_path)
        print(f"  Saved: {img_path}")
    
    print(f"✅ Saved {min(10, tensors.shape[0])} images")

def main():
    """Main function to demonstrate tensor loading and viewing."""
    import torch
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tensors = torch.load(TENSOR_FILE, weights_only=False)

    print(f"✅ Loaded tensors successfully!")
    print(f"📊 Tensor shape: {tensors.shape}")
    print(f"📊 Tensor dtype: {tensors.dtype}")
    print(f"📊 Number of images: {tensors.shape[0]}")
    
    # Get tensor statistics
    print(f"📈 Statistics:")
    print(f"   Min value: {torch.min(tensors):.4f}")
    print(f"   Max value: {torch.max(tensors):.4f}")
    tensors_float = tensors.float()
    print(f"   Mean value: {torch.mean(tensors_float):.4f}")
    print(f"   Std value: {torch.std(tensors_float):.4f}")

    end_index = min(START_INDEX + COUNT, tensors.shape[0])
    for i in range(START_INDEX, end_index):
        tensor = tensors[i]
        image = tensor.permute(1, 2, 0).numpy().astype('uint8')
        img = Image.fromarray(image)
        img.save(os.path.join(OUTPUT_DIR, f"image_{i}.png"))
        print(f"Saved image_{i}.png")

if __name__ == "__main__":
    main() 