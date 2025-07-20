import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "datasets"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_OUTPUT_DIR = os.path.join(TRAIN_DIR, "semi-synthetic")
TEST_OUTPUT_DIR = os.path.join(TEST_DIR, "semi-synthetic")
REAL_DIR = os.path.join(TRAIN_DIR, "real")
SYN_DIR = os.path.join(TRAIN_DIR, "synthetic")
TRAIN_TARGET_NUM_IMAGES = 200000
TEST_TARGET_NUM_IMAGES = 20000
IMAGES_PER_FILE = 5000
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Setup logging
logging.basicConfig(filename='semi_synthetic_generation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting semi-synthetic image generation")

# Ensure output directories exist
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# ==========================
# DATA LOADING
# ==========================
def load_images(directory, class_name):
    """
    Load images from .pt files in the specified directory.
    """
    image_tensors = []
    total_images = 0
    pt_files = sorted(glob.glob(os.path.join(directory, "*.pt")))
    if not pt_files:
        logging.error(f"No .pt files found in {directory}")
        raise ValueError(f"No .pt files found in {directory}")
    
    for pt_file in tqdm(pt_files, desc=f"Loading {class_name}", leave=False):
        try:
            images = torch.load(pt_file, map_location="cpu").to(torch.uint8)
            if len(images.shape) != 4 or images.shape[1] not in [1, 3] or images.shape[2:] != (256, 256):
                logging.warning(f"Invalid tensor shape in {pt_file}: {images.shape}")
                continue
            logging.info(f"{pt_file}: Pixel range [min={images.min().item():.2f}, max={images.max().item():.2f}]")
            print(f"{pt_file}: Pixel range [min={images.min().item():.2f}, max={images.max().item():.2f}]")
            num_images = images.shape[0]
            image_tensors.append((pt_file, num_images, images))
            total_images += num_images
            images = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            logging.error(f"Error loading {pt_file}: {e}")
            continue
    
    logging.info(f"Loaded {total_images} {class_name} images from {len(pt_files)} .pt files")
    print(f"Loaded {total_images} {class_name} images from {len(pt_files)} .pt files")
    return image_tensors, total_images

# ==========================
# SEMI-SYNTHETIC GENERATION
# ==========================
def generate_semi_synthetic(real_tensors, syn_tensors, target_num_images, images_per_file, output_dir, dataset_type="train"):
    """
    Generate semi-synthetic images by blending real and synthetic images, keeping pixel values in [0, 255].
    """
    real_indices = []
    for file_idx, (pt_file, num_images, _) in enumerate(real_tensors):
        real_indices.extend([(file_idx, i) for i in range(num_images)])
    syn_indices = []
    for file_idx, (pt_file, num_images, _) in enumerate(syn_tensors):
        syn_indices.extend([(file_idx, i) for i in range(num_images)])

    if len(real_indices) < 1 or len(syn_indices) < 1:
        logging.error("Insufficient real or synthetic images")
        raise ValueError("Insufficient real or synthetic images")

    np.random.shuffle(real_indices)
    np.random.shuffle(syn_indices)

    generated_images = 0
    current_batch = []
    file_count = 0

    with torch.no_grad():
        for i in tqdm(range(target_num_images), desc=f"Generating semi-synthetic ({dataset_type})"):
            real_idx = real_indices[i % len(real_indices)]
            syn_idx = syn_indices[i % len(syn_indices)]
            real_file_idx, real_img_idx = real_idx
            syn_file_idx, syn_img_idx = syn_idx

            real_img = real_tensors[real_file_idx][2][real_img_idx]
            syn_img = syn_tensors[syn_file_idx][2][syn_img_idx]

            if real_img.shape[0] == 1:
                real_img = real_img.repeat(3, 1, 1)
            if syn_img.shape[0] == 1:
                syn_img = syn_img.repeat(3, 1, 1)

            alpha = np.random.uniform(0.3, 0.7)
            semi_img = alpha * real_img + (1 - alpha) * syn_img
            semi_img = torch.clamp(semi_img, 0, 255) # Ensure output stays in [0, 255]

            current_batch.append(semi_img)

            if len(current_batch) == images_per_file:
                batch_tensor = torch.stack(current_batch)
                logging.info(f"Batch {file_count} pixel range [min={batch_tensor.min().item():.2f}, max={batch_tensor.max().item():.2f}]")
                print(f"Batch {file_count} pixel range [min={batch_tensor.min().item():.2f}, max={batch_tensor.max().item():.2f}]")
                output_path = os.path.join(output_dir, f"semi_{file_count:04d}.pt")
                try:
                    torch.save(batch_tensor, output_path)
                    logging.info(f"Saved {len(current_batch)} images to {output_path}")
                    print(f"Saved {len(current_batch)} images to {output_path}")
                    generated_images += len(current_batch)
                    current_batch = []
                    file_count += 1
                    batch_tensor = None
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    logging.error(f"Error saving {output_path}: {e}")
                    print(f"Error saving {output_path}: {e}")

    if current_batch:
        batch_tensor = torch.stack(current_batch)
        logging.info(f"Batch {file_count} pixel range [min={batch_tensor.min().item():.2f}, max={batch_tensor.max().item():.2f}]")
        print(f"Batch {file_count} pixel range [min={batch_tensor.min().item():.2f}, max={batch_tensor.max().item():.2f}]")
        output_path = os.path.join(output_dir, f"semi_{file_count:04d}.pt")
        try:
            torch.save(batch_tensor, output_path)
            logging.info(f"Saved {len(current_batch)} images to {output_path}")
            print(f"Saved {len(current_batch)} images to {output_path}")
            generated_images += len(current_batch)
        except Exception as e:
            logging.error(f"Error saving {output_path}: {e}")
            print(f"Error saving {output_path}: {e}")

    return generated_images

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    # Load real and synthetic images
    real_tensors, real_count = load_images(REAL_DIR, "real")
    syn_tensors, syn_count = load_images(SYN_DIR, "synthetic")

    # Verify counts
    expected_counts = {'real': 95000, 'synthetic': 95000}
    for class_name, count in expected_counts.items():
        actual = real_count if class_name == 'real' else syn_count
        if actual != count:
            logging.warning(f"Mismatch in {class_name}: Expected {count}, Got {actual}")
            print(f"Warning: Mismatch in {class_name}: Expected {count}, Got {actual}")

    # Generate training semi-synthetic images
    train_generated = generate_semi_synthetic(real_tensors, syn_tensors, TRAIN_TARGET_NUM_IMAGES, IMAGES_PER_FILE, TRAIN_OUTPUT_DIR, "train")
    logging.info(f"Generated {train_generated} semi-synthetic images in {TRAIN_OUTPUT_DIR}")
    print(f"Generated {train_generated} semi-synthetic images in {TRAIN_OUTPUT_DIR}")
    if train_generated != TRAIN_TARGET_NUM_IMAGES:
        logging.warning(f"Expected {TRAIN_TARGET_NUM_IMAGES} training semi-synthetic images, generated {train_generated}")
        print(f"Warning: Expected {TRAIN_TARGET_NUM_IMAGES} training semi-synthetic images, generated {train_generated}")

    # Generate test semi-synthetic images
    test_generated = generate_semi_synthetic(real_tensors, syn_tensors, TEST_TARGET_NUM_IMAGES, IMAGES_PER_FILE, TEST_OUTPUT_DIR, "test")
    logging.info(f"Generated {test_generated} semi-synthetic images in {TEST_OUTPUT_DIR}")
    print(f"Generated {test_generated} semi-synthetic images in {TEST_OUTPUT_DIR}")
    if test_generated != TEST_TARGET_NUM_IMAGES:
        logging.warning(f"Expected {TEST_TARGET_NUM_IMAGES} test semi-synthetic images, generated {test_generated}")
        print(f"Warning: Expected {TEST_TARGET_NUM_IMAGES} test semi-synthetic images, generated {test_generated}")

if __name__ == "__main__":
    main()
