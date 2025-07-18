import os
import glob
import torch
import logging

# Setup logging
logging.basicConfig(filename='dataset_check.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = "datasets"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
class_names = ['real', 'synthetic', 'semi-synthetic']

def check_dataset():
    print(f"Base directory: {BASE_DIR}, exists: {os.path.exists(BASE_DIR)}")
    logging.info(f"Base directory: {BASE_DIR}, exists: {os.path.exists(BASE_DIR)}")
    print(f"Train directory: {TRAIN_DIR}, exists: {os.path.exists(TRAIN_DIR)}")
    logging.info(f"Train directory: {TRAIN_DIR}, exists: {os.path.exists(TRAIN_DIR)}")
    print(f"Test directory: {TEST_DIR}, exists: {os.path.exists(TEST_DIR)}")
    logging.info(f"Test directory: {TEST_DIR}, exists: {os.path.exists(TEST_DIR)}")
    
    for dir_path, dir_name in [(TRAIN_DIR, "train"), (TEST_DIR, "test")]:
        if not os.path.exists(dir_path):
            print(f"{dir_name} directory missing!")
            logging.error(f"{dir_name} directory missing!")
            continue
        print(f"\nChecking {dir_name} directory: {dir_path}")
        logging.info(f"Checking {dir_name} directory: {dir_path}")
        for class_name in class_names:
            class_dir = os.path.join(dir_path, class_name)
            print(f"  {class_name} directory exists: {os.path.exists(class_dir)}")
            logging.info(f"  {class_name} directory exists: {os.path.exists(class_dir)}")
            if not os.path.exists(class_dir):
                logging.warning(f"Directory {class_dir} not found.")
                continue
            pt_files = glob.glob(os.path.join(class_dir, "*.pt"))
            print(f"  {class_name}: {len(pt_files)} .pt files found")
            logging.info(f"  {class_name}: {len(pt_files)} .pt files found")
            for pt_file in pt_files:
                try:
                    images = torch.load(pt_file, map_location="cpu")
                    print(f"    {pt_file}: shape={images.shape}, min={images.min()}, max={images.max()}")
                    logging.info(f"    {pt_file}: shape={images.shape}, min={images.min()}, max={images.max()}")
                    if images.max() > 1.0 or images.min() < 0.0:
                        print(f"    Warning: Values out of range [0,1] in {pt_file}")
                        logging.warning(f"Image values out of range [0,1] in {pt_file}")
                except Exception as e:
                    print(f"    Error loading {pt_file}: {e}")
                    logging.error(f"Error loading {pt_file}: {e}")

if __name__ == "__main__":
    check_dataset()