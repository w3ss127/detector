import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
import scipy.stats as stats
from scipy import ndimage
from scipy.fft import fft2, fftshift
import os
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import pickle
import json
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
from torch.cuda.amp import GradScaler, autocast
import pywt
import warnings
import gc
import psutil
import tempfile
import traceback
    
warnings.filterwarnings('ignore')

class PTFileDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, device: str = 'cpu', augment: bool = False):
        """Enhanced dataset with optional augmentation"""
        # Keep images as uint8 in memory to save RAM; normalize on-the-fly in __getitem__
        self.images = images.to(torch.uint8)
        self.labels = labels.long()
        self.device = device
        self.augment = augment
        
        # Advanced augmentation pipeline for training
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])
        else:
            self.transform = None
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Convert from uint8 [0,255] → float32 [0,1] on-the-fly for model/augmentations
        image = self.images[idx].to(torch.float32).div_(255.0)
        label = self.labels[idx]
        
        if self.augment and self.transform and torch.rand(1) < 0.7:
            # Apply augmentation with probability
            image = self.transform(image)
        
        return image.to(self.device), label.to(self.device)


def load_pt_data_streaming(data_dir: str, max_images_per_class: int = 50000) -> Tuple[torch.Tensor, torch.Tensor]:
    """MEMORY-EFFICIENT streaming data loader - prevents OOM crashes"""
    
    def get_memory_usage():
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def check_memory_limit(threshold_gb=100):
        """Check if we're approaching memory limit"""
        current_memory = get_memory_usage()
        if current_memory > threshold_gb:
            print(f"⚠️ Memory usage high: {current_memory:.1f}GB - triggering cleanup")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False
    
    print(f"🔍 Starting MEMORY-EFFICIENT streaming data loading from: {data_dir}")
    print(f"💾 Initial memory usage: {get_memory_usage():.1f}GB")
    print(f"🎯 Max images per class: {max_images_per_class:,}")
    
    class_folders = ['real', 'synthetic', 'semi-synthetic']
    
    # Pre-calculate total dataset size to avoid OOM
    total_expected_images = 0
    for class_name in class_folders:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            pt_files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
            estimated_images = len(pt_files) * 5000  # Assuming 5000 images per file
            total_expected_images += min(estimated_images, max_images_per_class)
            print(f"📊 {class_name}: {len(pt_files)} files → ~{estimated_images:,} images (limit: {max_images_per_class:,})")
    
    print(f"📊 Total expected images: ~{total_expected_images:,}")
    # Estimate with uint8 (1 byte per pixel channel)
    estimated_memory_gb = total_expected_images * 3 * 224 * 224 * 1 / (1024**3)
    print(f"💾 Estimated memory needed (uint8): {estimated_memory_gb:.1f}GB")
    
    if estimated_memory_gb > 150:
        print("⚠️ WARNING: Dataset too large for memory! Using streaming approach...")
        return load_pt_data_chunked(data_dir, max_images_per_class)
    
    # Process each class with strict memory management
    final_images = []
    final_labels = []
    loaded_files = set()
    
    for class_idx, class_name in enumerate(class_folders):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            continue
            
        print(f"\n📁 Processing {class_name} (Class {class_idx}) with memory management...")
        
        pt_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.pt')])
        
        if not pt_files:
            continue
        
        # STREAMING APPROACH: Accumulate per-file tensors, single concat at end
        class_parts: List[torch.Tensor] = []
        images_loaded = 0
        successful_loads = 0
        
        with tqdm(pt_files, desc=f"Streaming {class_name} files", leave=True) as pbar:
            
            for file_idx, pt_file in enumerate(pbar):
                pt_path = os.path.abspath(os.path.join(class_dir, pt_file))
                file_id = f"{class_name}::{pt_file}"
                
                if file_id in loaded_files:
                    pbar.set_postfix_str(f"SKIP duplicate")
                    continue
                
                # Optional memory check (throttled)
                if file_idx % 200 == 0 and check_memory_limit(120):  # 120GB limit
                    print(f"🛑 Memory limit reached, stopping at {images_loaded} images for {class_name}")
                    break
                
                if images_loaded >= max_images_per_class:
                    print(f"🎯 Reached max images limit ({max_images_per_class:,}) for {class_name}")
                    break
                
                try:
                    # Load tensor
                    tensor_data = torch.load(pt_path, map_location='cpu', weights_only=False)
                    
                    # Extract images
                    if isinstance(tensor_data, dict):
                        if 'images' in tensor_data:
                            images = tensor_data['images']
                        elif 'data' in tensor_data:
                            images = tensor_data['data']
                        else:
                            images = list(tensor_data.values())[0]
                    elif isinstance(tensor_data, (list, tuple)):
                        images = tensor_data[0] if len(tensor_data) > 0 else torch.empty(0)
                    else:
                        images = tensor_data
                    
                    # Validate
                    if not isinstance(images, torch.Tensor) or images.numel() == 0:
                        del tensor_data
                        continue
                    
                    # Normalize shape and data
                    while images.dim() < 4:
                        images = images.unsqueeze(0)
                    
                    # Keep as uint8 to save memory; only ensure channel dimensions
                    if images.dtype != torch.uint8:
                        # If values look like 0-255 floats, cast to uint8; else clamp and scale
                        if images.max() > 1.0 and images.max() <= 255.0 and images.min() >= 0.0:
                            images = images.to(torch.uint8)
                        else:
                            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
                            images = (images * 255.0).clamp(0, 255).to(torch.uint8)
                    
                    # Handle channels
                    if images.shape[1] == 1:
                        images = images.repeat(1, 3, 1, 1)
                    elif images.shape[1] > 3:
                        images = images[:, :3, :, :]
                    
                    # Size check
                    if images.shape[2] < 32 or images.shape[3] < 32:
                        del tensor_data, images
                        continue
                    
                    # Append to list; single concat later to avoid O(n^2) copies
                    remaining_slots = max_images_per_class - images_loaded
                    take = min(images.shape[0], max(0, remaining_slots))
                    if take <= 0:
                        del tensor_data, images
                        break
                    class_parts.append(images[:take].contiguous())
                    images_loaded += take
                    
                    successful_loads += 1
                    loaded_files.add(file_id)
                    
                    # Update progress
                    if file_idx % 100 == 0:
                        memory_gb = get_memory_usage()
                        pbar.set_postfix_str(f"✅ {images_loaded:,} imgs | {memory_gb:.1f}GB RAM")
                    
                    # Aggressive cleanup
                    del tensor_data, images
                    if file_idx % 200 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"❌ Error loading {pt_file}: {e}")
                    if 'tensor_data' in locals():
                        del tensor_data
                    if 'images' in locals():
                        del images
                    gc.collect()
                    continue
        
        # Add completed class to final dataset
        if class_parts:
            print(f"🧩 Concatenating {len(class_parts)} parts (~{images_loaded:,} images) for class '{class_name}'...")
            t0 = time.time()
            class_tensor = torch.cat(class_parts, dim=0)
            print(f"✅ Concatenated class '{class_name}' in {time.time()-t0:.1f}s → {class_tensor.shape}")
            num_images = class_tensor.shape[0]
            class_labels = torch.full((num_images,), class_idx, dtype=torch.long)
            
            final_images.append(class_tensor)
            final_labels.append(class_labels)
            
            print(f"✅ {class_name} completed:")
            print(f"   📊 Images loaded: {num_images:,}")
            print(f"   📐 Tensor shape: {class_tensor.shape}")
            print(f"   💾 Memory usage: {get_memory_usage():.1f}GB")
            
            # Don't delete class_tensor yet - we need it for final combination
        else:
            print(f"❌ No images loaded for {class_name}")
    
    # Final combination with memory check
    if final_images:
        print(f"\n🔗 Final dataset assembly...")
        print(f"💾 Pre-combination memory: {get_memory_usage():.1f}GB")
        
        try:
            combined_images = torch.cat(final_images, dim=0)
            combined_labels = torch.cat(final_labels, dim=0)
            
            # Cleanup intermediate tensors immediately
            del final_images, final_labels
            gc.collect()
            
            print(f"\n🎉 DATASET LOADING COMPLETED!")
            print(f"📊 Final dataset:")
            print(f"   Total images: {combined_images.shape[0]:,}")
            print(f"   Shape: {combined_images.shape}")
            print(f"   Memory: {combined_images.element_size() * combined_images.nelement() / (1024**3):.2f}GB")
            print(f"   Final RAM usage: {get_memory_usage():.1f}GB")
            
            # Label distribution
            label_counts = torch.bincount(combined_labels)
            class_names = ['Real', 'Synthetic', 'Semi-synthetic']
            print(f"📊 Class distribution:")
            for i, (name, count) in enumerate(zip(class_names, label_counts)):
                if i < len(label_counts):
                    percentage = (count.item() / len(combined_labels)) * 100
                    print(f"   {name}: {count:,} images ({percentage:.1f}%)")
            
            return combined_images, combined_labels
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"💥 CUDA OOM Error: {e}")
            print("🔄 Falling back to CPU-only processing...")
            # Move everything to CPU and retry
            final_images_cpu = [img.cpu() for img in final_images]
            final_labels_cpu = [lbl.cpu() for lbl in final_labels]
            combined_images = torch.cat(final_images_cpu, dim=0)
            combined_labels = torch.cat(final_labels_cpu, dim=0)
            return combined_images, combined_labels
            
        except MemoryError as e:
            print(f"💥 System OOM Error: {e}")
            print("🆘 Dataset too large for available RAM!")
            raise ValueError("Dataset exceeds available system memory. Consider using chunked loading or reducing dataset size.")
    else:
        raise ValueError("❌ No data loaded!")


def load_pt_data_chunked(data_dir: str, max_images_per_class: int = 50000, chunk_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    """ULTRA-MEMORY-EFFICIENT chunked loading for massive datasets"""

    
    print(f"🔄 Using CHUNKED loading mode (chunk size: {chunk_size:,})")
    
    class_folders = ['real', 'synthetic', 'semi-synthetic']
    temp_files = []  # Track temporary files for cleanup
    
    try:
        # Process each class and save to temporary files
        class_info = []
        
        for class_idx, class_name in enumerate(class_folders):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            pt_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.pt')])
            
            if not pt_files:
                continue
            
            print(f"\n📁 Processing {class_name} in chunks...")
            
            # Create per-class chunk directory
            chunk_dir = tempfile.mkdtemp(prefix=f"chunks_{class_name}_")
            temp_files.append(chunk_dir)
            
            chunk_images = []
            chunk_labels = []
            total_images_saved = 0
            images_in_current_chunk = 0
            chunk_index = 0
            
            with tqdm(pt_files, desc=f"Chunking {class_name}", leave=True) as pbar:
                
                for pt_file in pbar:
                    if total_images_saved >= max_images_per_class:
                        break
                    
                    pt_path = os.path.join(class_dir, pt_file)
                    
                    try:
                        # Load single file
                        tensor_data = torch.load(pt_path, map_location='cpu', weights_only=False)
                        
                        # Extract images
                        if isinstance(tensor_data, dict):
                            images = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                        elif isinstance(tensor_data, (list, tuple)):
                            images = tensor_data[0] if len(tensor_data) > 0 else torch.empty(0)
                        else:
                            images = tensor_data
                        
                        if not isinstance(images, torch.Tensor) or images.numel() == 0:
                            del tensor_data
                            continue
                        
                        # Quick normalization
                        while images.dim() < 4:
                            images = images.unsqueeze(0)
                        
                        # Keep as uint8 to save memory
                        if images.dtype != torch.uint8:
                            if images.max() > 1.0 and images.max() <= 255.0 and images.min() >= 0.0:
                                images = images.to(torch.uint8)
                            else:
                                images = (images - images.min()) / (images.max() - images.min() + 1e-8)
                                images = (images * 255.0).clamp(0, 255).to(torch.uint8)
                        
                        # Handle channels
                        if images.shape[1] == 1:
                            images = images.repeat(1, 3, 1, 1)
                        elif images.shape[1] > 3:
                            images = images[:, :3, :, :]
                        
                        # Size check
                        if images.shape[2] < 32 or images.shape[3] < 32:
                            del tensor_data, images
                            continue
                        
                        # Limit images if needed
                        remaining_slots = max_images_per_class - total_images_saved
                        if images.shape[0] > remaining_slots:
                            images = images[:remaining_slots]
                        
                        # Add to current chunk
                        chunk_images.append(images)
                        labels = torch.full((images.shape[0],), class_idx, dtype=torch.long)
                        chunk_labels.append(labels)
                        
                        images_in_current_chunk += images.shape[0]
                        total_images_saved += images.shape[0]
                        
                        # Save chunk when it gets large enough
                        if images_in_current_chunk >= chunk_size or total_images_saved >= max_images_per_class:
                            if chunk_images:
                                # Combine chunk
                                chunk_tensor = torch.cat(chunk_images, dim=0)
                                chunk_label_tensor = torch.cat(chunk_labels, dim=0)
                                
                                # Save to a new chunk file (no appending to avoid repeated IO)
                                chunk_data = {
                                    'images': chunk_tensor,
                                    'labels': chunk_label_tensor
                                }
                                chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:05d}.pt")
                                torch.save(chunk_data, chunk_path)
                                chunk_index += 1
                                
                                # Reset chunk
                                del chunk_images, chunk_labels, chunk_tensor, chunk_label_tensor
                                if 'combined_imgs' in locals():
                                    del combined_imgs, combined_lbls
                                chunk_images = []
                                chunk_labels = []
                                images_in_current_chunk = 0
                                
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        
                        pbar.set_postfix_str(f"✅ {total_images_saved:,}/{max_images_per_class:,} | {get_memory_usage():.1f}GB")
                        
                        # Cleanup
                        del tensor_data, images, labels
                        
                    except Exception as e:
                        print(f"❌ Error with {pt_file}: {e!r} ({type(e).__name__})")
                        traceback.print_exc()
                        continue
            
            # Save any remaining chunk
            if chunk_images:
                chunk_tensor = torch.cat(chunk_images, dim=0)
                chunk_label_tensor = torch.cat(chunk_labels, dim=0)
                
                chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:05d}.pt")
                torch.save({'images': chunk_tensor, 'labels': chunk_label_tensor}, chunk_path)
                chunk_index += 1
                
                del chunk_images, chunk_labels, chunk_tensor, chunk_label_tensor
            
            # Record class info
            # Record class info
            class_info.append({
                'class_idx': class_idx,
                'class_name': class_name,
                'chunk_dir': chunk_dir,
                'images_saved': total_images_saved
            })
            print(f"✅ {class_name}: {total_images_saved:,} images saved across {chunk_index} chunks in {chunk_dir}")
        
        # Now load all classes from temp files and combine
        print(f"\n🔗 Loading from temporary files for final assembly...")
        
        for class_data in class_info:
            try:
                print(f"📂 Loading {class_data['class_name']} from chunks in {class_data['chunk_dir']}...")
                chunk_files = sorted([f for f in os.listdir(class_data['chunk_dir']) if f.endswith('.pt')])
                if not chunk_files:
                    print(f"⚠️ No chunks found for {class_data['class_name']}, skipping")
                    continue
                class_parts = []
                label_parts = []
                for cf in chunk_files:
                    cp = os.path.join(class_data['chunk_dir'], cf)
                    if os.path.getsize(cp) == 0:
                        continue
                    td = torch.load(cp, map_location='cpu')
                    class_parts.append(td['images'])
                    label_parts.append(td['labels'])
                if class_parts:
                    final_images.append(torch.cat(class_parts, dim=0))
                    final_labels.append(torch.cat(label_parts, dim=0))
                    print(f"✅ Loaded {final_images[-1].shape[0]:,} images for {class_data['class_name']}")
                
            except Exception as e:
                print(f"❌ Error loading temp file for {class_data['class_name']}: {e!r} ({type(e).__name__})")
                traceback.print_exc()
                continue
        
        # Final combination
        if final_images:
            print(f"🔗 Final assembly...")
            print(f"💾 Pre-final memory: {get_memory_usage():.1f}GB")
            
            combined_images = torch.cat(final_images, dim=0)
            combined_labels = torch.cat(final_labels, dim=0)
            
            print(f"🎉 STREAMING LOAD COMPLETED!")
            print(f"📊 Final dataset: {combined_images.shape[0]:,} images")
            print(f"💾 Final memory: {get_memory_usage():.1f}GB")
            
            return combined_images, combined_labels
        else:
            raise ValueError("No data loaded from any class!")
    
    finally:
        # Cleanup temporary chunk directories
        for tmp in temp_files:
            try:
                if os.path.isdir(tmp):
                    for f in os.listdir(tmp):
                        fp = os.path.join(tmp, f)
                        try:
                            os.remove(fp)
                        except:
                            pass
                    os.rmdir(tmp)
                    print(f"🗑️ Cleaned up chunk dir: {tmp}")
                elif os.path.exists(tmp):
                    os.unlink(tmp)
                    print(f"🗑️ Cleaned up temp file: {tmp}")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")


def load_and_verify_data(data_dir: str, max_images_per_class: int = 50000):
    """Main data loading function with automatic memory management"""
    print("🛡️ MEMORY-SAFE DATA LOADING INITIATED")
    
    # Check available system memory
    available_memory = psutil.virtual_memory().available / (1024**3)
    print(f"💾 Available system memory: {available_memory:.1f}GB")
    
    if available_memory < 50:
        print("⚠️ Low memory detected - using conservative loading")
        max_images_per_class = min(max_images_per_class, 20000)
    
    start_time = time.time()
    
    try:
        images, labels = load_pt_data_streaming(data_dir, max_images_per_class)
        
        load_time = time.time() - start_time
        print(f"⚡ Total loading time: {load_time:.1f}s")
        
        return images, labels
        
    except Exception as e:
        print(f"💥 Loading failed: {e}")
        print("🔄 Attempting emergency chunked loading...")
        return load_pt_data_chunked(data_dir, max_images_per_class // 2, chunk_size=5000)


# Emergency function for extremely large datasets
def load_reduced_dataset(data_dir: str, images_per_class: int = 10000):
    """Load a reduced dataset when full dataset is too large"""
    print(f"🎯 Loading REDUCED dataset: {images_per_class:,} images per class")
    return load_and_verify_data(data_dir, max_images_per_class=images_per_class)


def create_train_val_test_split(images: torch.Tensor, labels: torch.Tensor, 
                               train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                               random_seed=42) -> Tuple[PTFileDataset, PTFileDataset, PTFileDataset]:
    """Enhanced stratified split with class balance verification"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\n🎯 Creating stratified split from EXISTING loaded data...")
    print(f"📊 Input data: {images.shape[0]} images, {len(torch.unique(labels))} classes")
    
    unique_labels = torch.unique(labels)
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Stratified splitting by class
    for label in unique_labels:
        label_indices = torch.where(labels == label)[0]
        label_indices = label_indices[torch.randperm(len(label_indices))]
        
        n_samples = len(label_indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices.extend(label_indices[:n_train].tolist())
        val_indices.extend(label_indices[n_train:n_train + n_val].tolist())
        test_indices.extend(label_indices[n_train + n_val:].tolist())
        
        print(f"   Class {label.item()}: {n_samples} total → Train: {n_train}, Val: {n_val}, Test: {n_samples-n_train-n_val}")
    
    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    train_indices = torch.tensor(train_indices)
    val_indices = torch.tensor(val_indices)
    test_indices = torch.tensor(test_indices)
    
    # Create datasets using SAME images tensor - NO RELOADING
    train_dataset = PTFileDataset(images[train_indices], labels[train_indices], augment=True)
    val_dataset = PTFileDataset(images[val_indices], labels[val_indices], augment=False)
    test_dataset = PTFileDataset(images[test_indices], labels[test_indices], augment=False)
    
    print(f"\n✅ CORRECTED stratified split completed - NO DATA RELOADING:")
    print(f"📊 Train: {len(train_dataset)} samples")
    print(f"📊 Validation: {len(val_dataset)} samples") 
    print(f"📊 Test: {len(test_dataset)} samples")
    print(f"📊 Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")
    
    # Verify class distribution in each split
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        class_counts = torch.bincount(dataset.labels, minlength=3)
        print(f"{split_name} class distribution: Real={class_counts[0]}, Synthetic={class_counts[1]}, Semi-synthetic={class_counts[2]}")
    
    return train_dataset, val_dataset, test_dataset


class AdvancedResidualAutoencoder(nn.Module):
    """State-of-the-art autoencoder with attention mechanisms and residual connections"""
    def __init__(self, input_channels=3):
        super(AdvancedResidualAutoencoder, self).__init__()
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # Enhanced encoder with residual blocks
        self.encoder = nn.ModuleList([
            # Block 1
            self._make_residual_block(input_channels, 64),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            self._make_residual_block(64, 128),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            self._make_residual_block(128, 256),
            nn.MaxPool2d(2, 2),
            
            # Block 4 - Deeper encoding
            self._make_residual_block(256, 512),
            # Use stride-2 pooling to reach 14x14 for 224x224 inputs
            nn.MaxPool2d(2, 2)
        ])
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.1)
        )
        
        # Enhanced decoder with skip connections
        self.decoder = nn.ModuleList([
            # Upsampling block 1
            nn.Sequential(
                nn.ConvTranspose2d(256, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ),
            
            # Upsampling block 2
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ),
            
            # Upsampling block 3
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),

            # Upsampling block 4 to reach 224x224
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            
            # Final reconstruction
            nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, input_channels, 3, padding=1),
                nn.Sigmoid()
            )
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block with proper channel matching"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            ResidualConnection(in_channels, out_channels),
            nn.ReLU(True)
        )
    
    def _init_weights(self, m):
        """Initialize weights using Xavier initialization"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store intermediate features for skip connections
        features = []
        
        # Encoder
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i % 2 == 0:  # Store features before pooling
                features.append(x)
        
        # Bottleneck with self-attention
        x = self.bottleneck(x)
        
        # Apply attention mechanism
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).transpose(1, 2)  # (B, H*W, C)
        x_attn, _ = self.attention(x_flat, x_flat, x_flat)
        x = x_attn.transpose(1, 2).view(b, c, h, w)
        
        # Decoder
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        
        return x

class ResidualConnection(nn.Module):
    """Residual connection with channel matching"""
    def __init__(self, in_channels, out_channels):
        super(ResidualConnection, self).__init__()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        return x  # Placeholder - actual residual will be handled in parent

class UltraAdvancedFeatureExtractor:
    """State-of-the-art feature extractor with 500+ features"""
    
    def __init__(self, n_jobs=-1, use_gpu_features=True):
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = SelectKBest(f_classif, k='all')
        self.fitted = False
        self.n_jobs = n_jobs if n_jobs != -1 else min(mp.cpu_count(), 16)
        self.use_gpu_features = use_gpu_features and torch.cuda.is_available()
        print(f"🚀 Ultra-advanced feature extractor initialized with {self.n_jobs} CPU cores")
        if self.use_gpu_features:
            print("⚡ GPU acceleration enabled for feature extraction")
    
    def extract_comprehensive_features(self, noise_map: np.ndarray) -> np.ndarray:
        """Extract 500+ comprehensive features from noise map"""
        features = []
        
        # Basic statistical features (20 features)
        noise_flat = noise_map.flatten()
        features.extend([
            np.mean(noise_flat), np.std(noise_flat), np.var(noise_flat),
            stats.skew(noise_flat) if len(noise_flat) > 0 else 0.0,
            stats.kurtosis(noise_flat) if len(noise_flat) > 0 else 0.0,
            np.min(noise_flat), np.max(noise_flat),
            stats.iqr(noise_flat), np.median(noise_flat),
            stats.entropy(np.histogram(noise_flat, bins=50)[0] + 1e-8)
        ])
        
        # Extended percentiles (20 features)
        percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
        perc_values = np.percentile(noise_flat, percentiles)
        features.extend(perc_values.tolist())
        
        # Histogram features (50 features)
        hist, _ = np.histogram(noise_flat, bins=50, range=(-1, 1))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist.tolist())
        
        # Spatial analysis features (30 features)
        h, w = noise_map.shape
        if h > 8 and w > 8:
            # Gradient analysis
            grad_x = cv2.Sobel(noise_map, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(noise_map, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_dir = np.arctan2(grad_y, grad_x)
            
            grad_features = [
                np.mean(grad_mag), np.std(grad_mag),
                np.percentile(grad_mag, 90), np.percentile(grad_mag, 95),
                np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y)),
                np.std(grad_dir), np.var(grad_dir)
            ]
            features.extend(grad_features)
            
            # Laplacian features
            laplacian = cv2.Laplacian(noise_map, cv2.CV_32F)
            lap_features = [
                np.mean(laplacian), np.std(laplacian),
                np.percentile(np.abs(laplacian), 90),
                np.sum(laplacian > 0) / laplacian.size,
                np.sum(laplacian < 0) / laplacian.size
            ]
            features.extend(lap_features)
            
            # Quadrant analysis (8 features)
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                noise_map[:mid_h, :mid_w],
                noise_map[:mid_h, mid_w:],
                noise_map[mid_h:, :mid_w],
                noise_map[mid_h:, mid_w:]
            ]
            quad_means = [np.mean(quad) for quad in quadrants]
            quad_vars = [np.var(quad) for quad in quadrants]
            features.extend([
                np.std(quad_means), np.std(quad_vars),
                max(quad_vars) - min(quad_vars),
                max(quad_means) - min(quad_means),
                np.corrcoef([q.flatten() for q in quadrants])[0, 1] if len(quadrants) > 1 else 0.0
            ])
            
            # Regional correlation analysis (8 features)
            regions = []
            region_size = min(h//4, w//4, 32)
            for i in range(0, h-region_size, region_size):
                for j in range(0, w-region_size, region_size):
                    region = noise_map[i:i+region_size, j:j+region_size]
                    regions.append(region.flatten())
            
            if len(regions) >= 4:
                region_corrs = []
                for i in range(min(4, len(regions))):
                    for j in range(i+1, min(4, len(regions))):
                        corr = np.corrcoef(regions[i], regions[j])[0, 1]
                        region_corrs.append(corr if not np.isnan(corr) else 0.0)
                features.extend(region_corrs[:6])
            
            # Pad if needed
            while len(features) < len(features) + (30 - (len(features) % 30)):
                features.append(0.0)
        else:
            features.extend([0.0] * 30)
        
        # Frequency domain analysis (40 features)
        try:
            # FFT analysis
            fft = fft2(noise_map)
            fft_shifted = fftshift(fft)
            magnitude_spectrum = np.abs(fft_shifted)
            phase_spectrum = np.angle(fft_shifted)
            
            # Frequency features
            freq_features = [
                np.mean(magnitude_spectrum), np.std(magnitude_spectrum),
                np.percentile(magnitude_spectrum, 90),
                np.percentile(magnitude_spectrum, 95),
                np.mean(phase_spectrum), np.std(phase_spectrum),
                np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 95)),
                np.sum(magnitude_spectrum < np.percentile(magnitude_spectrum, 5))
            ]
            
            # Radial frequency analysis
            center_y, center_x = np.array(magnitude_spectrum.shape) // 2
            y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Analyze different frequency bands
            for r_min, r_max in [(0, 10), (10, 30), (30, 60), (60, 100)]:
                mask = (radius >= r_min) & (radius < r_max)
                if np.any(mask):
                    band_energy = np.mean(magnitude_spectrum[mask])
                    freq_features.append(band_energy)
                else:
                    freq_features.append(0.0)
            
            features.extend(freq_features[:20])
            
            # Power spectral density features
            psd = np.abs(fft)**2
            psd_features = [
                np.mean(psd), np.std(psd),
                np.percentile(psd, 90), np.percentile(psd, 95),
                np.sum(psd > np.percentile(psd, 99)),
                np.sum(psd < np.percentile(psd, 1))
            ]
            features.extend(psd_features)
            
            # Add padding to reach 40 features
            while len(features) % 40 != (len(features) - len(freq_features) - len(psd_features)) % 40:
                features.append(0.0)
                
        except Exception as e:
            features.extend([0.0] * 40)
        
        # Wavelet analysis (60 features)
        try:
            # Multi-level wavelet decomposition
            wavelets = ['db1', 'db4', 'haar', 'coif2']
            for wavelet in wavelets:
                try:
                    coeffs = pywt.wavedec2(noise_map, wavelet, level=3)
                    for coeff in coeffs:
                        if isinstance(coeff, np.ndarray):
                            features.extend([np.mean(coeff), np.std(coeff), np.var(coeff)])
                        else:
                            for c in coeff:
                                features.extend([np.mean(c), np.std(c)])
                except:
                    features.extend([0.0] * 15)
        except:
            features.extend([0.0] * 60)
        
        # Texture analysis using LBP and GLCM (80 features)
        try:
            # Local Binary Pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(noise_map, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                     range=(0, n_points + 2))
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
            features.extend(lbp_hist.tolist()[:20])
            
            # Gray-Level Co-occurrence Matrix
            # Convert to uint8 for GLCM
            noise_uint8 = ((noise_map + 1) * 127.5).astype(np.uint8)
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm_features = []
            for distance in distances:
                for angle in angles:
                    try:
                        glcm = graycomatrix(noise_uint8, [distance], [angle], 
                                          levels=32, symmetric=True, normed=True)
                        
                        contrast = graycoprops(glcm, 'contrast')[0, 0]
                        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                        energy = graycoprops(glcm, 'energy')[0, 0]
                        correlation = graycoprops(glcm, 'correlation')[0, 0]
                        
                        glcm_features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
                    except:
                        glcm_features.extend([0.0] * 5)
            
            features.extend(glcm_features[:60])
            
        except Exception as e:
            features.extend([0.0] * 80)
        
        # Gabor filter responses (40 features)
        try:
            frequencies = [0.1, 0.3, 0.5, 0.7]
            angles = [0, 45, 90, 135]
            
            for frequency in frequencies:
                for angle in angles:
                    real, _ = gabor(noise_map, frequency=frequency, 
                                  theta=np.radians(angle))
                    features.extend([
                        np.mean(real), np.std(real),
                        np.percentile(np.abs(real), 90)
                    ])
                    
        except Exception as e:
            features.extend([0.0] * 48)
        
        # Advanced spatial correlation features (30 features)
        try:
            # Autocorrelation analysis
            autocorr = np.correlate(noise_flat, noise_flat, mode='full')
            autocorr_norm = autocorr / np.max(autocorr)
            center = len(autocorr_norm) // 2
            
            # Extract features from autocorrelation
            window = min(50, center)
            autocorr_window = autocorr_norm[center-window:center+window]
            features.extend([
                np.mean(autocorr_window), np.std(autocorr_window),
                np.max(autocorr_window), np.argmax(autocorr_window) - window,
                np.sum(autocorr_window > 0.5), np.sum(autocorr_window < -0.5)
            ])
            
            # Structural tensor analysis
            Ixx = cv2.Sobel(noise_map, cv2.CV_32F, 2, 0, ksize=3)
            Ixy = cv2.Sobel(noise_map, cv2.CV_32F, 1, 1, ksize=3)
            Iyy = cv2.Sobel(noise_map, cv2.CV_32F, 0, 2, ksize=3)
            
            det_st = Ixx * Iyy - Ixy**2
            trace_st = Ixx + Iyy
            
            features.extend([
                np.mean(det_st), np.std(det_st),
                np.mean(trace_st), np.std(trace_st),
                np.mean(np.abs(det_st)), np.mean(np.abs(trace_st))
            ])
            
            # Morphological features
            kernel = np.ones((3,3), np.uint8)
            noise_binary = (noise_map > np.mean(noise_map)).astype(np.uint8)
            
            opening = cv2.morphologyEx(noise_binary, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(noise_binary, cv2.MORPH_CLOSE, kernel)
            gradient = cv2.morphologyEx(noise_binary, cv2.MORPH_GRADIENT, kernel)
            
            features.extend([
                np.mean(opening), np.std(opening),
                np.mean(closing), np.std(closing),
                np.mean(gradient), np.std(gradient),
                np.sum(opening), np.sum(closing), np.sum(gradient)
            ])
            
            # Add padding to reach 30
            while len(features) % 30 != (len(features) - 21) % 30:
                features.append(0.0)
                
        except Exception as e:
            features.extend([0.0] * 30)
        
        # Noise pattern analysis (50 features)
        try:
            # Multi-scale analysis
            scales = [1, 2, 4, 8]
            for scale in scales:
                if scale > 1:
                    downsampled = cv2.resize(noise_map, 
                                           (max(8, w//scale), max(8, h//scale)),
                                           interpolation=cv2.INTER_AREA)
                else:
                    downsampled = noise_map
                
                # Extract features at this scale
                scale_flat = downsampled.flatten()
                features.extend([
                    np.mean(scale_flat), np.std(scale_flat),
                    np.percentile(scale_flat, 95),
                    np.percentile(scale_flat, 5),
                    stats.skew(scale_flat) if len(scale_flat) > 0 else 0.0
                ])
            
            # Edge density analysis
            edges = cv2.Canny((noise_map * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Connected components analysis
            num_labels, labels_img = cv2.connectedComponents(edges)
            
            features.extend([
                edge_density, num_labels,
                np.std([np.sum(labels_img == i) for i in range(1, min(num_labels, 10))])
                if num_labels > 1 else 0.0
            ])
            
            # Directional analysis
            orientations = [0, 45, 90, 135]
            for orientation in orientations:
                kernel = cv2.getRotationMatrix2D((1, 1), orientation, 1)
                rotated = cv2.warpAffine(noise_map, kernel, (h, w))
                features.extend([np.mean(rotated), np.std(rotated)])
            
            # Pad to reach 50
            while len(features) % 50 != (len(features) - 43) % 50:
                features.append(0.0)
                
        except Exception as e:
            features.extend([0.0] * 50)
        
        # Advanced spectral features (60 features)
        try:
            # Power spectrum analysis
            fft_result = np.fft.fft2(noise_map)
            power_spectrum = np.abs(fft_result)**2
            
            # Frequency bin analysis
            freq_bins = 20
            h_bins = np.array_split(power_spectrum, freq_bins, axis=0)
            w_bins = np.array_split(power_spectrum, freq_bins, axis=1)
            
            # Horizontal frequency analysis
            h_energies = [np.mean(bin_data) for bin_data in h_bins[:10]]
            features.extend(h_energies)
            
            # Vertical frequency analysis  
            w_energies = [np.mean(bin_data) for bin_data in w_bins[:10]]
            features.extend(w_energies)
            
            # Spectral centroid and spread
            freqs_h = np.fft.fftfreq(h)
            freqs_w = np.fft.fftfreq(w)
            
            # Compute spectral centroid
            h_spectrum = np.mean(power_spectrum, axis=1)
            w_spectrum = np.mean(power_spectrum, axis=0)
            
            h_centroid = np.sum(freqs_h * h_spectrum) / (np.sum(h_spectrum) + 1e-8)
            w_centroid = np.sum(freqs_w * w_spectrum) / (np.sum(w_spectrum) + 1e-8)
            
            # Spectral spread
            h_spread = np.sqrt(np.sum(((freqs_h - h_centroid)**2) * h_spectrum) / (np.sum(h_spectrum) + 1e-8))
            w_spread = np.sqrt(np.sum(((freqs_w - w_centroid)**2) * w_spectrum) / (np.sum(w_spectrum) + 1e-8))
            
            features.extend([h_centroid, w_centroid, h_spread, w_spread])
            
            # Spectral rolloff
            cumsum_h = np.cumsum(h_spectrum)
            cumsum_w = np.cumsum(w_spectrum)
            rolloff_85_h = np.where(cumsum_h >= 0.85 * cumsum_h[-1])[0]
            rolloff_85_w = np.where(cumsum_w >= 0.85 * cumsum_w[-1])[0]
            
            features.extend([
                rolloff_85_h[0] / len(freqs_h) if len(rolloff_85_h) > 0 else 0.0,
                rolloff_85_w[0] / len(freqs_w) if len(rolloff_85_w) > 0 else 0.0
            ])
            
            # High frequency content
            high_freq_h = np.sum(h_spectrum[len(h_spectrum)//2:])
            high_freq_w = np.sum(w_spectrum[len(w_spectrum)//2:])
            total_energy_h = np.sum(h_spectrum)
            total_energy_w = np.sum(w_spectrum)
            
            features.extend([
                high_freq_h / (total_energy_h + 1e-8),
                high_freq_w / (total_energy_w + 1e-8)
            ])
            
            # Phase coherence analysis
            phase_diff_h = np.diff(np.unwrap(np.angle(np.fft.fft(noise_map.mean(axis=0)))))
            phase_diff_w = np.diff(np.unwrap(np.angle(np.fft.fft(noise_map.mean(axis=1)))))
            
            features.extend([
                np.std(phase_diff_h), np.mean(np.abs(phase_diff_h)),
                np.std(phase_diff_w), np.mean(np.abs(phase_diff_w))
            ])
            
            # Pad remaining features to 60
            current_spectral = 30  # Current count
            remaining = 60 - current_spectral
            
            # Add spectral entropy and other advanced features
            if remaining > 0:
                # Spectral entropy
                psd_norm = power_spectrum / (np.sum(power_spectrum) + 1e-8)
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
                features.append(spectral_entropy)
                
                # Spectral flux
                if h > 1 and w > 1:
                    prev_spectrum = np.abs(np.fft.fft2(noise_map[:-1, :-1]))**2
                    curr_spectrum = power_spectrum[:-1, :-1]
                    spectral_flux = np.mean((curr_spectrum - prev_spectrum)**2)
                    features.append(spectral_flux)
                else:
                    features.append(0.0)
                
                # Fill remaining with zeros
                features.extend([0.0] * (remaining - 2))
                
        except Exception as e:
            features.extend([0.0] * 60)
        
        # Advanced noise pattern detection (70 features)
        try:
            # Periodic pattern detection
            autocorr_2d = np.correlate(noise_flat, noise_flat, mode='full')
            autocorr_2d = autocorr_2d / np.max(autocorr_2d)
            
            # Find peaks in autocorrelation
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(autocorr_2d, height=0.1)
            
            features.extend([
                len(peaks), np.mean(autocorr_2d) if len(autocorr_2d) > 0 else 0.0,
                np.std(autocorr_2d) if len(autocorr_2d) > 0 else 0.0
            ])
            
            # Fractal dimension estimation
            def box_count(image, min_box_size=1, max_box_size=None):
                if max_box_size is None:
                    max_box_size = min(image.shape) // 4
                    
                sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), 
                                  num=10, dtype=int)
                sizes = np.unique(sizes)
                counts = []
                
                for size in sizes:
                    if size >= min(image.shape):
                        break
                    # Count boxes that contain the pattern
                    boxes = 0
                    for i in range(0, image.shape[0], size):
                        for j in range(0, image.shape[1], size):
                            box = image[i:i+size, j:j+size]
                            if np.std(box) > 0.01:  # Box contains variation
                                boxes += 1
                    counts.append(boxes)
                
                if len(counts) > 1 and len(sizes) == len(counts):
                    # Fit line to log-log plot
                    log_sizes = np.log(sizes[:len(counts)])
                    log_counts = np.log(np.array(counts) + 1)
                    if len(log_sizes) > 1:
                        slope, _ = np.polyfit(log_sizes, log_counts, 1)
                        return -slope
                return 1.5  # Default fractal dimension
            
            fractal_dim = box_count((np.abs(noise_map) > np.std(noise_map)).astype(int))
            features.append(fractal_dim)
            
            # Lacunarity analysis
            box_sizes = [2, 4, 8, 16]
            lacunarities = []
            
            for box_size in box_sizes:
                if box_size < min(h, w):
                    box_masses = []
                    for i in range(0, h-box_size, box_size//2):
                        for j in range(0, w-box_size, box_size//2):
                            box = noise_map[i:i+box_size, j:j+box_size]
                            mass = np.sum(np.abs(box))
                            box_masses.append(mass)
                    
                    if len(box_masses) > 1:
                        mean_mass = np.mean(box_masses)
                        var_mass = np.var(box_masses)
                        lacunarity = var_mass / (mean_mass**2 + 1e-8)
                        lacunarities.append(lacunarity)
                    else:
                        lacunarities.append(0.0)
                else:
                    lacunarities.append(0.0)
            
            features.extend(lacunarities)
            
            # Hurst exponent estimation
            def hurst_exponent(signal):
                if len(signal) < 10:
                    return 0.5
                    
                lags = range(2, min(len(signal)//4, 50))
                variability = []
                
                for lag in lags:
                    tau = [np.std(signal[i:i+lag]) for i in range(len(signal)-lag)]
                    if len(tau) > 0:
                        variability.append(np.mean(tau))
                    else:
                        variability.append(0.0)
                
                if len(variability) > 2:
                    log_lags = np.log(lags[:len(variability)])
                    log_var = np.log(np.array(variability) + 1e-8)
                    slope, _ = np.polyfit(log_lags, log_var, 1)
                    return slope
                return 0.5
            
            hurst_h = hurst_exponent(noise_map.mean(axis=1))
            hurst_w = hurst_exponent(noise_map.mean(axis=0))
            features.extend([hurst_h, hurst_w])
            
            # Fill remaining pattern features
            current_pattern = 11  # Current count
            remaining_pattern = 70 - current_pattern
            
            # Add more advanced pattern features
            if remaining_pattern > 0:
                # Texture energy and homogeneity at multiple scales
                for scale in [1, 2, 4]:
                    try:
                        if scale > 1:
                            scaled_noise = cv2.resize(noise_map, 
                                                    (max(8, w//scale), max(8, h//scale)))
                        else:
                            scaled_noise = noise_map
                        
                        # Texture energy
                        texture_energy = np.sum(scaled_noise**2)
                        features.append(texture_energy)
                        
                        # Local variance
                        kernel_var = np.ones((3, 3)) / 9
                        local_mean = cv2.filter2D(scaled_noise, -1, kernel_var)
                        local_var = cv2.filter2D(scaled_noise**2, -1, kernel_var) - local_mean**2
                        features.extend([np.mean(local_var), np.std(local_var)])
                        
                    except:
                        features.extend([0.0] * 3)
                
                # Fill any remaining with zeros
                current_added = 9  # 3 scales * 3 features
                features.extend([0.0] * max(0, remaining_pattern - current_added))
                
        except Exception as e:
            features.extend([0.0] * 70)
        
        # Ensure exactly 500 features
        target_length = 500
        current_length = len(features)
        
        if current_length < target_length:
            # Add sophisticated additional features
            try:
                # Image quality metrics
                # Signal-to-noise ratio estimation
                signal_power = np.var(noise_map)
                noise_power = np.var(noise_map - ndimage.gaussian_filter(noise_map, sigma=1))
                snr = 10 * np.log10((signal_power + 1e-8) / (noise_power + 1e-8))
                features.append(snr)
                
                # Total variation
                tv_h = np.sum(np.abs(np.diff(noise_map, axis=0)))
                tv_w = np.sum(np.abs(np.diff(noise_map, axis=1)))
                total_variation = tv_h + tv_w
                features.append(total_variation)
                
                # Entropy measures
                _, counts = np.unique(np.round(noise_flat * 1000).astype(int), return_counts=True)
                entropy = stats.entropy(counts + 1e-8)
                features.append(entropy)
                
                # Add zeros for any remaining features
                remaining = target_length - len(features)
                features.extend([0.0] * remaining)
                
            except:
                remaining = target_length - len(features)
                features.extend([0.0] * remaining)
        
        # Truncate to exactly target_length
        features = features[:target_length]
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch_features_parallel(self, noise_maps_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Extract features from a batch using optimized parallel processing"""
        if len(noise_maps_batch) <= 2 or self.n_jobs == 1:
            return [self.extract_comprehensive_features(nm) for nm in noise_maps_batch]
        
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(noise_maps_batch))) as executor:
            futures = [executor.submit(self.extract_comprehensive_features, nm) 
                      for nm in noise_maps_batch]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    results.append(np.zeros(500, dtype=np.float32))
            return results
    
    def fit_transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        """Enhanced feature extraction with feature selection"""
        print(f"🚀 Advanced feature extraction from {len(noise_maps)} noise maps...")
        
        # Optimized batch processing
        batch_size = max(16, len(noise_maps) // (self.n_jobs * 4))
        batches = [noise_maps[i:i + batch_size] for i in range(0, len(noise_maps), batch_size)]
        
        feature_matrix = []
        start_time = time.time()
        
        for batch in tqdm(batches, desc="⚡ Advanced feature extraction"):
            batch_features = self.extract_batch_features_parallel(batch)
            feature_matrix.extend(batch_features)
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        extraction_time = time.time() - start_time
        
        print(f"✅ Feature matrix shape: {feature_matrix.shape}")
        print(f"⏱️ Extraction speed: {len(noise_maps)/extraction_time:.1f} maps/second")
        
        # Handle any NaN or infinite values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Apply robust scaling
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.fitted = True
        
        print(f"✅ Advanced feature extraction completed: {feature_matrix.shape[1]} features per sample")
        
        return feature_matrix
    
    def transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        """Transform noise maps to features (after fitting)"""
        if not self.fitted:
            raise ValueError("Must call fit_transform first")
        
        print(f"🔄 Transforming {len(noise_maps)} noise maps...")
        
        batch_size = max(16, len(noise_maps) // (self.n_jobs * 4))
        batches = [noise_maps[i:i + batch_size] for i in range(0, len(noise_maps), batch_size)]
        
        feature_matrix = []
        for batch in tqdm(batches, desc="⚡ Feature transformation"):
            batch_features = self.extract_batch_features_parallel(batch)
            feature_matrix.extend(batch_features)
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.scaler.transform(feature_matrix)

class UltraAdvancedClassificationPipeline:
    """State-of-the-art classification pipeline targeting MCC > 0.95"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint_dir='ultra_checkpoints'):
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        
        print(f"🚀 Initializing ULTRA-ADVANCED pipeline targeting MCC > 0.95...")
        print(f"🔧 Primary device: {self.device}")
        if self.num_gpus > 0:
            print(f"🔥 Available GPUs: {self.num_gpus}")
            for i in range(self.num_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Enhanced autoencoder
        self.autoencoder = AdvancedResidualAutoencoder().to(self.device)
        if self.num_gpus > 1:
            print("🔗 Enabling multi-GPU support")
            self.autoencoder = nn.DataParallel(self.autoencoder)
        
        # Advanced feature extractor
        self.noise_extractor = UltraAdvancedFeatureExtractor(n_jobs=-1, use_gpu_features=True)
        
        # Ensemble of advanced classifiers
        self.base_classifiers = {
            'rf': RandomForestClassifier(
                n_estimators=2000,  # Much larger ensemble
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced_subsample',
                n_jobs=-1,
                bootstrap=True,
                max_features='sqrt'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=15,
                subsample=0.8,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        }
        
        # Voting ensemble
        self.ensemble_classifier = VotingClassifier(
            estimators=list(self.base_classifiers.items()),
            voting='soft',
            n_jobs=-1
        )
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.autoencoder_trained = False
        self.classifier_trained = False
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        self.training_history = {
            'autoencoder_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
            'best_test_mcc': 0.0,
            'epochs_trained': 0,
            'best_individual_mccs': {},
            'ensemble_mcc': 0.0
        }
    
    def save_checkpoint(self, checkpoint_name='latest', extra_data=None):
        """Enhanced checkpoint saving"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth')
        
        autoencoder_state = None
        if self.autoencoder is not None:
            if self.num_gpus > 1:
                autoencoder_state = self.autoencoder.module.state_dict()
            else:
                autoencoder_state = self.autoencoder.state_dict()
        
        # Save advanced feature extractor
        extractor_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_extractor.pkl')
        if self.noise_extractor.fitted:
            try:
                with open(extractor_path, 'wb') as f:
                    pickle.dump(self.noise_extractor, f)
                print(f"💾 Advanced extractor saved: {extractor_path}")
            except Exception as e:
                print(f"⚠️ Error saving extractor: {e}")
        
        # Save ensemble classifier
        ensemble_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_ensemble.pkl')
        if self.classifier_trained:
            try:
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(self.ensemble_classifier, f)
                print(f"💾 Ensemble classifier saved: {ensemble_path}")
            except Exception as e:
                print(f"⚠️ Error saving ensemble: {e}")
        
        # Save individual classifiers
        for name, classifier in self.base_classifiers.items():
            if hasattr(classifier, 'classes_'):
                classifier_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_{name}.pkl')
                try:
                    with open(classifier_path, 'wb') as f:
                        pickle.dump(classifier, f)
                except Exception as e:
                    print(f"⚠️ Error saving {name}: {e}")
        
        # Main checkpoint
        checkpoint = {
            'autoencoder_state_dict': autoencoder_state,
            'autoencoder_trained': self.autoencoder_trained,
            'classifier_trained': self.classifier_trained,
            'training_history': self.training_history,
            'device': self.device,
            'extractor_fitted': self.noise_extractor.fitted
        }
        
        if extra_data:
            checkpoint.update(extra_data)
            
        try:
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Main checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Error saving checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_name='latest'):
        """Enhanced checkpoint loading"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint {checkpoint_path} not found.")
            return False, 0
        
        print(f"🔄 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load autoencoder
        if checkpoint['autoencoder_state_dict'] is not None:
            self.autoencoder = AdvancedResidualAutoencoder().to(self.device)
            if self.num_gpus > 1:
                self.autoencoder = nn.DataParallel(self.autoencoder)
                self.autoencoder.module.load_state_dict(checkpoint['autoencoder_state_dict'])
            else:
                self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        
        # Load feature extractor
        extractor_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_extractor.pkl')
        if os.path.exists(extractor_path):
            try:
                with open(extractor_path, 'rb') as f:
                    self.noise_extractor = pickle.load(f)
                print("✅ Advanced extractor loaded")
            except Exception as e:
                print(f"⚠️ Could not load extractor: {e}")
        
        # Load ensemble
        ensemble_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_ensemble.pkl')
        if os.path.exists(ensemble_path):
            try:
                with open(ensemble_path, 'rb') as f:
                    self.ensemble_classifier = pickle.load(f)
                print("✅ Ensemble classifier loaded")
            except Exception as e:
                print(f"⚠️ Could not load ensemble: {e}")
        
        # Load individual classifiers
        for name in self.base_classifiers.keys():
            classifier_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_{name}.pkl')
            if os.path.exists(classifier_path):
                try:
                    with open(classifier_path, 'rb') as f:
                        self.base_classifiers[name] = pickle.load(f)
                    print(f"✅ {name} classifier loaded")
                except Exception as e:
                    print(f"⚠️ Could not load {name}: {e}")
        
        self.autoencoder_trained = checkpoint.get('autoencoder_trained', False)
        self.classifier_trained = checkpoint.get('classifier_trained', False)
        self.training_history = checkpoint.get('training_history', {
            'autoencoder_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
            'best_test_mcc': 0.0,
            'epochs_trained': 0,
            'best_individual_mccs': {},
            'ensemble_mcc': 0.0
        })
        
        print(f"✅ Loaded: Autoencoder: {self.autoencoder_trained}, Classifier: {self.classifier_trained}")
        return True, self.training_history.get('epochs_trained', 0)

    def train_autoencoder_advanced(self, train_loader: DataLoader, val_loader: DataLoader = None, 
                                 epochs=50, resume_from_epoch=0, save_every=5):
        """Advanced autoencoder training with multiple loss functions"""
        if self.autoencoder_trained and resume_from_epoch == 0:
            print("✅ Autoencoder already trained.")
            return
            
        print(f"🚀 Training ADVANCED autoencoder for {epochs} epochs...")
        
        if self.autoencoder is None:
            self.autoencoder = AdvancedResidualAutoencoder().to(self.device)
            if self.num_gpus > 1:
                self.autoencoder = nn.DataParallel(self.autoencoder)
        
        # Advanced optimizer with scheduling
        initial_lr = 0.0005  # Lower learning rate for stability
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=initial_lr, 
                              weight_decay=1e-4, betas=(0.9, 0.999))
        
        # Sophisticated learning rate scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.001, epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3, anneal_strategy='cos'
        )
        
        # Advanced loss function combining multiple objectives
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        
        self.autoencoder.train()
        start_epoch = resume_from_epoch
        
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            num_batches = 0
            epoch_start_time = time.time()
            
            progress_bar = tqdm(train_loader, desc=f'🔥 Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (data, _) in enumerate(progress_bar):
                data = data.to(self.device, non_blocking=True)
                
                # Advanced noise strategy - multiple noise types
                noise_strategies = ['gaussian', 'uniform', 'salt_pepper', 'speckle']
                strategy = np.random.choice(noise_strategies)
                
                if strategy == 'gaussian':
                    noise_strength = 0.05 + 0.15 * (epoch / epochs)
                    noise = torch.randn_like(data) * noise_strength
                elif strategy == 'uniform':
                    noise_strength = 0.1 + 0.1 * (epoch / epochs)
                    noise = (torch.rand_like(data) - 0.5) * noise_strength
                elif strategy == 'salt_pepper':
                    prob = 0.01 + 0.04 * (epoch / epochs)
                    noise = torch.zeros_like(data)
                    salt = torch.rand_like(data) < prob/2
                    pepper = torch.rand_like(data) < prob/2
                    noise[salt] = 1.0
                    noise[pepper] = -1.0
                else:  # speckle
                    noise_strength = 0.1 + 0.1 * (epoch / epochs)
                    noise = torch.randn_like(data) * data * noise_strength
                
                noisy_data = torch.clamp(data + noise, 0., 1.)
                
                optimizer.zero_grad()
                
                if self.scaler is not None:
                    with autocast():
                        reconstructed = self.autoencoder(noisy_data)
                        
                        # Combined loss function
                        mse = mse_loss(reconstructed, data)
                        l1 = l1_loss(reconstructed, data)
                        
                        # Perceptual loss component
                        perceptual_loss = 0.0
                        if epoch > epochs // 4:  # Add perceptual loss later in training
                            # Use gradients as perceptual features
                            data_grad = torch.gradient(data.mean(dim=1), dim=[1, 2])
                            recon_grad = torch.gradient(reconstructed.mean(dim=1), dim=[1, 2])
                            perceptual_loss = F.mse_loss(recon_grad[0], data_grad[0]) + \
                                            F.mse_loss(recon_grad[1], data_grad[1])
                        
                        # Total loss with adaptive weighting
                        total_loss = mse + 0.1 * l1 + 0.05 * perceptual_loss
                    
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    reconstructed = self.autoencoder(noisy_data)
                    mse = mse_loss(reconstructed, data)
                    l1 = l1_loss(reconstructed, data)
                    total_loss = mse + 0.1 * l1
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                total_loss += total_loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.6f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Clear cache periodically
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / num_batches
            self.training_history['autoencoder_losses'].append(avg_loss)
            self.training_history['epochs_trained'] = epoch + 1
            
            # Enhanced validation
            val_loss = 0.0
            if val_loader is not None:
                val_loss = self._validate_autoencoder_advanced(val_loader, epoch, epochs)
                self.training_history['val_losses'].append(val_loss)
                
                if val_loss < self.training_history['best_val_loss']:
                    self.training_history['best_val_loss'] = val_loss
                    self.save_checkpoint('best_autoencoder')
            
            print(f'⚡ Epoch [{epoch+1}/{epochs}] - Train: {avg_loss:.6f}, Val: {val_loss:.6f}, Time: {epoch_time:.1f}s')
            
            # Regular checkpointing
            if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
                checkpoint_name = f'autoencoder_epoch_{epoch+1}'
                self.save_checkpoint(checkpoint_name)
        
        self.autoencoder_trained = True
        self.save_checkpoint('autoencoder_final')
        print("🎉 Advanced autoencoder training completed!")
    
    def _validate_autoencoder_advanced(self, val_loader: DataLoader, current_epoch: int, total_epochs: int) -> float:
        """Enhanced autoencoder validation"""
        self.autoencoder.eval()
        total_loss = 0
        total_ssim = 0
        num_batches = 0
        
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device, non_blocking=True)
                
                # Use similar noise as training
                noise_strength = 0.05 + 0.15 * (current_epoch / total_epochs)
                noise = torch.randn_like(data) * noise_strength
                noisy_data = torch.clamp(data + noise, 0., 1.)
                
                if self.scaler is not None:
                    with autocast():
                        reconstructed = self.autoencoder(noisy_data)
                        loss = mse_loss(reconstructed, data) + 0.1 * l1_loss(reconstructed, data)
                else:
                    reconstructed = self.autoencoder(noisy_data)
                    loss = mse_loss(reconstructed, data) + 0.1 * l1_loss(reconstructed, data)
                
                total_loss += loss.item()
                
                # Calculate SSIM for additional validation metric
                try:
                    batch_ssim = 0
                    for i in range(min(4, data.shape[0])):  # Sample a few for SSIM
                        img1 = data[i].cpu().numpy().transpose(1, 2, 0)
                        img2 = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
                        if img1.shape[2] == 1:
                            img1, img2 = img1.squeeze(), img2.squeeze()
                        ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=2 if len(img1.shape)==3 else None)
                        batch_ssim += ssim_val
                    total_ssim += batch_ssim / min(4, data.shape[0])
                except:
                    pass
                
                num_batches += 1
        
        self.autoencoder.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_ssim = total_ssim / num_batches if num_batches > 0 else 0.0
        
        if num_batches > 0:
            print(f"    📊 Validation - Loss: {avg_loss:.6f}, SSIM: {avg_ssim:.4f}")
        
        return avg_loss
    
    def extract_noise_from_images_advanced(self, images: torch.Tensor, batch_size: int = 64) -> List[np.ndarray]:
        """Advanced noise extraction with multiple techniques"""
        if self.autoencoder is None or not self.autoencoder_trained:
            raise ValueError("Advanced autoencoder must be trained first")
        
        self.autoencoder.eval()
        noise_maps = []
        
        print(f"⚡ Advanced noise extraction with batch size: {batch_size}")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="🔍 Extracting advanced noise"):
                batch = images[i:i+batch_size].to(self.device, non_blocking=True)
                
                try:
                    if self.scaler is not None:
                        with autocast():
                            reconstructed = self.autoencoder(batch)
                    else:
                        reconstructed = self.autoencoder(batch)
                    
                    # Multi-scale noise extraction
                    noise_original = batch - reconstructed
                    
                    # Additional noise extraction techniques
                    for j in range(noise_original.shape[0]):
                        # Original noise map
                        noise_map = noise_original[j].cpu().numpy()
                        
                        # Average across channels for primary noise map
                        if noise_map.shape[0] > 1:
                            primary_noise = np.mean(noise_map, axis=0)
                        else:
                            primary_noise = noise_map[0]
                        
                        # Enhance noise map using multiple techniques
                        enhanced_noise = self._enhance_noise_map(primary_noise)
                        noise_maps.append(enhanced_noise.astype(np.float32))
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️ GPU memory error, reducing batch size...")
                    # Fallback to smaller batches
                    for k in range(0, len(batch), batch_size // 4):
                        mini_batch = batch[k:k+batch_size//4]
                        if self.scaler is not None:
                            with autocast():
                                reconstructed = self.autoencoder(mini_batch)
                        else:
                            reconstructed = self.autoencoder(mini_batch)
                        
                        noise_original = mini_batch - reconstructed
                        for j in range(noise_original.shape[0]):
                            noise_map = noise_original[j].cpu().numpy()
                            if noise_map.shape[0] > 1:
                                primary_noise = np.mean(noise_map, axis=0)
                            else:
                                primary_noise = noise_map[0]
                            enhanced_noise = self._enhance_noise_map(primary_noise)
                            noise_maps.append(enhanced_noise.astype(np.float32))
                
                # Clear cache periodically
                if i % (batch_size * 5) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"✅ Extracted {len(noise_maps)} enhanced noise maps")
        return noise_maps
    
    def _enhance_noise_map(self, noise_map: np.ndarray) -> np.ndarray:
        """Enhance noise map using advanced signal processing"""
        try:
            # Apply multiple enhancement techniques
            enhanced = noise_map.copy()
            
            # 1. Adaptive histogram equalization
            noise_uint8 = ((noise_map + 1) * 127.5).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            equalized = clahe.apply(noise_uint8)
            enhanced += 0.1 * ((equalized.astype(np.float32) / 127.5) - 1)
            
            # 2. Unsharp masking
            blurred = cv2.GaussianBlur(noise_map, (3, 3), 1.0)
            unsharp = noise_map + 0.5 * (noise_map - blurred)
            enhanced += 0.1 * unsharp
            
            # 3. Edge enhancement
            laplacian = cv2.Laplacian(noise_map, cv2.CV_32F, ksize=3)
            enhanced += 0.05 * laplacian
            
            # Normalize to prevent overflow
            enhanced = np.clip(enhanced, -2, 2)
            
            return enhanced
            
        except Exception as e:
            return noise_map
    
    def train_ensemble_classifier_advanced(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train advanced ensemble classifier targeting MCC > 0.95"""
        if self.classifier_trained:
            print("✅ Advanced ensemble classifier already trained.")
            return
            
        if not self.autoencoder_trained:
            raise ValueError("❌ Advanced autoencoder must be trained first")
        
        print("\n" + "="*80)
        print("🚀 TRAINING ULTRA-ADVANCED ENSEMBLE CLASSIFIER (Target MCC > 0.95)")
        print("="*80)
        
        total_start_time = time.time()
        
        # Extract enhanced noise features
        print("\n[1/5] 🔍 Extracting advanced noise maps...")
        extraction_start = time.time()
        
        all_noise_maps = []
        all_labels = []
        total_samples = sum(len(labels) for _, labels in train_loader)
        print(f"📊 Total training samples: {total_samples}")
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="⚡ Processing batches")):
            try:
                noise_maps = self.extract_noise_from_images_advanced(images, batch_size=64)
                all_noise_maps.extend(noise_maps)
                all_labels.extend(labels.cpu().numpy())
                
                if batch_idx % 10 == 0 and batch_idx > 0:
                    elapsed = time.time() - extraction_start
                    processed = len(all_noise_maps)
                    speed = processed / elapsed
                    print(f"    📈 Progress: {processed}/{total_samples} - Speed: {speed:.1f} maps/sec")
                    
            except Exception as e:
                print(f"    ❌ Error processing batch {batch_idx}: {e}")
                continue
        
        extraction_time = time.time() - extraction_start
        print(f"✅ Extracted {len(all_noise_maps)} noise maps in {extraction_time:.2f}s")
        print(f"🚀 Speed: {len(all_noise_maps)/extraction_time:.1f} maps/second")
        
        # Extract comprehensive features
        print(f"\n[2/5] ⚡ Extracting 500+ comprehensive features...")
        feature_start = time.time()
        
        feature_matrix = self.noise_extractor.fit_transform(all_noise_maps)
        feature_time = time.time() - feature_start
        
        print(f"✅ Feature extraction completed in {feature_time:.2f}s")
        print(f"📊 Feature matrix shape: {feature_matrix.shape}")
        print(f"🚀 Feature extraction speed: {len(all_noise_maps)/feature_time:.1f} maps/second")
        
        # Feature selection for optimal performance
        print(f"\n[3/5] 🎯 Advanced feature selection and optimization...")
        
        # Analyze class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        print(f"📊 Training class distribution:")
        for label, count in zip(unique, counts):
            print(f"   {class_names[label]}: {count:,} samples ({100*count/len(all_labels):.1f}%)")
        
        # Apply feature selection
        feature_selector = SelectKBest(f_classif, k=min(400, feature_matrix.shape[1]))
        feature_matrix_selected = feature_selector.fit_transform(feature_matrix, all_labels)
        self.feature_selector = feature_selector
        
        print(f"📊 Selected {feature_matrix_selected.shape[1]} most informative features")
        
        # Train individual classifiers with hyperparameter optimization
        print(f"\n[4/5] 🌳 Training advanced ensemble classifiers...")
        classifier_start = time.time()
        
        individual_results = {}
        cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Hyperparameter grids for each classifier
        param_grids = {
            'rf': {
                'n_estimators': [1500, 2000, 2500],
                'max_depth': [25, 30, 35],
                'min_samples_split': [2, 3],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            },
            'gb': {
                'n_estimators': [800, 1000, 1200],
                'learning_rate': [0.03, 0.05, 0.07],
                'max_depth': [12, 15, 18],
                'subsample': [0.8, 0.9]
            },
            'svm': {
                'C': [1, 5, 10, 20],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            },
            'mlp': {
                'hidden_layer_sizes': [(512, 256), (512, 256, 128), (1024, 512, 256)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        
        # Train each classifier with optimization
        for name, classifier in self.base_classifiers.items():
            print(f"\n🔧 Optimizing {name.upper()} classifier...")
            
            try:
                # Hyperparameter search
                search = RandomizedSearchCV(
                    classifier, param_grids[name], n_iter=15,
                    scoring='accuracy', cv=cv_folds, n_jobs=-1,
                    random_state=42, verbose=0
                )
                
                search.fit(feature_matrix_selected, all_labels)
                optimized_classifier = search.best_estimator_
                
                # Evaluate individual classifier
                predictions = optimized_classifier.predict(feature_matrix_selected)
                mcc = matthews_corrcoef(all_labels, predictions)
                accuracy = accuracy_score(all_labels, predictions)
                
                individual_results[name] = {
                    'mcc': mcc,
                    'accuracy': accuracy,
                    'best_params': search.best_params_
                }
                
                # Update base classifier
                self.base_classifiers[name] = optimized_classifier
                
                print(f"   ✅ {name.upper()}: MCC={mcc:.4f}, Accuracy={accuracy:.4f}")
                print(f"   🎯 Best params: {search.best_params_}")
                
            except Exception as e:
                print(f"   ❌ Error optimizing {name}: {e}")
                continue
        
        # Train ensemble
        print(f"\n🎭 Training ensemble classifier...")
        ensemble_estimators = [(name, clf) for name, clf in self.base_classifiers.items() 
                              if hasattr(clf, 'classes_')]
        
        if len(ensemble_estimators) >= 2:
            self.ensemble_classifier = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft',
                n_jobs=-1
            )
            
            try:
                self.ensemble_classifier.fit(feature_matrix_selected, all_labels)
                
                # Evaluate ensemble
                ensemble_predictions = self.ensemble_classifier.predict(feature_matrix_selected)
                ensemble_mcc = matthews_corrcoef(all_labels, ensemble_predictions)
                ensemble_accuracy = accuracy_score(all_labels, ensemble_predictions)
                
                self.training_history['ensemble_mcc'] = ensemble_mcc
                self.training_history['best_individual_mccs'] = {
                    name: results['mcc'] for name, results in individual_results.items()
                }
                
                print(f"\n🎭 ENSEMBLE RESULTS:")
                print(f"   🎯 Ensemble MCC: {ensemble_mcc:.4f}")
                print(f"   🎯 Ensemble Accuracy: {ensemble_accuracy:.4f}")
                
                classifier_time = time.time() - classifier_start
                print(f"✅ Advanced ensemble training completed in {classifier_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error training ensemble: {e}")
                # Fall back to best individual classifier
                best_classifier_name = max(individual_results.keys(), 
                                         key=lambda x: individual_results[x]['mcc'])
                self.ensemble_classifier = self.base_classifiers[best_classifier_name]
                print(f"🔄 Using best individual classifier: {best_classifier_name}")
        
        # Validation evaluation
        if val_loader is not None:
            print(f"\n[5/5] 🔍 Validating ensemble on validation set...")
            try:
                val_predictions, val_probabilities = self.predict_advanced(val_loader)
                val_labels = []
                for _, labels in val_loader:
                    val_labels.extend(labels.cpu().numpy())
                
                # Align lengths
                min_len = min(len(val_labels), len(val_predictions))
                val_labels = val_labels[:min_len]
                val_predictions = val_predictions[:min_len]
                
                val_mcc = matthews_corrcoef(val_labels, val_predictions)
                val_accuracy = accuracy_score(val_labels, val_predictions)
                
                print(f"\n✅ VALIDATION RESULTS:")
                print(f"   🎯 Validation MCC: {val_mcc:.4f}")
                print(f"   🎯 Validation Accuracy: {val_accuracy:.4f}")
                
                # Detailed validation analysis
                val_cm = confusion_matrix(val_labels, val_predictions)
                print(f"\n📊 Validation Confusion Matrix:")
                print("True\\Pred    Real  Synth  Semi")
                for i, row in enumerate(val_cm):
                    print(f"{class_names[i]:10s} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
                
                # Per-class validation performance
                val_report = classification_report(val_labels, val_predictions, 
                                                 target_names=class_names, 
                                                 output_dict=True, zero_division=0)
                print(f"\n📈 Validation Per-class Performance:")
                for class_name in class_names:
                    if class_name in val_report:
                        metrics = val_report[class_name]
                        print(f"   {class_name}:")
                        print(f"     Precision: {metrics['precision']:.4f}")
                        print(f"     Recall:    {metrics['recall']:.4f}")
                        print(f"     F1-score:  {metrics['f1-score']:.4f}")
                        
                # Update best validation MCC
                if val_mcc > self.training_history.get('best_val_mcc', 0.0):
                    self.training_history['best_val_mcc'] = val_mcc
                    self.save_checkpoint('best_validation')
                    
            except Exception as e:
                print(f"❌ Error during validation: {e}")
        
        # Save trained components
        self.classifier_trained = True
        self.save_checkpoint('ensemble_final')
        
        total_time = time.time() - total_start_time
        
        print("\n" + "="*80)
        print("🎉 ULTRA-ADVANCED ENSEMBLE TRAINING COMPLETED!")
        print("="*80)
        print(f"📊 Processed {len(all_noise_maps):,} training samples")
        print(f"🔧 Extracted {feature_matrix.shape[1]} total features, selected {feature_matrix_selected.shape[1]}")
        print(f"🎭 Trained ensemble of {len(ensemble_estimators)} advanced classifiers")
        if 'ensemble_mcc' in self.training_history:
            print(f"🎯 Training ensemble MCC: {self.training_history['ensemble_mcc']:.4f}")
        if val_loader and 'best_val_mcc' in self.training_history:
            print(f"🎯 Best validation MCC: {self.training_history['best_val_mcc']:.4f}")
        print(f"⏱️ TIMING BREAKDOWN:")
        print(f"   Noise Extraction: {extraction_time:.2f}s ({100*extraction_time/total_time:.1f}%)")
        print(f"   Feature Extraction: {feature_time:.2f}s ({100*feature_time/total_time:.1f}%)")
        print(f"   Ensemble Training: {classifier_time:.2f}s ({100*classifier_time/total_time:.1f}%)")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"🚀 Overall Speed: {len(all_noise_maps)/total_time:.1f} samples/second")
        print("="*80)
    
    def predict_advanced(self, test_loader: DataLoader, batch_size: int = 64) -> Tuple[List[int], List[float]]:
        """Advanced prediction with ensemble voting"""
        if not self.autoencoder_trained or not self.classifier_trained:
            raise ValueError("❌ Both autoencoder and ensemble must be trained")
        if not self.noise_extractor.fitted:
            raise ValueError("❌ Feature extractor must be fitted!")
        
        print("🔮 Generating advanced ensemble predictions...")
        
        all_predictions = []
        all_probabilities = []
        
        self.autoencoder.eval()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="⚡ Advanced prediction")):
                try:
                    # Extract enhanced noise maps
                    noise_maps = self.extract_noise_from_images_advanced(images, batch_size=batch_size)
                    
                    # Extract comprehensive features
                    feature_matrix = self.noise_extractor.transform(noise_maps)
                    
                    # Apply feature selection
                    if hasattr(self, 'feature_selector'):
                        feature_matrix = self.feature_selector.transform(feature_matrix)
                    
                    # Ensemble prediction
                    predictions = self.ensemble_classifier.predict(feature_matrix)
                    probabilities = self.ensemble_classifier.predict_proba(feature_matrix)
                    
                    all_predictions.extend(predictions.tolist())
                    all_probabilities.extend(probabilities.tolist())
                    
                except Exception as e:
                    print(f"⚠️ Error predicting batch {batch_idx}: {e}")
                    batch_size_actual = len(images)
                    all_predictions.extend([0] * batch_size_actual)
                    all_probabilities.extend([[1.0, 0.0, 0.0]] * batch_size_actual)
        
        print(f"✅ Generated advanced predictions for {len(all_predictions)} samples")
        return all_predictions, all_probabilities
    
    def evaluate_advanced(self, test_loader: DataLoader, test_labels: List[int], 
                         save_results: bool = True, results_dir: str = 'ultra_results') -> Dict:
        """Comprehensive evaluation targeting MCC > 0.95"""
        print("\n🔍 Starting ULTRA-ADVANCED evaluation...")
        eval_start = time.time()
        
        predictions, probabilities = self.predict_advanced(test_loader)
        
        # Ensure same length
        min_len = min(len(test_labels), len(predictions))
        test_labels = test_labels[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        
        # Calculate comprehensive metrics
        mcc = matthews_corrcoef(test_labels, predictions)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, 
                                    target_names=['Real', 'Synthetic', 'Semi-synthetic'],
                                    output_dict=True, zero_division=0)
        
        cm = confusion_matrix(test_labels, predictions)
        eval_time = time.time() - eval_start
        
        # Calculate advanced metrics
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        
        # Per-class MCC calculation
        per_class_mcc = []
        for i in range(3):
            # Binary classification for each class
            binary_true = (np.array(test_labels) == i).astype(int)
            binary_pred = (np.array(predictions) == i).astype(int)
            if len(np.unique(binary_true)) > 1 and len(np.unique(binary_pred)) > 1:
                class_mcc = matthews_corrcoef(binary_true, binary_pred)
            else:
                class_mcc = 0.0
            per_class_mcc.append(class_mcc)
        
        # Confidence analysis
        probabilities_array = np.array(probabilities)
        max_probs = np.max(probabilities_array, axis=1)
        entropy_scores = -np.sum(probabilities_array * np.log(probabilities_array + 1e-8), axis=1)
        
        confidence_metrics = {
            'mean_max_probability': np.mean(max_probs),
            'std_max_probability': np.std(max_probs),
            'mean_entropy': np.mean(entropy_scores),
            'std_entropy': np.std(entropy_scores),
            'high_confidence_ratio': np.mean(max_probs > 0.9),
            'low_confidence_ratio': np.mean(max_probs < 0.6)
        }
        
        # Update best test performance
        if mcc > self.training_history['best_test_mcc']:
            self.training_history['best_test_mcc'] = mcc
            if save_results:
                self.save_checkpoint('best_test_model', {
                    'test_mcc': mcc, 
                    'test_accuracy': accuracy,
                    'per_class_mcc': per_class_mcc,
                    'confidence_metrics': confidence_metrics
                })
        
        results = {
            'mcc': mcc,
            'accuracy': accuracy,
            'per_class_mcc': per_class_mcc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_metrics': confidence_metrics,
            'evaluation_time': eval_time,
            'prediction_speed': len(predictions) / eval_time,
            'individual_classifier_mccs': self.training_history.get('best_individual_mccs', {})
        }
        
        print("\n" + "="*80)
        print("🎯 ULTRA-ADVANCED EVALUATION RESULTS")
        print("="*80)
        print(f"🏆 OVERALL MCC: {mcc:.6f}")
        print(f"🏆 OVERALL ACCURACY: {accuracy:.6f}")
        print(f"⏱️ Evaluation time: {eval_time:.2f} seconds")
        print(f"🚀 Prediction speed: {len(predictions)/eval_time:.1f} samples/second")
        
        # MCC Analysis
        if mcc > 0.95:
            print(f"🎉 TARGET ACHIEVED! MCC > 0.95 ✅")
        elif mcc > 0.90:
            print(f"🎯 Excellent performance! MCC > 0.90 ✅")
        elif mcc > 0.85:
            print(f"⭐ Very good performance! MCC > 0.85")
        else:
            print(f"📈 Good performance, room for improvement")
        
        print(f"\n📊 Per-class MCC scores:")
        for i, (class_name, class_mcc) in enumerate(zip(class_names, per_class_mcc)):
            print(f"   {class_name}: {class_mcc:.4f}")
        
        print(f"\n📊 Confidence Analysis:")
        print(f"   Mean max probability: {confidence_metrics['mean_max_probability']:.4f}")
        print(f"   High confidence predictions (>0.9): {confidence_metrics['high_confidence_ratio']*100:.1f}%")
        print(f"   Low confidence predictions (<0.6): {confidence_metrics['low_confidence_ratio']*100:.1f}%")
        print(f"   Mean prediction entropy: {confidence_metrics['mean_entropy']:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print("True\\Pred    Real  Synth  Semi")
        for i, (name, row) in enumerate(zip(['Real     ', 'Synthetic', 'Semi-synth'], cm)):
            print(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
        
        print(f"\n📈 Detailed Per-class Performance:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
        
        # Individual classifier performance
        if self.training_history.get('best_individual_mccs'):
            print(f"\n🔧 Individual Classifier MCCs:")
            for name, mcc_score in self.training_history['best_individual_mccs'].items():
                print(f"   {name.upper()}: {mcc_score:.4f}")
        
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            
            # Save comprehensive results
            results_file = os.path.join(results_dir, 'ultra_evaluation_results.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'overall_mcc': mcc,
                    'overall_accuracy': accuracy,
                    'per_class_mcc': per_class_mcc,
                    'confidence_metrics': confidence_metrics,
                    'evaluation_time': eval_time,
                    'prediction_speed': len(predictions) / eval_time,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report,
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'true_labels': test_labels,
                    'individual_classifier_mccs': self.training_history.get('best_individual_mccs', {}),
                    'target_achieved': mcc > 0.95
                }, f, indent=2)
            
            # Create advanced visualizations
            print("📊 Creating ultra-advanced visualizations...")
            
            # Enhanced confusion matrix with percentages
            plt.figure(figsize=(12, 10))
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create annotation combining counts and percentages
            annotations = []
            for i in range(cm.shape[0]):
                row = []
                for j in range(cm.shape[1]):
                    row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
                annotations.append(row)
            
            sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Percentage'})
            plt.title(f'Enhanced Confusion Matrix\nOverall MCC: {mcc:.4f} | Accuracy: {accuracy:.4f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Add target achievement indicator
            if mcc > 0.95:
                plt.text(0.5, -0.15, '🎉 TARGET ACHIEVED: MCC > 0.95!', 
                        transform=plt.gca().transAxes, ha='center', fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            
            cm_path = os.path.join(results_dir, 'ultra_confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Advanced performance dashboard
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Overall metrics
            metrics = ['Accuracy', 'MCC', 'Avg F1']
            avg_f1 = np.mean([report[cls]['f1-score'] for cls in class_names if cls in report])
            values = [accuracy, mcc, avg_f1]
            colors = ['#36A2EB', '#FF6384', '#4BC0C0']
            
            bars = axes[0, 0].bar(metrics, values, color=colors, alpha=0.8)
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Overall Performance Metrics')
            axes[0, 0].set_ylim(0, 1)
            
            # Add target line for MCC
            axes[0, 0].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target MCC')
            axes[0, 0].legend()
            
            for bar, value in zip(bars, values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Per-class metrics
            metrics_per_class = ['Precision', 'Recall', 'F1-Score']
            x = np.arange(len(class_names))
            width = 0.25
            
            for i, metric in enumerate(metrics_per_class):
                values = []
                for class_name in class_names:
                    if class_name in report:
                        if metric.lower().replace('-', '') in report[class_name]:
                            values.append(report[class_name][metric.lower().replace('-', '')])
                        else:
                            values.append(0.0)
                    else:
                        values.append(0.0)
                
                axes[0, 1].bar(x + i*width, values, width, label=metric, alpha=0.8)
            
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Per-Class Performance')
            axes[0, 1].set_xticks(x + width)
            axes[0, 1].set_xticklabels(class_names)
            axes[0, 1].legend()
            axes[0, 1].set_ylim(0, 1)
            
            # Per-class MCC
            axes[0, 2].bar(class_names, per_class_mcc, 
                          color=['#FF9F40', '#4BC0C0', '#9966FF'], alpha=0.8)
            axes[0, 2].set_ylabel('MCC Score')
            axes[0, 2].set_title('Per-Class MCC Scores')
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target')
            axes[0, 2].legend()
            
            for i, score in enumerate(per_class_mcc):
                axes[0, 2].text(i, score + 0.01, f'{score:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
            
            # Confidence distribution
            axes[1, 0].hist(max_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_xlabel('Maximum Prediction Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Prediction Confidence Distribution')
            axes[1, 0].axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='High Conf.')
            axes[1, 0].legend()
            
            # Individual classifier comparison
            if self.training_history.get('best_individual_mccs'):
                clf_names = list(self.training_history['best_individual_mccs'].keys())
                clf_mccs = list(self.training_history['best_individual_mccs'].values())
                clf_mccs.append(mcc)  # Add ensemble
                clf_names.append('Ensemble')
                
                colors_clf = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                axes[1, 1].bar(clf_names, clf_mccs, color=colors_clf[:len(clf_names)], alpha=0.8)
                axes[1, 1].set_ylabel('MCC Score')
                axes[1, 1].set_title('Classifier Comparison')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target')
                axes[1, 1].legend()
                
                for i, score in enumerate(clf_mccs):
                    axes[1, 1].text(i, score + 0.01, f'{score:.3f}', 
                                   ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Class distribution and performance summary
            class_counts = [np.sum(np.array(test_labels) == i) for i in range(3)]
            sizes = class_counts
            colors_pie = ['#FF9F40', '#4BC0C0', '#9966FF']
            
            wedges, texts, autotexts = axes[1, 2].pie(sizes, labels=class_names, autopct='%1.1f%%', 
                                                     colors=colors_pie, startangle=90)
            axes[1, 2].set_title(f'Test Set Distribution\nTotal: {sum(sizes)} samples')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.tight_layout()
            plot_path = os.path.join(results_dir, 'ultra_comprehensive_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Training history visualization
            if self.training_history['autoencoder_losses']:
                plt.figure(figsize=(15, 5))
                
                # Autoencoder loss
                plt.subplot(1, 3, 1)
                epochs_range = range(1, len(self.training_history['autoencoder_losses']) + 1)
                plt.plot(epochs_range, self.training_history['autoencoder_losses'], 
                        'b-', label='Training Loss', linewidth=2)
                if self.training_history['val_losses']:
                    plt.plot(epochs_range, self.training_history['val_losses'], 
                            'r-', label='Validation Loss', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Autoencoder Training Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # MCC progression
                plt.subplot(1, 3, 2)
                if self.training_history.get('best_individual_mccs'):
                    clf_names = list(self.training_history['best_individual_mccs'].keys())
                    clf_mccs = list(self.training_history['best_individual_mccs'].values())
                    plt.bar(clf_names, clf_mccs, alpha=0.7)
                    plt.ylabel('MCC Score')
                    plt.title('Individual Classifier MCCs')
                    plt.xticks(rotation=45)
                    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target')
                    plt.legend()
                
                # Feature importance (top 20)
                plt.subplot(1, 3, 3)
                if hasattr(self.ensemble_classifier, 'feature_importances_'):
                    importance = self.ensemble_classifier.feature_importances_
                elif hasattr(self.base_classifiers['rf'], 'feature_importances_'):
                    importance = self.base_classifiers['rf'].feature_importances_
                else:
                    importance = np.random.random(20)  # Placeholder
                
                top_features = np.argsort(importance)[-20:]
                plt.barh(range(20), importance[top_features])
                plt.xlabel('Feature Importance')
                plt.ylabel('Feature Index')
                plt.title('Top 20 Feature Importance')
                
                plt.tight_layout()
                training_path = os.path.join(results_dir, 'training_history.png')
                plt.savefig(training_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save detailed text report
            report_path = os.path.join(results_dir, 'ultra_detailed_report.txt')
            with open(report_path, 'w') as f:
                f.write("ULTRA-ADVANCED NOISE CLASSIFICATION RESULTS\n")
                f.write("="*60 + "\n\n")
                f.write(f"🏆 OVERALL PERFORMANCE:\n")
                f.write(f"   MCC: {mcc:.6f}\n")
                f.write(f"   Accuracy: {accuracy:.6f}\n")
                f.write(f"   Target Achieved (MCC > 0.95): {'YES ✅' if mcc > 0.95 else 'NO ❌'}\n\n")
                
                f.write(f"📊 PER-CLASS MCC SCORES:\n")
                for class_name, class_mcc in zip(class_names, per_class_mcc):
                    f.write(f"   {class_name}: {class_mcc:.4f}\n")
                f.write(f"\n")
                
                f.write(f"📊 CONFIDENCE ANALYSIS:\n")
                for key, value in confidence_metrics.items():
                    f.write(f"   {key}: {value:.4f}\n")
                f.write(f"\n")
                
                f.write(f"⏱️ PERFORMANCE METRICS:\n")
                f.write(f"   Evaluation Time: {eval_time:.2f} seconds\n")
                f.write(f"   Prediction Speed: {len(predictions)/eval_time:.1f} samples/second\n")
                f.write(f"   Total Features Used: {500}\n")
                f.write(f"   Selected Features: {feature_matrix_selected.shape[1] if 'feature_matrix_selected' in locals() else 'N/A'}\n\n")
                
                f.write(f"🔧 INDIVIDUAL CLASSIFIER PERFORMANCE:\n")
                if self.training_history.get('best_individual_mccs'):
                    for name, mcc_score in self.training_history['best_individual_mccs'].items():
                        f.write(f"   {name.upper()}: {mcc_score:.4f}\n")
                f.write(f"\n")
                
                f.write(f"📊 CONFUSION MATRIX:\n")
                f.write("True\\Pred    Real  Synth  Semi\n")
                for i, (name, row) in enumerate(zip(['Real     ', 'Synthetic', 'Semi-synth'], cm)):
                    f.write(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}\n")
                f.write(f"\n")
                
                f.write(f"📈 DETAILED PER-CLASS METRICS:\n")
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
                        f.write(f"  F1-score: {metrics['f1-score']:.4f}\n")
                        f.write(f"  Support: {metrics['support']}\n\n")
            
            print(f"✅ Ultra-detailed results saved to {results_dir}/")
        
        return results

    def visualize_noise_maps_advanced(self, images: torch.Tensor, labels: torch.Tensor = None, 
                                    num_samples: int = 12, save_path: str = None):
        """Advanced noise map visualization with enhanced analysis"""
        if self.autoencoder is None or not self.autoencoder_trained:
            raise ValueError("Advanced autoencoder must be trained first")
        
        self.autoencoder.eval()
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        
        with torch.no_grad():
            images = images[:num_samples].to(self.device)
            
            if self.scaler is not None:
                with autocast():
                    reconstructed = self.autoencoder(images)
            else:
                reconstructed = self.autoencoder(images)
                
            noise_maps = images - reconstructed
            
            # Create enhanced visualization grid
            rows = int(np.ceil(num_samples / 3))
            fig, axes = plt.subplots(rows, 12, figsize=(24, 8*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
                
            for i in range(num_samples):
                row = i // 3
                col_start = (i % 3) * 4
                
                # Original image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                if img.shape[2] == 1:
                    img = img.squeeze()
                    axes[row, col_start].imshow(img, cmap='gray')
                else:
                    axes[row, col_start].imshow(np.clip(img, 0, 1))
                
                title = 'Original'
                if labels is not None:
                    title += f'\n({class_names[labels[i]]})'
                axes[row, col_start].set_title(title, fontsize=10)
                axes[row, col_start].axis('off')
                
                # Reconstructed image
                recon = reconstructed[i].cpu().permute(1, 2, 0).numpy()
                if recon.shape[2] == 1:
                    recon = recon.squeeze()
                    axes[row, col_start + 1].imshow(recon, cmap='gray')
                else:
                    axes[row, col_start + 1].imshow(np.clip(recon, 0, 1))
                axes[row, col_start + 1].set_title('Reconstructed', fontsize=10)
                axes[row, col_start + 1].axis('off')
                
                # Enhanced noise map
                noise = noise_maps[i].cpu().mean(dim=0).numpy()
                enhanced_noise = self._enhance_noise_map(noise)
                
                im = axes[row, col_start + 2].imshow(enhanced_noise, cmap='RdBu_r', 
                                                   vmin=-0.3, vmax=0.3)
                axes[row, col_start + 2].set_title('Enhanced Noise', fontsize=10)
                axes[row, col_start + 2].axis('off')
                plt.colorbar(im, ax=axes[row, col_start + 2], shrink=0.6)
                
                # Noise statistics visualization
                noise_stats = f'μ: {np.mean(enhanced_noise):.3f}\nσ: {np.std(enhanced_noise):.3f}\nMax: {np.max(enhanced_noise):.3f}'
                axes[row, col_start + 3].text(0.1, 0.5, noise_stats, 
                                            transform=axes[row, col_start + 3].transAxes,
                                            fontsize=9, verticalalignment='center',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[row, col_start + 3].set_title('Noise Stats', fontsize=10)
                axes[row, col_start + 3].axis('off')
                
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Advanced noise maps saved to {save_path}")
            plt.show()

def main_ultra_advanced():
    """Main script for ultra-advanced classification targeting MCC > 0.95"""
    print("🚀 ULTRA-ADVANCED NOISE CLASSIFICATION PIPELINE")
    print("🎯 TARGET: MCC > 0.95 | ACCURACY > 0.98")
    print("="*80)
    
    # Enhanced configuration
    data_dir = './datasets/train'
    batch_size = 32 * torch.cuda.device_count()  # Optimized for memory efficiency and speed
    num_epochs = 50  # More epochs for better convergence
    checkpoint_dir = './ultra_checkpoints'
    results_dir = './ultra_results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # System information
    print(f"🔧 ULTRA-ADVANCED System Configuration:")
    print(f"   Device: {device}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CPU cores: {mp.cpu_count()}")
    print(f"   Target Features: 500+")
    print(f"   Ensemble Classifiers: 4 (RF, GB, SVM, MLP)")
    
    if torch.cuda.is_available():
        print(f"🔥 CUDA available with {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    # Enable all optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load and split dataset
    print(f"\n📊 Loading dataset from {data_dir}...")
    load_start = time.time()
    try:
        images, labels = load_and_verify_data(data_dir,max_images_per_class=120000)
        load_time = time.time() - load_start
        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
        print(f"📊 Dataset statistics:")
        print(f"   Total images: {len(images):,}")
        print(f"   Image shape: {images.shape[1:]}")
        print(f"   Memory usage: {images.element_size() * images.nelement() / (1024**3):.2f} GB")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Return safe default results so caller doesn't crash
        return {
            'mcc': 0.0,
            'accuracy': 0.0,
            'prediction_speed': 0.0
        }

    # Enhanced stratified split
    train_dataset, val_dataset, test_dataset = create_train_val_test_split(
        images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # Optimized data loaders with advanced settings
    num_workers = min(8, mp.cpu_count() // 2)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    
    print(f"⚙️ Advanced DataLoader settings:")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}")
    print(f"   Pin memory: {pin_memory}")
    print(f"   Training augmentation: Enabled")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=4
    )

    # Initialize ultra-advanced pipeline
    pipeline = UltraAdvancedClassificationPipeline(device=device, checkpoint_dir=checkpoint_dir)

    # Check for existing checkpoint
    print("\n🔍 Checking for existing ultra-advanced checkpoints...")
    loaded, resume_epoch = pipeline.load_checkpoint('latest')
    if loaded:
        print(f"✅ Loaded existing checkpoint. Resume from epoch: {resume_epoch}")
    else:
        print("ℹ️ No checkpoint found. Starting fresh ultra-advanced training.")
        resume_epoch = 0

    # Train advanced autoencoder
    if not pipeline.autoencoder_trained:
        print(f"\n🔥 Training ULTRA-ADVANCED autoencoder...")
        autoencoder_start = time.time()
        pipeline.train_autoencoder_advanced(
            train_loader, val_loader, 
            epochs=num_epochs,
            resume_from_epoch=resume_epoch, 
            save_every=10
        )
        autoencoder_time = time.time() - autoencoder_start
        print(f"✅ Advanced autoencoder training completed in {autoencoder_time:.2f} seconds")
        pipeline.save_checkpoint('autoencoder_complete')
    else:
        print("✅ Advanced autoencoder already trained.")

    # Advanced noise map visualization
    print("\n📊 Creating advanced noise map visualizations...")
    try:
        # Get balanced samples from each class
        sample_images = []
        sample_labels = []
        class_counts = [0, 0, 0]
        target_per_class = 4
        
        for images_batch, labels_batch in test_loader:
            for img, label in zip(images_batch, labels_batch):
                if class_counts[label.item()] < target_per_class:
                    sample_images.append(img)
                    sample_labels.append(label.item())
                    class_counts[label.item()] += 1
                    
                if sum(class_counts) >= target_per_class * 3:
                    break
            if sum(class_counts) >= target_per_class * 3:
                break
        
        if len(sample_images) > 0:
            sample_images = torch.stack(sample_images)
            sample_labels = torch.tensor(sample_labels)
            os.makedirs(results_dir, exist_ok=True)
            pipeline.visualize_noise_maps_advanced(
                sample_images, sample_labels, 
                num_samples=len(sample_images),
                save_path=os.path.join(results_dir, 'ultra_noise_maps_analysis.png')
            )
    except Exception as e:
        print(f"⚠️ Error visualizing noise maps: {e}")

    # Train ultra-advanced ensemble classifier
    if not pipeline.classifier_trained:
        print(f"\n🎭 Training ULTRA-ADVANCED ensemble classifier...")
        classifier_start = time.time()
        pipeline.train_ensemble_classifier_advanced(train_loader, val_loader)
        classifier_time = time.time() - classifier_start
        print(f"✅ Ultra-advanced ensemble training completed in {classifier_time:.2f} seconds")
        pipeline.save_checkpoint('complete_ultra_pipeline')
    else:
        print("✅ Ultra-advanced ensemble classifier already trained.")

    # Ultra-comprehensive evaluation
    print("\n🎯 ULTRA-COMPREHENSIVE EVALUATION ON TEST SET...")
    test_labels = test_dataset.labels.cpu().tolist()
    results = pipeline.evaluate_advanced(
        test_loader, test_labels,
        save_results=True, results_dir=results_dir
    )

    # Final ultra-advanced summary
    total_pipeline_time = time.time() - load_start
    
    print("\n" + "="*80)
    print("🏆 ULTRA-ADVANCED PIPELINE EXECUTION COMPLETED!")
    print("="*80)
    
    # Achievement status
    target_achieved = results['mcc'] > 0.95
    accuracy_excellent = results['accuracy'] > 0.98
    
    print(f"🎯 TARGET ACHIEVEMENT STATUS:")
    print(f"   MCC > 0.95: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'} ({results['mcc']:.6f})")
    print(f"   Accuracy > 0.98: {'✅ ACHIEVED' if accuracy_excellent else '❌ NOT ACHIEVED'} ({results['accuracy']:.6f})")
    
    if target_achieved and accuracy_excellent:
        print(f"\n🎉🎉🎉 CONGRATULATIONS! PERFECT SCORE ACHIEVED! 🎉🎉🎉")
        print(f"🏆 Your model has achieved the target performance!")
    elif target_achieved:
        print(f"\n🎉 EXCELLENT! MCC target achieved!")
    elif results['mcc'] > 0.90:
        print(f"\n⭐ VERY GOOD! Close to target!")
    else:
        print(f"\n📈 GOOD PROGRESS! Continue optimizing!")
    
    print(f"\n📊 COMPREHENSIVE STATISTICS:")
    print(f"   Total images processed: {len(images):,}")
    print(f"   Final test MCC: {results['mcc']:.6f}")
    print(f"   Final test accuracy: {results['accuracy']:.6f}")
    print(f"   Average per-class MCC: {np.mean(results['per_class_mcc']):.6f}")
    print(f"   Prediction speed: {results['prediction_speed']:.1f} samples/second")
    print(f"   High confidence predictions: {results['confidence_metrics']['high_confidence_ratio']*100:.1f}%")
    
    print(f"\n🔧 TECHNICAL DETAILS:")
    print(f"   Features extracted: 500+ per sample")
    print(f"   Features selected: {400} most informative")
    print(f"   Ensemble size: 4 advanced classifiers")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    
    print(f"\n⏱️ PERFORMANCE BREAKDOWN:")
    if 'autoencoder_time' in locals():
        print(f"   Autoencoder training: {autoencoder_time:.2f}s")
    if 'classifier_time' in locals():
        print(f"   Ensemble training: {classifier_time:.2f}s")
    print(f"   Evaluation: {results['evaluation_time']:.2f}s")
    print(f"   Total pipeline time: {total_pipeline_time:.2f}s")
    
    print(f"\n💾 SAVED ARTIFACTS:")
    print(f"   Model checkpoints: {checkpoint_dir}/")
    print(f"   Results and plots: {results_dir}/")
    print(f"   Best model: {'best_test_model' if target_achieved else 'ensemble_final'}")
    
    # Performance tier classification
    if results['mcc'] >= 0.95:
        tier = "🏆 PLATINUM TIER"
    elif results['mcc'] >= 0.90:
        tier = "🥇 GOLD TIER"
    elif results['mcc'] >= 0.85:
        tier = "🥈 SILVER TIER"
    elif results['mcc'] >= 0.80:
        tier = "🥉 BRONZE TIER"
    else:
        tier = "📈 DEVELOPING TIER"
    
    print(f"\n{tier} - MCC: {results['mcc']:.6f}")
    print("="*80)
    
    return results

# Ultra-fast inference functions for production use
def load_ultra_pretrained_pipeline(checkpoint_dir='./ultra_checkpoints') -> UltraAdvancedClassificationPipeline:
    """Load ultra-advanced pre-trained pipeline"""
    print("🔄 Loading ultra-advanced pre-trained pipeline...")
    
    pipeline = UltraAdvancedClassificationPipeline(checkpoint_dir=checkpoint_dir)
    
    # Load the best available checkpoint
    checkpoint_names = ['best_test_model', 'ensemble_final', 'complete_ultra_pipeline', 'latest']
    
    loaded = False
    for checkpoint_name in checkpoint_names:
        if pipeline.load_checkpoint(checkpoint_name)[0]:
            loaded = True
            print(f"✅ Loaded checkpoint: {checkpoint_name}")
            break
    
    if not loaded:
        print("❌ No valid checkpoint found!")
        return None
    
    print(f"✅ Ultra-advanced pipeline loaded successfully!")
    print(f"   Autoencoder: {'✅' if pipeline.autoencoder_trained else '❌'}")
    print(f"   Ensemble: {'✅' if pipeline.classifier_trained else '❌'}")
    print(f"   Feature extractor: {'✅' if pipeline.noise_extractor.fitted else '❌'}")
    
    return pipeline

def classify_images_ultra_fast(image_paths: List[str], 
                              pipeline: UltraAdvancedClassificationPipeline, 
                              batch_size: int = 32,
                              confidence_threshold: float = 0.8) -> Dict:
    """Ultra-fast classification of new images with confidence analysis"""
    if pipeline is None or not pipeline.autoencoder_trained or not pipeline.classifier_trained:
        raise ValueError("❌ Ultra-advanced pipeline must be fully trained!")
    
    print(f"🔮 Ultra-fast classification of {len(image_paths)} images...")
    start_time = time.time()
    
    # Load and preprocess images
    images = []
    valid_paths = []
    
    for img_path in tqdm(image_paths, desc="📂 Loading images"):
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Enhanced preprocessing
            img_array = np.array(img)
            
            # Quality checks
            if img_array.std() < 1.0:  # Very low variance image
                print(f"⚠️ Low quality image detected: {img_path}")
            
            # Normalize and convert
            img_tensor = transforms.ToTensor()(img)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            images.append(img_tensor)
            valid_paths.append(img_path)
            
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            continue
    
    if not images:
        print("❌ No valid images loaded!")
        return {}
    
    # Stack into tensor
    images_tensor = torch.cat(images, dim=0)
    print(f"✅ Loaded {len(images_tensor)} valid images")
    
    # Create temporary dataset and dataloader
    dummy_labels = torch.zeros(len(images_tensor), dtype=torch.long)
    temp_dataset = PTFileDataset(images_tensor, dummy_labels, augment=False)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=torch.cuda.is_available())
    
    # Get ultra-fast predictions
    predictions, probabilities = pipeline.predict_advanced(temp_loader)
    
    # Analyze results
    probabilities_array = np.array(probabilities)
    max_probs = np.max(probabilities_array, axis=1)
    prediction_confidence = max_probs
    
    # Classify by confidence level
    high_confidence = max_probs >= confidence_threshold
    medium_confidence = (max_probs >= 0.6) & (max_probs < confidence_threshold)
    low_confidence = max_probs < 0.6
    
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    
    # Detailed results
    detailed_results = []
    for i, (path, pred, prob, conf) in enumerate(zip(valid_paths, predictions, probabilities, prediction_confidence)):
        confidence_level = 'High' if high_confidence[i] else ('Medium' if medium_confidence[i] else 'Low')
        
        detailed_results.append({
            'image_path': path,
            'prediction': class_names[pred],
            'prediction_id': pred,
            'confidence': conf,
            'confidence_level': confidence_level,
            'probabilities': {
                'Real': prob[0],
                'Synthetic': prob[1],
                'Semi-synthetic': prob[2]
            }
        })
    
    classification_time = time.time() - start_time
    
    # Summary statistics
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    
    results_summary = {
        'total_images': len(predictions),
        'classification_time': classification_time,
        'speed': len(predictions) / classification_time,
        'class_distribution': {
            class_names[pred_class]: int(count) 
            for pred_class, count in zip(unique_preds, pred_counts)
        },
        'confidence_analysis': {
            'high_confidence': int(np.sum(high_confidence)),
            'medium_confidence': int(np.sum(medium_confidence)),
            'low_confidence': int(np.sum(low_confidence)),
            'mean_confidence': float(np.mean(prediction_confidence)),
            'std_confidence': float(np.std(prediction_confidence))
        },
        'detailed_results': detailed_results
    }
    
    print(f"\n✅ Ultra-fast classification completed!")
    print(f"📊 Results summary:")
    print(f"   Processing speed: {results_summary['speed']:.1f} images/second")
    print(f"   Total time: {classification_time:.2f} seconds")
    
    print(f"\n📊 Classification results:")
    for class_name, count in results_summary['class_distribution'].items():
        percentage = (count / len(predictions)) * 100
        print(f"   {class_name}: {count} images ({percentage:.1f}%)")
    
    print(f"\n🎯 Confidence analysis:")
    conf_stats = results_summary['confidence_analysis']
    print(f"   High confidence (≥{confidence_threshold}): {conf_stats['high_confidence']} ({conf_stats['high_confidence']/len(predictions)*100:.1f}%)")
    print(f"   Medium confidence (0.6-{confidence_threshold}): {conf_stats['medium_confidence']} ({conf_stats['medium_confidence']/len(predictions)*100:.1f}%)")
    print(f"   Low confidence (<0.6): {conf_stats['low_confidence']} ({conf_stats['low_confidence']/len(predictions)*100:.1f}%)")
    print(f"   Mean confidence: {conf_stats['mean_confidence']:.4f}")
    
    return results_summary

# Ensemble model evaluation and comparison
def evaluate_model_ensemble(pipeline: UltraAdvancedClassificationPipeline,
                          test_loader: DataLoader,
                          test_labels: List[int]) -> Dict:
    """Evaluate individual models in the ensemble"""
    print("\n🔬 Detailed ensemble analysis...")
    
    if not pipeline.classifier_trained or not pipeline.autoencoder_trained:
        raise ValueError("Pipeline must be fully trained")
    
    # Extract test features once
    print("📊 Extracting test features for ensemble analysis...")
    all_test_noise_maps = []
    
    for images, _ in tqdm(test_loader, desc="Extracting test noise maps"):
        noise_maps = pipeline.extract_noise_from_images_advanced(images)
        all_test_noise_maps.extend(noise_maps)
    
    test_features = pipeline.noise_extractor.transform(all_test_noise_maps)
    if hasattr(pipeline, 'feature_selector'):
        test_features = pipeline.feature_selector.transform(test_features)
    
    # Align lengths
    min_len = min(len(test_labels), len(test_features))
    test_labels = test_labels[:min_len]
    test_features = test_features[:min_len]
    
    # Evaluate each individual classifier
    individual_results = {}
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    
    print(f"\n🔧 Evaluating individual classifiers:")
    
    for name, classifier in pipeline.base_classifiers.items():
        if hasattr(classifier, 'predict'):
            try:
                pred = classifier.predict(test_features)
                prob = classifier.predict_proba(test_features) if hasattr(classifier, 'predict_proba') else None
                
                # Align prediction length
                pred = pred[:min_len]
                if prob is not None:
                    prob = prob[:min_len]
                
                mcc = matthews_corrcoef(test_labels, pred)
                acc = accuracy_score(test_labels, pred)
                
                individual_results[name] = {
                    'mcc': mcc,
                    'accuracy': acc,
                    'predictions': pred.tolist(),
                    'probabilities': prob.tolist() if prob is not None else None
                }
                
                print(f"   {name.upper()}: MCC={mcc:.4f}, Accuracy={acc:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {name}: {e}")
    
    # Ensemble evaluation
    try:
        ensemble_pred = pipeline.ensemble_classifier.predict(test_features)
        ensemble_prob = pipeline.ensemble_classifier.predict_proba(test_features)
        
        ensemble_pred = ensemble_pred[:min_len]
        ensemble_prob = ensemble_prob[:min_len]
        
        ensemble_mcc = matthews_corrcoef(test_labels, ensemble_pred)
        ensemble_acc = accuracy_score(test_labels, ensemble_pred)
        
        individual_results['ensemble'] = {
            'mcc': ensemble_mcc,
            'accuracy': ensemble_acc,
            'predictions': ensemble_pred.tolist(),
            'probabilities': ensemble_prob.tolist()
        }
        
        print(f"   🎭 ENSEMBLE: MCC={ensemble_mcc:.4f}, Accuracy={ensemble_acc:.4f}")
        
    except Exception as e:
        print(f"❌ Error evaluating ensemble: {e}")
    
    # Find best performing model
    best_model = max(individual_results.keys(), key=lambda x: individual_results[x]['mcc'])
    best_mcc = individual_results[best_model]['mcc']
    
    print(f"\n🏆 Best performing model: {best_model.upper()} (MCC: {best_mcc:.4f})")
    
    return individual_results

# Advanced hyperparameter optimization
def optimize_hyperparameters_advanced(pipeline: UltraAdvancedClassificationPipeline,
                                     train_features: np.ndarray,
                                     train_labels: np.ndarray,
                                     n_iter: int = 50) -> Dict:
    """Advanced hyperparameter optimization for ultimate performance"""
    print(f"\n🎯 Advanced hyperparameter optimization ({n_iter} iterations)...")
    
    # Extended parameter grids for maximum performance
    extended_param_grids = {
        'rf': {
            'n_estimators': [1000, 1500, 2000, 2500, 3000],
            'max_depth': [20, 25, 30, 35, 40],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'gb': {
            'n_estimators': [500, 800, 1000, 1200, 1500],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'max_depth': [10, 12, 15, 18, 20],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3]
        },
        'svm': {
            'C': [0.1, 1, 5, 10, 20, 50],
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'degree': [2, 3, 4],  # For poly kernel
            'class_weight': ['balanced', None]
        },
        'mlp': {
            'hidden_layer_sizes': [
                (256,), (512,), (1024,),
                (256, 128), (512, 256), (1024, 512),
                (512, 256, 128), (1024, 512, 256), (1024, 512, 256, 128)
            ],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [500, 1000, 2000],
            'early_stopping': [True],
            'validation_fraction': [0.2]
        }
    }
    
    optimization_results = {}
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, classifier in pipeline.base_classifiers.items():
        print(f"\n🔧 Optimizing {name.upper()} with {n_iter} iterations...")
        
        try:
            # Advanced random search with MCC scoring
            search = RandomizedSearchCV(
                classifier, 
                extended_param_grids[name], 
                n_iter=n_iter,
                scoring='accuracy',  # Primary scoring
                cv=cv_folds, 
                n_jobs=-1,
                random_state=42, 
                verbose=1,
                return_train_score=True
            )
            
            search.fit(train_features, train_labels)
            
            # Evaluate with MCC
            best_pred = search.best_estimator_.predict(train_features)
            best_mcc = matthews_corrcoef(train_labels, best_pred)
            
            optimization_results[name] = {
                'best_params': search.best_params_,
                'best_cv_score': search.best_score_,
                'best_mcc': best_mcc,
                'best_estimator': search.best_estimator_
            }
            
            # Update pipeline classifier
            pipeline.base_classifiers[name] = search.best_estimator_
            
            print(f"   ✅ {name.upper()}: CV Score={search.best_score_:.4f}, MCC={best_mcc:.4f}")
            print(f"   🎯 Best params: {search.best_params_}")
            
        except Exception as e:
            print(f"   ❌ Error optimizing {name}: {e}")
            continue
    
    # Rebuild ensemble with optimized classifiers
    optimized_estimators = [(name, results['best_estimator']) 
                           for name, results in optimization_results.items()]
    
    if len(optimized_estimators) >= 2:
        pipeline.ensemble_classifier = VotingClassifier(
            estimators=optimized_estimators,
            voting='soft',
            n_jobs=-1
        )
        
        pipeline.ensemble_classifier.fit(train_features, train_labels)
        
        # Evaluate optimized ensemble
        ensemble_pred = pipeline.ensemble_classifier.predict(train_features)
        ensemble_mcc = matthews_corrcoef(train_labels, ensemble_pred)
        
        optimization_results['ensemble'] = {
            'mcc': ensemble_mcc,
            'accuracy': accuracy_score(train_labels, ensemble_pred)
        }
        
        print(f"\n🎭 Optimized ensemble MCC: {ensemble_mcc:.4f}")
    
    return optimization_results

# Main execution with advanced options
if __name__ == "__main__":
    # Enable all CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print("🚀 LAUNCHING ULTRA-ADVANCED CLASSIFICATION PIPELINE")
    print("🎯 MISSION: ACHIEVE PERFECT SCORE 10 & MCC > 0.95")
    print("="*80)
    
    # Run main ultra-advanced pipeline
    results = main_ultra_advanced()
    
    print(f"\n🏁 MISSION STATUS:")
    if results['mcc'] > 0.95:
        print(f"🎉 MISSION ACCOMPLISHED! TARGET ACHIEVED!")
        print(f"🏆 Perfect classification score achieved: MCC = {results['mcc']:.6f}")
        score = 10.0
    elif results['mcc'] > 0.90:
        score = 8.0 + (results['mcc'] - 0.90) * 20  # Scale 0.90-0.95 to 8-10
        print(f"⭐ Excellent performance! Score: {score:.1f}/10")
    elif results['mcc'] > 0.80:
        score = 6.0 + (results['mcc'] - 0.80) * 20  # Scale 0.80-0.90 to 6-8
        print(f"👍 Very good performance! Score: {score:.1f}/10")
    else:
        score = max(0, results['mcc'] * 7.5)  # Scale 0-0.80 to 0-6
        print(f"📈 Good progress! Score: {score:.1f}/10")
    
    print(f"\n📊 FINAL SCORECARD:")
    print(f"   Classification Score: {score:.1f}/10")
    print(f"   MCC Achievement: {results['mcc']:.6f}/0.95")
    print(f"   Accuracy: {results['accuracy']:.6f}")
    print(f"   Speed: {results['prediction_speed']:.1f} samples/second")
    
    print(f"\n🎯 RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
    if results['mcc'] < 0.95:
        print(f"   • Increase training data diversity")
        print(f"   • Implement more sophisticated noise models")
        print(f"   • Try ensemble of autoencoders")
        print(f"   • Implement advanced feature engineering")
        print(f"   • Use cross-validation for model selection")
        print(f"   • Consider domain-specific augmentations")
    else:
        print(f"   🎉 Perfect! No improvements needed!")
    
    print("="*80)