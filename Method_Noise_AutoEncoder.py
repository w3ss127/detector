import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import json
import seaborn as sns
import pickle
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import pywt
from functools import partial
import gc
import psutil
import warnings
import random
from pathlib import Path
import h5py
import threading
from queue import Queue
import shutil

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_gb': mem_info.rss / (1024**3),
        'vms_gb': mem_info.vms / (1024**3),
        'percent': process.memory_percent()
    }

def process_class_folder(class_info):
    """Process a single class folder - moved to module level for pickling"""
    class_name, class_dir, max_files = class_info
    class_mapping = {'real': 0, 'synthetic': 1, 'semi-synthetic': 2}
    class_idx = class_mapping[class_name]
    
    if not os.path.exists(class_dir):
        print(f"⚠️ Directory {class_dir} not found")
        return [], []
    
    # Get all .pt files
    pt_files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
    pt_files.sort()
    
    if max_files and len(pt_files) > max_files:
        # Use stratified sampling instead of just taking first N
        indices = np.linspace(0, len(pt_files)-1, max_files, dtype=int)
        pt_files = [pt_files[i] for i in indices]
    
    print(f"📂 {class_name}: Processing {len(pt_files):,} files")
    
    class_paths = []
    class_labels = []
    
    # Process files to get individual image paths
    for pt_file in pt_files:
        pt_path = os.path.join(class_dir, pt_file)
        try:
            # Quick check for file validity and count images
            tensor_data = torch.load(pt_path, map_location='cpu', weights_only=False)
            
            if isinstance(tensor_data, dict):
                if 'images' in tensor_data:
                    images = tensor_data['images']
                elif 'data' in tensor_data:
                    images = tensor_data['data']
                else:
                    images = list(tensor_data.values())[0]
            else:
                images = tensor_data
            
            # Count images in this file
            if images.dim() == 4:
                num_images = images.shape[0]
            elif images.dim() == 3:
                num_images = 1
            else:
                continue
            
            # Add entries for each image (we'll handle indexing in dataset)
            for _ in range(num_images):
                class_paths.append(pt_path)
                class_labels.append(class_idx)
            
            del tensor_data, images
            
        except Exception as e:
            print(f"⚠️ Error processing {pt_file}: {e}")
            continue
    
    return class_paths, class_labels

class HighPerformancePTFileDataset(Dataset):
    """High-performance dataset with intelligent caching and multi-threading"""
    
    def __init__(self, file_paths: List[str], labels: List[int], 
                 device: str = 'cuda', transform=None, prefetch_factor: int = 4,
                 cache_dir: str = None, use_memory_mapping: bool = True):
        self.file_paths = file_paths
        self.labels = labels
        self.device = device
        self.transform = transform
        self.prefetch_factor = prefetch_factor
        self.use_memory_mapping = use_memory_mapping
        
        # Create cache directory for preprocessed data
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._build_cache()
        else:
            self.cache_dir = None
        
        # Memory mapping for faster data access
        if use_memory_mapping and cache_dir:
            self._setup_memory_mapping()
        
        # Threading for data loading
        self.loader_pool = ThreadPoolExecutor(max_workers=8)
        self.prefetch_queue = Queue(maxsize=prefetch_factor * 32)
        self._start_prefetching()
    
    def _build_cache(self):
        """Build cache of preprocessed images"""
        print(f"🔧 Building dataset cache in {self.cache_dir}")
        cache_file = self.cache_dir / 'dataset_cache.h5'
        
        if cache_file.exists():
            print("✅ Cache already exists, skipping build")
            return
        
        with h5py.File(cache_file, 'w') as f:
            # Create datasets for images and labels
            img_shape = self._get_sample_shape()
            images_ds = f.create_dataset(
                'images', 
                shape=(len(self.file_paths), *img_shape), 
                dtype=np.float32,
                compression='lzf',
                chunks=True
            )
            labels_ds = f.create_dataset('labels', data=self.labels, dtype=np.int32)
            
            # Process images in parallel
            batch_size = 1000
            for i in tqdm(range(0, len(self.file_paths), batch_size), desc="Caching images"):
                end_idx = min(i + batch_size, len(self.file_paths))
                batch_paths = self.file_paths[i:end_idx]
                
                # Load batch in parallel
                with ProcessPoolExecutor(max_workers=16) as executor:
                    batch_images = list(executor.map(self._load_and_preprocess_image, batch_paths))
                
                # Store in cache
                for j, img in enumerate(batch_images):
                    if img is not None:
                        images_ds[i + j] = img
    
    def _setup_memory_mapping(self):
        """Setup memory mapping for cached data"""
        cache_file = self.cache_dir / 'dataset_cache.h5'
        if cache_file.exists():
            self.cache_file = h5py.File(cache_file, 'r')
            self.images_ds = self.cache_file['images']
            self.labels_ds = self.cache_file['labels']
            print(f"✅ Memory mapping enabled for {len(self.images_ds)} cached images")
        else:
            self.cache_file = None
    
    def _get_sample_shape(self) -> Tuple[int, int, int]:
        """Get sample image shape"""
        sample_path = self.file_paths[0]
        img = self._load_and_preprocess_image(sample_path)
        return img.shape if img is not None else (3, 224, 224)
    
    def _load_and_preprocess_image(self, file_path: str) -> Optional[np.ndarray]:
        """Load and preprocess a single image"""
        try:
            # Load tensor
            tensor_data = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Extract image from tensor
            if isinstance(tensor_data, dict):
                if 'images' in tensor_data:
                    image = tensor_data['images']
                elif 'data' in tensor_data:
                    image = tensor_data['data']
                else:
                    image = list(tensor_data.values())[0]
            else:
                image = tensor_data
            
            # Handle batch dimension
            if image.dim() == 4:
                image = image[0]
            elif image.dim() == 3:
                pass
            else:
                return None
            
            # Convert to numpy and normalize
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            else:
                image = image.float()
                if image.max() > 1.0:
                    image = image / 255.0
            
            # Ensure correct shape (C, H, W) and size
            if image.shape[1:] != (224, 224):
                image = F.interpolate(image.unsqueeze(0), size=(224, 224), 
                                    mode='bilinear', align_corners=False)[0]
            
            return image.numpy().astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            return None
    
    def _start_prefetching(self):
        """Start background prefetching"""
        def prefetch_worker():
            for idx in range(len(self)):
                try:
                    item = self._get_item_cached(idx)
                    self.prefetch_queue.put((idx, item))
                except Exception as e:
                    print(f"Prefetch error for index {idx}: {e}")
        
        # Start prefetch thread
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _get_item_cached(self, idx):
        """Get item from cache or load from disk"""
        if self.cache_file and hasattr(self, 'images_ds'):
            # Load from cache
            image = torch.from_numpy(self.images_ds[idx].copy())
            label = int(self.labels_ds[idx])
        else:
            # Load from disk
            file_path = self.file_paths[idx]
            image_np = self._load_and_preprocess_image(file_path)
            if image_np is None:
                image = torch.zeros((3, 224, 224), dtype=torch.float32)
            else:
                image = torch.from_numpy(image_np)
            label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        return self._get_item_cached(idx)
    
    def __del__(self):
        if hasattr(self, 'cache_file') and self.cache_file:
            self.cache_file.close()
        if hasattr(self, 'loader_pool'):
            self.loader_pool.shutdown(wait=False)

def load_dataset_parallel(data_dir: str, max_files_per_class: int = None) -> Tuple[List[str], List[int]]:
    """Load dataset paths in parallel with better organization"""
    print(f"🔍 Loading dataset from: {data_dir}")
    
    all_file_paths = []
    all_labels = []
    
    class_folders = ['real', 'synthetic', 'semi-synthetic']
    
    # Process classes in parallel
    class_info_list = [
        (class_name, os.path.join(data_dir, class_name), max_files_per_class)
        for class_name in class_folders
    ]
    
    try:
        with ProcessPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(process_class_folder, class_info_list))
    except Exception as e:
        print(f"⚠️ Parallel processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing
        results = []
        for class_info in class_info_list:
            results.append(process_class_folder(class_info))
    
    # Combine results
    for class_paths, class_labels in results:
        all_file_paths.extend(class_paths)
        all_labels.extend(class_labels)
    
    if not all_file_paths:
        raise ValueError("❌ No valid data found!")
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total images: {len(all_file_paths):,}")
    
    # Class distribution
    unique, counts = np.unique(all_labels, return_counts=True)
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    for label, count in zip(unique, counts):
        if label < len(class_names):
            print(f"   {class_names[label]}: {count:,} images ({100*count/len(all_labels):.1f}%)")
    
    return all_file_paths, all_labels

def create_stratified_splits(file_paths: List[str], labels: List[int], 
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                           random_seed=42, cache_dir: str = None) -> Tuple:
    """Create stratified dataset splits with caching"""
    print(f"\n🔄 Creating stratified dataset splits...")
    
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    # Convert to numpy arrays
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    # Stratified split to ensure balanced classes
    unique_labels = np.unique(labels)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        
        n_samples = len(label_indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:])
    
    # Shuffle indices to mix classes
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    # Create cache directories
    train_cache = os.path.join(cache_dir, 'train') if cache_dir else None
    val_cache = os.path.join(cache_dir, 'val') if cache_dir else None
    test_cache = os.path.join(cache_dir, 'test') if cache_dir else None
    
    # Create datasets
    train_dataset = HighPerformancePTFileDataset(
        file_paths[train_indices].tolist(),
        labels[train_indices].tolist(),
        cache_dir=train_cache
    )
    
    val_dataset = HighPerformancePTFileDataset(
        file_paths[val_indices].tolist(),
        labels[val_indices].tolist(),
        cache_dir=val_cache
    )
    
    test_dataset = HighPerformancePTFileDataset(
        file_paths[test_indices].tolist(),
        labels[test_indices].tolist(),
        cache_dir=test_cache
    )
    
    print(f"📊 Dataset splits created:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Validation: {len(val_dataset):,} samples")
    print(f"   Test: {len(test_dataset):,} samples")
    
    return train_dataset, val_dataset, test_dataset

class EfficientResNet(nn.Module):
    """Efficient ResNet-like architecture for denoising"""
    
    def __init__(self, input_channels=3, base_filters=32):
        super(EfficientResNet, self).__init__()
        
        self.input_channels = input_channels
        
        # Initial conv
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Encoder blocks
        self.enc_block1 = self._make_layer(base_filters, base_filters * 2, 2)
        self.enc_block2 = self._make_layer(base_filters * 2, base_filters * 4, 2) 
        self.enc_block3 = self._make_layer(base_filters * 4, base_filters * 8, 2)
        self.enc_block4 = self._make_layer(base_filters * 8, base_filters * 16, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 16, base_filters * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Decoder blocks
        self.dec_block4 = self._make_decoder_layer(base_filters * 16, base_filters * 8)
        self.dec_block3 = self._make_decoder_layer(base_filters * 8, base_filters * 4)
        self.dec_block2 = self._make_decoder_layer(base_filters * 4, base_filters * 2)
        self.dec_block1 = self._make_decoder_layer(base_filters * 2, base_filters)
        
        # Final layers
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(base_filters, base_filters, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_filters, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.initial_conv(x)  # /4
        x2 = self.enc_block1(x1)  # /8
        x3 = self.enc_block2(x2)  # /16
        x4 = self.enc_block3(x3)  # /32
        x5 = self.enc_block4(x4)  # /64
        
        # Bottleneck
        bottleneck = self.bottleneck(x5)
        
        # Decoder with skip connections
        d4 = self.dec_block4(bottleneck)  # /32
        d4 = d4 + x4  # Skip connection
        
        d3 = self.dec_block3(d4)         # /16
        d3 = d3 + x3  # Skip connection
        
        d2 = self.dec_block2(d3)         # /8
        d2 = d2 + x2  # Skip connection
        
        d1 = self.dec_block1(d2)         # /4
        d1 = d1 + x1  # Skip connection
        
        # Final upsampling
        output = self.final_upsample(d1)  # /1
        
        return output

class AdvancedNoiseGenerator:
    """Advanced noise generation for training"""
    
    @staticmethod
    def add_gaussian_noise(images, strength=0.1):
        noise = torch.randn_like(images) * strength
        return noise
    
    @staticmethod
    def add_salt_pepper_noise(images, prob=0.05):
        noise = torch.zeros_like(images)
        salt_mask = torch.rand_like(images) < prob/2
        pepper_mask = torch.rand_like(images) > (1 - prob/2)
        noise[salt_mask] = 0.5
        noise[pepper_mask] = -0.5
        return noise
    
    @staticmethod
    def add_speckle_noise(images, strength=0.1):
        noise = torch.randn_like(images) * strength * images
        return noise
    
    @staticmethod
    def generate_mixed_noise(images, epoch=0, max_epochs=100):
        """Generate stable noise that doesn't increase too aggressively"""
        batch_size = images.shape[0]
        
        # More conservative noise strength progression
        progress = min(epoch / max_epochs, 1.0)
        base_strength = 0.02 + 0.08 * progress  # Start smaller, increase more gradually
        
        total_noise = torch.zeros_like(images)
        
        # Always add Gaussian noise (primary noise type)
        gaussian_noise = AdvancedNoiseGenerator.add_gaussian_noise(images, base_strength)
        total_noise += gaussian_noise
        
        # Add salt-pepper noise more conservatively
        if epoch > max_epochs * 0.5:  # Start later
            sp_noise = AdvancedNoiseGenerator.add_salt_pepper_noise(images, 0.01 + 0.02 * progress)
            total_noise += sp_noise * 0.2  # Reduced weight
        
        # Add speckle noise very conservatively
        if epoch > max_epochs * 0.7:  # Start even later
            speckle_noise = AdvancedNoiseGenerator.add_speckle_noise(images, base_strength * 0.3)
            total_noise += speckle_noise * 0.1  # Much reduced weight
        
        return total_noise

class CombinedLoss(nn.Module):
    """Combined loss function with multiple components"""
    
    def __init__(self, mse_weight=1.0, l1_weight=0.5, perceptual_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def perceptual_loss(self, pred, target):
        """Simple perceptual loss using gradients"""
        # Compute gradients
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = (self.mse_weight * mse + 
                     self.l1_weight * l1 + 
                     self.perceptual_weight * perceptual)
        
        return total_loss, mse, l1, perceptual

class FastNoiseFeatureExtractor:
    """Ultra-fast feature extraction optimized for large datasets"""
    
    def __init__(self, n_jobs=-1):
        self.scaler = StandardScaler()
        self.fitted = False
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
    
    def extract_features_batch(self, noise_maps: np.ndarray) -> np.ndarray:
        """Extract features from batch of noise maps efficiently"""
        if len(noise_maps.shape) == 2:
            noise_maps = noise_maps[np.newaxis, ...]
        
        batch_size, h, w = noise_maps.shape
        features = np.zeros((batch_size, 32), dtype=np.float32)
        
        # Flatten for vectorized operations
        noise_flat = noise_maps.reshape(batch_size, -1)
        
        # Basic statistics (vectorized)
        features[:, 0] = np.mean(noise_flat, axis=1)
        features[:, 1] = np.std(noise_flat, axis=1)
        features[:, 2] = np.var(noise_flat, axis=1)
        
        # Percentiles
        percentiles = np.percentile(noise_flat, [10, 25, 50, 75, 90], axis=1)
        features[:, 3:8] = percentiles.T
        features[:, 8] = percentiles[4] - percentiles[0]  # Range
        
        # Histogram features (simplified)
        for i in range(batch_size):
            hist, _ = np.histogram(noise_flat[i], bins=8, range=(-1, 1))
            hist = hist / (np.sum(hist) + 1e-8)
            features[i, 9:17] = hist
        
        # Simple gradient features
        if h > 4 and w > 4:
            grad_x = np.diff(noise_maps, axis=2)
            grad_y = np.diff(noise_maps, axis=1)
            grad_mag = np.sqrt(grad_x[:, :, :-1]**2 + grad_y[:, :-1, :]**2)
            
            features[:, 17] = np.mean(grad_mag.reshape(batch_size, -1), axis=1)
            features[:, 18] = np.std(grad_mag.reshape(batch_size, -1), axis=1)
        
        # Spatial variation
        if h > 8 and w > 8:
            mid_h, mid_w = h // 2, w // 2
            q1 = noise_maps[:, :mid_h, :mid_w]
            q2 = noise_maps[:, :mid_h, mid_w:]
            q3 = noise_maps[:, mid_h:, :mid_w]
            q4 = noise_maps[:, mid_h:, mid_w:]
            
            quad_vars = np.stack([
                np.var(q1.reshape(batch_size, -1), axis=1),
                np.var(q2.reshape(batch_size, -1), axis=1),
                np.var(q3.reshape(batch_size, -1), axis=1),
                np.var(q4.reshape(batch_size, -1), axis=1)
            ], axis=1)
            
            features[:, 19] = np.std(quad_vars, axis=1)
            features[:, 20] = np.max(quad_vars, axis=1) - np.min(quad_vars, axis=1)
        
        # Fill remaining features with simple statistics
        for i in range(21, 32):
            if i < noise_flat.shape[1]:
                features[:, i] = noise_flat[:, i % noise_flat.shape[1]]
        
        return features
    
    def fit_transform(self, noise_maps_list: List[np.ndarray]) -> np.ndarray:
        """Fit and transform noise maps to features"""
        print(f"⚡ Fast feature extraction from {len(noise_maps_list)} noise maps...")
        
        # Process in batches for efficiency
        batch_size = 1000
        all_features = []
        
        for i in tqdm(range(0, len(noise_maps_list), batch_size), desc="Extracting features"):
            end_idx = min(i + batch_size, len(noise_maps_list))
            batch_maps = np.stack(noise_maps_list[i:end_idx])
            
            batch_features = self.extract_features_batch(batch_maps)
            all_features.append(batch_features)
        
        feature_matrix = np.vstack(all_features)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.scaler.transform(feature_matrix)

class MultiGPUImageClassificationPipeline:
    """High-performance multi-GPU pipeline for 480k+ images"""
    
    def __init__(self, checkpoint_dir='./checkpoints_multi_gpu', world_size=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU setup
        if world_size is None:
            self.world_size = torch.cuda.device_count()
        else:
            self.world_size = world_size
            
        print(f"🚀 Multi-GPU Pipeline Initialized")
        print(f"💻 Available GPUs: {self.world_size}")
        for i in range(self.world_size):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Models and components
        self.autoencoder = None
        self.noise_extractor = FastNoiseFeatureExtractor()
        self.classifier = None
        self.scaler = None
        
        # Training state
        self.autoencoder_trained = False
        self.classifier_trained = False
        self.noise_generator = AdvancedNoiseGenerator()
        
        # Training history
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
    
    def train_autoencoder_distributed(self, train_dataset, val_dataset, epochs=50, 
                                    batch_size=32, learning_rate=1e-3):
        """Train autoencoder using distributed training"""
        print(f"\n{'='*80}")
        print(f"🚀 DISTRIBUTED AUTOENCODER TRAINING - {epochs} EPOCHS")
        print(f"{'='*80}")
        
        if self.world_size > 1:
            mp.spawn(self._train_autoencoder_worker, 
                    args=(train_dataset, val_dataset, epochs, batch_size, learning_rate),
                    nprocs=self.world_size, join=True)
        else:
            # Single GPU training
            self._train_autoencoder_single_gpu(train_dataset, val_dataset, epochs, batch_size, learning_rate)
        
        self.autoencoder_trained = True
        print(f"✅ Autoencoder training completed!")
    
    def _train_autoencoder_worker(self, rank, train_dataset, val_dataset, epochs, batch_size, learning_rate):
        """Distributed training worker for autoencoder"""
        # Setup distributed training
        setup_distributed(rank, self.world_size)
        
        # Initialize model
        sample_batch = next(iter(DataLoader(train_dataset, batch_size=1)))
        input_channels = sample_batch[0].shape[1]
        
        model = EfficientResNet(input_channels=input_channels).cuda(rank)
        model = DDP(model, device_ids=[rank])
        
        # Distributed samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=rank)
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=8, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        scaler = GradScaler()
        
        criterion = CombinedLoss()
        
        if rank == 0:
            print(f"🔧 Training configuration:")
            print(f"   Batch size per GPU: {batch_size}")
            print(f"   Total batch size: {batch_size * self.world_size}")
            print(f"   Train batches per epoch: {len(train_loader)}")
            print(f"   Learning rate: {learning_rate}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            
            # Training
            epoch_losses = []
            progress_bar = tqdm(train_loader, desc=f'GPU {rank} Epoch {epoch+1}/{epochs}', 
                              disable=rank != 0)
            
            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.cuda(rank, non_blocking=True)
                
                # Generate progressive noise
                noise = self.noise_generator.generate_mixed_noise(images, epoch, epochs)
                noisy_images = torch.clamp(images + noise, 0., 1.)
                
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    reconstructed = model(noisy_images)
                    loss, mse_loss, l1_loss, perceptual_loss = criterion(reconstructed, images)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_losses.append(loss.item())
                
                if rank == 0:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        'MSE': f'{mse_loss.item():.4f}',
                        'L1': f'{l1_loss.item():.4f}',
                        'Perc': f'{perceptual_loss.item():.4f}'
                    })
                
                # Memory cleanup
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
            
            scheduler.step()
            
            # Validation
            if rank == 0:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for images, _ in val_loader:
                        images = images.cuda(rank, non_blocking=True)
                        noise = self.noise_generator.generate_mixed_noise(images, epoch, epochs)
                        noisy_images = torch.clamp(images + noise, 0., 1.)
                        
                        with autocast('cuda'):
                            reconstructed = model(noisy_images)
                            loss, _, _, _ = criterion(reconstructed, images)
                        
                        val_losses.append(loss.item())
                
                avg_train_loss = np.mean(epoch_losses)
                avg_val_loss = np.mean(val_losses)
                
                print(f'\nEpoch [{epoch+1}/{epochs}]:')
                print(f'  Train Loss: {avg_train_loss:.6f}')
                print(f'  Val Loss: {avg_val_loss:.6f}')
                print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_autoencoder_checkpoint(model.module, optimizer, scheduler, 
                                                    avg_val_loss, epoch, 'best_autoencoder')
                
                # Regular checkpoint
                if (epoch + 1) % 10 == 0:
                    self._save_autoencoder_checkpoint(model.module, optimizer, scheduler, 
                                                    avg_val_loss, epoch, f'epoch_{epoch+1}')
        
        # Store the trained model
        if rank == 0:
            self.autoencoder = model.module
        
        cleanup_distributed()
    
    def _train_autoencoder_single_gpu(self, train_dataset, val_dataset, epochs, batch_size, learning_rate):
        """Single GPU training fallback"""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        sample_batch = next(iter(DataLoader(train_dataset, batch_size=1)))
        input_channels = sample_batch[0].shape[1]
        
        model = EfficientResNet(input_channels=input_channels).to(device)
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        # Optimizer and criterion
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        scaler = GradScaler() if torch.cuda.is_available() else None
        criterion = CombinedLoss()
        
        print(f"🔧 Single GPU training configuration:")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Train batches: {len(train_loader)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.to(device, non_blocking=True)
                
                # Generate noise
                noise = self.noise_generator.generate_mixed_noise(images, epoch, epochs)
                noisy_images = torch.clamp(images + noise, 0., 1.)
                
                optimizer.zero_grad()
                
                if scaler:
                    with autocast('cuda'):
                        reconstructed = model(noisy_images)
                        loss, mse_loss, l1_loss, perceptual_loss = criterion(reconstructed, images)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    reconstructed = model(noisy_images)
                    loss, mse_loss, l1_loss, perceptual_loss = criterion(reconstructed, images)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                epoch_losses.append(loss.item())
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'MSE': f'{mse_loss.item():.4f}',
                    'L1': f'{l1_loss.item():.4f}',
                    'Perc': f'{perceptual_loss.item():.4f}'
                })
                
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device, non_blocking=True)
                    noise = self.noise_generator.generate_mixed_noise(images, epoch, epochs)
                    noisy_images = torch.clamp(images + noise, 0., 1.)
                    
                    if scaler:
                        with autocast('cuda'):
                            reconstructed = model(noisy_images)
                            loss, _, _, _ = criterion(reconstructed, images)
                    else:
                        reconstructed = model(noisy_images)
                        loss, _, _, _ = criterion(reconstructed, images)
                    
                    val_losses.append(loss.item())
            
            avg_train_loss = np.mean(epoch_losses)
            avg_val_loss = np.mean(val_losses)
            
            print(f'\nEpoch [{epoch+1}/{epochs}]:')
            print(f'  Train Loss: {avg_train_loss:.6f}')
            print(f'  Val Loss: {avg_val_loss:.6f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_autoencoder_checkpoint(model, optimizer, scheduler, 
                                                avg_val_loss, epoch, 'best_autoencoder')
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_autoencoder_checkpoint(model, optimizer, scheduler, 
                                                avg_val_loss, epoch, f'epoch_{epoch+1}')
        
        self.autoencoder = model
    
    def _save_autoencoder_checkpoint(self, model, optimizer, scheduler, val_loss, epoch, name):
        """Save autoencoder checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'{name}.pth'
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'epoch': epoch,
            'input_channels': model.input_channels
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {name} (Val Loss: {val_loss:.6f})")
    
    def load_autoencoder_checkpoint(self, name='best_autoencoder'):
        """Load autoencoder checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'{name}.pth'
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize model
        self.autoencoder = EfficientResNet(input_channels=checkpoint['input_channels']).to(device)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.autoencoder_trained = True
        
        print(f"✅ Loaded autoencoder: {name} (Val Loss: {checkpoint['val_loss']:.6f})")
        return True
    
    def extract_noise_maps_fast(self, dataset, max_samples=None, batch_size=64):
        """Fast noise map extraction with GPU acceleration"""
        if not self.autoencoder_trained:
            raise ValueError("Autoencoder must be trained first!")
        
        device = next(self.autoencoder.parameters()).device
        self.autoencoder.eval()
        
        # Limit samples if specified
        if max_samples:
            indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
            subset_paths = [dataset.file_paths[i] for i in indices]
            subset_labels = [dataset.labels[i] for i in indices]
            
            # Create temporary dataset
            temp_dataset = HighPerformancePTFileDataset(subset_paths, subset_labels)
        else:
            temp_dataset = dataset
        
        dataloader = DataLoader(
            temp_dataset, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True
        )
        
        noise_maps = []
        labels_used = []
        
        print(f"🔍 Extracting noise maps from {len(temp_dataset)} samples...")
        
        with torch.no_grad():
            for batch_idx, (images, batch_labels) in enumerate(tqdm(dataloader, desc="Extracting noise")):
                images = images.to(device, non_blocking=True)
                
                try:
                    with autocast('cuda'):
                        reconstructed = self.autoencoder(images)
                    
                    # Calculate noise
                    noise = images - reconstructed
                    
                    # Convert to numpy and average across channels
                    noise_np = noise.cpu().numpy()
                    for i in range(noise_np.shape[0]):
                        noise_map = np.mean(noise_np[i], axis=0).astype(np.float32)
                        noise_maps.append(noise_map)
                        labels_used.append(batch_labels[i].item())
                    
                except torch.cuda.OutOfMemoryError:
                    print("⚠️ GPU OOM, processing smaller batches...")
                    for i in range(images.shape[0]):
                        single_img = images[i:i+1]
                        with autocast('cuda'):
                            reconstructed = self.autoencoder(single_img)
                        noise = single_img - reconstructed
                        noise_map = np.mean(noise.cpu().numpy()[0], axis=0).astype(np.float32)
                        noise_maps.append(noise_map)
                        labels_used.append(batch_labels[i].item())
                
                # Memory cleanup
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        print(f"✅ Extracted {len(noise_maps)} noise maps")
        return noise_maps, labels_used
    
    def train_classifier_optimized(self, train_dataset, val_dataset=None, max_samples=100000):
        """Train classifier with optimized feature extraction"""
        print(f"\n{'='*70}")
        print(f"🌳 OPTIMIZED CLASSIFIER TRAINING")
        print(f"{'='*70}")
        
        if not self.autoencoder_trained:
            raise ValueError("Autoencoder must be trained first!")
        
        # Extract noise maps
        print(f"[1/3] 🔍 Extracting noise maps (max {max_samples:,} samples)...")
        train_noise_maps, train_labels = self.extract_noise_maps_fast(
            train_dataset, max_samples=max_samples, batch_size=32
        )
        
        print(f"📊 Training samples: {len(train_noise_maps):,}")
        unique, counts = np.unique(train_labels, return_counts=True)
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        for label, count in zip(unique, counts):
            if label < len(class_names):
                print(f"   {class_names[label]}: {count:,} samples")
        
        # Extract features
        print(f"[2/3] ⚡ Feature extraction...")
        feature_matrix = self.noise_extractor.fit_transform(train_noise_maps)
        
        print(f"📊 Feature matrix shape: {feature_matrix.shape}")
        
        # Train classifier with hyperparameter tuning
        print(f"[3/3] 🎯 Training classifier...")
        
        # Optimized Random Forest parameters for large dataset
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
            warm_start=False
        )
        
        # Train classifier
        start_time = time.time()
        self.classifier.fit(feature_matrix, train_labels)
        training_time = time.time() - start_time
        
        # Training metrics
        train_predictions = self.classifier.predict(feature_matrix)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_mcc = matthews_corrcoef(train_labels, train_predictions)
        
        print(f"✅ Classifier training completed in {training_time:.2f}s")
        print(f"🎯 Training Results:")
        print(f"   Accuracy: {train_accuracy:.4f}")
        print(f"   MCC: {train_mcc:.4f}")
        
        # Save classifier and scaler
        classifier_path = self.checkpoint_dir / 'classifier.pkl'
        scaler_path = self.checkpoint_dir / 'feature_scaler.pkl'
        
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.noise_extractor.scaler, f)
        
        self.classifier_trained = True
        
        # Validation if provided
        if val_dataset:
            print(f"🔍 Validating...")
            val_predictions, val_probas = self.predict_dataset(val_dataset, max_samples=10000)
            val_labels = val_dataset.labels[:len(val_predictions)]
            
            val_accuracy = accuracy_score(val_labels, val_predictions)
            val_mcc = matthews_corrcoef(val_labels, val_predictions)
            
            print(f"📊 Validation Results:")
            print(f"   Accuracy: {val_accuracy:.4f}")
            print(f"   MCC: {val_mcc:.4f}")
        
        # Cleanup
        del train_noise_maps, feature_matrix
        gc.collect()
        torch.cuda.empty_cache()
        
        print("✅ CLASSIFIER TRAINING COMPLETED!")
    
    def predict_dataset(self, dataset, max_samples=None, batch_size=64):
        """Generate predictions for dataset"""
        if not (self.autoencoder_trained and self.classifier_trained):
            raise ValueError("Both autoencoder and classifier must be trained!")
        
        print(f"🔮 Generating predictions...")
        
        # Extract noise maps
        noise_maps, used_labels = self.extract_noise_maps_fast(
            dataset, max_samples=max_samples, batch_size=batch_size
        )
        
        # Extract features and predict
        feature_matrix = self.noise_extractor.transform(noise_maps)
        predictions = self.classifier.predict(feature_matrix)
        probabilities = self.classifier.predict_proba(feature_matrix)
        
        # Cleanup
        del noise_maps, feature_matrix
        gc.collect()
        
        return predictions.tolist(), probabilities.tolist()
    
    def load_trained_models(self):
        """Load all trained models"""
        success = True
        
        # Load autoencoder
        if not self.load_autoencoder_checkpoint('best_autoencoder'):
            success = False
        
        # Load classifier
        classifier_path = self.checkpoint_dir / 'classifier.pkl'
        scaler_path = self.checkpoint_dir / 'feature_scaler.pkl'
        
        if classifier_path.exists() and scaler_path.exists():
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.noise_extractor.scaler = pickle.load(f)
            
            self.noise_extractor.fitted = True
            self.classifier_trained = True
            print("✅ Loaded classifier and scaler")
        else:
            print("❌ Classifier or scaler not found")
            success = False
        
        return success

def main():
    """Main execution optimized for 480k images"""
    print("🚀 MULTI-GPU OPTIMIZED PIPELINE FOR 480K+ IMAGES")
    print("="*80)
    
    # Configuration
    config = {
        'data_dir': './datasets/train',
        'cache_dir': './dataset_cache',
        'checkpoint_dir': './checkpoints_multi_gpu',
        'results_dir': './results_multi_gpu',
        'batch_size': 32,  # Per GPU
        'autoencoder_epochs': 60,
        'learning_rate': 1e-3,
        'max_files_per_class': None,  # Use all files
        'max_classifier_samples': 150000,  # For classifier training
    }
    
    # Create directories
    for dir_path in [config['cache_dir'], config['checkpoint_dir'], config['results_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"💾 Initial memory: {get_memory_usage()['rss_gb']:.2f} GB")
    
    # Load dataset
    print(f"\n📊 Loading complete dataset...")
    start_time = time.time()
    try:
        file_paths, labels = load_dataset_parallel(
            config['data_dir'], max_files_per_class=config['max_files_per_class']
        )
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded in {load_time:.2f}s: {len(file_paths):,} images")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None
    
    # Create datasets with caching
    print(f"\n🔄 Creating high-performance datasets...")
    train_dataset, val_dataset, test_dataset = create_stratified_splits(
        file_paths, labels, cache_dir=config['cache_dir']
    )
    
    print(f"💾 Memory after dataset creation: {get_memory_usage()['rss_gb']:.2f} GB")
    
    # Initialize pipeline
    pipeline = MultiGPUImageClassificationPipeline(
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Check for existing models
    existing_autoencoder = (Path(config['checkpoint_dir']) / 'best_autoencoder.pth').exists()
    existing_classifier = (Path(config['checkpoint_dir']) / 'classifier.pkl').exists()
    
    if existing_autoencoder and existing_classifier:
        print(f"\n🔄 Found existing models. Use them? (y/n): ", end="")
        choice = input().strip().lower()
        if choice == 'y':
            if pipeline.load_trained_models():
                print("✅ Using existing trained models")
                skip_training = True
            else:
                skip_training = False
        else:
            skip_training = False
    else:
        skip_training = False
    
    if not skip_training:
        # Train autoencoder
        if not existing_autoencoder:
            print(f"\n🎯 Training autoencoder...")
            start_time = time.time()
            try:
                pipeline.train_autoencoder_distributed(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=config['autoencoder_epochs'],
                    batch_size=config['batch_size'],
                    learning_rate=config['learning_rate']
                )
                train_time = time.time() - start_time
                print(f"✅ Autoencoder training completed in {train_time/60:.1f} minutes")
            except Exception as e:
                print(f"❌ Autoencoder training failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            pipeline.load_autoencoder_checkpoint('best_autoencoder')
        
        # Train classifier
        if not existing_classifier:
            print(f"\n🌳 Training classifier...")
            start_time = time.time()
            try:
                pipeline.train_classifier_optimized(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    max_samples=config['max_classifier_samples']
                )
                train_time = time.time() - start_time
                print(f"✅ Classifier training completed in {train_time/60:.1f} minutes")
            except Exception as e:
                print(f"❌ Classifier training failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # Load existing classifier
            classifier_path = Path(config['checkpoint_dir']) / 'classifier.pkl'
            scaler_path = Path(config['checkpoint_dir']) / 'feature_scaler.pkl'
            with open(classifier_path, 'rb') as f:
                pipeline.classifier = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                pipeline.noise_extractor.scaler = pickle.load(f)
            pipeline.noise_extractor.fitted = True
            pipeline.classifier_trained = True
    
    # Evaluate on test set
    print(f"\n📊 Final evaluation on test set...")
    start_time = time.time()
    try:
        # Use larger sample for final evaluation
        test_samples = min(50000, len(test_dataset))
        predictions, probabilities = pipeline.predict_dataset(
            test_dataset, max_samples=test_samples, batch_size=64
        )
        
        test_labels = test_dataset.labels[:len(predictions)]
        eval_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(test_labels, predictions)
        mcc = matthews_corrcoef(test_labels, predictions)
        cm = confusion_matrix(test_labels, predictions)
        
        print(f"✅ Evaluation completed in {eval_time:.2f}s")
        print(f"\n🎯 FINAL RESULTS:")
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        print(f"📈 Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"📊 Test samples evaluated: {len(predictions):,}")
        
        # Detailed results
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        print(f"\n📊 Confusion Matrix:")
        print("True\\Pred    Real  Synth  Semi")
        for i, (name, row) in enumerate(zip(['Real     ', 'Synthetic', 'Semi-synth'], cm)):
            print(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
        
        # Per-class metrics
        report = classification_report(
            test_labels, predictions, target_names=class_names, output_dict=True
        )
        print(f"\n📈 Per-Class Performance:")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name:12}: F1={metrics['f1-score']:.4f}, "
                      f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        
        # Class distribution in test set
        unique_test, counts_test = np.unique(test_labels, return_counts=True)
        print(f"\n📊 Test Set Distribution:")
        for label, count in zip(unique_test, counts_test):
            if label < len(class_names):
                print(f"  {class_names[label]:12}: {count:,} samples ({100*count/len(test_labels):.1f}%)")
        
        # Save comprehensive results
        results = {
            'final_metrics': {
                'accuracy': float(accuracy),
                'mcc': float(mcc),
                'test_samples': len(predictions)
            },
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                class_name: {
                    'f1_score': float(report[class_name]['f1-score']),
                    'precision': float(report[class_name]['precision']),
                    'recall': float(report[class_name]['recall']),
                    'support': int(report[class_name]['support'])
                } for class_name in class_names if class_name in report
            },
            'training_config': config,
            'dataset_info': {
                'total_images': len(file_paths),
                'train_images': len(train_dataset),
                'val_images': len(val_dataset),
                'test_images': len(test_dataset),
                'test_evaluated': len(predictions)
            },
            'model_info': {
                'autoencoder_epochs': config['autoencoder_epochs'],
                'batch_size_per_gpu': config['batch_size'],
                'total_gpus': pipeline.world_size,
                'effective_batch_size': config['batch_size'] * pipeline.world_size
            }
        }
        
        # Save results
        results_file = Path(config['results_dir']) / 'comprehensive_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions for analysis
        predictions_file = Path(config['results_dir']) / 'test_predictions.npz'
        np.savez_compressed(
            predictions_file,
            predictions=np.array(predictions),
            true_labels=np.array(test_labels),
            probabilities=np.array(probabilities)
        )
        
        # Performance summary
        final_memory = get_memory_usage()
        total_params = sum(p.numel() for p in pipeline.autoencoder.parameters())
        
        print(f"\n📋 PERFORMANCE SUMMARY:")
        print(f"   Dataset Size: {len(file_paths):,} total images")
        print(f"   Training Images: {len(train_dataset):,}")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   Test MCC: {mcc:.4f}")
        print(f"   Model Parameters: {total_params:,}")
        print(f"   GPUs Used: {pipeline.world_size}")
        print(f"   Final Memory: {final_memory['rss_gb']:.2f} GB")
        
        # Create visualization
        try:
            create_results_visualization(results, config['results_dir'])
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
        
        print(f"\n{'='*80}")
        print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Accuracy: {accuracy:.4f} | MCC: {mcc:.4f}")
        print(f"💾 Results saved to: {config['results_dir']}")
        print(f"{'='*80}")
        
        return results
        
    except Exception as e:
        print(f"❌ Final evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_results_visualization(results, results_dir):
    """Create comprehensive result visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8')
    results_dir = Path(results_dir)
    
    # Set up the plotting parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-GPU Image Classification Results', fontsize=20, fontweight='bold')
    
    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = np.array(results['confusion_matrix'])
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Per-class Performance
    ax2 = axes[0, 1]
    metrics_data = results['per_class_metrics']
    classes = list(metrics_data.keys())
    f1_scores = [metrics_data[c]['f1_score'] for c in classes]
    precisions = [metrics_data[c]['precision'] for c in classes]
    recalls = [metrics_data[c]['recall'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax2.bar(x - width, f1_scores, width, label='F1-Score', alpha=0.8)
    ax2.bar(x, precisions, width, label='Precision', alpha=0.8)
    ax2.bar(x + width, recalls, width, label='Recall', alpha=0.8)
    
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Class Performance Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # 3. Dataset Distribution
    ax3 = axes[1, 0]
    dataset_info = results['dataset_info']
    sizes = [dataset_info['train_images'], dataset_info['val_images'], 
             dataset_info['test_images']]
    labels = ['Train', 'Validation', 'Test']
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                       startangle=90, explode=(0.05, 0, 0))
    ax3.set_title('Dataset Split Distribution')
    
    # 4. Model Performance Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"""
MODEL PERFORMANCE SUMMARY

Final Test Accuracy: {results['final_metrics']['accuracy']:.4f}
Matthews Correlation: {results['final_metrics']['mcc']:.4f}
Test Samples: {results['final_metrics']['test_samples']:,}

TRAINING CONFIGURATION
Total Images: {dataset_info['total_images']:,}
Autoencoder Epochs: {results['model_info']['autoencoder_epochs']}
Batch Size per GPU: {results['model_info']['batch_size_per_gpu']}
Total GPUs: {results['model_info']['total_gpus']}
Effective Batch Size: {results['model_info']['effective_batch_size']}

DATASET BREAKDOWN
Training: {dataset_info['train_images']:,} images
Validation: {dataset_info['val_images']:,} images  
Test: {dataset_info['test_images']:,} images
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = results_dir / 'results_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'results_visualization.pdf', bbox_inches='tight')
    plt.close()
    
    # Create separate detailed confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Detailed Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {results_dir}")

def check_system_requirements():
    """Check system requirements and provide recommendations"""
    print("🔍 SYSTEM REQUIREMENTS CHECK")
    print("="*50)
    
    # GPU check
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("❌ No CUDA GPUs detected!")
        print("   This pipeline requires at least 1 GPU for optimal performance")
        return False
    
    print(f"✅ GPUs detected: {gpu_count}")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Memory check
    total_memory = psutil.virtual_memory().total / (1024**3)
    available_memory = psutil.virtual_memory().available / (1024**3)
    print(f"✅ System RAM: {total_memory:.1f} GB (Available: {available_memory:.1f} GB)")
    
    if available_memory < 16:
        print("⚠️ Warning: Less than 16GB available RAM. Consider closing other applications.")
    
    # CPU check
    cpu_count = multiprocessing.cpu_count()
    print(f"✅ CPU cores: {cpu_count}")
    
    if cpu_count < 8:
        print("⚠️ Warning: Less than 8 CPU cores. Data loading might be slower.")
    
    # Disk space check (approximate)
    disk_usage = psutil.disk_usage('.')
    free_space = disk_usage.free / (1024**3)
    print(f"✅ Available disk space: {free_space:.1f} GB")
    
    if free_space < 100:
        print("⚠️ Warning: Less than 100GB free space. Ensure adequate space for caching.")
    
    print("="*50)
    return True

def print_usage_instructions():
    """Print usage instructions"""
    print("\n📖 USAGE INSTRUCTIONS")
    print("="*50)
    print("1. Prepare your dataset:")
    print("   └── datasets/train/")
    print("       ├── real/           (160k+ .pt files)")
    print("       ├── synthetic/      (160k+ .pt files)")
    print("       └── semi-synthetic/ (160k+ .pt files)")
    print()
    print("2. Each .pt file should contain:")
    print("   - PyTorch tensor with shape (C, H, W) or (N, C, H, W)")
    print("   - Images should be in RGB format")
    print("   - Values in range [0, 1] or [0, 255]")
    print()
    print("3. Run the pipeline:")
    print("   python pipeline.py")
    print()
    print("4. The pipeline will:")
    print("   - Load and cache the dataset")
    print("   - Train a denoising autoencoder (distributed)")
    print("   - Extract noise features")
    print("   - Train a Random Forest classifier")
    print("   - Evaluate on test set")
    print("="*50)

if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    try:
        print_usage_instructions()
        
        # Check system requirements
        if not check_system_requirements():
            print("❌ System requirements not met. Exiting.")
            exit(1)
        
        # Run main pipeline
        results = main()
        
        if results:
            print(f"\n✅ SUCCESS! Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            print(f"📊 Results saved with comprehensive metrics and visualizations")
        else:
            print(f"\n❌ PIPELINE FAILED!")
            exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)