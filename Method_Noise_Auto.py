import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from skimage.feature import local_binary_pattern
from scipy import stats
from scipy.fft import fft2, fftshift
import os
from typing import Tuple, List, Dict
from tqdm import tqdm
import pickle
import json
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import time
from torch.cuda.amp import GradScaler, autocast
import warnings
import gc
import psutil
import tempfile
import traceback
import glob
import shutil
import sys
from collections import OrderedDict
import h5py
import tenacity

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if torch.cuda.is_available():
    mp.set_start_method('spawn', force=True)

warnings.filterwarnings('ignore')

class EfficientPTFileDataset(Dataset):
    def __init__(self, image_metadata: List[Dict], device: str = 'cpu', 
                 augment: bool = False, cache_size: int = 500):
        self.image_metadata = image_metadata
        self.device = device
        self.augment = augment
        self.cache_size = cache_size
        self.file_cache = OrderedDict()
        
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
        
        print(f"✅ EfficientPTFileDataset created: {len(self.image_metadata)} samples")
        print(f"🗂️ File cache size: {cache_size} files")
        self._print_dataset_stats()
    
    def _print_dataset_stats(self):
        labels = [item['label'] for item in self.image_metadata]
        unique, counts = np.unique(labels, return_counts=True)
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        print(f"📊 Dataset distribution:")
        for label, count in zip(unique, counts):
            if label < len(class_names):
                print(f"   {class_names[label]}: {count:,} samples ({100*count/len(labels):.1f}%)")
    
    def _load_file_with_cache(self, file_path: str) -> torch.Tensor:
        if file_path in self.file_cache:
            self.file_cache.move_to_end(file_path)
            return self.file_cache[file_path]
        
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            if isinstance(data, dict) and 'images' in data:
                images = data['images']
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], torch.Tensor):
                    images = torch.stack(data)
                else:
                    images = torch.tensor(data)
            elif isinstance(data, torch.Tensor):
                images = data
            else:
                raise ValueError(f"Invalid data format in {file_path}")
            
            if len(self.file_cache) >= self.cache_size:
                self.file_cache.popitem(last=False)
            
            self.file_cache[file_path] = images
            return images
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    
    def clear_cache(self):
        self.file_cache.clear()
        gc.collect()
        print("🗑️ File cache cleared")
    
    def load_all_images(self) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"🔄 Loading all {len(self.image_metadata)} images into memory...")
        images_tensor = torch.zeros(len(self.image_metadata), 3, 64, 64, dtype=torch.float32)
        labels_tensor = torch.zeros(len(self.image_metadata), dtype=torch.long)
        
        for idx in tqdm(range(len(self.image_metadata)), desc="Loading images"):
            try:
                metadata = self.image_metadata[idx]
                file_path = metadata['file_path']
                image_idx = metadata['image_idx']
                label = metadata['label']
                
                images_data = self._load_file_with_cache(file_path)
                if image_idx >= images_data.shape[0]:
                    print(f"⚠️ Image index {image_idx} out of range in {file_path}, using index 0")
                    image_idx = 0
                
                image = images_data[image_idx].clone()
                
                if len(image.shape) == 2:
                    image = image.unsqueeze(0).repeat(3, 1, 1)
                elif len(image.shape) != 3:
                    print(f"⚠️ Invalid image shape: {image.shape}, creating fallback")
                    image = torch.zeros(3, 64, 64, dtype=torch.float32)
                else:
                    if image.shape[0] == 1:
                        image = image.repeat(3, 1, 1)
                    elif image.shape[0] > 3:
                        image = image[:3, :, :]
                    elif image.shape[0] != 3:
                        print(f"⚠️ Invalid channel count: {image.shape[0]}, creating fallback")
                        image = torch.zeros(3, 64, 64, dtype=torch.float32)
                
                if image.shape[1] < 32 or image.shape[2] < 32:
                    image = F.interpolate(image.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
                
                if image.dtype != torch.float32:
                    image = image.float()
                if image.max() > 1.0:
                    image = image / 255.0
                image = torch.clamp(image, 0.0, 1.0)
                
                images_tensor[idx] = image
                labels_tensor[idx] = label
                
                if idx % 1000 == 0:
                    print(f"Loaded {idx} images, memory: {psutil.Process().memory_info().rss / (1024**3):.2f}GB")
                    self.clear_cache()
                    
            except Exception as e:
                print(f"❌ Error loading image {idx}: {e}")
                images_tensor[idx] = torch.zeros(3, 64, 64, dtype=torch.float32)
                labels_tensor[idx] = label
        
        print(f"✅ Loaded {len(images_tensor)} images, memory: {images_tensor.element_size() * images_tensor.nelement() / (1024**3):.2f}GB")
        return images_tensor, labels_tensor
    
    def __len__(self):
        return len(self.image_metadata)
    
    def __getitem__(self, idx):
        if idx >= len(self.image_metadata):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_metadata)}")
        
        metadata = self.image_metadata[idx]
        file_path = metadata['file_path']
        image_idx = metadata['image_idx']
        label = metadata['label']
        
        try:
            images = self._load_file_with_cache(file_path)
            if image_idx >= images.shape[0]:
                print(f"⚠️ Image index {image_idx} out of range in {file_path}, using index 0")
                image_idx = 0
            
            image = images[image_idx].clone()
            
            if len(image.shape) == 2:
                image = image.unsqueeze(0).repeat(3, 1, 1)
            elif len(image.shape) != 3:
                print(f"⚠️ Invalid image shape: {image.shape}, creating fallback")
                image = torch.zeros(3, 64, 64, dtype=torch.float32)
            else:
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                elif image.shape[0] > 3:
                    image = image[:3, :, :]
                elif image.shape[0] != 3:
                    print(f"⚠️ Invalid channel count: {image.shape[0]}, creating fallback")
                    image = torch.zeros(3, 64, 64, dtype=torch.float32)
            
            if image.shape[1] < 32 or image.shape[2] < 32:
                image = F.interpolate(image.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
            
            if image.dtype != torch.float32:
                image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
            image = torch.clamp(image, 0.0, 1.0)
            
            if self.augment and self.transform and torch.rand(1) < 0.7:
                try:
                    image = self.transform(image)
                except Exception:
                    pass
            
            image = image.to('cpu')
            label = torch.tensor(label, dtype=torch.long).to('cpu')
            return image, label
            
        except Exception as e:
            print(f"❌ Error in __getitem__ for {file_path}[{image_idx}]: {e}")
            fallback_image = torch.zeros(3, 64, 64, dtype=torch.float32).to('cpu')
            fallback_label = torch.tensor(label, dtype=torch.long).to('cpu')
            return fallback_image, fallback_label

def create_dataset_metadata_optimized(data_dir: str, max_images_per_class: int = 24000) -> List[Dict]:
    print(f"🔍 Creating dataset metadata from: {data_dir}")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"❌ Data directory does not exist: {data_dir}")
    
    class_folders = ['real', 'synthetic', 'semi-synthetic']
    class_to_label = {name: idx for idx, name in enumerate(class_folders)}
    metadata = []
    total_files_processed = 0
    total_images_found = 0
    
    for class_name in class_folders:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️ Class directory {class_dir} not found, skipping.")
            continue
        
        pt_files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))
        print(f"📁 Found {len(pt_files)} .pt files in {class_dir}")
        
        if not pt_files:
            print(f"⚠️ No .pt files found in {class_dir}")
            continue
        
        class_image_count = 0
        
        for pt_file in tqdm(pt_files, desc=f"Processing {class_name} files"):
            if class_image_count >= max_images_per_class:
                break
                
            try:
                data = torch.load(pt_file, map_location='cpu', weights_only=False)
                
                if isinstance(data, dict) and 'images' in data:
                    num_images = data['images'].shape[0]
                elif isinstance(data, list):
                    num_images = len(data)
                elif isinstance(data, torch.Tensor):
                    num_images = data.shape[0]
                else:
                    print(f"⚠️ Invalid .pt file format in {pt_file}, skipping.")
                    continue
                
                images_to_add = min(num_images, max_images_per_class - class_image_count)
                for img_idx in range(images_to_add):
                    metadata.append({
                        'file_path': pt_file,
                        'image_idx': img_idx,
                        'label': class_to_label[class_name],
                        'class_name': class_name,
                        'file_info': {
                            'total_images_in_file': num_images,
                            'file_size_mb': os.path.getsize(pt_file) / (1024 * 1024)
                        }
                    })
                    class_image_count += 1
                    total_images_found += 1
                
                total_files_processed += 1
                del data
                
                if class_image_count >= max_images_per_class:
                    print(f"   ✅ Reached max images for {class_name}: {class_image_count}")
                    break
                    
            except Exception as e:
                print(f"⚠️ Error processing {pt_file}: {e}")
                continue
    
    if not metadata:
        raise ValueError("❌ No valid data found in dataset!")
    
    print(f"\n✅ Dataset metadata created:")
    print(f"   📊 Total files processed: {total_files_processed}")
    print(f"   📊 Total images found: {total_images_found:,}")
    print(f"   📊 Metadata entries: {len(metadata):,}")
    
    labels = [item['label'] for item in metadata]
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    print(f"   📊 Class distribution:")
    for label, count in zip(unique, counts):
        if label < len(class_names):
            print(f"      {class_names[label]}: {count:,} images")
    
    return metadata

def create_stratified_splits_from_metadata(metadata: List[Dict], results_dir: str) -> Dict[str, List[Dict]]:
    print(f"\n🔀 Creating stratified splits from {len(metadata)} samples...")
    
    labels = [item['label'] for item in metadata]
    
    train_metadata, temp_metadata, train_labels, temp_labels = train_test_split(
        metadata, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    val_metadata, test_metadata, val_labels, test_labels = train_test_split(
        temp_metadata, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    splits = {
        'train': train_metadata,
        'val': val_metadata,
        'test': test_metadata
    }
    
    os.makedirs(results_dir, exist_ok=True)
    splits_path = os.path.join(results_dir, 'dataset_splits_metadata.json')
    
    try:
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"💾 Dataset splits saved to {splits_path}")
    except Exception as e:
        print(f"⚠️ Error saving splits: {e}")
    
    print(f"\n✅ Stratified splits created:")
    print(f"   📊 Train: {len(train_metadata):,} samples")
    print(f"   📊 Validation: {len(val_metadata):,} samples")
    print(f"   📊 Test: {len(test_metadata):,} samples")
    print(f"   📊 Total: {len(train_metadata) + len(val_metadata) + len(test_metadata):,} samples")
    
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    for split_name, split_data in [("Train", train_metadata), ("Val", val_metadata), ("Test", test_metadata)]:
        split_labels = [item['label'] for item in split_data]
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"   📊 {split_name} distribution:", end=" ")
        for label, count in zip(unique, counts):
            if label < len(class_names):
                print(f"{class_names[label]}={count:,}", end=" ")
        print()
    
    return splits

class AdvancedResidualAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(AdvancedResidualAutoencoder, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        self.encoder = nn.ModuleList([
            self._make_residual_block(input_channels, 64),
            nn.MaxPool2d(2, 2),
            self._make_residual_block(64, 128),
            nn.MaxPool2d(2, 2),
            self._make_residual_block(128, 256),
            nn.MaxPool2d(2, 2),
            self._make_residual_block(256, 512),
            nn.MaxPool2d(2, 2)
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.1)
        )
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, input_channels, 3, padding=1),
                nn.Sigmoid()
            )
        ])
        
        self.apply(self._init_weights)
    
    def _make_residual_block(self, in_channels, out_channels):
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
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i % 2 == 0:
                features.append(x)
        x = self.bottleneck(x)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).transpose(1, 2)
        x_attn, _ = self.attention(x_flat, x_flat, x_flat)
        x = x_attn.transpose(1, 2).view(b, c, h, w)
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConnection, self).__init__()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        return x

class UltraAdvancedFeatureExtractor:
    def __init__(self, n_jobs=-1, use_gpu_features=True):
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(f_classif, k='all')
        self.fitted = False
        self.n_jobs = min(n_jobs if n_jobs != -1 else mp.cpu_count(), 8)
        self.use_gpu_features = use_gpu_features and torch.cuda.is_available()
        print(f"🚀 Ultra-advanced feature extractor initialized with {self.n_jobs} CPU cores")
        if self.use_gpu_features:
            print("⚡ GPU acceleration enabled for feature extraction")
    
    def extract_comprehensive_features(self, noise_map: np.ndarray) -> np.ndarray:
        features = []
        noise_flat = noise_map.flatten()
        
        features.extend([
            np.mean(noise_flat), np.std(noise_flat), np.var(noise_flat),
            stats.skew(noise_flat) if len(noise_flat) > 0 else 0.0,
            stats.kurtosis(noise_flat) if len(noise_flat) > 0 else 0.0,
            np.min(noise_flat), np.max(noise_flat),
            stats.iqr(noise_flat), np.median(noise_flat),
            stats.entropy(np.histogram(noise_flat, bins=50)[0] + 1e-8)
        ])
        
        percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
        perc_values = np.percentile(noise_flat, percentiles)
        features.extend(perc_values.tolist())
        
        hist, _ = np.histogram(noise_flat, bins=50, range=(-1, 1))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist.tolist())
        
        h, w = noise_map.shape
        if h > 8 and w > 8:
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
            
            try:
                fft = fft2(noise_map)
                fft_shifted = fftshift(fft)
                magnitude_spectrum = np.abs(fft_shifted)
                
                freq_features = [
                    np.mean(magnitude_spectrum), np.std(magnitude_spectrum),
                    np.percentile(magnitude_spectrum, 90),
                    np.percentile(magnitude_spectrum, 95)
                ]
                features.extend(freq_features)
            except:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 12)
        
        try:
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(noise_map, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
            features.extend(lbp_hist.tolist())
        except:
            features.extend([0.0] * 10)
        
        target_length = 150
        current_length = len(features)
        if current_length < target_length:
            features.extend([0.0] * (target_length - current_length))
        features = features[:target_length]
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch_features_parallel(self, noise_maps_batch: List[np.ndarray]) -> List[np.ndarray]:
        if len(noise_maps_batch) <= 2 or self.n_jobs == 1:
            return [self.extract_comprehensive_features(nm) for nm in noise_maps_batch]
        
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(noise_maps_batch))) as executor:
            futures = [executor.submit(self.extract_comprehensive_features, nm) for nm in noise_maps_batch]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    results.append(np.zeros(150, dtype=np.float32))
            return results
    
    def fit_transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        print(f"🚀 Advanced feature extraction from {len(noise_maps)} noise maps...")
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
        
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.fitted = True
        print(f"✅ Advanced feature extraction completed: {feature_matrix.shape[1]} features per sample")
        return feature_matrix
    
    def transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
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

class OptimizedClassificationPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint_dir='ultra_checkpoints'):
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        print(f"🚀 Initializing Optimized Classification Pipeline...")
        print(f"🔧 Primary device: {self.device}")
        print(f"🔥 Available GPUs: {self.num_gpus}")
        
        self.autoencoder = None
        self.noise_extractor = UltraAdvancedFeatureExtractor(n_jobs=-1, use_gpu_features=True)
        
        self.base_classifiers = {
            'rf': RandomForestClassifier(
                n_estimators=2000, max_depth=30, min_samples_split=2, min_samples_leaf=1,
                random_state=42, class_weight='balanced_subsample', n_jobs=-1, bootstrap=True, max_features='sqrt'
            ),
            'hgb': HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=15, random_state=42
            ),
            'sgd_svc': SGDClassifier(
                loss='modified_huber', penalty='l2', alpha=0.0001, learning_rate='optimal',
                random_state=42, class_weight='balanced'
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam',
                alpha=0.001, learning_rate='adaptive', random_state=42
            )
        }
        
        self.ensemble_classifier = VotingClassifier(
            estimators=list(self.base_classifiers.items()), voting='soft', n_jobs=-1
        )
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.autoencoder_loaded = False
        self.classifier_trained = False
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.training_history = {'best_test_mcc': 0.0, 'best_individual_mccs': {}, 'ensemble_mcc': 0.0}
    
    def load_pretrained_autoencoder(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_autoencoder.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Pre-trained autoencoder not found at: {checkpoint_path}")
        
        try:
            print(f"🔄 Loading pre-trained autoencoder from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            self.autoencoder = AdvancedResidualAutoencoder().to(self.device)
            if self.num_gpus > 1:
                self.autoencoder = nn.DataParallel(self.autoencoder)
            
            if 'autoencoder_state_dict' in checkpoint and checkpoint['autoencoder_state_dict'] is not None:
                if self.num_gpus > 1 and hasattr(self.autoencoder, 'module'):
                    self.autoencoder.module.load_state_dict(checkpoint['autoencoder_state_dict'], strict=False)
                else:
                    self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'], strict=False)
                
                print("✅ Autoencoder weights loaded successfully")
                
                self.autoencoder.eval()
                print("🔍 Validating loaded autoencoder...")
                test_input = torch.randn(1, 3, 64, 64).to(self.device)
                with torch.no_grad():
                    test_output = self.autoencoder(test_input)
                if test_output.shape == test_input.shape:
                    print("✅ Autoencoder validation successful")
                    self.autoencoder_loaded = True
                else:
                    raise ValueError(f"❌ Autoencoder output shape mismatch: {test_output.shape} vs {test_input.shape}")
                del test_input, test_output
            else:
                raise ValueError("❌ No autoencoder state dict found in checkpoint")
                
        except Exception as e:
            print(f"❌ Error loading autoencoder: {e}")
            raise
    
    def extract_noise_from_images_advanced(self, images: torch.Tensor, batch_size: int = 128) -> List[np.ndarray]:
        if not self.autoencoder_loaded:
            raise ValueError("❌ Pre-trained autoencoder must be loaded first")
        
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
                    
                    noise_original = batch - reconstructed
                    
                    for j in range(noise_original.shape[0]):
                        noise_map = noise_original[j].cpu().numpy()
                        if noise_map.shape[0] > 1:
                            primary_noise = np.mean(noise_map, axis=0)
                        else:
                            primary_noise = noise_map[0]
                        
                        enhanced_noise = self._enhance_noise_map(primary_noise)
                        noise_maps.append(enhanced_noise.astype(np.float32))
                        
                    del batch, reconstructed, noise_original
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️ GPU memory error, reducing batch size...")
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
                        
                        del mini_batch, reconstructed, noise_original
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"✅ Extracted {len(noise_maps)} enhanced noise maps")
        return noise_maps
    
    def _enhance_noise_map(self, noise_map: np.ndarray) -> np.ndarray:
        try:
            enhanced = noise_map.copy()
            noise_uint8 = ((noise_map + 1) * 127.5).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            equalized = clahe.apply(noise_uint8)
            enhanced += 0.1 * ((equalized.astype(np.float32) / 127.5) - 1)
            
            blurred = cv2.GaussianBlur(noise_map, (3, 3), 1.0)
            unsharp = noise_map + 0.5 * (noise_map - blurred)
            enhanced += 0.1 * unsharp
            
            laplacian = cv2.Laplacian(noise_map, cv2.CV_32F, ksize=3)
            enhanced += 0.05 * laplacian
            
            enhanced = np.clip(enhanced, -2, 2)
            return enhanced
        except Exception:
            return noise_map
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(BlockingIOError),
        before_sleep=lambda retry_state: print(f"Retrying HDF5 file creation (attempt {retry_state.attempt_number})...")
    )
    def _create_hdf5_file(self, temp_path: str):
        return h5py.File(temp_path, 'w')
    
    def train_ensemble_classifier(self, train_dataset: EfficientPTFileDataset, val_loader: DataLoader = None, batch_size: int = 128):
        if self.classifier_trained:
            print("✅ Ensemble classifier already trained.")
            return
            
        if not self.autoencoder_loaded:
            raise ValueError("❌ Pre-trained autoencoder must be loaded first")
        
        print("\n" + "="*80)
        print(f"🚀 TRAINING ENSEMBLE CLASSIFIER WITH PRE-TRAINED AUTOENCODER (Using {self.num_gpus} GPUs)")
        print("="*80)
        
        total_start_time = time.time()
        
        print("\n[1/4] 🔄 Loading all training images...")
        loading_start = time.time()
        all_images, all_labels = train_dataset.load_all_images()
        loading_time = time.time() - loading_start
        print(f"✅ Loaded {len(all_images)} images in {loading_time:.2f}s")
        
        print("\n[2/4] 🔍 Extracting noise maps and features in batches...")
        extraction_start = time.time()
        feature_output_path = os.path.join(self.checkpoint_dir, 'features.h5')
        batch_idx = 0
        subset_noise_maps = []
        subset_labels = []
        subset_size = 20000
        use_hdf5 = True
        
        def print_memory_usage(step: str):
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"[{step}] Memory usage: {mem_info.rss / (1024**3):.2f}GB")
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f'features_temp_{os.getpid()}.h5')
        feature_output_dir = os.path.join(self.checkpoint_dir, 'features')
        
        try:
            with self._create_hdf5_file(temp_path) as f:
                for i in tqdm(range(0, len(all_images), batch_size), desc="⚡ Processing batches"):
                    print_memory_usage(f"Batch {batch_idx} start")
                    batch_images = all_images[i:i+batch_size]
                    batch_labels = all_labels[i:i+batch_size]
                    
                    noise_maps = self.extract_noise_from_images_advanced(batch_images, batch_size=batch_size)
                    
                    if len(subset_noise_maps) < subset_size:
                        remaining = subset_size - len(subset_noise_maps)
                        subset_noise_maps.extend(noise_maps[:remaining])
                        subset_labels.extend(batch_labels.cpu().numpy()[:remaining])
                    
                    if len(subset_noise_maps) >= subset_size and not self.noise_extractor.fitted:
                        print(f"\nFitting feature extractor on subset of {subset_size} samples...")
                        _ = self.noise_extractor.fit_transform(subset_noise_maps)
                        del subset_noise_maps
                        gc.collect()
                    
                    batch_features = self.noise_extractor.transform(noise_maps)
                    f.create_dataset(f'features_{batch_idx}', data=batch_features)
                    f.create_dataset(f'labels_{batch_idx}', data=batch_labels.cpu().numpy())
                    
                    all_images[i:i+batch_size] = torch.tensor([])
                    batch_idx += 1
                    
                    del noise_maps, batch_features, batch_images
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print_memory_usage(f"Batch {batch_idx} end")
            
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            shutil.move(temp_path, feature_output_path)
            print(f"💾 Moved HDF5 file to {feature_output_path}")
            
        except Exception as e:
            print(f"❌ Failed to create or move HDF5 file: {e}, falling back to NumPy")
            use_hdf5 = False
            os.makedirs(feature_output_dir, exist_ok=True)
            batch_idx = 0
            all_images, all_labels = train_dataset.load_all_images()  # Reload images
            for i in tqdm(range(0, len(all_images), batch_size), desc="⚡ Processing batches (NumPy)"):
                print_memory_usage(f"Batch {batch_idx} start")
                batch_images = all_images[i:i+batch_size]
                batch_labels = all_labels[i:i+batch_size]
                
                noise_maps = self.extract_noise_from_images_advanced(batch_images, batch_size=batch_size)
                
                if len(subset_noise_maps) < subset_size:
                    remaining = subset_size - len(subset_noise_maps)
                    subset_noise_maps.extend(noise_maps[:remaining])
                    subset_labels.extend(batch_labels.cpu().numpy()[:remaining])
                
                if len(subset_noise_maps) >= subset_size and not self.noise_extractor.fitted:
                    print(f"\nFitting feature extractor on subset of {subset_size} samples...")
                    _ = self.noise_extractor.fit_transform(subset_noise_maps)
                    del subset_noise_maps
                    gc.collect()
                
                batch_features = self.noise_extractor.transform(noise_maps)
                np.save(os.path.join(feature_output_dir, f'features_{batch_idx}.npy'), batch_features)
                np.save(os.path.join(feature_output_dir, f'labels_{batch_idx}.npy'), batch_labels.cpu().numpy())
                
                all_images[i:i+batch_size] = torch.tensor([])
                batch_idx += 1
                
                del noise_maps, batch_features, batch_images
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print_memory_usage(f"Batch {batch_idx} end")
        
        del all_images, all_labels
        gc.collect()
        print_memory_usage("After freeing images")
        
        extraction_time = time.time() - extraction_start
        print(f"✅ Extracted features for {batch_idx} batches in {extraction_time:.2f}s")
        
        print(f"\n[3/4] 🎯 Feature selection on subset...")
        subset_features = self.noise_extractor.transform(subset_noise_maps[:subset_size])
        self.feature_selector = SelectKBest(f_classif, k=min(100, subset_features.shape[1]))
        self.feature_selector.fit(subset_features, subset_labels[:subset_size])
        print(f"📊 Selected {self.feature_selector.get_support().sum()} most informative features")
        del subset_features, subset_labels
        gc.collect()
        
        print(f"\n[4/4] 🌳 Training classifiers on all features...")
        classifier_start = time.time()
        all_features = []
        all_labels = []
        
        if use_hdf5:
            with h5py.File(feature_output_path, 'r') as f:
                num_batches = len([k for k in f.keys() if k.startswith('features_')])
                for b in tqdm(range(num_batches), desc="Loading features"):
                    print_memory_usage(f"Loading batch {b}")
                    batch_features = f[f'features_{b}'][:]
                    batch_labels = f[f'labels_{b}'][:]
                    batch_features_selected = self.feature_selector.transform(batch_features)
                    all_features.append(batch_features_selected)
                    all_labels.append(batch_labels)
                    del batch_features, batch_features_selected
                    gc.collect()
        else:
            num_batches = len(glob.glob(os.path.join(feature_output_dir, 'features_*.npy')))
            for b in tqdm(range(num_batches), desc="Loading features (NumPy)"):
                print_memory_usage(f"Loading batch {b}")
                batch_features = np.load(os.path.join(feature_output_dir, f'features_{b}.npy'))
                batch_labels = np.load(os.path.join(feature_output_dir, f'labels_{b}.npy'))
                batch_features_selected = self.feature_selector.transform(batch_features)
                all_features.append(batch_features_selected)
                all_labels.append(batch_labels)
                del batch_features, batch_features_selected
                gc.collect()
        
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        print(f"✅ Loaded feature matrix: {all_features.shape}")
        print(f"✅ Loaded labels: {all_labels.shape}")
        
        print(f"📊 Training class distribution:")
        unique, counts = np.unique(all_labels, return_counts=True)
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        for label, count in zip(unique, counts):
            print(f"   {class_names[label]}: {count:,} samples ({100*count/len(all_labels):.1f}%)")
        
        for name, classifier in self.base_classifiers.items():
            print(f"Training {name} classifier...")
            classifier.fit(all_features, all_labels)
        
        self.ensemble_classifier.estimators_ = [(name, clf) for name, clf in self.base_classifiers.items() if hasattr(clf, 'classes_')]
        
        if val_loader is not None:
            print(f"\n🔍 Validating on validation set...")
            val_predictions, val_probabilities = self.predict(val_loader, batch_size=batch_size)
            val_labels = []
            for _, labels in val_loader:
                val_labels.extend(labels.cpu().numpy())
            
            min_len = min(len(val_labels), len(val_predictions))
            val_labels = val_labels[:min_len]
            val_predictions = val_predictions[:min_len]
            
            val_mcc = matthews_corrcoef(val_labels, val_predictions)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            print(f"\n✅ VALIDATION RESULTS:")
            print(f"   🎯 Validation MCC: {val_mcc:.4f}")
            print(f"   🎯 Validation Accuracy: {val_accuracy:.4f}")
        
        classifier_time = time.time() - classifier_start
        print(f"✅ Ensemble training completed in {classifier_time:.2f}s")
        
        self.classifier_trained = True
        self.save_classifier()
        
        total_time = time.time() - total_start_time
        print("\n" + "="*80)
        print("🎉 ENSEMBLE TRAINING COMPLETED!")
        print("="*80)
        print(f"📊 Processed {len(all_labels):,} training samples")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print("="*80)
    
    def predict(self, test_loader: DataLoader, batch_size: int = 128) -> Tuple[List[int], List[float]]:
        if not self.autoencoder_loaded or not self.classifier_trained:
            raise ValueError("❌ Both autoencoder and ensemble must be loaded/trained")
        if not self.noise_extractor.fitted:
            raise ValueError("❌ Feature extractor must be fitted!")
        
        print("🔮 Generating ensemble predictions...")
        all_predictions = []
        all_probabilities = []
        
        self.autoencoder.eval()
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="⚡ Prediction")):
                try:
                    images = images.to(self.device, non_blocking=True)
                    noise_maps = self.extract_noise_from_images_advanced(images, batch_size=batch_size)
                    feature_matrix = self.noise_extractor.transform(noise_maps)
                    
                    if hasattr(self, 'feature_selector'):
                        feature_matrix = self.feature_selector.transform(feature_matrix)
                    
                    predictions = self.ensemble_classifier.predict(feature_matrix)
                    probabilities = self.ensemble_classifier.predict_proba(feature_matrix)
                    
                    all_predictions.extend(predictions.tolist())
                    all_probabilities.extend(probabilities.tolist())
                except Exception as e:
                    print(f"⚠️ Error predicting batch {batch_idx}: {e}")
                    batch_size_actual = len(images)
                    all_predictions.extend([0] * batch_size_actual)
                    all_probabilities.extend([[1.0, 0.0, 0.0]] * batch_size_actual)
                
                del images, noise_maps, feature_matrix
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"✅ Generated predictions for {len(all_predictions)} samples")
        return all_predictions, all_probabilities
    
    def evaluate(self, test_loader: DataLoader, test_labels: List[int], 
                save_results: bool = True, results_dir: str = 'results') -> Dict:
        print("\n🔍 Starting evaluation...")
        eval_start = time.time()
        
        predictions, probabilities = self.predict(test_loader)
        
        min_len = min(len(test_labels), len(predictions))
        test_labels = test_labels[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        
        mcc = matthews_corrcoef(test_labels, predictions)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, 
                                    target_names=['Real', 'Synthetic', 'Semi-synthetic'],
                                    output_dict=True, zero_division=0)
        cm = confusion_matrix(test_labels, predictions)
        eval_time = time.time() - eval_start
        
        per_class_mcc = []
        for i in range(3):
            binary_true = (np.array(test_labels) == i).astype(int)
            binary_pred = (np.array(predictions) == i).astype(int)
            if len(np.unique(binary_true)) > 1 and len(np.unique(binary_pred)) > 1:
                class_mcc = matthews_corrcoef(binary_true, binary_pred)
            else:
                class_mcc = 0.0
            per_class_mcc.append(class_mcc)
        
        probabilities_array = np.array(probabilities)
        max_probs = np.max(probabilities_array, axis=1)
        entropy_scores = -np.sum(probabilities_array * np.log(probabilities_array + 1e-8), axis=1)
        
        confidence_metrics = {
            'mean_max_probability': np.mean(max_probs),
            'std_max_probability': np.std(max_probs),
            'mean_entropy': np.mean(entropy_scores),
            'high_confidence_ratio': np.mean(max_probs > 0.9),
            'low_confidence_ratio': np.mean(max_probs < 0.6)
        }
        
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
        print("🎯 EVALUATION RESULTS")
        print("="*80)
        print(f"🏆 OVERALL MCC: {mcc:.6f}")
        print(f"🏆 OVERALL ACCURACY: {accuracy:.6f}")
        print(f"⏱️ Evaluation time: {eval_time:.2f} seconds")
        
        if mcc > 0.95:
            print(f"🎉 TARGET ACHIEVED: MCC > 0.95! ✅")
        elif mcc > 0.90:
            print(f"🎯 Excellent performance! MCC > 0.90 ✅")
        
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        print(f"\n📊 Per-class MCC scores:")
        for i, (class_name, class_mcc) in enumerate(zip(class_names, per_class_mcc)):
            print(f"   {class_name}: {class_mcc:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print("True\\Pred    Real  Synth  Semi")
        for i, (name, row) in enumerate(zip(['Real     ', 'Synthetic', 'Semi-synth'], cm)):
            print(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
        
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            
            results_file = os.path.join(results_dir, 'evaluation_results.json')
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
            
            plt.figure(figsize=(10, 8))
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            annotations = []
            for i in range(cm.shape[0]):
                row = []
                for j in range(cm.shape[1]):
                    row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
                annotations.append(row)
            
            sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Percentage'})
            plt.title(f'Confusion Matrix\nMCC: {mcc:.4f} | Accuracy: {accuracy:.4f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if mcc > 0.95:
                plt.text(0.5, -0.15, '🎉 TARGET ACHIEVED: MCC > 0.95!', 
                        transform=plt.gca().transAxes, ha='center', fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            
            cm_path = os.path.join(results_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Results saved to: {results_dir}")
        
        print("="*80)
        return results
    
    def save_classifier(self):
        try:
            extractor_path = os.path.join(self.checkpoint_dir, 'feature_extractor.pkl')
            with open(extractor_path, 'wb') as f:
                pickle.dump(self.noise_extractor, f)
            print(f"💾 Feature extractor saved: {extractor_path}")
            
            if hasattr(self, 'feature_selector'):
                selector_path = os.path.join(self.checkpoint_dir, 'feature_selector.pkl')
                with open(selector_path, 'wb') as f:
                    pickle.dump(self.feature_selector, f)
                print(f"💾 Feature selector saved: {selector_path}")
            
            ensemble_path = os.path.join(self.checkpoint_dir, 'ensemble_classifier.pkl')
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble_classifier, f)
            print(f"💾 Ensemble classifier saved: {ensemble_path}")
            
            for name, classifier in self.base_classifiers.items():
                if hasattr(classifier, 'classes_'):
                    classifier_path = os.path.join(self.checkpoint_dir, f'{name}_classifier.pkl')
                    with open(classifier_path, 'wb') as f:
                        pickle.dump(classifier, f)
                    print(f"💾 {name} classifier saved: {classifier_path}")
            
            history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"💾 Training history saved: {history_path}")
            
        except Exception as e:
            print(f"⚠️ Error saving classifiers: {e}")
    
    def load_classifier(self):
        try:
            extractor_path = os.path.join(self.checkpoint_dir, 'feature_extractor.pkl')
            if os.path.exists(extractor_path):
                with open(extractor_path, 'rb') as f:
                    self.noise_extractor = pickle.load(f)
                print("✅ Feature extractor loaded")
            
            selector_path = os.path.join(self.checkpoint_dir, 'feature_selector.pkl')
            if os.path.exists(selector_path):
                with open(selector_path, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                print("✅ Feature selector loaded")
            
            ensemble_path = os.path.join(self.checkpoint_dir, 'ensemble_classifier.pkl')
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'rb') as f:
                    self.ensemble_classifier = pickle.load(f)
                self.classifier_trained = True
                print("✅ Ensemble classifier loaded")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error loading classifiers: {e}")
            return False

def main_optimized(data_dir: str = "./datasets/train", 
                  max_images_per_class: int = 24000,
                  batch_size: int = 128,
                  results_dir: str = "./results",
                  checkpoint_dir: str = "./ultra_checkpoints"):
    print("🚀 OPTIMIZED CLASSIFICATION PIPELINE FOR LARGE DATASETS")
    print("🎯 Memory-efficient feature extraction with pre-trained autoencoder")
    print("="*80)
    print(f"📊 Configuration:")
    print(f"   Data directory: {data_dir}")
    print(f"   Max images per class: {max_images_per_class:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Results directory: {results_dir}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    print("="*80)
    
    pipeline = OptimizedClassificationPipeline(device='cuda' if torch.cuda.is_available() else 'cpu',
                                            checkpoint_dir=checkpoint_dir)
    
    try:
        if torch.cuda.is_available():
            torch.cuda.init()
        
        pipeline.load_pretrained_autoencoder()
        
        metadata = create_dataset_metadata_optimized(data_dir, max_images_per_class)
        
        splits = create_stratified_splits_from_metadata(metadata, results_dir)
        
        train_dataset = EfficientPTFileDataset(
            splits['train'],
            device=pipeline.device,
            augment=False,
            cache_size=500
        )
        val_dataset = EfficientPTFileDataset(
            splits['val'],
            device=pipeline.device,
            augment=False,
            cache_size=200
        )
        test_dataset = EfficientPTFileDataset(
            splits['test'],
            device=pipeline.device,
            augment=False,
            cache_size=200
        )
        
        num_workers = min(mp.cpu_count(), 4) if torch.cuda.is_available() else 0
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        pipeline.train_ensemble_classifier(train_dataset, val_loader, batch_size=batch_size)
        
        test_labels = [item['label'] for item in splits['test']]
        results = pipeline.evaluate(
            test_loader,
            test_labels,
            save_results=True,
            results_dir=results_dir
        )
        
        train_dataset.clear_cache()
        val_dataset.clear_cache()
        test_dataset.clear_cache()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\n" + "="*80)
        print("🎉 PIPELINE EXECUTION COMPLETED")
        print("="*80)
        print(f"🏆 Final MCC: {results['mcc']:.6f}")
        print(f"🏆 Final Accuracy: {results['accuracy']:.6f}")
        print(f"⏱️ Prediction speed: {results['prediction_speed']:.1f} samples/second")
        print(f"📊 Confidence metrics:")
        for metric, value in results['confidence_metrics'].items():
            print(f"   {metric}: {value:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        traceback.print_exc()
        return None
    finally:
        if 'train_dataset' in locals():
            train_dataset.clear_cache()
        if 'val_dataset' in locals():
            val_dataset.clear_cache()
        if 'test_dataset' in locals():
            test_dataset.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Image Classification Pipeline")
    parser.add_argument('--data-dir', default='./datasets/train', 
                       help='Directory containing training data')
    parser.add_argument('--max-images-per-class', type=int, default=24000,
                       help='Maximum number of images per class')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for feature extraction')
    parser.add_argument('--results-dir', default='./results',
                       help='Directory to save results')
    parser.add_argument('--checkpoint-dir', default='./ultra_checkpoints',
                       help='Directory for model checkpoints')
    
    args = parser.parse_args()
    
    results = main_optimized(
        data_dir=args.data_dir,
        max_images_per_class=args.max_images_per_class,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    if results is not None:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)