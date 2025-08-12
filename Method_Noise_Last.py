import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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
import glob
import shutil
import sys

warnings.filterwarnings('ignore')

class PTFileDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], device: str = 'cpu', augment: bool = False):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.device = device
        self.augment = augment
        self.file_indices = []
        self._build_file_indices()
        
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
    
    def _build_file_indices(self):
        """Map each index to (file_path, image_idx_within_file)."""
        print("🔍 Building file indices for dataset...")
        total_files = len(self.image_paths)
        
        for file_idx, file_path in enumerate(tqdm(self.image_paths, desc="📂 Indexing files", unit="files")):
            try:
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                if isinstance(data, dict) and 'images' in data:
                    num_images = data['images'].shape[0]
                elif isinstance(data, list):
                    num_images = len(data)
                elif isinstance(data, torch.Tensor):
                    num_images = data.shape[0]
                else:
                    continue
                for i in range(num_images):
                    self.file_indices.append((file_path, i))
                    
                # Progress update every 100 files
                if (file_idx + 1) % 100 == 0:
                    print(f"   📊 Processed {file_idx + 1}/{total_files} files, found {len(self.file_indices)} images so far")
                    
            except Exception as e:
                if file_idx % 50 == 0:  # Only print errors occasionally to avoid spam
                    print(f"   ⚠️ Error reading {file_path}: {e}")
                continue
        
        print(f"✅ File indexing complete: {len(self.file_indices)} total images from {total_files} files")
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        file_path, image_idx = self.file_indices[idx]
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            if isinstance(data, dict) and 'images' in data:
                image = data['images'][image_idx]
            elif isinstance(data, list):
                image = data[image_idx]
            elif isinstance(data, torch.Tensor):
                image = data[image_idx]
            else:
                raise ValueError(f"Invalid data format in {file_path}")
            
            if image.dtype != torch.uint8:
                if image.max() > 1.0 and image.max() <= 255.0 and image.min() >= 0.0:
                    image = image.to(torch.uint8)
                else:
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    image = (image * 255.0).clamp(0, 255).to(torch.uint8)
            
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] > 3:
                image = image[:3, :, :]
            
            if image.shape[1] < 32 or image.shape[2] < 32:
                raise ValueError(f"Invalid image size in {file_path}: {image.shape}")
            
            image = image.float() / 255.0
            if self.augment and self.transform and torch.rand(1) < 0.7:
                image = self.transform(image)
            
            label = self.labels[idx]
            return image.to(self.device), label.to(self.device)
        except Exception as e:
            print(f"⚠️ Error loading image {idx} from {file_path}: {e}")
            return torch.zeros(3, 224, 224, dtype=torch.float32).to(self.device), self.labels[idx].to(self.device)

def collect_image_paths_and_labels(data_dir: str, max_images_per_class: int = 160000) -> Tuple[List[str], List[int]]:
    """
    Collect file paths and labels for images without loading into memory.
    Returns: (image_paths, labels)
    """
    print(f"\n🔍 COLLECTING IMAGE PATHS AND LABELS")
    print(f"📂 Data directory: {data_dir}")
    print(f"📊 Max images per class: {max_images_per_class:,}")
    print("="*60)
    
    class_folders = ['real', 'synthetic', 'semi-synthetic']
    class_to_label = {name: idx for idx, name in enumerate(class_folders)}
    image_paths = []
    labels = []
    skipped_files = 0
    
    for class_idx, class_name in enumerate(class_folders):
        print(f"\n📁 Processing class [{class_idx+1}/3]: {class_name.upper()}")
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"⚠️ Class directory {class_dir} not found, skipping.")
            continue
        
        pt_files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))
        print(f"   📄 Found {len(pt_files)} .pt files in directory")
        
        if len(pt_files) > max_images_per_class:
            pt_files = pt_files[:max_images_per_class]
            print(f"   ✂️ Limited to first {max_images_per_class} files")
        
        if not pt_files:
            print(f"   ⚠️ No .pt files found in {class_dir}, skipping.")
            continue
        
        class_images_count = 0
        for file_idx, pt_file in enumerate(tqdm(pt_files, desc=f"   🔄 Processing {class_name} files", unit="files")):
            try:
                data = torch.load(pt_file, map_location='cpu', weights_only=False)
                if isinstance(data, dict) and 'images' in data:
                    num_images = data['images'].shape[0]
                elif isinstance(data, list):
                    num_images = len(data)
                elif isinstance(data, torch.Tensor):
                    num_images = data.shape[0]
                else:
                    print(f"   ⚠️ Invalid .pt file format in {pt_file}, skipping.")
                    skipped_files += 1
                    continue
                
                for _ in range(num_images):
                    image_paths.append(pt_file)
                    labels.append(class_to_label[class_name])
                    class_images_count += 1
                
                # Progress update every 50 files
                if (file_idx + 1) % 50 == 0:
                    print(f"      📊 Processed {file_idx + 1}/{len(pt_files)} files, {class_images_count:,} images so far")
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {pt_file}: {e}")
                skipped_files += 1
                continue
        
        print(f"   ✅ Class {class_name}: {class_images_count:,} images from {len(pt_files)} files")
    
    if not image_paths:
        raise ValueError("❌ No valid data loaded from dataset!")
    
    print(f"\n✅ COLLECTION COMPLETE:")
    print(f"   📊 Total images: {len(image_paths):,}")
    print(f"   📄 Skipped files: {skipped_files:,}")
    print(f"   🎯 Success rate: {100*(len(image_paths))/(len(image_paths)+skipped_files):.1f}%")
    
    # Print class distribution
    class_counts = np.bincount(labels)
    print(f"\n📊 CLASS DISTRIBUTION:")
    for i, count in enumerate(class_counts):
        class_name = ['Real', 'Synthetic', 'Semi-synthetic'][i]
        percentage = 100 * count / len(labels)
        print(f"   {class_name}: {count:,} images ({percentage:.1f}%)")
    
    return image_paths, labels

def save_splits_to_json(image_paths: List[str], labels: List[int], results_dir: str) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Split image paths into train/val/test and save to JSON.
    Returns: (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    print(f"\n📊 CREATING DATASET SPLITS")
    print("="*50)
    
    print("🔄 Performing stratified split (70% train, 15% val, 15% test)...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    splits = {
        'train': {'paths': train_paths, 'labels': train_labels},
        'val': {'paths': val_paths, 'labels': val_labels},
        'test': {'paths': test_paths, 'labels': test_labels}
    }
    
    print(f"💾 Saving splits to JSON...")
    json_path = os.path.join(results_dir, 'dataset_splits.json')
    try:
        os.makedirs(results_dir, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(splits, f, indent=4)
        print(f"   ✅ Dataset splits saved to {json_path}")
    except Exception as e:
        print(f"   ⚠️ Error saving JSON: {e}")
    
    print(f"\n✅ STRATIFIED SPLIT COMPLETED:")
    print(f"   📊 Train: {len(train_paths):,} samples")
    print(f"   📊 Validation: {len(val_paths):,} samples")
    print(f"   📊 Test: {len(test_paths):,} samples")
    print(f"   📊 Total: {len(train_paths) + len(val_paths) + len(test_paths):,} samples")
    
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    splits_data = [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]
    
    print(f"\n📊 CLASS DISTRIBUTION PER SPLIT:")
    for split_name, split_labels in splits_data:
        class_counts = np.bincount(split_labels, minlength=3)
        total = sum(class_counts)
        print(f"   {split_name}:")
        for i, count in enumerate(class_counts):
            percentage = 100 * count / total if total > 0 else 0
            print(f"     {class_names[i]}: {count:,} ({percentage:.1f}%)")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

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
        self.n_jobs = n_jobs if n_jobs != -1 else min(mp.cpu_count(), 16)
        self.use_gpu_features = use_gpu_features and torch.cuda.is_available()
        print(f"🚀 Ultra-advanced feature extractor initialized with {self.n_jobs} CPU cores")
        if self.use_gpu_features:
            print("⚡ GPU acceleration enabled for feature extraction")
    
    def extract_comprehensive_features(self, noise_map: np.ndarray) -> np.ndarray:
        """Extract 500+ comprehensive features from noise map with detailed progress"""
        features = []
        
        # 1. Basic statistical features
        noise_flat = noise_map.flatten()
        features.extend([
            np.mean(noise_flat), np.std(noise_flat), np.var(noise_flat),
            stats.skew(noise_flat) if len(noise_flat) > 0 else 0.0,
            stats.kurtosis(noise_flat) if len(noise_flat) > 0 else 0.0,
            np.min(noise_flat), np.max(noise_flat),
            stats.iqr(noise_flat), np.median(noise_flat),
            stats.entropy(np.histogram(noise_flat, bins=50)[0] + 1e-8)
        ])
        
        # 2. Percentile features
        percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
        perc_values = np.percentile(noise_flat, percentiles)
        features.extend(perc_values.tolist())
        
        # 3. Histogram features
        hist, _ = np.histogram(noise_flat, bins=50, range=(-1, 1))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist.tolist())
        
        # 4. Gradient and edge features
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
            
            # Laplacian features
            laplacian = cv2.Laplacian(noise_map, cv2.CV_32F)
            lap_features = [
                np.mean(laplacian), np.std(laplacian),
                np.percentile(np.abs(laplacian), 90),
                np.sum(laplacian > 0) / laplacian.size,
                np.sum(laplacian < 0) / laplacian.size
            ]
            features.extend(lap_features)
            
            # Spatial features
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
            
            # Regional correlation features
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
            
            while len(features) < len(features) + (30 - (len(features) % 30)):
                features.append(0.0)
        else:
            features.extend([0.0] * 30)
        
        # 5. Frequency domain features
        try:
            fft = fft2(noise_map)
            fft_shifted = fftshift(fft)
            magnitude_spectrum = np.abs(fft_shifted)
            phase_spectrum = np.angle(fft_shifted)
            
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
            
            while len(features) % 40 != (len(features) - len(freq_features) - len(psd_features)) % 40:
                features.append(0.0)
        except Exception as e:
            features.extend([0.0] * 40)
        
        # 6. Wavelet features
        try:
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
        
        # 7. Texture features (LBP and GLCM)
        try:
            # Local Binary Pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(noise_map, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
            features.extend(lbp_hist.tolist()[:20])
            
            # Gray Level Co-occurrence Matrix
            noise_uint8 = ((noise_map + 1) * 127.5).astype(np.uint8)
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm_features = []
            
            for distance in distances:
                for angle in angles:
                    try:
                        glcm = graycomatrix(noise_uint8, [distance], [angle], levels=32, symmetric=True, normed=True)
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
        
        # 8. Gabor filter features
        try:
            frequencies = [0.1, 0.3, 0.5, 0.7]
            angles = [0, 45, 90, 135]
            for frequency in frequencies:
                for angle in angles:
                    real, _ = gabor(noise_map, frequency=frequency, theta=np.radians(angle))
                    features.extend([
                        np.mean(real), np.std(real),
                        np.percentile(np.abs(real), 90)
                    ])
        except Exception as e:
            features.extend([0.0] * 48)
        
        # 9. Advanced pattern analysis
        try:
            # Autocorrelation
            autocorr = np.correlate(noise_flat, noise_flat, mode='full')
            autocorr_norm = autocorr / np.max(autocorr)
            center = len(autocorr_norm) // 2
            window = min(50, center)
            autocorr_window = autocorr_norm[center-window:center+window]
            features.extend([
                np.mean(autocorr_window), np.std(autocorr_window),
                np.max(autocorr_window), np.argmax(autocorr_window) - window,
                np.sum(autocorr_window > 0.5), np.sum(autocorr_window < -0.5)
            ])
            
            # Structure tensor
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
            
            while len(features) % 30 != (len(features) - 21) % 30:
                features.append(0.0)
        except Exception as e:
            features.extend([0.0] * 30)
        
        # 10. Multi-scale analysis
        try:
            scales = [1, 2, 4, 8]
            for scale in scales:
                if scale > 1:
                    downsampled = cv2.resize(noise_map, (max(8, w//scale), max(8, h//scale)), interpolation=cv2.INTER_AREA)
                else:
                    downsampled = noise_map
                scale_flat = downsampled.flatten()
                features.extend([
                    np.mean(scale_flat), np.std(scale_flat),
                    np.percentile(scale_flat, 95),
                    np.percentile(scale_flat, 5),
                    stats.skew(scale_flat) if len(scale_flat) > 0 else 0.0
                ])
            
            # Edge analysis
            edges = cv2.Canny((noise_map * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            num_labels, labels_img = cv2.connectedComponents(edges)
            features.extend([
                edge_density, num_labels,
                np.std([np.sum(labels_img == i) for i in range(1, min(num_labels, 10))]) if num_labels > 1 else 0.0
            ])
            
            # Directional analysis
            orientations = [0, 45, 90, 135]
            for orientation in orientations:
                kernel = cv2.getRotationMatrix2D((1, 1), orientation, 1)
                rotated = cv2.warpAffine(noise_map, kernel, (h, w))
                features.extend([np.mean(rotated), np.std(rotated)])
            
            while len(features) % 50 != (len(features) - 43) % 50:
                features.append(0.0)
        except Exception as e:
            features.extend([0.0] * 50)
        
        # 11. Advanced spectral analysis
        try:
            fft_result = np.fft.fft2(noise_map)
            power_spectrum = np.abs(fft_result)**2
            
            # Frequency binning
            freq_bins = 20
            h_bins = np.array_split(power_spectrum, freq_bins, axis=0)
            w_bins = np.array_split(power_spectrum, freq_bins, axis=1)
            h_energies = [np.mean(bin_data) for bin_data in h_bins[:10]]
            features.extend(h_energies)
            w_energies = [np.mean(bin_data) for bin_data in w_bins[:10]]
            features.extend(w_energies)
            
            # Spectral centroids and spread
            freqs_h = np.fft.fftfreq(h)
            freqs_w = np.fft.fftfreq(w)
            h_spectrum = np.mean(power_spectrum, axis=1)
            w_spectrum = np.mean(power_spectrum, axis=0)
            
            h_centroid = np.sum(freqs_h * h_spectrum) / (np.sum(h_spectrum) + 1e-8)
            w_centroid = np.sum(freqs_w * w_spectrum) / (np.sum(w_spectrum) + 1e-8)
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
            
            # High frequency energy ratio
            high_freq_h = np.sum(h_spectrum[len(h_spectrum)//2:])
            high_freq_w = np.sum(w_spectrum[len(w_spectrum)//2:])
            total_energy_h = np.sum(h_spectrum)
            total_energy_w = np.sum(w_spectrum)
            features.extend([
                high_freq_h / (total_energy_h + 1e-8),
                high_freq_w / (total_energy_w + 1e-8)
            ])
            
            # Phase coherence
            phase_diff_h = np.diff(np.unwrap(np.angle(np.fft.fft(noise_map.mean(axis=0)))))
            phase_diff_w = np.diff(np.unwrap(np.angle(np.fft.fft(noise_map.mean(axis=1)))))
            features.extend([
                np.std(phase_diff_h), np.mean(np.abs(phase_diff_h)),
                np.std(phase_diff_w), np.mean(np.abs(phase_diff_w))
            ])
            
            current_spectral = 30
            remaining = 60 - current_spectral
            if remaining > 0:
                psd_norm = power_spectrum / (np.sum(power_spectrum) + 1e-8)
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
                features.append(spectral_entropy)
                
                if h > 1 and w > 1:
                    prev_spectrum = np.abs(np.fft.fft2(noise_map[:-1, :-1]))**2
                    curr_spectrum = power_spectrum[:-1, :-1]
                    spectral_flux = np.mean((curr_spectrum - prev_spectrum)**2)
                    features.append(spectral_flux)
                else:
                    features.append(0.0)
                
                features.extend([0.0] * (remaining - 2))
        except Exception as e:
            features.extend([0.0] * 60)
        
        # 12. Fractal and complexity features
        try:
            # Autocorrelation 2D
            autocorr_2d = np.correlate(noise_flat, noise_flat, mode='full')
            autocorr_2d = autocorr_2d / np.max(autocorr_2d)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(autocorr_2d, height=0.1)
            features.extend([
                len(peaks), np.mean(autocorr_2d) if len(autocorr_2d) > 0 else 0.0,
                np.std(autocorr_2d) if len(autocorr_2d) > 0 else 0.0
            ])
            
            # Box counting fractal dimension
            def box_count(image, min_box_size=1, max_box_size=None):
                if max_box_size is None:
                    max_box_size = min(image.shape) // 4
                sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=10, dtype=int)
                sizes = np.unique(sizes)
                counts = []
                for size in sizes:
                    if size >= min(image.shape):
                        break
                    boxes = 0
                    for i in range(0, image.shape[0], size):
                        for j in range(0, image.shape[1], size):
                            box = image[i:i+size, j:j+size]
                            if np.std(box) > 0.01:
                                boxes += 1
                    counts.append(boxes)
                if len(counts) > 1 and len(sizes) == len(counts):
                    log_sizes = np.log(sizes[:len(counts)])
                    log_counts = np.log(np.array(counts) + 1)
                    if len(log_sizes) > 1:
                        slope, _ = np.polyfit(log_sizes, log_counts, 1)
                        return -slope
                return 1.5
            
            fractal_dim = box_count((np.abs(noise_map) > np.std(noise_map)).astype(int))
            features.append(fractal_dim)
            
            # Lacunarity
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
            
            # Hurst exponent
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
            
            current_pattern = 11
            remaining_pattern = 70 - current_pattern
            if remaining_pattern > 0:
                for scale in [1, 2, 4]:
                    try:
                        if scale > 1:
                            scaled_noise = cv2.resize(noise_map, (max(8, w//scale), max(8, h//scale)))
                        else:
                            scaled_noise = noise_map
                        texture_energy = np.sum(scaled_noise**2)
                        features.append(texture_energy)
                        
                        kernel_var = np.ones((3, 3)) / 9
                        local_mean = cv2.filter2D(scaled_noise, -1, kernel_var)
                        local_var = cv2.filter2D(scaled_noise**2, -1, kernel_var) - local_mean**2
                        features.extend([np.mean(local_var), np.std(local_var)])
                    except:
                        features.extend([0.0] * 3)
                
                current_added = 9
                features.extend([0.0] * max(0, remaining_pattern - current_added))
        except Exception as e:
            features.extend([0.0] * 70)
        
        # 13. Final features to reach target length
        target_length = 500
        current_length = len(features)
        if current_length < target_length:
            try:
                # Signal-to-noise ratio
                signal_power = np.var(noise_map)
                noise_power = np.var(noise_map - ndimage.gaussian_filter(noise_map, sigma=1))
                snr = 10 * np.log10((signal_power + 1e-8) / (noise_power + 1e-8))
                features.append(snr)
                
                # Total variation
                tv_h = np.sum(np.abs(np.diff(noise_map, axis=0)))
                tv_w = np.sum(np.abs(np.diff(noise_map, axis=1)))
                total_variation = tv_h + tv_w
                features.append(total_variation)
                
                # Information entropy
                _, counts = np.unique(np.round(noise_flat * 1000).astype(int), return_counts=True)
                entropy = stats.entropy(counts + 1e-8)
                features.append(entropy)
                
                remaining = target_length - len(features)
                features.extend([0.0] * remaining)
            except:
                remaining = target_length - len(features)
                features.extend([0.0] * remaining)
        
        features = features[:target_length]
        return np.array(features, dtype=np.float32)
    
    def extract_batch_features_parallel(self, noise_maps_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Extract features from batch with progress tracking"""
        if len(noise_maps_batch) <= 2 or self.n_jobs == 1:
            return [self.extract_comprehensive_features(nm) for nm in noise_maps_batch]
        
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(noise_maps_batch))) as executor:
            futures = [executor.submit(self.extract_comprehensive_features, nm) for nm in noise_maps_batch]
            results = []
            for i, future in enumerate(futures):
                try:
                    results.append(future.result())
                    if (i + 1) % 50 == 0:
                        print(f"      🔬 Processed {i + 1}/{len(futures)} noise maps in batch")
                except Exception as e:
                    print(f"      ⚠️ Feature extraction error for sample {i}: {e}")
                    results.append(np.zeros(500, dtype=np.float32))
            return results
    
    def fit_transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        """Extract features with detailed progress tracking"""
        print(f"\n🚀 ADVANCED FEATURE EXTRACTION")
        print(f"📊 Processing {len(noise_maps):,} noise maps...")
        print(f"🔧 Using {self.n_jobs} CPU cores for parallel processing")
        print("="*60)
        
        batch_size = max(16, len(noise_maps) // (self.n_jobs * 4))
        batches = [noise_maps[i:i + batch_size] for i in range(0, len(noise_maps), batch_size)]
        feature_matrix = []
        
        start_time = time.time()
        total_processed = 0
        
        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()
            print(f"\n📦 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} samples)...")
            
            batch_features = self.extract_batch_features_parallel(batch)
            feature_matrix.extend(batch_features)
            total_processed += len(batch)
            
            batch_time = time.time() - batch_start_time
            elapsed_total = time.time() - start_time
            avg_speed = total_processed / elapsed_total
            eta = (len(noise_maps) - total_processed) / avg_speed if avg_speed > 0 else 0
            
            print(f"   ✅ Batch {batch_idx + 1} completed in {batch_time:.2f}s")
            print(f"   📊 Progress: {total_processed}/{len(noise_maps):,} ({100*total_processed/len(noise_maps):.1f}%)")
            print(f"   🚀 Speed: {avg_speed:.1f} maps/sec")
            print(f"   ⏱️ ETA: {eta/60:.1f} minutes")
            
            # Memory cleanup
            if batch_idx % 5 == 0 and batch_idx > 0:
                gc.collect()
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                print(f"   💾 Memory usage: {memory_usage:.1f} GB")
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        extraction_time = time.time() - start_time
        
        print(f"\n✅ FEATURE EXTRACTION COMPLETE:")
        print(f"   📊 Feature matrix shape: {feature_matrix.shape}")
        print(f"   ⏱️ Total time: {extraction_time:.2f} seconds")
        print(f"   🚀 Overall speed: {len(noise_maps)/extraction_time:.1f} maps/second")
        
        # Data cleaning and scaling
        print(f"\n🧹 Cleaning and scaling features...")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for constant features
        std_features = np.std(feature_matrix, axis=0)
        constant_features = np.sum(std_features < 1e-8)
        if constant_features > 0:
            print(f"   ⚠️ Found {constant_features} constant features (will be handled by scaler)")
        
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.fitted = True
        
        print(f"✅ Feature extraction and scaling completed!")
        print(f"   📊 {feature_matrix.shape[1]} features per sample")
        print(f"   🎯 Ready for classification training")
        
        return feature_matrix
    
    def transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        """Transform new noise maps using fitted scaler"""
        if not self.fitted:
            raise ValueError("Must call fit_transform first")
        
        print(f"\n🔄 TRANSFORMING {len(noise_maps):,} noise maps...")
        
        batch_size = max(16, len(noise_maps) // (self.n_jobs * 4))
        batches = [noise_maps[i:i + batch_size] for i in range(0, len(noise_maps), batch_size)]
        feature_matrix = []
        
        start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(batches, desc="⚡ Feature transformation", unit="batches")):
            batch_features = self.extract_batch_features_parallel(batch)
            feature_matrix.extend(batch_features)
            
            if (batch_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                processed = (batch_idx + 1) * batch_size
                speed = processed / elapsed
                print(f"   📊 Processed {processed}/{len(noise_maps):,} - Speed: {speed:.1f} maps/sec")
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        transform_time = time.time() - start_time
        print(f"✅ Transformation complete in {transform_time:.2f}s")
        
        return self.scaler.transform(feature_matrix)

class UltraAdvancedClassificationPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint_dir='checkpoints_enhanced'):
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        print(f"\n🚀 INITIALIZING ULTRA-ADVANCED CLASSIFICATION PIPELINE")
        print(f"🎯 Target: MCC > 0.95")
        print("="*70)
        print(f"🔧 Primary device: {self.device}")
        if self.num_gpus > 0:
            print(f"🔥 Available GPUs: {self.num_gpus}")
            for i in range(self.num_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        self.autoencoder = None
        self.noise_extractor = UltraAdvancedFeatureExtractor(n_jobs=-1, use_gpu_features=True)
        
        # Advanced ensemble classifiers
        self.base_classifiers = {
            'rf': RandomForestClassifier(
                n_estimators=2000,
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
        
        print(f"✅ Pipeline initialized with {len(self.base_classifiers)} base classifiers")
    
    def save_checkpoint(self, checkpoint_name='latest', extra_data=None):
        """Save checkpoint with progress tracking"""
        print(f"\n💾 SAVING CHECKPOINT: {checkpoint_name}")
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth')
        autoencoder_state = None
        
        if self.autoencoder is not None:
            if self.num_gpus > 1:
                autoencoder_state = self.autoencoder.module.state_dict()
            else:
                autoencoder_state = self.autoencoder.state_dict()
            print("   ✅ Autoencoder state saved")
        
        # Save feature extractor
        extractor_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_extractor.pkl')
        if self.noise_extractor.fitted:
            try:
                with open(extractor_path, 'wb') as f:
                    pickle.dump(self.noise_extractor, f)
                print(f"   ✅ Feature extractor saved: {os.path.basename(extractor_path)}")
            except Exception as e:
                print(f"   ⚠️ Error saving extractor: {e}")
        
        # Save ensemble classifier
        ensemble_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_ensemble.pkl')
        if self.classifier_trained:
            try:
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(self.ensemble_classifier, f)
                print(f"   ✅ Ensemble classifier saved: {os.path.basename(ensemble_path)}")
            except Exception as e:
                print(f"   ⚠️ Error saving ensemble: {e}")
        
        # Save individual classifiers
        saved_classifiers = 0
        for name, classifier in self.base_classifiers.items():
            if hasattr(classifier, 'classes_'):
                classifier_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_{name}.pkl')
                try:
                    with open(classifier_path, 'wb') as f:
                        pickle.dump(classifier, f)
                    saved_classifiers += 1
                except Exception as e:
                    print(f"   ⚠️ Error saving {name}: {e}")
        
        if saved_classifiers > 0:
            print(f"   ✅ {saved_classifiers} individual classifiers saved")
        
        # Save main checkpoint
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
            print(f"   ✅ Main checkpoint saved: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"   ⚠️ Error saving checkpoint: {e}")
        
        print("💾 Checkpoint save complete!")
    
    def load_checkpoint(self, checkpoint_name='latest'):
        """Load checkpoint with detailed progress tracking"""
        print(f"\n🔄 LOADING CHECKPOINT: {checkpoint_name}")
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth')
        if not os.path.exists(checkpoint_path):
            print(f"   ⚠️ Checkpoint {checkpoint_path} not found.")
            return False, 0
        
        print(f"   📂 Loading main checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load autoencoder
        if checkpoint['autoencoder_state_dict'] is not None:
            print("   🔄 Loading autoencoder...")
            self.autoencoder = AdvancedResidualAutoencoder().to(self.device)
            if self.num_gpus > 1:
                self.autoencoder = nn.DataParallel(self.autoencoder)
                self.autoencoder.module.load_state_dict(checkpoint['autoencoder_state_dict'])
            else:
                self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            print("   ✅ Autoencoder loaded")
        
        # Load feature extractor
        extractor_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_extractor.pkl')
        if os.path.exists(extractor_path):
            try:
                print("   🔄 Loading feature extractor...")
                with open(extractor_path, 'rb') as f:
                    self.noise_extractor = pickle.load(f)
                print("   ✅ Feature extractor loaded")
            except Exception as e:
                print(f"   ⚠️ Could not load extractor: {e}")
        
        # Load ensemble classifier
        ensemble_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_ensemble.pkl')
        if os.path.exists(ensemble_path):
            try:
                print("   🔄 Loading ensemble classifier...")
                with open(ensemble_path, 'rb') as f:
                    self.ensemble_classifier = pickle.load(f)
                print("   ✅ Ensemble classifier loaded")
            except Exception as e:
                print(f"   ⚠️ Could not load ensemble: {e}")
        
        # Load individual classifiers
        loaded_classifiers = 0
        for name in self.base_classifiers.keys():
            classifier_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_{name}.pkl')
            if os.path.exists(classifier_path):
                try:
                    with open(classifier_path, 'rb') as f:
                        self.base_classifiers[name] = pickle.load(f)
                    loaded_classifiers += 1
                except Exception as e:
                    print(f"   ⚠️ Could not load {name}: {e}")
        
        if loaded_classifiers > 0:
            print(f"   ✅ {loaded_classifiers} individual classifiers loaded")
        
        # Load training state
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
        
        print(f"✅ CHECKPOINT LOADED SUCCESSFULLY:")
        print(f"   🤖 Autoencoder trained: {self.autoencoder_trained}")
        print(f"   🎭 Classifier trained: {self.classifier_trained}")
        print(f"   📊 Epochs trained: {self.training_history.get('epochs_trained', 0)}")
        print(f"   🎯 Best test MCC: {self.training_history.get('best_test_mcc', 0.0):.4f}")
        
        return True, self.training_history.get('epochs_trained', 0)

    def extract_noise_from_images_advanced(self, images: torch.Tensor, batch_size: int = 64) -> List[np.ndarray]:
        """Extract noise maps with detailed progress tracking"""
        if self.autoencoder is None or not self.autoencoder_trained:
            raise ValueError("Advanced autoencoder must be trained first")
        
        self.autoencoder.eval()
        noise_maps = []
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        print(f"🔍 EXTRACTING NOISE MAPS:")
        print(f"   📊 Total images: {len(images):,}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Total batches: {total_batches}")
        
        start_time = time.time()
        processed_images = 0
        
        with torch.no_grad():
            for batch_idx in range(0, len(images), batch_size):
                batch_start = time.time()
                batch = images[batch_idx:batch_idx+batch_size].to(self.device, non_blocking=True)
                current_batch_size = len(batch)
                
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
                    
                    processed_images += current_batch_size
                    batch_time = time.time() - batch_start
                    
                    # Progress update every 10 batches
                    if (batch_idx // batch_size + 1) % 10 == 0 or (batch_idx // batch_size + 1) == total_batches:
                        elapsed_total = time.time() - start_time
                        avg_speed = processed_images / elapsed_total
                        eta = (len(images) - processed_images) / avg_speed if avg_speed > 0 else 0
                        
                        print(f"   📊 Batch {batch_idx//batch_size + 1}/{total_batches} - "
                              f"Speed: {avg_speed:.1f} imgs/sec - "
                              f"ETA: {eta:.1f}s")
                
                except torch.cuda.OutOfMemoryError:
                    print(f"   ⚠️ GPU memory error in batch {batch_idx//batch_size + 1}, using smaller batches...")
                    # Process with smaller batches
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
                
                # Clean up GPU memory periodically
                if (batch_idx // batch_size + 1) % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        print(f"✅ NOISE EXTRACTION COMPLETE:")
        print(f"   📊 Extracted {len(noise_maps):,} enhanced noise maps")
        print(f"   ⏱️ Total time: {total_time:.2f} seconds")
        print(f"   🚀 Speed: {len(noise_maps)/total_time:.1f} maps/second")
        
        return noise_maps
    
    def _enhance_noise_map(self, noise_map: np.ndarray) -> np.ndarray:
        """Enhance noise map with multiple techniques"""
        try:
            enhanced = noise_map.copy()
            
            # Contrast enhancement
            noise_uint8 = ((noise_map + 1) * 127.5).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            equalized = clahe.apply(noise_uint8)
            enhanced += 0.1 * ((equalized.astype(np.float32) / 127.5) - 1)
            
            # Unsharp masking
            blurred = cv2.GaussianBlur(noise_map, (3, 3), 1.0)
            unsharp = noise_map + 0.5 * (noise_map - blurred)
            enhanced += 0.1 * unsharp
            
            # Edge enhancement
            laplacian = cv2.Laplacian(noise_map, cv2.CV_32F, ksize=3)
            enhanced += 0.05 * laplacian
            
            # Clip values
            enhanced = np.clip(enhanced, -2, 2)
            return enhanced
        except Exception as e:
            return noise_map
    
    def train_ensemble_classifier_advanced(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train ensemble classifier with comprehensive progress tracking"""
        if self.classifier_trained:
            print("✅ Advanced ensemble classifier already trained.")
            return
        
        if not self.autoencoder_trained:
            raise ValueError("❌ Advanced autoencoder must be trained first")
        
        print("\n" + "="*80)
        print("🚀 TRAINING ULTRA-ADVANCED ENSEMBLE CLASSIFIER")
        print("🎯 TARGET: MCC > 0.95")
        print("="*80)
        
        total_start_time = time.time()
        
        # Phase 1: Extract noise maps
        print("\n[PHASE 1/5] 🔍 EXTRACTING ADVANCED NOISE MAPS...")
        extraction_start = time.time()
        
        all_noise_maps = []
        all_labels = []
        total_samples = sum(len(labels) for _, labels in train_loader)
        print(f"📊 Total training samples to process: {total_samples:,}")
        
        processed_batches = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            try:
                print(f"\n🔄 Processing training batch {batch_idx + 1}...")
                noise_maps = self.extract_noise_from_images_advanced(images, batch_size=64)
                all_noise_maps.extend(noise_maps)
                all_labels.extend(labels.cpu().numpy())
                processed_batches += 1
                
                if batch_idx % 5 == 0 and batch_idx > 0:
                    elapsed = time.time() - extraction_start
                    processed = len(all_noise_maps)
                    speed = processed / elapsed
                    eta = (total_samples - processed) / speed if speed > 0 else 0
                    print(f"📈 Overall progress: {processed:,}/{total_samples:,} ({100*processed/total_samples:.1f}%)")
                    print(f"🚀 Overall speed: {speed:.1f} maps/sec, ETA: {eta/60:.1f} min")
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                continue
        
        extraction_time = time.time() - extraction_start
        print(f"\n✅ PHASE 1 COMPLETE:")
        print(f"   📊 Extracted {len(all_noise_maps):,} noise maps")
        print(f"   ⏱️ Time: {extraction_time:.2f} seconds")
        print(f"   🚀 Speed: {len(all_noise_maps)/extraction_time:.1f} maps/second")
        
        # Phase 2: Extract comprehensive features
        print(f"\n[PHASE 2/5] ⚡ EXTRACTING 500+ COMPREHENSIVE FEATURES...")
        feature_start = time.time()
        
        feature_matrix = self.noise_extractor.fit_transform(all_noise_maps)
        
        feature_time = time.time() - feature_start
        print(f"\n✅ PHASE 2 COMPLETE:")
        print(f"   📊 Feature matrix shape: {feature_matrix.shape}")
        print(f"   ⏱️ Time: {feature_time:.2f} seconds")
        print(f"   🚀 Speed: {len(all_noise_maps)/feature_time:.1f} maps/second")
        
        # Phase 3: Data analysis and feature selection
        print(f"\n[PHASE 3/5] 🎯 ADVANCED FEATURE SELECTION AND OPTIMIZATION...")
        
        unique, counts = np.unique(all_labels, return_counts=True)
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        print(f"\n📊 TRAINING CLASS DISTRIBUTION:")
        for label, count in zip(unique, counts):
            percentage = 100 * count / len(all_labels)
            print(f"   {class_names[label]}: {count:,} samples ({percentage:.1f}%)")
        
        # Feature selection
        print(f"\n🔍 Selecting most informative features...")
        feature_selector = SelectKBest(f_classif, k=min(400, feature_matrix.shape[1]))
        feature_matrix_selected = feature_selector.fit_transform(feature_matrix, all_labels)
        self.feature_selector = feature_selector
        
        print(f"✅ Selected {feature_matrix_selected.shape[1]} most informative features")
        print(f"📊 Feature reduction: {feature_matrix.shape[1]} → {feature_matrix_selected.shape[1]}")
        
        # Phase 4: Train advanced ensemble classifiers
        print(f"\n[PHASE 4/5] 🌳 TRAINING ADVANCED ENSEMBLE CLASSIFIERS...")
        classifier_start = time.time()
        
        individual_results = {}
        cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Hyperparameter grids for optimization
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
        
        total_classifiers = len(self.base_classifiers)
        for clf_idx, (name, classifier) in enumerate(self.base_classifiers.items()):
            print(f"\n🔧 [{clf_idx+1}/{total_classifiers}] Optimizing {name.upper()} classifier...")
            
            try:
                # Hyperparameter optimization
                search_start = time.time()
                search = RandomizedSearchCV(
                    classifier, param_grids[name], n_iter=15,
                    scoring='accuracy', cv=cv_folds, n_jobs=-1,
                    random_state=42, verbose=0
                )
                
                print(f"   🔄 Running randomized search with 15 iterations...")
                search.fit(feature_matrix_selected, all_labels)
                search_time = time.time() - search_start
                
                optimized_classifier = search.best_estimator_
                
                # Evaluate optimized classifier
                print(f"   📊 Evaluating optimized {name.upper()}...")
                predictions = optimized_classifier.predict(feature_matrix_selected)
                mcc = matthews_corrcoef(all_labels, predictions)
                accuracy = accuracy_score(all_labels, predictions)
                
                individual_results[name] = {
                    'mcc': mcc,
                    'accuracy': accuracy,
                    'best_params': search.best_params_,
                    'search_time': search_time
                }
                
                self.base_classifiers[name] = optimized_classifier
                
                print(f"   ✅ {name.upper()} optimization complete:")
                print(f"      🎯 MCC: {mcc:.4f}")
                print(f"      🎯 Accuracy: {accuracy:.4f}")
                print(f"      ⏱️ Search time: {search_time:.1f}s")
                print(f"      🔧 Best params: {search.best_params_}")
                
            except Exception as e:
                print(f"   ❌ Error optimizing {name}: {e}")
                # Use default classifier if optimization fails
                try:
                    classifier.fit(feature_matrix_selected, all_labels)
                    predictions = classifier.predict(feature_matrix_selected)
                    mcc = matthews_corrcoef(all_labels, predictions)
                    accuracy = accuracy_score(all_labels, predictions)
                    individual_results[name] = {
                        'mcc': mcc,
                        'accuracy': accuracy,
                        'best_params': 'default',
                        'search_time': 0
                    }
                    print(f"   🔄 Using default {name.upper()}: MCC={mcc:.4f}")
                except Exception as e2:
                    print(f"   ❌ Failed to train default {name}: {e2}")
                    continue
        
        # Train ensemble classifier
        print(f"\n🎭 TRAINING ENSEMBLE CLASSIFIER...")
        ensemble_estimators = [(name, clf) for name, clf in self.base_classifiers.items() 
                              if hasattr(clf, 'classes_')]
        
        if len(ensemble_estimators) >= 2:
            self.ensemble_classifier = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft',
                n_jobs=-1
            )
            
            try:
                ensemble_train_start = time.time()
                print(f"   🔄 Fitting ensemble with {len(ensemble_estimators)} classifiers...")
                self.ensemble_classifier.fit(feature_matrix_selected, all_labels)
                ensemble_train_time = time.time() - ensemble_train_start
                
                print(f"   📊 Evaluating ensemble performance...")
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
                print(f"   ⏱️ Training time: {ensemble_train_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error training ensemble: {e}")
                # Fall back to best individual classifier
                best_classifier_name = max(individual_results.keys(), 
                                         key=lambda x: individual_results[x]['mcc'])
                self.ensemble_classifier = self.base_classifiers[best_classifier_name]
                print(f"🔄 Using best individual classifier: {best_classifier_name.upper()}")
        else:
            print("❌ Not enough classifiers for ensemble, using best individual")
            if individual_results:
                best_classifier_name = max(individual_results.keys(), 
                                         key=lambda x: individual_results[x]['mcc'])
                self.ensemble_classifier = self.base_classifiers[best_classifier_name]
        
        classifier_time = time.time() - classifier_start
        print(f"\n✅ PHASE 4 COMPLETE:")
        print(f"   ⏱️ Total classifier training time: {classifier_time:.2f} seconds")
        
        # Phase 5: Validation (if validation loader provided)
        if val_loader is not None:
            print(f"\n[PHASE 5/5] 🔍 VALIDATING ON VALIDATION SET...")
            
            try:
                val_start = time.time()
                print(f"   🔄 Extracting validation predictions...")
                val_predictions, val_probabilities = self.predict_advanced(val_loader)
                
                # Get validation labels
                val_labels = []
                for _, labels in val_loader:
                    val_labels.extend(labels.cpu().numpy())
                
                # Ensure same length
                min_len = min(len(val_labels), len(val_predictions))
                val_labels = val_labels[:min_len]
                val_predictions = val_predictions[:min_len]
                
                # Calculate metrics
                val_mcc = matthews_corrcoef(val_labels, val_predictions)
                val_accuracy = accuracy_score(val_labels, val_predictions)
                val_cm = confusion_matrix(val_labels, val_predictions)
                
                val_time = time.time() - val_start
                
                print(f"\n✅ VALIDATION RESULTS:")
                print(f"   🎯 Validation MCC: {val_mcc:.4f}")
                print(f"   🎯 Validation Accuracy: {val_accuracy:.4f}")
                print(f"   ⏱️ Validation time: {val_time:.2f}s")
                
                # Detailed validation analysis
                print(f"\n📊 VALIDATION CONFUSION MATRIX:")
                print("True\\Pred    Real  Synth  Semi")
                for i, row in enumerate(val_cm):
                    print(f"{class_names[i]:10s} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
                
                # Per-class metrics
                val_report = classification_report(val_labels, val_predictions, 
                                                 target_names=class_names, 
                                                 output_dict=True, zero_division=0)
                
                print(f"\n📈 VALIDATION PER-CLASS PERFORMANCE:")
                for class_name in class_names:
                    if class_name in val_report:
                        metrics = val_report[class_name]
                        print(f"   {class_name}:")
                        print(f"     Precision: {metrics['precision']:.4f}")
                        print(f"     Recall:    {metrics['recall']:.4f}")
                        print(f"     F1-score:  {metrics['f1-score']:.4f}")
                
                # Save best validation model
                if val_mcc > self.training_history.get('best_val_mcc', 0.0):
                    self.training_history['best_val_mcc'] = val_mcc
                    print(f"   🏆 New best validation MCC! Saving checkpoint...")
                    self.save_checkpoint('best_validation')
                
            except Exception as e:
                print(f"❌ Error during validation: {e}")
                import traceback
                traceback.print_exc()
        
        # Mark classifier as trained and save final checkpoint
        self.classifier_trained = True
        self.save_checkpoint('ensemble_final')
        
        # Final summary
        total_time = time.time() - total_start_time
        print("\n" + "="*80)
        print("🎉 ULTRA-ADVANCED ENSEMBLE TRAINING COMPLETED!")
        print("="*80)
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"   📁 Processed {len(all_noise_maps):,} training samples")
        print(f"   🔧 Extracted {feature_matrix.shape[1]} total features")
        print(f"   🎯 Selected {feature_matrix_selected.shape[1]} best features")
        print(f"   🎭 Trained ensemble of {len(ensemble_estimators)} classifiers")
        
        if 'ensemble_mcc' in self.training_history:
            training_mcc = self.training_history['ensemble_mcc']
            print(f"   🏆 Training ensemble MCC: {training_mcc:.4f}")
            if training_mcc > 0.95:
                print(f"   🎉 TARGET ACHIEVED: MCC > 0.95! ✅")
            elif training_mcc > 0.90:
                print(f"   🎯 Excellent performance: MCC > 0.90! ⭐")
        
        if val_loader and 'best_val_mcc' in self.training_history:
            val_mcc = self.training_history['best_val_mcc']
            print(f"   🔍 Best validation MCC: {val_mcc:.4f}")
        
        print(f"\n⏱️ TIMING BREAKDOWN:")
        print(f"   Phase 1 (Noise Extraction): {extraction_time:.2f}s ({100*extraction_time/total_time:.1f}%)")
        print(f"   Phase 2 (Feature Extraction): {feature_time:.2f}s ({100*feature_time/total_time:.1f}%)")
        print(f"   Phase 4 (Classifier Training): {classifier_time:.2f}s ({100*classifier_time/total_time:.1f}%)")
        print(f"   Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        print(f"\n🚀 PERFORMANCE METRICS:")
        print(f"   Overall Speed: {len(all_noise_maps)/total_time:.1f} samples/second")
        print(f"   Memory Efficiency: Processed {len(all_noise_maps):,} samples")
        
        print(f"\n🔧 INDIVIDUAL CLASSIFIER PERFORMANCE:")
        for name, results in individual_results.items():
            print(f"   {name.upper()}: MCC={results['mcc']:.4f}, Accuracy={results['accuracy']:.4f}")
        
        print("="*80)
    
    def predict_advanced(self, test_loader: DataLoader, batch_size: int = 64) -> Tuple[List[int], List[float]]:
        """Generate predictions with detailed progress tracking"""
        if not self.autoencoder_trained or not self.classifier_trained:
            raise ValueError("❌ Both autoencoder and ensemble must be trained")
        
        if not self.noise_extractor.fitted:
            raise ValueError("❌ Feature extractor must be fitted!")
        
        print(f"\n🔮 GENERATING ADVANCED ENSEMBLE PREDICTIONS")
        print("="*60)
        
        all_predictions = []
        all_probabilities = []
        total_batches = len(test_loader)
        
        print(f"📊 Test batches to process: {total_batches}")
        print(f"📦 Batch size: {batch_size}")
        
        self.autoencoder.eval()
        start_time = time.time()
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(test_loader):
                batch_start = time.time()
                current_batch_size = len(images)
                
                try:
                    # Extract noise maps
                    noise_maps = self.extract_noise_from_images_advanced(images, batch_size=batch_size)
                    
                    # Extract features
                    feature_matrix = self.noise_extractor.transform(noise_maps)
                    
                    # Apply feature selection if available
                    if hasattr(self, 'feature_selector'):
                        feature_matrix = self.feature_selector.transform(feature_matrix)
                    
                    # Generate predictions
                    predictions = self.ensemble_classifier.predict(feature_matrix)
                    probabilities = self.ensemble_classifier.predict_proba(feature_matrix)
                    
                    all_predictions.extend(predictions.tolist())
                    all_probabilities.extend(probabilities.tolist())
                    
                    processed_samples += current_batch_size
                    batch_time = time.time() - batch_start
                    
                    # Progress update every 5 batches
                    if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                        elapsed_total = time.time() - start_time
                        avg_speed = processed_samples / elapsed_total
                        eta = (total_batches * batch_size - processed_samples) / avg_speed if avg_speed > 0 else 0
                        
                        print(f"   📊 Batch {batch_idx + 1}/{total_batches} - "
                              f"Speed: {avg_speed:.1f} samples/sec - "
                              f"ETA: {eta:.1f}s")
                
                except Exception as e:
                    print(f"⚠️ Error predicting batch {batch_idx}: {e}")
                    # Fill with default predictions to maintain consistency
                    batch_size_actual = len(images)
                    all_predictions.extend([0] * batch_size_actual)
                    all_probabilities.extend([[1.0, 0.0, 0.0]] * batch_size_actual)
                    processed_samples += batch_size_actual
        
        total_time = time.time() - start_time
        print(f"\n✅ PREDICTION COMPLETE:")
        print(f"   📊 Generated predictions for {len(all_predictions):,} samples")
        print(f"   ⏱️ Total time: {total_time:.2f} seconds")
        print(f"   🚀 Speed: {len(all_predictions)/total_time:.1f} predictions/second")
        
        return all_predictions, all_probabilities
    
    def evaluate_advanced(self, test_loader: DataLoader, test_labels: List[int], 
                         save_results: bool = True, results_dir: str = 'ultra_results') -> Dict:
        """Comprehensive evaluation with detailed analysis and visualization"""
        print("\n" + "="*80)
        print("🔍 STARTING ULTRA-ADVANCED EVALUATION")
        print("="*80)
        
        eval_start = time.time()
        
        # Generate predictions
        print("\n📊 Generating predictions...")
        predictions, probabilities = self.predict_advanced(test_loader)
        
        # Ensure consistent
        def evaluate_advanced(self, test_loader: DataLoader, test_labels: List[int], 
                         save_results: bool = True, results_dir: str = 'ultra_results') -> Dict:
            """Comprehensive evaluation with detailed analysis and visualization"""
        print("\n" + "="*80)
        print("🔍 STARTING ULTRA-ADVANCED EVALUATION")
        print("="*80)
        
        eval_start = time.time()
        
        # Generate predictions
        print("\n[STEP 1/7] 📊 Generating predictions...")
        step_start = time.time()
        predictions, probabilities = self.predict_advanced(test_loader)
        step_time = time.time() - step_start
        print(f"✅ Predictions generated in {step_time:.2f}s")
        
        # Ensure consistent lengths
        print("\n[STEP 2/7] 🔧 Aligning predictions with labels...")
        step_start = time.time()
        min_len = min(len(test_labels), len(predictions))
        test_labels = test_labels[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        
        print(f"   📊 Aligned {min_len:,} samples")
        print(f"   Test labels shape: {len(test_labels)}")
        print(f"   Predictions shape: {len(predictions)}")
        step_time = time.time() - step_start
        print(f"✅ Alignment completed in {step_time:.4f}s")
        
        # Calculate comprehensive metrics
        print("\n[STEP 3/7] 📈 Calculating comprehensive metrics...")
        step_start = time.time()
        
        # Core metrics
        mcc = matthews_corrcoef(test_labels, predictions)
        accuracy = accuracy_score(test_labels, predictions)
        cm = confusion_matrix(test_labels, predictions)
        
        # Classification report
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        report = classification_report(test_labels, predictions, 
                                     target_names=class_names, 
                                     output_dict=True, zero_division=0)
        
        print(f"   🎯 Matthews Correlation Coefficient: {mcc:.6f}")
        print(f"   🎯 Overall Accuracy: {accuracy:.6f}")
        print(f"   📊 Confusion matrix calculated")
        print(f"   📋 Classification report generated")
        
        # Per-class accuracy
        per_class_accuracy = {}
        for i, class_name in enumerate(class_names):
            if i < len(cm):
                correct = cm[i, i]
                total = np.sum(cm[i, :])
                per_class_acc = correct / total if total > 0 else 0.0
                per_class_accuracy[class_name] = per_class_acc
                print(f"   📊 {class_name} accuracy: {per_class_acc:.6f}")
        
        step_time = time.time() - step_start
        print(f"✅ Metrics calculated in {step_time:.4f}s")
        
        # Detailed analysis
        print("\n[STEP 4/7] 🔬 Performing detailed analysis...")
        step_start = time.time()
        
        # Prediction confidence analysis
        probabilities_array = np.array(probabilities)
        max_probs = np.max(probabilities_array, axis=1)
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs),
            'median_confidence': np.median(max_probs)
        }
        
        print(f"   📊 Confidence statistics calculated:")
        for key, value in confidence_stats.items():
            print(f"      {key}: {value:.6f}")
        
        # Error analysis
        incorrect_indices = np.where(np.array(test_labels) != np.array(predictions))[0]
        error_analysis = {
            'total_errors': len(incorrect_indices),
            'error_rate': len(incorrect_indices) / len(test_labels),
            'low_confidence_errors': np.sum(max_probs[incorrect_indices] < 0.7),
            'high_confidence_errors': np.sum(max_probs[incorrect_indices] >= 0.9)
        }
        
        print(f"   🔍 Error analysis completed:")
        for key, value in error_analysis.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.6f}")
            else:
                print(f"      {key}: {value}")
        
        step_time = time.time() - step_start
        print(f"✅ Detailed analysis completed in {step_time:.4f}s")
        
        # Create results directory
        print(f"\n[STEP 5/7] 📁 Creating results directory...")
        step_start = time.time()
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            print(f"   📂 Results directory: {results_dir}")
        step_time = time.time() - step_start
        print(f"✅ Directory setup completed in {step_time:.4f}s")
        
        # Generate visualizations
        print(f"\n[STEP 6/7] 🎨 Generating visualizations...")
        step_start = time.time()
        
        if save_results:
            try:
                # Confusion matrix heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Confusion Matrix\nMCC: {mcc:.4f}, Accuracy: {accuracy:.4f}')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                cm_path = os.path.join(results_dir, 'confusion_matrix.png')
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   📊 Confusion matrix saved: {cm_path}")
                
                # Confidence distribution
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
                plt.title('Prediction Confidence Distribution')
                plt.xlabel('Maximum Probability')
                plt.ylabel('Frequency')
                plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(max_probs):.3f}')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                for i, class_name in enumerate(class_names):
                    class_mask = np.array(predictions) == i
                    if np.any(class_mask):
                        plt.hist(max_probs[class_mask], bins=30, alpha=0.6, 
                                label=f'{class_name}', density=True)
                plt.title('Confidence by Predicted Class')
                plt.xlabel('Maximum Probability')
                plt.ylabel('Density')
                plt.legend()
                
                conf_path = os.path.join(results_dir, 'confidence_analysis.png')
                plt.savefig(conf_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   📈 Confidence analysis saved: {conf_path}")
                
                # Per-class performance
                plt.figure(figsize=(15, 5))
                
                metrics_to_plot = ['precision', 'recall', 'f1-score']
                for i, metric in enumerate(metrics_to_plot):
                    plt.subplot(1, 3, i+1)
                    metric_values = [report[class_name][metric] for class_name in class_names 
                                   if class_name in report]
                    bars = plt.bar(class_names[:len(metric_values)], metric_values, 
                                  color=['skyblue', 'lightcoral', 'lightgreen'])
                    plt.title(f'{metric.capitalize()} by Class')
                    plt.ylabel(metric.capitalize())
                    plt.ylim(0, 1.1)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, metric_values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                f'{value:.3f}', ha='center', va='bottom')
                
                perf_path = os.path.join(results_dir, 'per_class_performance.png')
                plt.savefig(perf_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   📊 Per-class performance saved: {perf_path}")
                
            except Exception as e:
                print(f"   ⚠️ Error generating visualizations: {e}")
        
        step_time = time.time() - step_start
        print(f"✅ Visualizations completed in {step_time:.4f}s")
        
        # Save detailed results
        print(f"\n[STEP 7/7] 💾 Saving detailed results...")
        step_start = time.time()
        
        # Compile comprehensive results
        results = {
            'overall_metrics': {
                'mcc': float(mcc),
                'accuracy': float(accuracy),
                'total_samples': len(test_labels)
            },
            'per_class_metrics': {
                class_name: {
                    'precision': float(report[class_name]['precision']) if class_name in report else 0.0,
                    'recall': float(report[class_name]['recall']) if class_name in report else 0.0,
                    'f1_score': float(report[class_name]['f1-score']) if class_name in report else 0.0,
                    'support': int(report[class_name]['support']) if class_name in report else 0,
                    'accuracy': float(per_class_accuracy.get(class_name, 0.0))
                } for class_name in class_names
            },
            'confidence_analysis': confidence_stats,
            'error_analysis': error_analysis,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'training_history': self.training_history,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_evaluation_time': 0.0  # Will be updated below
        }
        
        if save_results:
            try:
                # Save JSON results
                results_path = os.path.join(results_dir, 'evaluation_results.json')
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"   📋 JSON results saved: {results_path}")
                
                # Save detailed text report
                report_path = os.path.join(results_dir, 'detailed_report.txt')
                with open(report_path, 'w') as f:
                    f.write("ULTRA-ADVANCED CLASSIFICATION EVALUATION REPORT\n")
                    f.write("="*60 + "\n\n")
                    
                    f.write(f"Timestamp: {results['timestamp']}\n")
                    f.write(f"Total Samples: {len(test_labels):,}\n\n")
                    
                    f.write("OVERALL PERFORMANCE:\n")
                    f.write(f"Matthews Correlation Coefficient: {mcc:.6f}\n")
                    f.write(f"Overall Accuracy: {accuracy:.6f}\n\n")
                    
                    f.write("PER-CLASS PERFORMANCE:\n")
                    for class_name in class_names:
                        if class_name in report:
                            f.write(f"\n{class_name}:\n")
                            f.write(f"  Precision: {report[class_name]['precision']:.6f}\n")
                            f.write(f"  Recall: {report[class_name]['recall']:.6f}\n")
                            f.write(f"  F1-Score: {report[class_name]['f1-score']:.6f}\n")
                            f.write(f"  Support: {report[class_name]['support']}\n")
                            f.write(f"  Accuracy: {per_class_accuracy.get(class_name, 0.0):.6f}\n")
                    
                    f.write(f"\nCONFIDENCE ANALYSIS:\n")
                    for key, value in confidence_stats.items():
                        f.write(f"{key}: {value:.6f}\n")
                    
                    f.write(f"\nERROR ANALYSIS:\n")
                    for key, value in error_analysis.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    
                    f.write(f"\nCONFUSION MATRIX:\n")
                    f.write("True\\Pred    Real  Synth  Semi\n")
                    for i, row in enumerate(cm):
                        f.write(f"{class_names[i]:10s} {row[0]:5d} {row[1]:6d} {row[2]:5d}\n")
                
                print(f"   📄 Detailed report saved: {report_path}")
                
                # Save raw predictions
                predictions_path = os.path.join(results_dir, 'predictions.npz')
                np.savez_compressed(predictions_path,
                                  predictions=predictions,
                                  probabilities=probabilities,
                                  true_labels=test_labels)
                print(f"   🔢 Raw predictions saved: {predictions_path}")
                
            except Exception as e:
                print(f"   ⚠️ Error saving results: {e}")
        
        step_time = time.time() - step_start
        print(f"✅ Results saved in {step_time:.4f}s")
        
        # Update total evaluation time
        total_eval_time = time.time() - eval_start
        results['total_evaluation_time'] = total_eval_time
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 ULTRA-ADVANCED EVALUATION COMPLETED!")
        print("="*80)
        
        print(f"\n📊 FINAL RESULTS SUMMARY:")
        print(f"   🎯 Matthews Correlation Coefficient: {mcc:.6f}")
        print(f"   🎯 Overall Accuracy: {accuracy:.6f}")
        print(f"   📊 Total Samples Evaluated: {len(test_labels):,}")
        print(f"   ⏱️ Total Evaluation Time: {total_eval_time:.2f} seconds")
        print(f"   🚀 Evaluation Speed: {len(test_labels)/total_eval_time:.1f} samples/second")
        
        # MCC Achievement Check
        if mcc > 0.95:
            print(f"\n🏆 OUTSTANDING ACHIEVEMENT!")
            print(f"   🎉 TARGET MCC > 0.95 ACHIEVED! (MCC = {mcc:.6f})")
            print(f"   ⭐ This represents exceptional classification performance!")
        elif mcc > 0.90:
            print(f"\n🎯 EXCELLENT PERFORMANCE!")
            print(f"   ⭐ MCC > 0.90 achieved! (MCC = {mcc:.6f})")
            print(f"   🚀 Very strong classification capability!")
        elif mcc > 0.80:
            print(f"\n✅ GOOD PERFORMANCE!")
            print(f"   👍 MCC > 0.80 achieved! (MCC = {mcc:.6f})")
            print(f"   📈 Solid classification performance!")
        else:
            print(f"\n⚠️ PERFORMANCE ANALYSIS:")
            print(f"   📊 Current MCC: {mcc:.6f}")
            print(f"   🎯 Target MCC: 0.95")
            print(f"   📈 Consider hyperparameter tuning or more training data")
        
        print(f"\n📁 Results saved to: {results_dir}")
        print("="*80)
        
        return results

    def train_autoencoder_advanced(self, train_loader: DataLoader, val_loader: DataLoader = None, 
                                  epochs: int = 100, learning_rate: float = 1e-4,
                                  patience: int = 15, min_delta: float = 1e-5):
        """Train advanced autoencoder with comprehensive progress tracking"""
        
        print("\n" + "="*80)
        print("🚀 TRAINING ULTRA-ADVANCED RESIDUAL AUTOENCODER")
        print("🎯 TARGET: High-quality noise extraction for MCC > 0.95")
        print("="*80)
        
        if self.autoencoder_trained:
            print("✅ Advanced autoencoder already trained.")
            return
        
        # Initialize autoencoder
        print(f"\n[SETUP] 🔧 Initializing autoencoder...")
        self.autoencoder = AdvancedResidualAutoencoder().to(self.device)
        
        if self.num_gpus > 1:
            print(f"   🔥 Using DataParallel with {self.num_gpus} GPUs")
            self.autoencoder = nn.DataParallel(self.autoencoder)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=learning_rate, 
                               weight_decay=1e-5, betas=(0.9, 0.999))
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=patience//2, 
            min_lr=1e-7, verbose=True
        )
        
        criterion = nn.MSELoss()
        
        print(f"   ✅ Optimizer: AdamW (lr={learning_rate})")
        print(f"   ✅ Scheduler: ReduceLROnPlateau")
        print(f"   ✅ Loss function: MSELoss")
        print(f"   ✅ Early stopping patience: {patience}")
        
        # Training variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        training_start_time = time.time()
        
        print(f"\n🚀 Starting training for up to {epochs} epochs...")
        print("="*80)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            print(f"\n[EPOCH {epoch+1}/{epochs}] 🔄 TRAINING PHASE")
            self.autoencoder.train()
            train_loss = 0.0
            train_batches = 0
            
            train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", 
                                    unit="batch", leave=False)
            
            for batch_idx, (images, _) in enumerate(train_progress_bar):
                batch_start = time.time()
                images = images.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                try:
                    if self.scaler is not None:
                        with autocast():
                            reconstructed = self.autoencoder(images)
                            loss = criterion(reconstructed, images)
                        
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        reconstructed = self.autoencoder(images)
                        loss = criterion(reconstructed, images)
                        loss.backward()
                        optimizer.step()
                    
                    batch_loss = loss.item()
                    train_loss += batch_loss
                    train_batches += 1
                    
                    batch_time = time.time() - batch_start
                    
                    # Update progress bar
                    train_progress_bar.set_postfix({
                        'Loss': f'{batch_loss:.6f}',
                        'Avg Loss': f'{train_loss/train_batches:.6f}',
                        'Batch Time': f'{batch_time:.3f}s'
                    })
                    
                    # Memory management
                    if batch_idx % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️ GPU memory error in training batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"⚠️ Error in training batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = train_loss / max(train_batches, 1)
            train_time = time.time() - epoch_start_time
            
            # Validation phase
            val_loss = float('inf')
            if val_loader is not None:
                print(f"\n[EPOCH {epoch+1}/{epochs}] 🔍 VALIDATION PHASE")
                self.autoencoder.eval()
                val_loss_sum = 0.0
                val_batches = 0
                
                val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", 
                                      unit="batch", leave=False)
                
                with torch.no_grad():
                    for batch_idx, (images, _) in enumerate(val_progress_bar):
                        try:
                            images = images.to(self.device, non_blocking=True)
                            
                            if self.scaler is not None:
                                with autocast():
                                    reconstructed = self.autoencoder(images)
                                    loss = criterion(reconstructed, images)
                            else:
                                reconstructed = self.autoencoder(images)
                                loss = criterion(reconstructed, images)
                            
                            batch_val_loss = loss.item()
                            val_loss_sum += batch_val_loss
                            val_batches += 1
                            
                            val_progress_bar.set_postfix({
                                'Loss': f'{batch_val_loss:.6f}',
                                'Avg Loss': f'{val_loss_sum/val_batches:.6f}'
                            })
                            
                        except Exception as e:
                            print(f"⚠️ Error in validation batch {batch_idx}: {e}")
                            continue
                
                val_loss = val_loss_sum / max(val_batches, 1)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_history['autoencoder_losses'].append(avg_train_loss)
            if val_loader is not None:
                self.training_history['val_losses'].append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            total_elapsed = time.time() - training_start_time
            
            # Epoch summary
            print(f"\n✅ EPOCH {epoch+1} SUMMARY:")
            print(f"   📊 Training Loss: {avg_train_loss:.8f}")
            if val_loader is not None:
                print(f"   📊 Validation Loss: {val_loss:.8f}")
                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    print(f"   🎯 New best validation loss! (Improvement: {improvement:.8f})")
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    self.training_history['best_val_loss'] = val_loss
                    
                    # Save best model
                    self.save_checkpoint('best_autoencoder')
                else:
                    epochs_without_improvement += 1
                    print(f"   ⚠️ No improvement for {epochs_without_improvement} epochs")
            
            print(f"   🔧 Learning Rate: {current_lr:.2e}")
            print(f"   ⏱️ Epoch Time: {epoch_time:.2f}s")
            print(f"   ⏱️ Total Elapsed: {total_elapsed/60:.1f}min")
            
            # Estimate remaining time
            avg_epoch_time = total_elapsed / (epoch + 1)
            eta_epochs = epochs - (epoch + 1)
            eta_time = avg_epoch_time * eta_epochs
            print(f"   📈 ETA: {eta_time/60:.1f}min ({eta_epochs} epochs remaining)")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.memory_reserved() / 1024**3
                print(f"   💾 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
            
            # Early stopping check
            if val_loader is not None and epochs_without_improvement >= patience:
                print(f"\n🛑 EARLY STOPPING triggered!")
                print(f"   📊 No improvement for {patience} epochs")
                print(f"   🏆 Best validation loss: {best_val_loss:.8f}")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"   💾 Saving checkpoint at epoch {epoch+1}...")
                self.save_checkpoint(f'autoencoder_epoch_{epoch+1}')
            
            print("-" * 80)
        
        # Training completion
        total_training_time = time.time() - training_start_time
        self.training_history['epochs_trained'] = epoch + 1
        self.autoencoder_trained = True
        
        print("\n" + "="*80)
        print("🎉 AUTOENCODER TRAINING COMPLETED!")
        print("="*80)
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"   📈 Epochs completed: {epoch+1}/{epochs}")
        print(f"   🏆 Final training loss: {avg_train_loss:.8f}")
        if val_loader is not None:
            print(f"   🏆 Best validation loss: {best_val_loss:.8f}")
        print(f"   ⏱️ Total training time: {total_training_time/60:.1f} minutes")
        print(f"   📊 Training batches per epoch: {train_batches}")
        if val_loader is not None:
            print(f"   📊 Validation batches per epoch: {val_batches}")
        
        # Save final model
        self.save_checkpoint('autoencoder_final')
        print(f"   💾 Final model saved")
        
        print("="*80)

def main_training_pipeline(data_dir: str, results_dir: str = 'ultra_advanced_results', 
                          max_images_per_class: int = 160000,
                          autoencoder_epochs: int = 100,
                          autoencoder_lr: float = 1e-4,
                          batch_size: int = 32):
    """Complete training pipeline with comprehensive progress tracking"""
    
    print("\n" + "="*100)
    print("🚀 STARTING ULTRA-ADVANCED CLASSIFICATION PIPELINE")
    print("🎯 TARGET: Matthews Correlation Coefficient > 0.95")
    print("="*100)
    
    pipeline_start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize pipeline
    print(f"\n[PHASE 1/6] 🔧 PIPELINE INITIALIZATION")
    print(f"📂 Data directory: {data_dir}")
    print(f"📊 Results directory: {results_dir}")
    print(f"🎯 Max images per class: {max_images_per_class:,}")
    print(f"🔧 Device: {device}")
    
    pipeline = UltraAdvancedClassificationPipeline(device=device, 
                                                  checkpoint_dir=os.path.join(results_dir, 'checkpoints'))
    
    # Data collection and splitting
    print(f"\n[PHASE 2/6] 📊 DATA COLLECTION AND PREPARATION")
    phase_start = time.time()
    
    image_paths, labels = collect_image_paths_and_labels(data_dir, max_images_per_class)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = save_splits_to_json(
        image_paths, labels, results_dir
    )
    
    phase_time = time.time() - phase_start
    print(f"✅ Data preparation completed in {phase_time/60:.1f} minutes")
    
    # Create data loaders
    print(f"\n[PHASE 3/6] 📦 CREATING DATA LOADERS")
    phase_start = time.time()
    
    print("   🔄 Creating training dataset...")
    train_dataset = PTFileDataset(train_paths, train_labels, device=device, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True if device == 'cuda' else False)
    
    print("   🔄 Creating validation dataset...")
    val_dataset = PTFileDataset(val_paths, val_labels, device=device, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True if device == 'cuda' else False)
    
    print("   🔄 Creating test dataset...")
    test_dataset = PTFileDataset(test_paths, test_labels, device=device, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True if device == 'cuda' else False)
    
    phase_time = time.time() - phase_start
    print(f"✅ Data loaders created in {phase_time:.2f} seconds")
    
    print(f"📊 Data loader summary:")
    print(f"   🚂 Training: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"   🔍 Validation: {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"   🧪 Test: {len(test_dataset):,} samples, {len(test_loader)} batches")
    
    # Check for existing checkpoints
    print(f"\n[PHASE 4/6] 🔄 CHECKING FOR EXISTING CHECKPOINTS")
    phase_start = time.time()
    
    checkpoint_loaded, epochs_trained = pipeline.load_checkpoint('autoencoder_final')
    if checkpoint_loaded:
        print(f"   ✅ Loaded existing checkpoint with {epochs_trained} epochs trained")
    else:
        print(f"   ℹ️ No existing checkpoint found, will train from scratch")
    
    phase_time = time.time() - phase_start
    print(f"✅ Checkpoint check completed in {phase_time:.2f} seconds")
    
    # Autoencoder training
    print(f"\n[PHASE 5/6] 🤖 AUTOENCODER TRAINING")
    phase_start = time.time()
    
    try:
        pipeline.train_autoencoder_advanced(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=autoencoder_epochs,
            learning_rate=autoencoder_lr,
            patience=15,
            min_delta=1e-5
        )
        autoencoder_success = True
    except Exception as e:
        print(f"❌ Autoencoder training failed: {e}")
        import traceback
        traceback.print_exc()
        autoencoder_success = False
    
    phase_time = time.time() - phase_start
    print(f"✅ Autoencoder phase completed in {phase_time/60:.1f} minutes")
    
    # Ensemble classifier training
    print(f"\n[PHASE 6/6] 🎭 ENSEMBLE CLASSIFIER TRAINING")
    phase_start = time.time()
    
    if autoencoder_success:
        try:
            pipeline.train_ensemble_classifier_advanced(
                train_loader=train_loader,
                val_loader=val_loader
            )
            classifier_success = True
        except Exception as e:
            print(f"❌ Ensemble classifier training failed: {e}")
            import traceback
            traceback.print_exc()
            classifier_success = False
    else:
        print(f"❌ Skipping classifier training due to autoencoder failure")
        classifier_success = False
    
    phase_time = time.time() - phase_start
    print(f"✅ Ensemble classifier phase completed in {phase_time/60:.1f} minutes")
    
    # Final evaluation
    if classifier_success:
        print(f"\n[FINAL EVALUATION] 🔍 COMPREHENSIVE MODEL EVALUATION")
        evaluation_start = time.time()
        
        try:
            print("🧪 Evaluating on test set...")
            evaluation_results = pipeline.evaluate_advanced(
                test_loader=test_loader,
                test_labels=test_labels,
                save_results=True,
                results_dir=results_dir
            )
            
            evaluation_time = time.time() - evaluation_start
            print(f"✅ Evaluation completed in {evaluation_time/60:.1f} minutes")
            
            # Final MCC achievement check
            final_mcc = evaluation_results['overall_metrics']['mcc']
            if final_mcc > 0.95:
                print(f"\n🏆 MISSION ACCOMPLISHED!")
                print(f"   🎉 TARGET MCC > 0.95 ACHIEVED!")
                print(f"   🎯 Final MCC: {final_mcc:.6f}")
            
        except Exception as e:
            print(f"❌ Final evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Pipeline completion summary
    total_pipeline_time = time.time() - pipeline_start_time
    
    print("\n" + "="*100)
    print("🎉 ULTRA-ADVANCED CLASSIFICATION PIPELINE COMPLETED!")
    print("="*100)
    
    print(f"\n📊 PIPELINE SUMMARY:")
    print(f"   ⏱️ Total pipeline time: {total_pipeline_time/3600:.1f} hours")
    print(f"   📂 Results saved to: {results_dir}")
    print(f"   🤖 Autoencoder training: {'✅ Success' if autoencoder_success else '❌ Failed'}")
    print(f"   🎭 Ensemble training: {'✅ Success' if classifier_success else '❌ Failed'}")
    
    if classifier_success and 'evaluation_results' in locals():
        final_mcc = evaluation_results['overall_metrics']['mcc']
        final_accuracy = evaluation_results['overall_metrics']['accuracy']
        print(f"   🎯 Final MCC: {final_mcc:.6f}")
        print(f"   🎯 Final Accuracy: {final_accuracy:.6f}")
        
        if final_mcc > 0.95:
            print(f"   🏆 TARGET ACHIEVED: MCC > 0.95! 🎉")
        elif final_mcc > 0.90:
            print(f"   ⭐ Excellent performance: MCC > 0.90!")
        elif final_mcc > 0.80:
            print(f"   ✅ Good performance: MCC > 0.80!")
    
    print(f"\n📁 All results, checkpoints, and visualizations saved to:")
    print(f"   {os.path.abspath(results_dir)}")
    
    print("="*100)
    
    return pipeline

def run_inference_pipeline(data_dir: str, checkpoint_dir: str = 'ultra_advanced_results/checkpoints',
                          results_dir: str = 'inference_results', batch_size: int = 32):
    """Run inference on new data using trained models"""
    
    print("\n" + "="*80)
    print("🔮 STARTING INFERENCE PIPELINE")
    print("="*80)
    
    inference_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize pipeline
    print(f"\n[STEP 1/5] 🔧 Initializing inference pipeline...")
    pipeline = UltraAdvancedClassificationPipeline(device=device, checkpoint_dir=checkpoint_dir)
    
    # Load trained models
    print(f"\n[STEP 2/5] 📥 Loading trained models...")
    checkpoint_loaded, epochs_trained = pipeline.load_checkpoint('ensemble_final')
    
    if not checkpoint_loaded:
        print("❌ Could not load trained models. Please train the pipeline first.")
        return None
    
    print(f"✅ Loaded models trained for {epochs_trained} epochs")
    
    # Collect inference data
    print(f"\n[STEP 3/5] 📊 Collecting inference data...")
    image_paths, labels = collect_image_paths_and_labels(data_dir, max_images_per_class=50000)
    
    # Create inference dataset
    print(f"\n[STEP 4/5] 📦 Creating inference dataset...")
    inference_dataset = PTFileDataset(image_paths, labels, device=device, augment=False)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=4, pin_memory=True if device == 'cuda' else False)
    
    print(f"📊 Inference dataset: {len(inference_dataset):,} samples, {len(inference_loader)} batches")
    
    # Run inference
    print(f"\n[STEP 5/5] 🔮 Running inference...")
    try:
        predictions, probabilities = pipeline.predict_advanced(inference_loader, batch_size=batch_size)
        
        # Save inference results
        os.makedirs(results_dir, exist_ok=True)
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'image_paths': image_paths,
            'true_labels': labels,
            'class_names': ['Real', 'Synthetic', 'Semi-synthetic'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(predictions)
        }
        
        # Save results
        results_path = os.path.join(results_dir, 'inference_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save numpy arrays
        np.savez_compressed(os.path.join(results_dir, 'inference_arrays.npz'),
                           predictions=predictions,
                           probabilities=probabilities,
                           true_labels=labels)
        
        # Calculate accuracy if true labels available
        if labels:
            accuracy = accuracy_score(labels, predictions)
            mcc = matthews_corrcoef(labels, predictions)
            print(f"\n📊 Inference Performance:")
            print(f"   🎯 Accuracy: {accuracy:.4f}")
            print(f"   🎯 MCC: {mcc:.4f}")
        
        inference_time = time.time() - inference_start
        print(f"\n✅ Inference completed in {inference_time:.2f} seconds")
        print(f"🚀 Speed: {len(predictions)/inference_time:.1f} predictions/second")
        print(f"📁 Results saved to: {results_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage and main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Advanced Classification Pipeline')
    parser.add_argument('--data_dir', type=str, default='datasets/train', 
                       help='Path to data directory containing real, synthetic, semi-synthetic folders')
    parser.add_argument('--results_dir', type=str, default='ultra_results',
                       help='Directory to save results and checkpoints')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train',
                       help='Mode: train new model or run inference with existing model')
    parser.add_argument('--max_images_per_class', type=int, default=160000,
                       help='Maximum images per class to process')
    parser.add_argument('--autoencoder_epochs', type=int, default=100,
                       help='Number of epochs for autoencoder training')
    parser.add_argument('--autoencoder_lr', type=float, default=1e-4,
                       help='Learning rate for autoencoder')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training and inference')
    parser.add_argument('--checkpoint_dir', type=str, default="ultra_checkpoints",
                       help='Checkpoint directory for inference mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Starting training mode...")
        pipeline = main_training_pipeline(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            max_images_per_class=args.max_images_per_class,
            autoencoder_epochs=args.autoencoder_epochs,
            autoencoder_lr=args.autoencoder_lr,
            batch_size=args.batch_size
        )
        
    elif args.mode == 'inference':
        print("🔮 Starting inference mode...")
        checkpoint_dir = args.checkpoint_dir or os.path.join(args.results_dir, 'checkpoints')
        results = run_inference_pipeline(
            data_dir=args.data_dir,
            checkpoint_dir=checkpoint_dir,
            results_dir=os.path.join(args.results_dir, 'inference'),
            batch_size=args.batch_size
        )
    
    print("\n🎉 Pipeline execution completed!")

# Additional utility functions for analysis and monitoring

def analyze_training_progress(results_dir: str):
    """Analyze and visualize training progress from saved checkpoints"""
    
    print(f"\n📈 ANALYZING TRAINING PROGRESS")
    print(f"📂 Results directory: {results_dir}")
    
    # Load training history
    checkpoint_path = os.path.join(results_dir, 'checkpoints', 'ensemble_final.pth')
    if not os.path.exists(checkpoint_path):
        print(f"❌ No training history found at {checkpoint_path}")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('training_history', {})
        
        print(f"\n📊 Training History Summary:")
        print(f"   🤖 Autoencoder epochs: {history.get('epochs_trained', 0)}")
        print(f"   🏆 Best validation loss: {history.get('best_val_loss', 'N/A')}")
        print(f"   🎯 Best test MCC: {history.get('best_test_mcc', 'N/A')}")
        print(f"   🎭 Ensemble MCC: {history.get('ensemble_mcc', 'N/A')}")
        
        # Plot training curves if available
        if 'autoencoder_losses' in history and history['autoencoder_losses']:
            plt.figure(figsize=(15, 5))
            
            # Autoencoder loss
            plt.subplot(1, 3, 1)
            epochs = range(1, len(history['autoencoder_losses']) + 1)
            plt.plot(epochs, history['autoencoder_losses'], 'b-', label='Training Loss')
            if 'val_losses' in history and history['val_losses']:
                plt.plot(epochs[:len(history['val_losses'])], history['val_losses'], 
                        'r-', label='Validation Loss')
            plt.title('Autoencoder Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Individual classifier MCCs
            plt.subplot(1, 3, 2)
            if 'best_individual_mccs' in history and history['best_individual_mccs']:
                classifiers = list(history['best_individual_mccs'].keys())
                mccs = list(history['best_individual_mccs'].values())
                plt.bar(classifiers, mccs, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
                plt.title('Individual Classifier Performance')
                plt.ylabel('Matthews Correlation Coefficient')
                plt.xticks(rotation=45)
                plt.grid(True, axis='y')
                
                # Add value labels on bars
                for i, (classifier, mcc) in enumerate(zip(classifiers, mccs)):
                    plt.text(i, mcc + 0.01, f'{mcc:.3f}', ha='center', va='bottom')
            
            # Performance comparison
            plt.subplot(1, 3, 3)
            performance_data = []
            labels = []
            
            if 'ensemble_mcc' in history and history['ensemble_mcc']:
                performance_data.append(history['ensemble_mcc'])
                labels.append('Ensemble MCC')
            
            if 'best_test_mcc' in history and history['best_test_mcc']:
                performance_data.append(history['best_test_mcc'])
                labels.append('Best Test MCC')
            
            if performance_data:
                plt.bar(labels, performance_data, color=['purple', 'orange'])
                plt.title('Final Performance Metrics')
                plt.ylabel('Matthews Correlation Coefficient')
                plt.ylim(0, 1.1)
                
                # Add target line at MCC = 0.95
                plt.axhline(y=0.95, color='red', linestyle='--', label='Target (0.95)')
                plt.legend()
                
                # Add value labels
                for i, (label, value) in enumerate(zip(labels, performance_data)):
                    plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            analysis_path = os.path.join(results_dir, 'training_analysis.png')
            plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Training analysis plot saved: {analysis_path}")
        
    except Exception as e:
        print(f"❌ Error analyzing training progress: {e}")

def monitor_system_resources():
    """Monitor system resources during training"""
    
    print(f"\n💻 SYSTEM RESOURCE MONITORING")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"🔧 CPU: {cpu_percent:.1f}% usage ({cpu_count} cores)")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"💾 RAM: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    # GPU usage (if available)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory_used = torch.cuda.memory_allocated(i) / 1024**3
            gpu_memory_total = torch.cuda.memory_reserved(i) / 1024**3
            print(f"🔥 GPU {i}: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB")
    
    # Disk usage
    disk = psutil.disk_usage('.')
    print(f"💽 Disk: {disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB ({disk.percent:.1f}%)")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent
    }