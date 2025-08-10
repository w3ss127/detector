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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from skimage.metrics import structural_similarity as ssim
import scipy.stats as stats
from scipy import ndimage
import os
from typing import Tuple, List, Dict
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

class PTFileDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, device: str = 'cpu'):
        """
        Dataset for .pt files containing image tensors
        Args:
            images: torch.Tensor of shape (N, C, H, W), dtype float32
            labels: torch.Tensor of shape (N,), dtype int64
            device: Device to move tensors to
        """
        self.images = images.float()  # Ensure float32
        self.labels = labels.long()   # Ensure int64
        self.device = device
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image.to(self.device), label.to(self.device)

def load_pt_data(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load all .pt files from the dataset directory structure
    Args:
        data_dir: Path to datasets/train directory containing {real,synthetic,semi-synthetic} folders
    Returns:
        images: Combined tensor of all images (float32, [0, 1])
        labels: Combined tensor of all labels (int64)
    """
    all_images = []
    all_labels = []
    
    class_folders = ['real', 'synthetic', 'semi-synthetic']
    class_mapping = {'real': 0, 'synthetic': 1, 'semi-synthetic': 2}
    
    for class_name in class_folders:
        class_idx = class_mapping[class_name]
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping...")
            continue
            
        print(f"Loading {class_name} images...")
        
        pt_files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
        pt_files.sort()
        
        class_images = []
        
        for pt_file in tqdm(pt_files, desc=f"Loading {class_name} files"):
            pt_path = os.path.join(class_dir, pt_file)
            
            try:
                tensor_data = torch.load(pt_path, map_location='cpu')
                
                if isinstance(tensor_data, dict):
                    if 'images' in tensor_data:
                        images = tensor_data['images']
                    elif 'data' in tensor_data:
                        images = tensor_data['data']
                    else:
                        images = list(tensor_data.values())[0]
                else:
                    images = tensor_data
                
                # Ensure tensor has correct shape (N, C, H, W)
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                elif images.dim() == 5:
                    images = images.squeeze()
                
                # Convert to float32 and normalize to [0, 1]
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                else:
                    images = images.float()
                    # Ensure values are in [0, 1] range
                    if images.max() > 1.0:
                        images = images / 255.0
                
                print(f"    Loaded {images.shape[0]} images with shape {images.shape[1:]} (dtype: {images.dtype})")
                class_images.append(images)
                
            except Exception as e:
                print(f"    Error loading {pt_file}: {e}. Skipping...")
                continue
        
        if class_images:
            class_tensor = torch.cat(class_images, dim=0)
            all_images.append(class_tensor)
            
            num_images = class_tensor.shape[0]
            class_labels = torch.full((num_images,), class_idx, dtype=torch.long)
            all_labels.append(class_labels)
            
            print(f"Total {class_name} images: {num_images}")
    
    if all_images:
        combined_images = torch.cat(all_images, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        
        print(f"\nTotal dataset: {combined_images.shape[0]} images")
        print(f"Image shape: {combined_images.shape[1:]}")
        print(f"Label distribution: {torch.bincount(combined_labels)}")
        
        return combined_images, combined_labels
    else:
        raise ValueError("No data loaded! Check your directory structure and .pt files.")

def create_train_val_test_split(images: torch.Tensor, labels: torch.Tensor, 
                               train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                               random_seed=42) -> Tuple[PTFileDataset, PTFileDataset, PTFileDataset]:
    """
    Split the dataset into train/validation/test sets with stratification
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Stratified split to maintain class balance
    unique_labels = torch.unique(labels)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label in unique_labels:
        label_indices = torch.where(labels == label)[0]
        label_indices = label_indices[torch.randperm(len(label_indices))]
        
        n_samples = len(label_indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices.extend(label_indices[:n_train].tolist())
        val_indices.extend(label_indices[n_train:n_train + n_val].tolist())
        test_indices.extend(label_indices[n_train + n_val:].tolist())
    
    # Shuffle the indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    train_indices = torch.tensor(train_indices)
    val_indices = torch.tensor(val_indices)
    test_indices = torch.tensor(test_indices)
    
    val_dataset = PTFileDataset(images[val_indices], labels[val_indices])
    train_dataset = PTFileDataset(images[train_indices], labels[train_indices])
    test_dataset = PTFileDataset(images[test_indices], labels[test_indices])
    
    print(f"\nStratified dataset split:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")
    
    for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        class_counts = torch.bincount(dataset.labels, minlength=3)
        print(f"{split_name} class distribution: Real={class_counts[0]}, Synthetic={class_counts[1]}, Semi-synthetic={class_counts[2]}")
    
    return train_dataset, val_dataset, test_dataset

class ImprovedDenoiseAutoencoder(nn.Module):
    """Improved autoencoder with residual connections and batch normalization"""
    def __init__(self, input_channels=3):
        super(ImprovedDenoiseAutoencoder, self).__init__()
        
        # Encoder with residual-like connections
        self.encoder = nn.ModuleList([
            # First block
            nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        
        # Decoder
        reconstructed = self.decoder(x)
        return reconstructed

class UltraFastNoiseDistributionExtractor:
    """Ultra-fast noise feature extractor with GPU acceleration and parallel processing"""
    
    def __init__(self, n_jobs=-1, use_gpu_features=True):
        self.scaler = StandardScaler()
        self.fitted = False
        self.n_jobs = n_jobs if n_jobs != -1 else min(mp.cpu_count(), 16)
        self.use_gpu_features = use_gpu_features and torch.cuda.is_available()
        print(f"🚀 Ultra-fast noise extractor initialized with {self.n_jobs} CPU cores")
        if self.use_gpu_features:
            print("⚡ GPU acceleration enabled for feature extraction")
    
    def extract_noise_features_ultra_fast(self, noise_map: np.ndarray) -> np.ndarray:
        """Extract features with GPU acceleration for large images"""
        if self.use_gpu_features and noise_map.size > 1024:
            try:
                noise_tensor = torch.from_numpy(noise_map).float()
                if torch.cuda.is_available():
                    noise_tensor = noise_tensor.cuda()
                return self._extract_gpu_features(noise_tensor).cpu().numpy()
            except:
                pass
        return self._extract_cpu_features_optimized(noise_map)
    
    def _extract_gpu_features(self, noise_tensor: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated feature extraction"""
        features = []
        
        # Basic statistical features
        noise_flat = noise_tensor.flatten()
        features.extend([
            torch.mean(noise_flat).item(),
            torch.std(noise_flat).item(),
            torch.var(noise_flat).item()
        ])
        
        # Fast percentiles using sorting
        sorted_noise = torch.sort(noise_flat)[0]
        n = len(sorted_noise)
        indices = [int(n * p) for p in [0.05, 0.25, 0.5, 0.75, 0.95]]
        percentiles = [sorted_noise[min(idx, n-1)].item() for idx in indices]
        features.extend(percentiles)
        features.append(percentiles[-1] - percentiles[0])  # Range
        
        # Fast histogram
        hist = torch.histc(noise_flat, bins=20, min=-1, max=1)
        hist = hist / (torch.sum(hist) + 1e-8)
        features.extend(hist.cpu().tolist())
        
        # GPU-accelerated gradient computation
        if noise_tensor.dim() == 2:
            noise_tensor = noise_tensor.unsqueeze(0).unsqueeze(0)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if torch.cuda.is_available():
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        grad_x = F.conv2d(noise_tensor, sobel_x, padding=1)
        grad_y = F.conv2d(noise_tensor, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        grad_flat = grad_mag.flatten()
        
        features.extend([
            torch.mean(grad_flat).item(),
            torch.std(grad_flat).item(),
            torch.quantile(grad_flat, 0.9).item(),
            torch.quantile(grad_flat, 0.95).item()
        ])
        
        # Spatial distribution analysis
        h, w = noise_tensor.shape[-2:]
        if h > 4 and w > 4:
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                noise_tensor[..., :mid_h, :mid_w],
                noise_tensor[..., :mid_h, mid_w:],
                noise_tensor[..., mid_h:, :mid_w],
                noise_tensor[..., mid_h:, mid_w:]
            ]
            quad_means = [torch.mean(quad).item() for quad in quadrants]
            quad_vars = [torch.var(quad).item() for quad in quadrants]
            features.extend([
                np.std(quad_means),
                np.std(quad_vars),
                max(quad_vars) - min(quad_vars)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Wavelet features (computed on CPU for compatibility)
        noise_cpu = noise_tensor.cpu().numpy().squeeze()
        try:
            coeffs = pywt.wavedec2(noise_cpu, 'db1', level=2)
            for coeff in coeffs:
                if isinstance(coeff, np.ndarray):
                    features.extend([np.mean(coeff), np.std(coeff)])
                else:
                    for c in coeff:
                        features.extend([np.mean(c), np.std(c)])
        except:
            # Fallback features if wavelet fails
            features.extend([0.0] * 10)
        
        # Ensure consistent feature length
        target_length = 50
        while len(features) < target_length:
            features.append(0.0)
        
        return torch.tensor(features[:target_length], dtype=torch.float32)
    
    def _extract_cpu_features_optimized(self, noise_map: np.ndarray) -> np.ndarray:
        """Optimized CPU feature extraction"""
        features = []
        
        # Basic statistical features
        noise_flat = noise_map.flatten()
        features.extend([
            np.mean(noise_flat),
            np.std(noise_flat),
            np.var(noise_flat),
            stats.skew(noise_flat) if len(noise_flat) > 0 else 0.0,
            stats.kurtosis(noise_flat) if len(noise_flat) > 0 else 0.0,
        ])
        
        # Fast percentiles
        percentiles = np.percentile(noise_flat, [5, 25, 50, 75, 95])
        features.extend(percentiles.tolist())
        features.append(percentiles[4] - percentiles[0])  # Range
        
        # Optimized histogram
        hist, _ = np.histogram(noise_flat, bins=20, range=(-1, 1))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist.tolist())
        
        # Gradient analysis (only for reasonably sized images)
        h, w = noise_map.shape
        if h > 8 and w > 8:
            grad_x = cv2.Sobel(noise_map, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(noise_map, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_flat = grad_mag.flatten()
            features.extend([
                np.mean(grad_flat),
                np.std(grad_flat),
                np.percentile(grad_flat, 90),
                np.percentile(grad_flat, 95)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Fast spatial distribution analysis
        if h > 4 and w > 4:
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                noise_map[:mid_h, :mid_w],
                noise_map[:mid_h, mid_w:],
                noise_map[mid_h:, :mid_w],
                noise_map[mid_h:, mid_w:]
            ]
            quad_vars = [np.var(quad) for quad in quadrants]
            features.extend([
                np.std(quad_vars),
                np.max(quad_vars) - np.min(quad_vars),
                np.mean(quad_vars)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Wavelet features
        try:
            coeffs = pywt.wavedec2(noise_map, 'db1', level=2)
            for coeff in coeffs:
                if isinstance(coeff, np.ndarray):
                    features.extend([np.mean(coeff), np.std(coeff)])
                else:
                    for c in coeff:
                        features.extend([np.mean(c), np.std(c)])
        except:
            # Fallback features if wavelet fails
            features.extend([0.0] * 10)
        
        # Ensure consistent feature length
        target_length = 50
        while len(features) < target_length:
            features.append(0.0)
        
        return np.array(features[:target_length], dtype=np.float32)
    
    def extract_batch_features_parallel(self, noise_maps_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Extract features from a batch using parallel processing"""
        if len(noise_maps_batch) <= 4 or self.n_jobs == 1:
            return [self.extract_noise_features_ultra_fast(nm) for nm in noise_maps_batch]
        
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(noise_maps_batch))) as executor:
            futures = [executor.submit(self.extract_noise_features_ultra_fast, nm) 
                      for nm in noise_maps_batch]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    results.append(np.zeros(50, dtype=np.float32))  # Updated to match target_length
            return results
    
    def fit_transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        """Ultra-fast feature extraction with parallel processing"""
        print(f"🚀 Ultra-fast feature extraction from {len(noise_maps)} noise maps...")
        
        # Determine optimal batch size based on available resources
        batch_size = max(32, len(noise_maps) // (self.n_jobs * 2))
        batches = [noise_maps[i:i + batch_size] for i in range(0, len(noise_maps), batch_size)]
        
        feature_matrix = []
        start_time = time.time()
        
        for batch in tqdm(batches, desc="⚡ Ultra-fast feature extraction"):
            batch_features = self.extract_batch_features_parallel(batch)
            feature_matrix.extend(batch_features)
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        extraction_time = time.time() - start_time
        
        print(f"✅ Feature matrix shape: {feature_matrix.shape}")
        print(f"⏱️ Extraction speed: {len(noise_maps)/extraction_time:.1f} maps/second")
        
        # Handle any NaN or infinite values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.fitted = True
        
        return feature_matrix
    
    def transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        """Transform noise maps to features (after fitting)"""
        if not self.fitted:
            raise ValueError("Must call fit_transform first")
        
        print(f"🔄 Transforming {len(noise_maps)} noise maps...")
        
        # Use same batching strategy
        batch_size = max(32, len(noise_maps) // (self.n_jobs * 2))
        batches = [noise_maps[i:i + batch_size] for i in range(0, len(noise_maps), batch_size)]
        
        feature_matrix = []
        for batch in tqdm(batches, desc="⚡ Feature transformation"):
            batch_features = self.extract_batch_features_parallel(batch)
            feature_matrix.extend(batch_features)
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.scaler.transform(feature_matrix)

class NoiseClassificationPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint_dir='noise_checkpoints'):
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        
        print(f"🚀 Initializing optimized pipeline...")
        print(f"🔧 Primary device: {self.device}")
        if self.num_gpus > 0:
            print(f"🔥 Available GPUs: {self.num_gpus}")
            for i in range(self.num_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        self.autoencoder = ImprovedDenoiseAutoencoder().to(self.device)
        if self.num_gpus > 1:
            print("🔗 Enabling multi-GPU support")
            self.autoencoder = nn.DataParallel(self.autoencoder)
        
        self.noise_extractor = UltraFastNoiseDistributionExtractor(n_jobs=-1, use_gpu_features=True)
        
        # Optimized classifier with hyperparameter tuning
        self.classifier = RandomForestClassifier(
            n_estimators=500,  # Increased for better performance
            max_depth=20,      # Increased depth
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced',
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
            'epochs_trained': 0
        }
    
    def save_checkpoint(self, checkpoint_name='latest', extra_data=None):
        """Save current training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth')
        
        autoencoder_state = None
        if self.autoencoder is not None:
            if self.num_gpus > 1:
                autoencoder_state = self.autoencoder.module.state_dict()
            else:
                autoencoder_state = self.autoencoder.state_dict()
        
        # Save noise extractor with consistent naming
        extractor_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_extractor.pkl')
        if self.noise_extractor.fitted:
            try:
                with open(extractor_path, 'wb') as f:
                    pickle.dump(self.noise_extractor, f)
                print(f"💾 Noise extractor saved: {extractor_path}")
            except Exception as e:
                print(f"⚠️ Error saving noise extractor: {e}")
        
        # Save classifier with consistent naming
        classifier_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_classifier.pkl')
        if self.classifier_trained:
            try:
                with open(classifier_path, 'wb') as f:
                    pickle.dump(self.classifier, f)
                print(f"💾 Classifier saved: {classifier_path}")
            except Exception as e:
                print(f"⚠️ Error saving classifier: {e}")
        
        # Also save with the legacy naming for backward compatibility
        if checkpoint_name in ['classifier_final', 'complete_pipeline', 'latest']:
            # Save with the naming expected by fast inference functions
            legacy_classifier_path = os.path.join(self.checkpoint_dir, 'random_forest_classifier.pkl')
            legacy_scaler_path = os.path.join(self.checkpoint_dir, 'feature_scaler.pkl')
            
            if self.classifier_trained:
                try:
                    with open(legacy_classifier_path, 'wb') as f:
                        pickle.dump(self.classifier, f)
                    print(f"💾 Legacy classifier saved: {legacy_classifier_path}")
                except Exception as e:
                    print(f"⚠️ Error saving legacy classifier: {e}")
                    
            if self.noise_extractor.fitted:
                try:
                    with open(legacy_scaler_path, 'wb') as f:
                        pickle.dump(self.noise_extractor.scaler, f)
                    print(f"💾 Legacy scaler saved: {legacy_scaler_path}")
                except Exception as e:
                    print(f"⚠️ Error saving legacy scaler: {e}")
        
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
            print(f"💾 Main checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Error saving main checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_name='latest'):
        """Load training state from checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint {checkpoint_path} not found.")
            return False, 0
        
        print(f"🔄 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load autoencoder
        if checkpoint['autoencoder_state_dict'] is not None:
            self.autoencoder = ImprovedDenoiseAutoencoder().to(self.device)
            if self.num_gpus > 1:
                self.autoencoder = nn.DataParallel(self.autoencoder)
                self.autoencoder.module.load_state_dict(checkpoint['autoencoder_state_dict'])
            else:
                self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            
        # Load noise extractor
        extractor_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_extractor.pkl')
        if os.path.exists(extractor_path):
            try:
                with open(extractor_path, 'rb') as f:
                    self.noise_extractor = pickle.load(f)
                print("✅ Noise extractor loaded")
            except Exception as e:
                print(f"⚠️ Could not load noise extractor: {e}")
        
        # Load classifier
        classifier_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_classifier.pkl')
        if os.path.exists(classifier_path):
            try:
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                print("✅ Classifier loaded")
            except Exception as e:
                print(f"⚠️ Could not load classifier: {e}")
        
        self.autoencoder_trained = checkpoint.get('autoencoder_trained', False)
        self.classifier_trained = checkpoint.get('classifier_trained', False)
        self.training_history = checkpoint.get('training_history', {
            'autoencoder_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
            'best_test_mcc': 0.0,
            'epochs_trained': 0
        })
        
        print(f"✅ Loaded: Autoencoder trained: {self.autoencoder_trained}, Classifier trained: {self.classifier_trained}")
        return True, self.training_history.get('epochs_trained', 0)

    def train_autoencoder(self, train_loader: DataLoader, val_loader: DataLoader = None, 
                         epochs=30, resume_from_epoch=0, save_every=1):
        """Train autoencoder with improved training strategy and GPU acceleration"""
        if self.autoencoder_trained and resume_from_epoch == 0:
            print("✅ Autoencoder already trained.")
            return
            
        print(f"🚀 Training autoencoder for {epochs} epochs with GPU acceleration")
        
        if self.autoencoder is None:
            self.autoencoder = ImprovedDenoiseAutoencoder().to(self.device)
            if self.num_gpus > 1:
                self.autoencoder = nn.DataParallel(self.autoencoder)
        
        # Optimized training configuration
        initial_lr = 0.001
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=initial_lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        criterion = nn.MSELoss()
        self.autoencoder.train()
        
        start_epoch = resume_from_epoch
        
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            num_batches = 0
            epoch_start_time = time.time()
            
            progress_bar = tqdm(train_loader, desc=f'🔥 Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (data, _) in enumerate(progress_bar):
                data = data.to(self.device, non_blocking=True)
                
                # Progressive noise strategy
                if epoch < epochs // 3:
                    noise_strength = 0.05
                elif epoch < 2 * epochs // 3:
                    noise_strength = 0.1
                else:
                    noise_strength = 0.15
                
                # Add different types of noise
                gaussian_noise = torch.randn_like(data) * noise_strength
                # Add some salt and pepper noise occasionally
                if np.random.random() < 0.3:
                    salt_pepper = torch.rand_like(data)
                    salt_pepper = (salt_pepper < 0.05).float() + (salt_pepper > 0.95).float() * -1
                    gaussian_noise += salt_pepper * 0.1
                
                noisy_data = torch.clamp(data + gaussian_noise, 0., 1.)
                
                optimizer.zero_grad()
                
                # Use mixed precision for speed
                if self.scaler is not None:
                    with autocast():
                        reconstructed = self.autoencoder(noisy_data)
                        loss = criterion(reconstructed, data)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    reconstructed = self.autoencoder(noisy_data)
                    loss = criterion(reconstructed, data)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Clear cache periodically
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / num_batches
            self.training_history['autoencoder_losses'].append(avg_loss)
            self.training_history['epochs_trained'] = epoch + 1
            
            # Validation
            val_loss = 0.0
            if val_loader is not None:
                val_loss = self._validate_autoencoder(val_loader, epoch, epochs)
                self.training_history['val_losses'].append(val_loss)
                scheduler.step(val_loss)
                
                if val_loss < self.training_history['best_val_loss']:
                    self.training_history['best_val_loss'] = val_loss
                    self.save_checkpoint('best_autoencoder')
            
            print(f'⚡ Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.1f}s')
            
            # Regular checkpointing
            if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
                checkpoint_name = f'autoencoder_epoch_{epoch+1}'
                self.save_checkpoint(checkpoint_name)
        
        self.autoencoder_trained = True
        self.save_checkpoint('autoencoder_final')
        print("🎉 Autoencoder training completed!")
    
    def _validate_autoencoder(self, val_loader: DataLoader, current_epoch: int, total_epochs: int) -> float:
        """Validate autoencoder performance with GPU acceleration"""
        self.autoencoder.eval()
        total_loss = 0
        num_batches = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device, non_blocking=True)
                
                # Use similar noise as training
                if current_epoch < total_epochs // 3:
                    noise_strength = 0.05
                elif current_epoch < 2 * total_epochs // 3:
                    noise_strength = 0.1
                else:
                    noise_strength = 0.15
                
                noise = torch.randn_like(data) * noise_strength
                noisy_data = torch.clamp(data + noise, 0., 1.)
                
                if self.scaler is not None:
                    with autocast():
                        reconstructed = self.autoencoder(noisy_data)
                        loss = criterion(reconstructed, data)
                else:
                    reconstructed = self.autoencoder(noisy_data)
                    loss = criterion(reconstructed, data)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.autoencoder.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def extract_noise_from_images_fast(self, images: torch.Tensor, batch_size: int = 128) -> List[np.ndarray]:
        """Extract noise maps using trained autoencoder with GPU acceleration"""
        if self.autoencoder is None or not self.autoencoder_trained:
            raise ValueError("Autoencoder must be trained first")
        
        self.autoencoder.eval()
        noise_maps = []
        
        # Use larger batch sizes for GPU acceleration
        effective_batch_size = batch_size * max(1, self.num_gpus)
        print(f"⚡ Extracting noise maps with batch size: {effective_batch_size}")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), effective_batch_size), desc="🔍 Extracting noise"):
                batch = images[i:i+effective_batch_size].to(self.device, non_blocking=True)
                
                try:
                    if self.scaler is not None:
                        with autocast():
                            reconstructed = self.autoencoder(batch)
                            noise = batch - reconstructed
                    else:
                        reconstructed = self.autoencoder(batch)
                        noise = batch - reconstructed
                    
                    noise_np = noise.cpu().numpy()
                    
                    for j in range(noise_np.shape[0]):
                        # Average across channels to get single noise map
                        noise_map = np.mean(noise_np[j], axis=0).astype(np.float32)
                        noise_maps.append(noise_map)
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️ GPU memory error, reducing batch size...")
                    # Fall back to smaller batches
                    for k in range(0, len(batch), batch_size // 2):
                        mini_batch = batch[k:k+batch_size//2]
                        if self.scaler is not None:
                            with autocast():
                                reconstructed = self.autoencoder(mini_batch)
                                noise = mini_batch - reconstructed
                        else:
                            reconstructed = self.autoencoder(mini_batch)
                            noise = mini_batch - reconstructed
                        noise_np = noise.cpu().numpy()
                        for j in range(noise_np.shape[0]):
                            noise_map = np.mean(noise_np[j], axis=0).astype(np.float32)
                            noise_maps.append(noise_map)
                
                # Clear cache periodically
                if i % (effective_batch_size * 10) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"✅ Extracted {len(noise_maps)} noise maps")
        return noise_maps
    
    def visualize_noise_maps(self, images: torch.Tensor, labels: torch.Tensor = None, 
                           num_samples: int = 9, save_path: str = None):
        """Visualize noise maps with class labels"""
        if self.autoencoder is None or not self.autoencoder_trained:
            raise ValueError("Autoencoder must be trained first")
        
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
            
            # Create subplot grid
            rows = int(np.ceil(num_samples / 3))
            fig, axes = plt.subplots(rows, 9, figsize=(18, 6*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
                
            for i in range(num_samples):
                row = i // 3
                col_start = (i % 3) * 3
                
                # Original image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                if img.shape[2] == 1:
                    img = img.squeeze()
                    axes[row, col_start].imshow(img, cmap='gray')
                else:
                    axes[row, col_start].imshow(np.clip(img, 0, 1))
                
                title = 'Original'
                if labels is not None:
                    title += f' ({class_names[labels[i]]})'
                axes[row, col_start].set_title(title)
                axes[row, col_start].axis('off')
                
                # Reconstructed image
                recon = reconstructed[i].cpu().permute(1, 2, 0).numpy()
                if recon.shape[2] == 1:
                    recon = recon.squeeze()
                    axes[row, col_start + 1].imshow(recon, cmap='gray')
                else:
                    axes[row, col_start + 1].imshow(np.clip(recon, 0, 1))
                axes[row, col_start + 1].set_title('Reconstructed')
                axes[row, col_start + 1].axis('off')
                
                # Noise map
                noise = noise_maps[i].cpu().mean(dim=0).numpy()
                im = axes[row, col_start + 2].imshow(noise, cmap='RdBu_r', 
                                                   vmin=-0.2, vmax=0.2)
                axes[row, col_start + 2].set_title('Noise Map')
                axes[row, col_start + 2].axis('off')
                plt.colorbar(im, ax=axes[row, col_start + 2], shrink=0.8)
                
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📊 Noise maps saved to {save_path}")
            plt.show()
    
    def train_noise_classifier(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train the noise distribution classifier with ultra-fast processing"""
        if self.classifier_trained:
            print("✅ Noise classifier already trained.")
            return
            
        if not self.autoencoder_trained:
            raise ValueError("❌ Autoencoder must be trained first")
        
        print("\n" + "="*70)
        print("🚀 STARTING ULTRA-FAST NOISE CLASSIFIER TRAINING")
        print("="*70)
        
        total_start_time = time.time()
        
        # Step 1: Extract noise features from training data with GPU acceleration
        print("\n[1/4] 🔍 Extracting noise maps from training data...")
        extraction_start = time.time()
        
        all_noise_maps = []
        all_labels = []
        total_samples = 0
        
        # Get total number of samples for progress tracking
        for _, labels in train_loader:
            total_samples += len(labels)
        print(f"📊 Total training samples to process: {total_samples}")
        
        # Process training data in larger batches for speed
        batch_count = 0
        samples_processed = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="⚡ Processing batches")):
            try:
                # Use fast noise extraction
                noise_maps = self.extract_noise_from_images_fast(images, batch_size=128)
                all_noise_maps.extend(noise_maps)
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1
                samples_processed += len(images)
                
                # Print periodic updates
                if batch_idx % 20 == 0 and batch_idx > 0:
                    elapsed = time.time() - extraction_start
                    speed = samples_processed / elapsed
                    print(f"    📈 Progress: {samples_processed}/{total_samples} samples ({100*samples_processed/total_samples:.1f}%) - Speed: {speed:.1f} samples/sec")
                    
            except Exception as e:
                print(f"    ❌ Error processing batch {batch_idx}: {e}")
                continue
        
        extraction_time = time.time() - extraction_start
        print(f"✅ Successfully processed {batch_count} batches in {extraction_time:.2f} seconds")
        print(f"✅ Extracted {len(all_noise_maps)} noise maps from {samples_processed} samples")
        print(f"🚀 Extraction speed: {len(all_noise_maps)/extraction_time:.1f} maps/second")
        
        if len(all_noise_maps) == 0:
            raise ValueError("❌ No noise maps extracted from training data")
        
        # Step 2: Ultra-fast feature extraction
        print(f"\n[2/4] ⚡ Computing noise distribution features with parallel processing...")
        feature_start = time.time()
        
        try:
            feature_matrix = self.noise_extractor.fit_transform(all_noise_maps)
            feature_time = time.time() - feature_start
            print(f"✅ Feature extraction completed in {feature_time:.2f} seconds")
            print(f"📊 Feature matrix shape: {feature_matrix.shape}")
            print(f"🚀 Feature extraction speed: {len(all_noise_maps)/feature_time:.1f} maps/second")
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            raise
        
        # Step 3: Train classifier with hyperparameter optimization
        print(f"\n[3/4] 🌳 Training optimized Random Forest classifier...")
        classifier_start = time.time()
        
        # Show class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"📊 Training class distribution:")
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        for label, count in zip(unique, counts):
            print(f"   {class_names[label]}: {count:,} samples ({100*count/len(all_labels):.1f}%)")
        
        # Hyperparameter tuning for better performance
        param_dist = {
            'n_estimators': [200, 500, 1000],
            'max_depth': [15, 20, 30, None],
            'min_samples_split': [3, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_search = RandomizedSearchCV(
            self.classifier, param_distributions=param_dist, n_iter=10,
            scoring='accuracy', cv=3, n_jobs=-1, random_state=42, verbose=1
        )
        
        print("🔍 Performing hyperparameter optimization...")
        rf_search.fit(feature_matrix, all_labels)
        self.classifier = rf_search.best_estimator_
        
        classifier_time = time.time() - classifier_start
        
        print(f"✅ Best parameters: {rf_search.best_params_}")
        print(f"✅ Best CV score: {rf_search.best_score_:.4f}")
        print(f"✅ Random Forest training completed in {classifier_time:.2f} seconds")
        
        # Step 4: Evaluate training performance
        print(f"\n[4/4] 📊 Evaluating classifier performance...")
        
        print("📈 Computing training metrics...")
        train_predictions = self.classifier.predict(feature_matrix)
        train_accuracy = accuracy_score(all_labels, train_predictions)
        train_mcc = matthews_corrcoef(all_labels, train_predictions)
        
        print(f"\n🎯 Training Results:")
        print(f"   Accuracy: {train_accuracy:.4f}")
        print(f"   MCC: {train_mcc:.4f}")
        
        # Training confusion matrix
        train_cm = confusion_matrix(all_labels, train_predictions)
        print(f"\n📊 Training Confusion Matrix:")
        print("True\\Pred    Real  Synth  Semi")
        for i, row in enumerate(train_cm):
            print(f"{class_names[i]:10s} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
        
        # Feature importance analysis
        if hasattr(self.classifier, 'feature_importances_'):
            feature_importance = self.classifier.feature_importances_
            top_features = np.argsort(feature_importance)[-10:]
            print(f"\n🔝 Top 10 Most Important Features:")
            for i, feat_idx in enumerate(reversed(top_features)):
                print(f"   {i+1:2d}. Feature {feat_idx:3d}: {feature_importance[feat_idx]:.4f}")
        
        # Validation evaluation
        if val_loader is not None:
            print(f"\n🔍 Validating on validation set...")
            try:
                val_predictions, val_probabilities = self.predict_fast(val_loader)
                val_labels = []
                for _, labels in val_loader:
                    val_labels.extend(labels.cpu().numpy())
                
                # Align lengths
                min_len = min(len(val_labels), len(val_predictions))
                val_labels = val_labels[:min_len]
                val_predictions = val_predictions[:min_len]
                
                val_accuracy = accuracy_score(val_labels, val_predictions)
                val_mcc = matthews_corrcoef(val_labels, val_predictions)
                
                print(f"\n✅ Validation Results:")
                print(f"   Accuracy: {val_accuracy:.4f}")
                print(f"   MCC: {val_mcc:.4f}")
                
                # Validation confusion matrix
                val_cm = confusion_matrix(val_labels, val_predictions)
                print(f"\n📊 Validation Confusion Matrix:")
                print("True\\Pred    Real  Synth  Semi")
                for i, row in enumerate(val_cm):
                    print(f"{class_names[i]:10s} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
                
                # Per-class performance
                val_report = classification_report(val_labels, val_predictions, 
                                                 target_names=class_names, 
                                                 output_dict=True, zero_division=0)
                print(f"\n📈 Per-class Validation Performance:")
                for class_name in class_names:
                    if class_name in val_report:
                        metrics = val_report[class_name]
                        print(f"   {class_name}:")
                        print(f"     Precision: {metrics['precision']:.4f}")
                        print(f"     Recall:    {metrics['recall']:.4f}")
                        print(f"     F1-score:  {metrics['f1-score']:.4f}")
                    
            except Exception as e:
                print(f"❌ Error during validation: {e}")
        
        # Save trained components
        self.classifier_trained = True
        
        # Save everything with the checkpoint system (this will now save correctly)
        self.save_checkpoint('classifier_final')
        
        total_time = time.time() - total_start_time
        
        print("\n" + "="*70)
        print("🎉 ULTRA-FAST NOISE CLASSIFIER TRAINING COMPLETED!")
        print("="*70)
        print(f"📊 Processed {len(all_noise_maps):,} training samples")
        print(f"🔧 Extracted {feature_matrix.shape[1]} noise features per sample")
        print(f"🌳 Trained Random Forest with {self.classifier.n_estimators} trees")
        print(f"🎯 Final training accuracy: {train_accuracy:.4f}")
        if val_loader:
            print(f"🎯 Final validation accuracy: {val_accuracy:.4f}")
        print(f"⏱️ TIMING BREAKDOWN:")
        print(f"   Noise Extraction: {extraction_time:.2f}s ({100*extraction_time/total_time:.1f}%)")
        print(f"   Feature Extraction: {feature_time:.2f}s ({100*feature_time/total_time:.1f}%)")
        print(f"   RF Training: {classifier_time:.2f}s ({100*classifier_time/total_time:.1f}%)")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"🚀 Overall Speed: {len(all_noise_maps)/total_time:.1f} samples/second")
        print(f"💾 Model saved to checkpoint: 'classifier_final'")
        print("="*70)
    
    def predict_fast(self, test_loader: DataLoader, batch_size: int = 128) -> Tuple[List[int], List[float]]:
        """Ultra-fast prediction with GPU acceleration"""
        if not self.autoencoder_trained or not self.classifier_trained:
            raise ValueError("❌ Both autoencoder and classifier must be trained")
        if not self.noise_extractor.fitted:
            raise ValueError("❌ Feature scaler must be fitted!")
        
        print("🔮 Generating ultra-fast predictions...")
        
        all_predictions = []
        all_probabilities = []
        
        self.autoencoder.eval()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="⚡ Predicting")):
                try:
                    # Extract noise maps with GPU acceleration
                    noise_maps = self.extract_noise_from_images_fast(images, batch_size=batch_size)
                    
                    # Convert to features using parallel processing
                    feature_matrix = self.noise_extractor.transform(noise_maps)
                    
                    # Predict
                    predictions = self.classifier.predict(feature_matrix)
                    probabilities = self.classifier.predict_proba(feature_matrix)
                    
                    all_predictions.extend(predictions.tolist())
                    all_probabilities.extend(probabilities.tolist())
                    
                except Exception as e:
                    print(f"⚠️ Error predicting batch {batch_idx}: {e}")
                    # Add dummy predictions to maintain alignment
                    batch_size_actual = len(images)
                    all_predictions.extend([0] * batch_size_actual)  # Default to class 0
                    all_probabilities.extend([[1.0, 0.0, 0.0]] * batch_size_actual)
        
        print(f"✅ Generated predictions for {len(all_predictions)} samples")
        return all_predictions, all_probabilities
    
    def evaluate(self, test_loader: DataLoader, test_labels: List[int], 
                 save_results: bool = True, results_dir: str = 'results') -> Dict:
        """Comprehensive evaluation with ultra-fast processing and visualization"""
        print("\n🔍 Starting comprehensive evaluation...")
        eval_start = time.time()
        
        predictions, probabilities = self.predict_fast(test_loader)
        
        # Ensure same length
        min_len = min(len(test_labels), len(predictions))
        test_labels = test_labels[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        
        # Calculate metrics
        mcc = matthews_corrcoef(test_labels, predictions)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, 
                                    target_names=['Real', 'Synthetic', 'Semi-synthetic'],
                                    output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        eval_time = time.time() - eval_start
        
        if mcc > self.training_history['best_test_mcc']:
            self.training_history['best_test_mcc'] = mcc
            if save_results:
                self.save_checkpoint('best_test_model', {'test_mcc': mcc, 'test_accuracy': accuracy})
        
        results = {
            'mcc': mcc,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'probabilities': probabilities,
            'evaluation_time': eval_time,
            'prediction_speed': len(predictions) / eval_time
        }
        
        print("\n" + "="*60)
        print("🎯 EVALUATION RESULTS")
        print("="*60)
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"📈 Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"⏱️ Evaluation time: {eval_time:.2f} seconds")
        print(f"🚀 Prediction speed: {len(predictions)/eval_time:.1f} samples/second")
        
        print("\n📊 Confusion Matrix:")
        print("True\\Pred    Real  Synth  Semi")
        class_names = ['Real     ', 'Synthetic', 'Semi-synth']
        for i, (name, row) in enumerate(zip(class_names, cm)):
            print(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
        
        print("\n📈 Per-class Metrics:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
        
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            
            # Save comprehensive results
            results_file = os.path.join(results_dir, 'evaluation_results.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'accuracy': accuracy,
                    'mcc': mcc,
                    'evaluation_time': eval_time,
                    'prediction_speed': len(predictions) / eval_time,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report,
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'true_labels': test_labels
                }, f, indent=2)
            
            # Save text summary
            results_path = os.path.join(results_dir, 'evaluation_summary.txt')
            with open(results_path, 'w') as f:
                f.write(f"ULTRA-FAST NOISE CLASSIFICATION RESULTS\n")
                f.write(f"="*50 + "\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n")
                f.write(f"Evaluation Time: {eval_time:.2f} seconds\n")
                f.write(f"Prediction Speed: {len(predictions)/eval_time:.1f} samples/second\n\n")
                f.write("Confusion Matrix:\n")
                f.write("True\\Pred    Real  Synth  Semi\n")
                for i, (name, row) in enumerate(zip(class_names, cm)):
                    f.write(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}\n")
                f.write("\nPer-class Metrics:\n")
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
                        f.write(f"  F1-score: {metrics['f1-score']:.4f}\n")
                        f.write(f"  Support: {metrics['support']}\n")
                        
            # Create enhanced visualizations
            print("📊 Creating enhanced visualizations...")
            
            # Confusion Matrix Heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Synthetic', 'Semi-synthetic'],
                       yticklabels=['Real', 'Synthetic', 'Semi-synthetic'])
            plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, MCC: {mcc:.4f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = os.path.join(results_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Performance metrics visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Overall metrics
            metrics = ['Accuracy', 'MCC']
            values = [accuracy, mcc]
            colors = ['#36A2EB', '#FF6384']
            bars = axes[0].bar(metrics, values, color=colors, alpha=0.8)
            axes[0].set_ylabel('Score')
            axes[0].set_title('Overall Performance')
            axes[0].set_ylim(0, 1)
            for bar, value in zip(bars, values):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Per-class F1 scores
            f1_scores = []
            for class_name in ['Real', 'Synthetic', 'Semi-synthetic']:
                if class_name in report:
                    f1_scores.append(report[class_name]['f1-score'])
                else:
                    f1_scores.append(0.0)
            
            axes[1].bar(['Real', 'Synthetic', 'Semi-synthetic'], f1_scores, 
                       color=['#FF9F40', '#4BC0C0', '#9966FF'], alpha=0.8)
            axes[1].set_ylabel('F1-Score')
            axes[1].set_title('Per-Class F1 Scores')
            axes[1].set_ylim(0, 1)
            for i, score in enumerate(f1_scores):
                axes[1].text(i, score + 0.01, f'{score:.3f}', 
                            ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            metrics_path = os.path.join(results_dir, 'performance_metrics.png')
            plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Results saved to {results_dir}/")
        
        return results

def main():
    """Main script to run the optimized noise classification pipeline"""
    print("🚀 OPTIMIZED NOISE CLASSIFICATION PIPELINE")
    print("="*70)
    
    # Configuration
    data_dir = './datasets/train'
    batch_size = 128  # Increased for better GPU utilization
    num_epochs = 30
    checkpoint_dir = './noise_checkpoints'
    results_dir = './results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # System information
    print(f"🔧 System Configuration:")
    print(f"   Device: {device}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CPU cores: {mp.cpu_count()}")
    
    if torch.cuda.is_available():
        print(f"🔥 CUDA available with {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print("⚠️ CUDA not available, using CPU")

    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Load and split dataset
    print(f"\n📊 Loading dataset from {data_dir}...")
    load_start = time.time()
    try:
        images, labels = load_pt_data(data_dir)
        load_time = time.time() - load_start
        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    train_dataset, val_dataset, test_dataset = create_train_val_test_split(
        images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # Optimized data loaders
    num_workers = min(8, mp.cpu_count() // 2)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    
    print(f"⚙️ DataLoader settings: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    # Initialize optimized pipeline
    pipeline = NoiseClassificationPipeline(device=device, checkpoint_dir=checkpoint_dir)

    # Check for existing checkpoint
    print("\n🔍 Checking for existing checkpoints...")
    loaded, resume_epoch = pipeline.load_checkpoint('latest')
    if loaded:
        print(f"✅ Loaded existing checkpoint. Resume from epoch: {resume_epoch}")
    else:
        print("ℹ️ No checkpoint found. Starting fresh training.")
        resume_epoch = 0

    # Train autoencoder if not already trained
    if not pipeline.autoencoder_trained:
        print(f"\n🔥 Training autoencoder with GPU acceleration...")
        autoencoder_start = time.time()
        pipeline.train_autoencoder(
            train_loader, val_loader, 
            epochs=num_epochs,
            resume_from_epoch=resume_epoch, 
            save_every=5  # Save less frequently for speed
        )
        autoencoder_time = time.time() - autoencoder_start
        print(f"✅ Autoencoder training completed in {autoencoder_time:.2f} seconds")
        pipeline.save_checkpoint('autoencoder_complete')
    else:
        print("✅ Autoencoder already trained.")

    # Visualize noise maps with samples from each class
    print("\n📊 Visualizing noise maps...")
    try:
        # Get samples from each class for visualization
        sample_images = []
        sample_labels = []
        class_counts = [0, 0, 0]
        
        for images_batch, labels_batch in test_loader:
            for img, label in zip(images_batch, labels_batch):
                if class_counts[label.item()] < 3:  # 3 samples per class
                    sample_images.append(img)
                    sample_labels.append(label.item())
                    class_counts[label.item()] += 1
                    
                if sum(class_counts) >= 9:
                    break
            if sum(class_counts) >= 9:
                break
        
        if len(sample_images) > 0:
            sample_images = torch.stack(sample_images)
            sample_labels = torch.tensor(sample_labels)
            os.makedirs(results_dir, exist_ok=True)
            pipeline.visualize_noise_maps(
                sample_images, sample_labels, 
                num_samples=len(sample_images),
                save_path=os.path.join(results_dir, 'noise_maps_by_class.png')
            )
    except Exception as e:
        print(f"⚠️ Error visualizing noise maps: {e}")

    # Train classifier if not already trained
    if not pipeline.classifier_trained:
        print(f"\n🌳 Training ultra-fast noise classifier...")
        classifier_start = time.time()
        pipeline.train_noise_classifier(train_loader, val_loader)
        classifier_time = time.time() - classifier_start
        print(f"✅ Classifier training completed in {classifier_time:.2f} seconds")
        pipeline.save_checkpoint('complete_pipeline')
    else:
        print("✅ Noise classifier already trained.")

    # Evaluate on test set
    print("\n🎯 Evaluating on test set...")
    test_labels = test_dataset.labels.cpu().tolist()
    results = pipeline.evaluate(
        test_loader, test_labels,
        save_results=True, results_dir=results_dir
    )

    # Plot comprehensive training metrics
    print("\n📊 Generating comprehensive training plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation losses
    if pipeline.training_history['autoencoder_losses']:
        epochs_range = range(1, len(pipeline.training_history['autoencoder_losses']) + 1)
        axes[0, 0].plot(epochs_range, pipeline.training_history['autoencoder_losses'], 
                       'b-', label='Training Loss', linewidth=2)
        if pipeline.training_history['val_losses']:
            axes[0, 0].plot(epochs_range, pipeline.training_history['val_losses'], 
                           'r-', label='Validation Loss', linewidth=2)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Autoencoder Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Final performance metrics
    metrics = ['Accuracy', 'MCC']
    values = [results['accuracy'], results['mcc']]
    colors = ['#36A2EB', '#FF6384']
    
    bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Final Test Performance')
    axes[0, 1].set_ylim(0, 1)
    
    for bar, value in zip(bars, values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Class-wise F1 scores
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    f1_scores = []
    
    for key in class_names:
        if key in results['classification_report']:
            f1_scores.append(results['classification_report'][key]['f1-score'])
        else:
            f1_scores.append(0.0)
    
    axes[1, 0].bar(class_names, f1_scores, color=['#FF9F40', '#4BC0C0', '#9966FF'], alpha=0.8)
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('Per-Class F1 Scores')
    axes[1, 0].set_ylim(0, 1)
    for i, score in enumerate(f1_scores):
        axes[1, 0].text(i, score + 0.01, f'{score:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Class distribution and speed metrics
    class_counts = torch.bincount(test_dataset.labels, minlength=3).numpy()
    axes[1, 1].pie(class_counts, labels=class_names, autopct='%1.1f%%', 
                  colors=['#FF9F40', '#4BC0C0', '#9966FF'])
    axes[1, 1].set_title(f'Test Set Distribution\nSpeed: {results["prediction_speed"]:.1f} samples/s')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'comprehensive_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Comprehensive results plot saved to {plot_path}")
    plt.show()

    # Final summary with performance metrics
    total_pipeline_time = time.time() - load_start if 'load_start' in locals() else 0
    
    print("\n" + "="*70)
    print("🎉 OPTIMIZED PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"📊 Dataset: {len(images):,} images processed")
    print(f"🎯 Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"📈 Final Test MCC: {results['mcc']:.4f}")
    print(f"🚀 Prediction Speed: {results['prediction_speed']:.1f} samples/second")
    print(f"⏱️ Total Pipeline Time: {total_pipeline_time:.2f} seconds")
    
    if pipeline.training_history['autoencoder_losses']:
        initial_loss = pipeline.training_history['autoencoder_losses'][0]
        final_loss = pipeline.training_history['autoencoder_losses'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"📉 Autoencoder Loss Improvement: {improvement:.1f}% ({initial_loss:.6f} → {final_loss:.6f})")
    
    print(f"💾 Model files saved in: {checkpoint_dir}")
    print(f"📊 Results saved in: {results_dir}")
    print(f"🔗 Multi-GPU acceleration: {torch.cuda.device_count()} GPUs used" if torch.cuda.is_available() else "🔧 CPU-only processing")
    print("="*70)
    
    return results

# Additional utility function for loading pre-trained models
def load_pretrained_pipeline(checkpoint_dir='./noise_checkpoints') -> NoiseClassificationPipeline:
    """Load a pre-trained pipeline from checkpoints"""
    print("🔄 Loading pre-trained pipeline...")
    
    pipeline = NoiseClassificationPipeline(checkpoint_dir=checkpoint_dir)
    
    # Try to load the best autoencoder
    autoencoder_loaded = False
    for checkpoint_name in ['best_autoencoder', 'autoencoder_final', 'latest']:
        checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_name}.pth')
        if os.path.exists(checkpoint_path):
            print(f"🔄 Loading autoencoder from {checkpoint_name}...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=pipeline.device)
                if checkpoint.get('autoencoder_state_dict') is not None:
                    if pipeline.num_gpus > 1:
                        pipeline.autoencoder = nn.DataParallel(pipeline.autoencoder)
                        pipeline.autoencoder.module.load_state_dict(checkpoint['autoencoder_state_dict'])
                    else:
                        pipeline.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
                    pipeline.autoencoder_trained = True
                    autoencoder_loaded = True
                    print(f"✅ Autoencoder loaded from {checkpoint_name}")
                    break
            except Exception as e:
                print(f"⚠️ Error loading {checkpoint_name}: {e}")
                continue
    
    if not autoencoder_loaded:
        print("❌ No valid autoencoder checkpoint found!")
        return None
    
    # Load classifier and extractor
    classifier_path = os.path.join(checkpoint_dir, 'random_forest_classifier.pkl')
    scaler_path = os.path.join(checkpoint_dir, 'feature_scaler.pkl')
    
    if os.path.exists(classifier_path) and os.path.exists(scaler_path):
        try:
            with open(classifier_path, 'rb') as f:
                pipeline.classifier = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                pipeline.noise_extractor.scaler = pickle.load(f)
            pipeline.noise_extractor.fitted = True
            pipeline.classifier_trained = True
            print("✅ Classifier and scaler loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading classifier/scaler: {e}")
    
    print(f"✅ Pipeline loaded successfully!")
    print(f"   Autoencoder trained: {pipeline.autoencoder_trained}")
    print(f"   Classifier trained: {pipeline.classifier_trained}")
    
    return pipeline

# Fast inference function for new data
def classify_new_images_fast(image_paths: List[str], pipeline: NoiseClassificationPipeline, 
                           batch_size: int = 64) -> Tuple[List[int], List[float]]:
    """Classify new images using the trained pipeline"""
    if pipeline is None or not pipeline.autoencoder_trained or not pipeline.classifier_trained:
        raise ValueError("❌ Pipeline must be fully trained!")
    
    print(f"🔮 Classifying {len(image_paths)} new images...")
    
    # Load images
    images = []
    for img_path in tqdm(image_paths, desc="📂 Loading images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = transforms.ToTensor()(img)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            images.append(img)
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            continue
    
    if not images:
        print("❌ No valid images loaded!")
        return [], []
    
    # Stack into tensor
    images_tensor = torch.cat(images, dim=0)
    
    # Create temporary dataset and dataloader
    dummy_labels = torch.zeros(len(images_tensor), dtype=torch.long)
    temp_dataset = PTFileDataset(images_tensor, dummy_labels)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Get predictions
    predictions, probabilities = pipeline.predict_fast(temp_loader)
    
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    print(f"✅ Classification completed!")
    print(f"📊 Results summary:")
    unique, counts = np.unique(predictions, return_counts=True)
    for pred_class, count in zip(unique, counts):
        print(f"   {class_names[pred_class]}: {count} images ({100*count/len(predictions):.1f}%)")
    
    return predictions, probabilities

if __name__ == "__main__":
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    main()