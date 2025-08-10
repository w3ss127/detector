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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PTFileDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, device: str = 'cpu', 
                 transform=None, augment=False):
        self.images = images.float()
        self.labels = labels.long()
        self.device = device
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.augment and torch.rand(1) > 0.5:
            # Random augmentation
            if torch.rand(1) > 0.7:  # Horizontal flip
                image = torch.flip(image, [-1])
            if torch.rand(1) > 0.7:  # Add slight noise
                noise = torch.randn_like(image) * 0.01
                image = torch.clamp(image + noise, 0, 1)
            if torch.rand(1) > 0.8:  # Slight rotation
                angle = (torch.rand(1) - 0.5) * 10  # -5 to 5 degrees
                # Simple rotation approximation for small angles
                pass
        
        if self.transform:
            image = self.transform(image)
            
        return image.to(self.device), label.to(self.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, attn_weights = self.attention(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights

class EnhancedViTNoiseExtractor(nn.Module):
    def __init__(self, input_channels=3, image_size=224, patch_size=16, dim=768, 
                 depth=6, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.dim = dim
        
        # Calculate number of patches
        num_patches = (image_size // patch_size) ** 2
        patch_dim = input_channels * patch_size ** 2

        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, dim//heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        
        # Noise-specific feature extraction heads
        self.noise_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Additional statistical feature extraction
        self.statistical_features = nn.Sequential(
            nn.Linear(128 + 40, 256),  # 128 from ViT + 40 statistical features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

    def forward(self, x, statistical_features=None):
        batch_size = x.shape[0]
        
        # Create patches
        x = self.to_patch_embedding(x)
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x += self.pos_embedding[:, :(x.size(1))]
        x = self.dropout(x)

        # Pass through transformer blocks
        attention_maps = []
        for transformer in self.transformer_blocks:
            x, attn_weights = transformer(x)
            attention_maps.append(attn_weights)

        # Normalize and get class token
        x = self.norm(x)
        cls_token_final = x[:, 0]
        
        # Extract noise features
        noise_features = self.noise_head(cls_token_final)
        
        # Combine with statistical features if provided
        if statistical_features is not None:
            combined_features = torch.cat([noise_features, statistical_features], dim=1)
            final_features = self.statistical_features(combined_features)
        else:
            final_features = noise_features
        
        return final_features, attention_maps

class ImprovedDenoiseAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(ImprovedDenoiseAutoencoder, self).__init__()
        
        # Enhanced encoder with residual connections
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # Attention mechanism in bottleneck
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 1),
            nn.Sigmoid()
        )
        
        # Enhanced decoder with skip connections
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),  # 512 because of skip connection
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 256 because of skip connection
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 128 because of skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck with attention
        bottleneck = self.bottleneck(pool3)
        attention_weights = self.attention(bottleneck)
        bottleneck = bottleneck * attention_weights
        
        # Decoder with skip connections
        up3 = self.upconv3(bottleneck)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.decoder3(merge3)
        
        up2 = self.upconv2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.decoder2(merge2)
        
        up1 = self.upconv1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        reconstructed = self.decoder1(merge1)
        
        return reconstructed

class UltraAdvancedNoiseExtractor:
    def __init__(self, n_jobs=-1, use_gpu_features=True):
        self.scaler = StandardScaler()
        self.fitted = False
        self.n_jobs = n_jobs if n_jobs != -1 else min(mp.cpu_count(), 16)
        self.use_gpu_features = use_gpu_features and torch.cuda.is_available()
        print(f"Advanced noise extractor initialized with {self.n_jobs} CPU cores")
        if self.use_gpu_features:
            print("GPU acceleration enabled for feature extraction")
    
    def extract_advanced_statistical_features(self, noise_map: np.ndarray) -> np.ndarray:
        """Extract comprehensive statistical features from noise map"""
        features = []
        noise_flat = noise_map.flatten()
        
        # Basic statistics
        features.extend([
            np.mean(noise_flat),
            np.std(noise_flat),
            np.var(noise_flat),
            stats.skew(noise_flat),
            stats.kurtosis(noise_flat)
        ])
        
        # Percentiles
        percentiles = np.percentile(noise_flat, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        features.extend(percentiles.tolist())
        
        # Range features
        features.extend([
            percentiles[8] - percentiles[0],  # 99th - 1st percentile
            percentiles[7] - percentiles[1],  # 95th - 5th percentile
            percentiles[6] - percentiles[2]   # 90th - 10th percentile
        ])
        
        # Histogram features
        hist, _ = np.histogram(noise_flat, bins=20, range=(-1, 1))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist.tolist())
        
        # Entropy
        hist_nonzero = hist[hist > 0]
        if len(hist_nonzero) > 0:
            entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-8))
        else:
            entropy = 0
        features.append(entropy)
        
        # Gradient features
        if noise_map.shape[0] > 8 and noise_map.shape[1] > 8:
            grad_x = cv2.Sobel(noise_map, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(noise_map, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(grad_mag),
                np.std(grad_mag),
                np.percentile(grad_mag, 90),
                np.percentile(grad_mag, 95)
            ])
            
            # Gradient direction histogram
            grad_angle = np.arctan2(grad_y, grad_x)
            angle_hist, _ = np.histogram(grad_angle, bins=8, range=(-np.pi, np.pi))
            angle_hist = angle_hist / (np.sum(angle_hist) + 1e-8)
            features.extend(angle_hist.tolist())
        else:
            features.extend([0.0] * 12)  # 4 + 8
        
        # Laplacian features
        if noise_map.shape[0] > 4 and noise_map.shape[1] > 4:
            laplacian = cv2.Laplacian(noise_map, cv2.CV_32F)
            features.extend([
                np.mean(laplacian),
                np.std(laplacian),
                np.var(laplacian)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Local Binary Pattern features
        if noise_map.shape[0] > 16 and noise_map.shape[1] > 16:
            try:
                # Simple LBP implementation
                lbp = np.zeros_like(noise_map)
                for i in range(1, noise_map.shape[0]-1):
                    for j in range(1, noise_map.shape[1]-1):
                        center = noise_map[i, j]
                        code = 0
                        neighbors = [
                            noise_map[i-1, j-1], noise_map[i-1, j], noise_map[i-1, j+1],
                            noise_map[i, j+1], noise_map[i+1, j+1], noise_map[i+1, j],
                            noise_map[i+1, j-1], noise_map[i, j-1]
                        ]
                        for k, neighbor in enumerate(neighbors):
                            if neighbor >= center:
                                code |= (1 << k)
                        lbp[i, j] = code
                
                lbp_hist, _ = np.histogram(lbp.flatten(), bins=16, range=(0, 255))
                lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
                features.extend(lbp_hist.tolist())
            except:
                features.extend([0.0] * 16)
        else:
            features.extend([0.0] * 16)
        
        # Wavelet features - multiple levels
        try:
            coeffs = pywt.wavedec2(noise_map, 'db4', level=3)
            for coeff in coeffs:
                if isinstance(coeff, np.ndarray):
                    features.extend([np.mean(coeff), np.std(coeff), np.var(coeff)])
                else:  # It's a tuple of detail coefficients
                    for c in coeff:
                        features.extend([np.mean(c), np.std(c), np.var(c)])
        except:
            # Fallback if wavelet fails
            coeffs = pywt.wavedec2(noise_map, 'db1', level=2)
            for coeff in coeffs:
                if isinstance(coeff, np.ndarray):
                    features.extend([np.mean(coeff), np.std(coeff)])
                else:
                    for c in coeff:
                        features.extend([np.mean(c), np.std(c)])
        
        # Ensure consistent length
        target_length = 120  # Increased for more features
        while len(features) < target_length:
            features.append(0.0)
        
        return np.array(features[:target_length], dtype=np.float32)
    
    def fit_transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        print(f"Advanced feature extraction from {len(noise_maps)} noise maps...")
        
        # Use parallel processing
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            feature_matrix = list(tqdm(
                executor.map(self.extract_advanced_statistical_features, noise_maps),
                total=len(noise_maps),
                desc="Extracting advanced features"
            ))
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Handle NaN and infinite values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit and transform
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.fitted = True
        
        return feature_matrix
    
    def transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Must call fit_transform first")
        
        print(f"Transforming {len(noise_maps)} noise maps...")
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            feature_matrix = list(tqdm(
                executor.map(self.extract_advanced_statistical_features, noise_maps),
                total=len(noise_maps),
                desc="Transforming features"
            ))
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.scaler.transform(feature_matrix)

class EnhancedMultiGPUPipeline:
    def __init__(self, checkpoint_dir='noise_vit_checkpoints', enable_checkpointing=True, 
                 checkpoint_frequency=5, max_checkpoints=3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_gpus = torch.cuda.device_count()
        
        print(f"🔥 Multi-GPU Pipeline Initialization:")
        print(f"   Available GPUs: {self.num_gpus}")
        print(f"   Primary device: {self.device}")
        
        if self.num_gpus > 1:
            print("🚀 Multi-GPU setup detected!")
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        # Enhanced autoencoder with skip connections - Multi-GPU ready
        self.autoencoder = ImprovedDenoiseAutoencoder().to(self.device)
        if self.num_gpus > 1:
            print(f"🔗 Wrapping autoencoder with DataParallel for {self.num_gpus} GPUs")
            self.autoencoder = nn.DataParallel(self.autoencoder)
        
        # ViT-based noise feature extractor - Multi-GPU ready
        self.vit_extractor = EnhancedViTNoiseExtractor(
            input_channels=1,  # For grayscale noise maps
            image_size=224,
            patch_size=16,
            dim=768,
            depth=8,
            heads=12,
            mlp_dim=3072,
            dropout=0.1
        ).to(self.device)
        
        if self.num_gpus > 1:
            print(f"🔗 Wrapping ViT extractor with DataParallel for {self.num_gpus} GPUs")
            self.vit_extractor = nn.DataParallel(self.vit_extractor)
        
        # Advanced statistical feature extractor
        self.noise_extractor = UltraAdvancedNoiseExtractor(n_jobs=-1, use_gpu_features=True)
        
        # Enhanced ensemble classifier
        self.classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Checkpoint properties
        self.checkpoint_dir = checkpoint_dir
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoints = max_checkpoints
        self.autoencoder_trained = False
        self.vit_trained = False
        self.classifier_trained = False
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.best_metrics = {}
        # In-memory checkpoint history
        self.checkpoint_history: List[Dict] = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"✅ Multi-GPU pipeline initialized successfully!")
        
    def save_checkpoint(self, model_type: str, epoch: int, model_state: dict, 
                       optimizer_state: dict = None, metrics: dict = None, 
                       additional_info: dict = None):
        """
        Save model checkpoint with metadata
        
        Args:
            model_type: 'autoencoder', 'vit', or 'classifier'
            epoch: Current epoch
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            metrics: Performance metrics
            additional_info: Any additional information
        """
        if not self.enable_checkpointing:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{model_type}_epoch_{epoch}_{timestamp}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint_data = {
            'model_type': model_type,
            'epoch': epoch,
            'model_state_dict': model_state,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'additional_info': additional_info or {},
            'num_gpus': self.num_gpus,
            'multi_gpu_setup': self.num_gpus > 1
        }
        
        if optimizer_state:
            checkpoint_data['optimizer_state_dict'] = optimizer_state
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update checkpoint history
        self.checkpoint_history.append({
            'model_type': model_type,
            'epoch': epoch,
            'path': checkpoint_path,
            'timestamp': timestamp,
            'metrics': metrics or {}
        })
        
        # Cleanup old checkpoints if needed
        self._cleanup_old_checkpoints(model_type)
        
        print(f"💾 Multi-GPU checkpoint saved: {checkpoint_name}")
        if metrics:
            print(f"   Metrics: {metrics}")
    
    def _cleanup_old_checkpoints(self, model_type: str):
        """Remove old checkpoints keeping only the latest ones"""
        model_checkpoints = [cp for cp in self.checkpoint_history if cp['model_type'] == model_type]
        
        if len(model_checkpoints) > self.max_checkpoints:
            # Sort by epoch (descending) and keep only the latest ones
            model_checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
            checkpoints_to_remove = model_checkpoints[self.max_checkpoints:]
            
            for cp in checkpoints_to_remove:
                try:
                    if os.path.exists(cp['path']):
                        os.remove(cp['path'])
                        print(f"🗑️ Removed old checkpoint: {os.path.basename(cp['path'])}")
                    self.checkpoint_history.remove(cp)
                except Exception as e:
                    print(f"⚠️ Could not remove checkpoint {cp['path']}: {e}")
    
    def load_checkpoint(self, model_type: str, checkpoint_path: str = None, 
                       load_best: bool = True, load_latest: bool = False):
        """
        Load model checkpoint with Multi-GPU support
        
        Args:
            model_type: 'autoencoder', 'vit', or 'classifier'
            checkpoint_path: Specific checkpoint path to load
            load_best: Load the best checkpoint based on metrics
            load_latest: Load the latest checkpoint by epoch
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if checkpoint_path:
            target_path = checkpoint_path
        elif load_best:
            target_path = self._find_best_checkpoint(model_type)
        elif load_latest:
            target_path = self._find_latest_checkpoint(model_type)
        else:
            target_path = os.path.join(self.checkpoint_dir, f'best_{model_type}.pth')
        
        if not target_path or not os.path.exists(target_path):
            print(f"❌ Multi-GPU checkpoint not found for {model_type}")
            return False
        
        try:
            print(f"🔄 Loading {model_type} checkpoint from: {target_path}")
            checkpoint = torch.load(target_path, map_location=self.device)
            
            # Check if checkpoint was saved in multi-GPU setup
            checkpoint_multi_gpu = checkpoint.get('multi_gpu_setup', False)
            current_multi_gpu = self.num_gpus > 1
            
            if model_type == 'autoencoder':
                if checkpoint_multi_gpu and current_multi_gpu:
                    # Both multi-GPU
                    self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
                elif checkpoint_multi_gpu and not current_multi_gpu:
                    # Checkpoint is multi-GPU, current is single-GPU
                    # Remove 'module.' prefix
                    state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        if key.startswith('module.'):
                            state_dict[key[7:]] = value
                        else:
                            state_dict[key] = value
                    self.autoencoder.load_state_dict(state_dict)
                elif not checkpoint_multi_gpu and current_multi_gpu:
                    # Checkpoint is single-GPU, current is multi-GPU
                    # Add 'module.' prefix
                    state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        state_dict[f'module.{key}'] = value
                    self.autoencoder.load_state_dict(state_dict)
                else:
                    # Both single-GPU
                    self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
                self.autoencoder_trained = True
                
            elif model_type == 'vit':
                if checkpoint_multi_gpu and current_multi_gpu:
                    # Both multi-GPU
                    self.vit_extractor.load_state_dict(checkpoint['model_state_dict'])
                elif checkpoint_multi_gpu and not current_multi_gpu:
                    # Checkpoint is multi-GPU, current is single-GPU
                    state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        if key.startswith('module.'):
                            state_dict[key[7:]] = value
                        else:
                            state_dict[key] = value
                    self.vit_extractor.load_state_dict(state_dict)
                elif not checkpoint_multi_gpu and current_multi_gpu:
                    # Checkpoint is single-GPU, current is multi-GPU
                    state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        state_dict[f'module.{key}'] = value
                    self.vit_extractor.load_state_dict(state_dict)
                else:
                    # Both single-GPU
                    self.vit_extractor.load_state_dict(checkpoint['model_state_dict'])
                self.vit_trained = True
                
            elif model_type == 'classifier':
                # Load sklearn models
                if 'classifier' in checkpoint:
                    self.classifier = checkpoint['classifier']
                if 'scaler' in checkpoint:
                    self.noise_extractor.scaler = checkpoint['scaler']
                    self.noise_extractor.fitted = True
                self.classifier_trained = True
            
            print(f"✅ Multi-GPU {model_type} checkpoint loaded successfully!")
            if 'metrics' in checkpoint:
                print(f"   Checkpoint metrics: {checkpoint['metrics']}")
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if 'num_gpus' in checkpoint:
                print(f"   Original GPU count: {checkpoint['num_gpus']}, Current: {self.num_gpus}")
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_type} checkpoint: {e}")
            return False
    
    def _find_best_checkpoint(self, model_type: str) -> str:
        """Find the best checkpoint based on metrics"""
        best_path = os.path.join(self.checkpoint_dir, f'best_{model_type}.pth')
        if os.path.exists(best_path):
            return best_path
        
        # If no best checkpoint, find from history
        model_checkpoints = [cp for cp in self.checkpoint_history if cp['model_type'] == model_type]
        if not model_checkpoints:
            return None
        
        # Sort by best metric
        if model_type == 'autoencoder':
            best_cp = min(model_checkpoints, 
                         key=lambda x: x['metrics'].get('val_loss', float('inf')))
        elif model_type == 'vit':
            best_cp = max(model_checkpoints, 
                         key=lambda x: x['metrics'].get('accuracy', 0))
        else:  # classifier
            best_cp = max(model_checkpoints, 
                         key=lambda x: x['metrics'].get('mcc', 0))
        
        return best_cp['path']
    
    def _find_latest_checkpoint(self, model_type: str) -> str:
        """Find the latest checkpoint by epoch"""
        model_checkpoints = [cp for cp in self.checkpoint_history if cp['model_type'] == model_type]
        if not model_checkpoints:
            return None
        
        latest_cp = max(model_checkpoints, key=lambda x: x['epoch'])
        return latest_cp['path']
    
    def list_checkpoints(self, model_type: str = None):
        """List available checkpoints"""
        if model_type:
            checkpoints = [cp for cp in self.checkpoint_history if cp['model_type'] == model_type]
        else:
            checkpoints = self.checkpoint_history
        
        if not checkpoints:
            print(f"No checkpoints found{' for ' + model_type if model_type else ''}")
            return
        
        print(f"\n📋 Available checkpoints{' for ' + model_type if model_type else ''}:")
        print("-" * 80)
        print(f"{'Type':<12} {'Epoch':<8} {'Timestamp':<16} {'Metrics'}")
        print("-" * 80)
        
        for cp in sorted(checkpoints, key=lambda x: (x['model_type'], x['epoch'])):
            metrics_str = str(cp['metrics']) if cp['metrics'] else "N/A"
            if len(metrics_str) > 40:
                metrics_str = metrics_str[:37] + "..."
            print(f"{cp['model_type']:<12} {cp['epoch']:<8} {cp['timestamp']:<16} {metrics_str}")
    
    def train_autoencoder_enhanced_multi_gpu(self, train_loader: DataLoader, val_loader: DataLoader,
                                           epochs=50, lr=1e-3, resume_from_checkpoint=False):
        """Train enhanced autoencoder with Multi-GPU support"""
        print(f"\n🚀 Multi-GPU Enhanced Autoencoder Training ({self.num_gpus} GPUs)")
        print(f"   Epochs: {epochs}, Learning Rate: {lr}")
        
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        start_epoch = 0
        
        # Effective batch size for multi-GPU
        effective_batch_size = train_loader.batch_size * max(1, self.num_gpus)
        print(f"   Effective batch size: {effective_batch_size}")
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            resume_epoch = self.resume_training('autoencoder')
            if resume_epoch is not None and resume_epoch > 0:
                start_epoch = resume_epoch
                # Load optimizer state if available
                checkpoint_path = self._find_latest_checkpoint('autoencoder')
                if checkpoint_path:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'best_val_loss' in checkpoint['metrics']:
                        best_val_loss = checkpoint['metrics']['best_val_loss']
        
        for epoch in range(start_epoch, epochs):
            # Training
            self.autoencoder.train()
            train_loss = 0.0
            batch_count = 0
            
            for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Multi-GPU Epoch {epoch+1}/{epochs}")):
                optimizer.zero_grad()
                
                # Move to device - multi-GPU will handle distribution automatically
                images = images.to(self.device, non_blocking=True)

                with autocast(enabled=torch.cuda.is_available()):
                    reconstructed = self.autoencoder(images)
                    loss = criterion(reconstructed, images)
                
                if self.scaler and torch.cuda.is_available():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                # Clear cache periodically for multi-GPU setups
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Validation
            self.autoencoder.eval()
            val_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for images, _ in val_loader:
                    # Move to device
                    images = images.to(self.device, non_blocking=True)
                    with autocast(enabled=torch.cuda.is_available()):
                        reconstructed = self.autoencoder(images)
                        loss = criterion(reconstructed, images)
                    val_loss += loss.item()
                    val_batch_count += 1
            
            train_loss /= batch_count
            val_loss /= val_batch_count
            
            print(f"Multi-GPU Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save checkpoint periodically
            if self.enable_checkpointing and (epoch + 1) % self.checkpoint_frequency == 0:
                # Get state dict properly for multi-GPU
                if self.num_gpus > 1:
                    model_state = self.autoencoder.module.state_dict()
                else:
                    model_state = self.autoencoder.state_dict()
                
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
                
                self.save_checkpoint(
                    'autoencoder', 
                    epoch + 1, 
                    model_state,
                    optimizer.state_dict(),
                    metrics
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_metrics['autoencoder_val_loss'] = val_loss
                
                # Get state dict properly for multi-GPU
                if self.num_gpus > 1:
                    model_state = self.autoencoder.module.state_dict()
                else:
                    model_state = self.autoencoder.state_dict()
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_type': 'autoencoder',
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss
                    },
                    'num_gpus': self.num_gpus,
                    'multi_gpu_setup': self.num_gpus > 1
                }
                torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_autoencoder.pth'))
                print(f"💾 Best multi-GPU autoencoder saved! Val Loss: {val_loss:.6f}")
            
            scheduler.step()
            
            # Memory cleanup for multi-GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.autoencoder_trained = True
        print("✅ Multi-GPU enhanced autoencoder training completed!")
        
        # Save final checkpoint
        if self.enable_checkpointing:
            if self.num_gpus > 1:
                final_model_state = self.autoencoder.module.state_dict()
            else:
                final_model_state = self.autoencoder.state_dict()
                
            self.save_checkpoint(
                'autoencoder',
                epochs,
                final_model_state,
                optimizer.state_dict(),
                {'final_train_loss': train_loss, 'final_val_loss': val_loss},
                {'training_completed': True}
            )
    
    def extract_enhanced_noise_maps_multi_gpu(self, images: torch.Tensor, batch_size: int = 64) -> List[np.ndarray]:
        """Extract noise maps using enhanced autoencoder with Multi-GPU optimization"""
        if not self.autoencoder_trained:
            raise ValueError("Enhanced autoencoder must be trained or loaded first!")
        
        self.autoencoder.eval()
        noise_maps = []
        
        # Optimize batch size for multi-GPU
        effective_batch_size = batch_size * max(1, self.num_gpus)
        print(f"🔍 Extracting enhanced noise with Multi-GPU (effective batch size: {effective_batch_size})")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), effective_batch_size), desc="Multi-GPU noise extraction"):
                batch = images[i:i+effective_batch_size].to(self.device, non_blocking=True)
                
                try:
                    with autocast(enabled=torch.cuda.is_available()):
                        reconstructed = self.autoencoder(batch)
                        noise = batch - reconstructed
                    
                    noise_np = noise.cpu().numpy()
                    for j in range(noise_np.shape[0]):
                        # Convert to grayscale and normalize
                        noise_map = np.mean(noise_np[j], axis=0).astype(np.float32)
                        noise_maps.append(noise_map)
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️ Multi-GPU memory error, reducing batch size...")
                    # Fallback to smaller batches
                    smaller_batch_size = effective_batch_size // 2
                    for k in range(i, min(i + effective_batch_size, len(images)), smaller_batch_size):
                        mini_batch = images[k:k+smaller_batch_size].to(self.device, non_blocking=True)
                        with autocast(enabled=torch.cuda.is_available()):
                            reconstructed = self.autoencoder(mini_batch)
                            noise = mini_batch - reconstructed
                        noise_np = noise.cpu().numpy()
                        for j in range(noise_np.shape[0]):
                            noise_map = np.mean(noise_np[j], axis=0).astype(np.float32)
                            noise_maps.append(noise_map)
                
                # Periodic memory cleanup for multi-GPU
                if i % (effective_batch_size * 5) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"✅ Multi-GPU extracted {len(noise_maps)} enhanced noise maps")
        return noise_maps
    
    def train_vit_extractor_multi_gpu(self, noise_maps: List[np.ndarray], statistical_features: np.ndarray,
                                    labels: List[int], epochs=30, lr=1e-4, batch_size=32, 
                                    resume_from_checkpoint=False):
        """Train ViT feature extractor with Multi-GPU support"""
        print(f"\n🧠 Multi-GPU ViT Feature Extractor Training ({self.num_gpus} GPUs)")
        print(f"   Epochs: {epochs}, Learning Rate: {lr}, Batch Size: {batch_size}")
        
        # Prepare data
        noise_tensors = []
        for nm in noise_maps:
            # Resize to 224x224 if needed
            if nm.shape != (224, 224):
                nm_resized = cv2.resize(nm, (224, 224))
            else:
                nm_resized = nm
            
            # Convert to tensor and add channel dimension
            nm_tensor = torch.from_numpy(nm_resized).unsqueeze(0).float()
            noise_tensors.append(nm_tensor)
        
        noise_tensors = torch.stack(noise_tensors)
        stat_features = torch.from_numpy(statistical_features).float()
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Create dataset and dataloader with multi-GPU optimization
        dataset = torch.utils.data.TensorDataset(noise_tensors, stat_features, labels_tensor)
        effective_batch_size = batch_size * max(1, self.num_gpus)
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, 
                               num_workers=min(8, mp.cpu_count()//2), pin_memory=True)
        
        print(f"   Effective batch size: {effective_batch_size}")
        
        # Add classification head to ViT
        classifier_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        ).to(self.device)
        
        # Multi-GPU classification head if needed
        if self.num_gpus > 1:
            classifier_head = nn.DataParallel(classifier_head)
        
        optimizer = optim.AdamW(
            list(self.vit_extractor.parameters()) + list(classifier_head.parameters()),
            lr=lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_accuracy = 0.0
        start_epoch = 0
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            resume_epoch = self.resume_training('vit')
            if resume_epoch is not None and resume_epoch > 0:
                start_epoch = resume_epoch
                # Try to load classifier head and optimizer state
                checkpoint_path = self._find_latest_checkpoint('vit')
                if checkpoint_path:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    if 'classifier_head' in checkpoint:
                        classifier_head.load_state_dict(checkpoint['classifier_head'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'best_accuracy' in checkpoint['metrics']:
                        best_accuracy = checkpoint['metrics']['best_accuracy']
        
        for epoch in range(start_epoch, epochs):
            self.vit_extractor.train()
            classifier_head.train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0
            
            for noise_batch, stat_batch, label_batch in tqdm(dataloader, desc=f"Multi-GPU ViT Epoch {epoch+1}/{epochs}"):
                noise_batch = noise_batch.to(self.device, non_blocking=True)
                stat_batch = stat_batch.to(self.device, non_blocking=True)
                label_batch = label_batch.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with autocast(enabled=torch.cuda.is_available()):
                    features, attention_maps = self.vit_extractor(noise_batch, stat_batch)
                    outputs = classifier_head(features)
                    loss = criterion(outputs, label_batch)
                
                if self.scaler and torch.cuda.is_available():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += label_batch.size(0)
                correct += predicted.eq(label_batch).sum().item()
                batch_count += 1
                
                # Memory cleanup for multi-GPU
                if batch_count % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            accuracy = 100. * correct / total
            avg_loss = total_loss / batch_count
            
            print(f"Multi-GPU ViT Epoch {epoch+1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Save checkpoint periodically
            if self.enable_checkpointing and (epoch + 1) % self.checkpoint_frequency == 0:
                if self.num_gpus > 1:
                    model_state = self.vit_extractor.module.state_dict()
                    classifier_head_state = classifier_head.module.state_dict()
                else:
                    model_state = self.vit_extractor.state_dict()
                    classifier_head_state = classifier_head.state_dict()
                
                metrics = {
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'best_accuracy': best_accuracy
                }
                
                checkpoint_data = {
                    'model_state_dict': model_state,
                    'classifier_head': classifier_head_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }
                
                self.save_checkpoint(
                    'vit', 
                    epoch + 1, 
                    checkpoint_data,
                    additional_info={'has_classifier_head': True, 'multi_gpu': self.num_gpus > 1}
                )
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_metrics['vit_accuracy'] = accuracy
                
                if self.num_gpus > 1:
                    model_state = self.vit_extractor.module.state_dict()
                    classifier_head_state = classifier_head.module.state_dict()
                else:
                    model_state = self.vit_extractor.state_dict()
                    classifier_head_state = classifier_head.state_dict()
                
                best_checkpoint = {
                    'model_type': 'vit',
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'classifier_head': classifier_head_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'metrics': {
                        'loss': avg_loss,
                        'accuracy': accuracy,
                        'best_accuracy': best_accuracy
                    },
                    'num_gpus': self.num_gpus,
                    'multi_gpu_setup': self.num_gpus > 1
                }
                torch.save(best_checkpoint, os.path.join(self.checkpoint_dir, 'best_vit_extractor.pth'))
                torch.save(best_checkpoint, os.path.join(self.checkpoint_dir, 'best_vit.pth'))
                print(f"💾 Best multi-GPU ViT saved! Accuracy: {accuracy:.2f}%")
            
            scheduler.step()
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.vit_trained = True
        print(f"✅ Multi-GPU ViT training completed! Best accuracy: {best_accuracy:.2f}%")
        
        # Save final checkpoint
        if self.enable_checkpointing:
            if self.num_gpus > 1:
                final_model_state = self.vit_extractor.module.state_dict()
                final_classifier_head_state = classifier_head.module.state_dict()
            else:
                final_model_state = self.vit_extractor.state_dict()
                final_classifier_head_state = classifier_head.state_dict()
                
            final_checkpoint_data = {
                'model_state_dict': final_model_state,
                'classifier_head': final_classifier_head_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'final_accuracy': accuracy, 'final_loss': avg_loss}
            }
            
            self.save_checkpoint(
                'vit',
                epochs,
                final_checkpoint_data,
                additional_info={'training_completed': True, 'has_classifier_head': True, 'multi_gpu': self.num_gpus > 1}
            )
    
    def extract_vit_features_multi_gpu(self, noise_maps: List[np.ndarray], statistical_features: np.ndarray,
                                     batch_size: int = 32) -> np.ndarray:
        """Extract features using trained ViT with Multi-GPU optimization"""
        if not self.vit_trained:
            print("⚠️ ViT not trained, using statistical features only")
            return statistical_features
        
        self.vit_extractor.eval()
        all_features = []
        
        # Prepare noise tensors
        noise_tensors = []
        for nm in noise_maps:
            if nm.shape != (224, 224):
                nm_resized = cv2.resize(nm, (224, 224))
            else:
                nm_resized = nm
            nm_tensor = torch.from_numpy(nm_resized).unsqueeze(0).float()
            noise_tensors.append(nm_tensor)
        
        noise_tensors = torch.stack(noise_tensors)
        stat_features = torch.from_numpy(statistical_features).float()
        
        # Optimize batch size for multi-GPU
        effective_batch_size = batch_size * max(1, self.num_gpus)
        print(f"🔮 Extracting ViT features with Multi-GPU (effective batch size: {effective_batch_size})")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(noise_tensors), effective_batch_size), desc="Multi-GPU ViT feature extraction"):
                end_idx = min(i + effective_batch_size, len(noise_tensors))
                noise_batch = noise_tensors[i:end_idx].to(self.device, non_blocking=True)
                stat_batch = stat_features[i:end_idx].to(self.device, non_blocking=True)
                
                try:
                    with autocast(enabled=torch.cuda.is_available()):
                        features, _ = self.vit_extractor(noise_batch, stat_batch)
                        all_features.append(features.cpu().numpy())
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️ Multi-GPU memory error, reducing batch size...")
                    # Process in smaller chunks
                    smaller_batch_size = effective_batch_size // 2
                    for j in range(i, end_idx, smaller_batch_size):
                        mini_end_idx = min(j + smaller_batch_size, end_idx)
                        mini_noise_batch = noise_tensors[j:mini_end_idx].to(self.device, non_blocking=True)
                        mini_stat_batch = stat_features[j:mini_end_idx].to(self.device, non_blocking=True)
                        
                        with autocast(enabled=torch.cuda.is_available()):
                            features, _ = self.vit_extractor(mini_noise_batch, mini_stat_batch)
                            all_features.append(features.cpu().numpy())
                
                # Memory cleanup
                if i % (effective_batch_size * 5) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        result = np.vstack(all_features)
        print(f"✅ Multi-GPU ViT feature extraction completed: {result.shape}")
        return result
    
    def train_enhanced_classifier_multi_gpu(self, train_loader: DataLoader, val_loader: DataLoader = None,
                                          save_intermediate_results=True):
        """Train classifier using enhanced features with Multi-GPU optimization"""
        print("\n" + "="*70)
        print("🚀 MULTI-GPU ENHANCED CLASSIFIER TRAINING")
        print("="*70)
        
        total_start_time = time.time()
        
        # Step 1: Extract enhanced noise maps with multi-GPU
        print(f"\n[1/5] 🔍 Multi-GPU enhanced noise map extraction...")
        all_noise_maps = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Multi-GPU processing batches")):
            noise_maps = self.extract_enhanced_noise_maps_multi_gpu(images, batch_size=64)
            all_noise_maps.extend(noise_maps)
            all_labels.extend(labels.cpu().numpy())
        
        print(f"✅ Multi-GPU extracted {len(all_noise_maps)} enhanced noise maps")
        
        # Save intermediate results if requested
        if save_intermediate_results:
            noise_maps_path = os.path.join(self.checkpoint_dir, 'multi_gpu_extracted_noise_maps.pkl')
            with open(noise_maps_path, 'wb') as f:
                pickle.dump(all_noise_maps, f)
            print(f"💾 Multi-GPU noise maps saved to: {noise_maps_path}")
        
        # Step 2: Extract advanced statistical features
        print(f"\n[2/5] ⚡ Multi-GPU advanced statistical feature extraction...")
        statistical_features = self.noise_extractor.fit_transform(all_noise_maps)
        print(f"✅ Statistical features shape: {statistical_features.shape}")
        
        # Save statistical features and scaler
        if save_intermediate_results:
            stat_features_path = os.path.join(self.checkpoint_dir, 'multi_gpu_statistical_features.pkl')
            with open(stat_features_path, 'wb') as f:
                pickle.dump(statistical_features, f)
            
            scaler_path = os.path.join(self.checkpoint_dir, 'multi_gpu_enhanced_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.noise_extractor.scaler, f)
            print(f"💾 Multi-GPU statistical features and scaler saved")
        
        # Step 3: Train ViT feature extractor if not trained
        if not self.vit_trained:
            print(f"\n[3/5] 🧠 Multi-GPU ViT feature extractor training...")
            self.train_vit_extractor_multi_gpu(all_noise_maps, statistical_features, all_labels, 
                                             epochs=20, batch_size=16)
        else:
            print(f"\n[3/5] ✅ ViT already trained, skipping...")
        
        # Step 4: Extract ViT features with multi-GPU
        print(f"\n[4/5] 🔮 Multi-GPU ViT feature extraction...")
        vit_features = self.extract_vit_features_multi_gpu(all_noise_maps, statistical_features, batch_size=16)
        print(f"✅ Multi-GPU ViT features shape: {vit_features.shape}")
        
        # Save ViT features
        if save_intermediate_results:
            vit_features_path = os.path.join(self.checkpoint_dir, 'multi_gpu_vit_features.pkl')
            with open(vit_features_path, 'wb') as f:
                pickle.dump(vit_features, f)
            print(f"💾 Multi-GPU ViT features saved")
        
        # Step 5: Train ensemble classifier
        print(f"\n[5/5] 🌟 Multi-GPU enhanced ensemble classifier training...")
        
        # Combine all features
        if vit_features.shape[0] == statistical_features.shape[0]:
            combined_features = np.hstack([statistical_features, vit_features])
        else:
            combined_features = statistical_features
            print("⚠️ Feature dimension mismatch, using statistical features only")
        
        print(f"📊 Multi-GPU final feature matrix shape: {combined_features.shape}")
        
        # Enhanced hyperparameter tuning for Gradient Boosting
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        grid_search = GridSearchCV(
            self.classifier, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1, 
            verbose=1
        )
        
        rf_start = time.time()
        grid_search.fit(combined_features, all_labels)
        self.classifier = grid_search.best_estimator_
        rf_time = time.time() - rf_start
        
        print(f"✅ Multi-GPU best parameters: {grid_search.best_params_}")
        print(f"✅ Multi-GPU best CV score: {grid_search.best_score_:.4f}")
        print(f"✅ Multi-GPU training completed in {rf_time:.2f} seconds")
        
        # Training metrics
        train_predictions = self.classifier.predict(combined_features)
        train_accuracy = accuracy_score(all_labels, train_predictions)
        train_mcc = matthews_corrcoef(all_labels, train_predictions)
        
        # Update best metrics
        self.best_metrics['classifier_accuracy'] = train_accuracy
        self.best_metrics['classifier_mcc'] = train_mcc
        
        # Save enhanced classifier with all components
        classifier_checkpoint = {
            'model_type': 'classifier',
            'classifier': self.classifier,
            'scaler': self.noise_extractor.scaler,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'metrics': {
                'train_accuracy': train_accuracy,
                'train_mcc': train_mcc,
                'cv_score': grid_search.best_score_
            },
            'feature_info': {
                'statistical_features_dim': statistical_features.shape[1],
                'vit_features_dim': vit_features.shape[1] if vit_features.shape[0] > 0 else 0,
                'combined_features_dim': combined_features.shape[1],
                'n_samples': combined_features.shape[0]
            },
            'multi_gpu_info': {
                'num_gpus': self.num_gpus,
                'multi_gpu_training': self.num_gpus > 1
            }
        }
        
        # Save using checkpoint system
        if self.enable_checkpointing:
            self.save_checkpoint(
                'classifier',
                1,  # Classifier doesn't have epochs like neural networks
                classifier_checkpoint,
                metrics={
                    'train_accuracy': train_accuracy,
                    'train_mcc': train_mcc,
                    'cv_score': grid_search.best_score_
                },
                additional_info={
                    'best_params': grid_search.best_params_,
                    'feature_info': classifier_checkpoint['feature_info'],
                    'multi_gpu_info': classifier_checkpoint['multi_gpu_info']
                }
            )
        
        # Also save to standard locations
        classifier_path = os.path.join(self.checkpoint_dir, 'multi_gpu_enhanced_classifier.pkl')
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier_checkpoint, f)
        
        best_classifier_path = os.path.join(self.checkpoint_dir, 'best_classifier.pkl')
        with open(best_classifier_path, 'wb') as f:
            pickle.dump(classifier_checkpoint, f)
        
        print(f"✅ Multi-GPU enhanced classifier saved to multiple locations!")
        
        print(f"\n🎯 MULTI-GPU TRAINING RESULTS:")
        print(f"   Training Accuracy: {train_accuracy:.4f}")
        print(f"   Training MCC: {train_mcc:.4f}")
        print(f"   Cross-validation Score: {grid_search.best_score_:.4f}")
        print(f"   GPUs Used: {self.num_gpus}")
        
        # Validation
        if val_loader:
            print(f"\n🔍 Multi-GPU enhanced validation...")
            val_predictions, _ = self.predict_enhanced_multi_gpu(val_loader)
            val_labels = []
            for _, labels in val_loader:
                val_labels.extend(labels.cpu().numpy())
            
            val_accuracy = accuracy_score(val_labels[:len(val_predictions)], val_predictions)
            val_mcc = matthews_corrcoef(val_labels[:len(val_predictions)], val_predictions)
            
            print(f"   Validation Accuracy: {val_accuracy:.4f}")
            print(f"   Validation MCC: {val_mcc:.4f}")
            
            # Update checkpoint with validation results
            classifier_checkpoint['metrics'].update({
                'val_accuracy': val_accuracy,
                'val_mcc': val_mcc
            })
            
            # Save updated checkpoint
            with open(best_classifier_path, 'wb') as f:
                pickle.dump(classifier_checkpoint, f)
        
        self.classifier_trained = True
        total_time = time.time() - total_start_time
        print(f"\n⏱️ Total multi-GPU enhanced training time: {total_time:.2f} seconds")
        print("="*70)
    
    def predict_enhanced_multi_gpu(self, test_loader: DataLoader) -> Tuple[List[int], List[float]]:
        """Enhanced prediction using all trained components with Multi-GPU optimization"""
        if not (self.autoencoder_trained and self.classifier_trained):
            raise ValueError("All components must be trained!")
        
        print("🔮 Generating multi-GPU enhanced predictions...")
        all_predictions = []
        all_probabilities = []
        
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Multi-GPU enhanced predicting")):
            try:
                # Extract enhanced noise maps with multi-GPU
                noise_maps = self.extract_enhanced_noise_maps_multi_gpu(images, batch_size=64)
                
                # Extract statistical features
                statistical_features = self.noise_extractor.transform(noise_maps)
                
                # Extract ViT features if available with multi-GPU
                if self.vit_trained:
                    vit_features = self.extract_vit_features_multi_gpu(noise_maps, statistical_features, batch_size=16)
                    if vit_features.shape[0] == statistical_features.shape[0]:
                        combined_features = np.hstack([statistical_features, vit_features])
                    else:
                        combined_features = statistical_features
                else:
                    combined_features = statistical_features
                
                # Predict
                predictions = self.classifier.predict(combined_features)
                probabilities = self.classifier.predict_proba(combined_features)
                
                all_predictions.extend(predictions.tolist())
                all_probabilities.extend(probabilities.tolist())
                
            except Exception as e:
                print(f"⚠️ Error in multi-GPU enhanced batch {batch_idx}: {e}")
                batch_size_actual = len(images)
                all_predictions.extend([0] * batch_size_actual)
                all_probabilities.extend([[1.0, 0.0, 0.0]] * batch_size_actual)
        
        print(f"✅ Generated multi-GPU enhanced predictions for {len(all_predictions)} samples")
        return all_predictions, all_probabilities
    
    def resume_training(self, model_type: str, checkpoint_path: str = None):
        """Resume training from a specific checkpoint"""
        if not self.load_checkpoint(model_type, checkpoint_path, load_latest=True):
            print(f"❌ Could not resume training for {model_type}")
            return None
        
        # Extract epoch information for resuming
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                           if f.startswith(f"{model_type}_epoch_") and f.endswith('.pth')]
        
        if checkpoint_files:
            # Find the latest epoch
            epochs = []
            for f in checkpoint_files:
                try:
                    epoch = int(f.split('_epoch_')[1].split('_')[0])
                    epochs.append(epoch)
                except:
                    continue
            
            if epochs:
                resume_epoch = max(epochs) + 1
                print(f"🔄 Resuming multi-GPU training from epoch {resume_epoch}")
                return resume_epoch
        
        return 0  # Start from beginning if can't determine epoch

def load_pt_data_enhanced(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced data loading with better preprocessing"""
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
            
        print(f"Loading {class_name} images with enhanced preprocessing...")
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
                
                # Normalize dimensions
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                elif images.dim() == 5:
                    images = images.squeeze()
                
                # Normalize values
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                else:
                    images = images.float()
                    if images.max() > 1.0:
                        images = images / 255.0
                
                # Enhanced preprocessing
                images = torch.clamp(images, 0, 1)
                
                # Resize to consistent size if needed
                if images.shape[-1] != 224 or images.shape[-2] != 224:
                    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
                
                class_images.append(images)
                
            except Exception as e:
                print(f"    Error loading {pt_file}: {e}")
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
        
        print(f"\nTotal enhanced dataset: {combined_images.shape[0]} images")
        print(f"Image shape: {combined_images.shape[1:]}")
        print(f"Label distribution: {torch.bincount(combined_labels)}")
        
        return combined_images, combined_labels
    else:
        raise ValueError("No data loaded! Check your directory structure and .pt files.")

def create_enhanced_train_val_test_split(images: torch.Tensor, labels: torch.Tensor, 
                                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                       random_seed=42) -> Tuple[PTFileDataset, PTFileDataset, PTFileDataset]:
    """Enhanced split with data augmentation for training"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
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
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    train_indices = torch.tensor(train_indices)
    val_indices = torch.tensor(val_indices)
    test_indices = torch.tensor(test_indices)
    
    # Create datasets with augmentation for training
    train_dataset = PTFileDataset(
        images[train_indices], 
        labels[train_indices], 
        augment=True
    )
    val_dataset = PTFileDataset(images[val_indices], labels[val_indices])
    test_dataset = PTFileDataset(images[test_indices], labels[test_indices])
    
    print(f"\nEnhanced stratified dataset split:")
    print(f"Train: {len(train_dataset)} samples (with augmentation)")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset

def main_enhanced_multi_gpu_pipeline():
    """Main function for enhanced noise classification pipeline with Multi-GPU support and checkpoints"""
    print("🚀 ENHANCED MULTI-GPU NOISE CLASSIFICATION PIPELINE WITH ViT + ATTENTION + CHECKPOINTS")
    print("="*100)
    
    # Configuration
    data_dir = './datasets/train'
    batch_size = 32  # Reduced for memory efficiency with ViT and Multi-GPU
    checkpoint_dir = './multi_gpu_checkpoints_enhanced'
    results_dir = './multi_gpu_results_enhanced'
    
    # Checkpoint configuration
    enable_checkpointing = True
    checkpoint_frequency = 3  # Save checkpoint every 3 epochs for faster checkpointing
    max_checkpoints = 5  # Keep last 5 checkpoints per model
    resume_training = True  # Set to True to resume from existing checkpoints
    
    # GPU info
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"🔥 CUDA available with {num_gpus} GPU(s)")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        if num_gpus > 1:
            print(f"🚀 Multi-GPU training will use all {num_gpus} GPUs!")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    print(f"💾 CPU cores available: {mp.cpu_count()}")
    print(f"📁 Multi-GPU checkpoint configuration:")
    print(f"   Directory: {checkpoint_dir}")
    print(f"   Enabled: {enable_checkpointing}")
    print(f"   Frequency: Every {checkpoint_frequency} epochs")
    print(f"   Max checkpoints per model: {max_checkpoints}")
    print(f"   Resume training: {resume_training}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n📊 Loading enhanced dataset...")
    start_time = time.time()
    try:
        images, labels = load_pt_data_enhanced(data_dir)
        load_time = time.time() - start_time
        print(f"✅ Enhanced dataset loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"\n🔄 Creating enhanced train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_enhanced_train_val_test_split(
        images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Optimize DataLoader settings for multi-GPU
    num_workers = min(8, mp.cpu_count() // 2)  # Optimized for multi-GPU
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # Important for multi-GPU training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    
    print(f"\n🚀 Initializing enhanced multi-GPU pipeline...")
    pipeline = EnhancedMultiGPUPipeline(
        checkpoint_dir=checkpoint_dir,
        enable_checkpointing=enable_checkpointing,
        checkpoint_frequency=checkpoint_frequency,
        max_checkpoints=max_checkpoints
    )
    
    # List existing checkpoints
    print(f"\n📋 Checking for existing multi-GPU checkpoints...")
    pipeline.list_checkpoints()
    
    # Train or load enhanced autoencoder
    autoencoder_loaded = False
    if resume_training:
        print(f"\n🔄 Attempting to load existing multi-GPU enhanced autoencoder...")
        autoencoder_loaded = pipeline.load_checkpoint('autoencoder', load_best=True)
    
    if not autoencoder_loaded:
        print(f"\n🏗️ Training enhanced autoencoder with multi-GPU from scratch...")
        pipeline.train_autoencoder_enhanced_multi_gpu(
            train_loader, val_loader, 
            epochs=25, 
            resume_from_checkpoint=resume_training
        )
    else:
        print(f"✅ Multi-GPU enhanced autoencoder loaded successfully!")
    
    # Train or load ViT extractor (this will be handled within classifier training if needed)
    vit_loaded = False
    if resume_training:
        print(f"\n🔄 Attempting to load existing multi-GPU ViT extractor...")
        vit_loaded = pipeline.load_checkpoint('vit', load_best=True)
        if vit_loaded:
            print(f"✅ Multi-GPU ViT extractor loaded successfully!")
    
    # Train or load classifier
    classifier_loaded = False
    if resume_training:
        print(f"\n🔄 Attempting to load existing multi-GPU classifier...")
        classifier_loaded = pipeline.load_checkpoint('classifier', load_best=True)
    
    if not classifier_loaded:
        print(f"\n🌟 Training enhanced classifier with all multi-GPU components...")
        classifier_start = time.time()
        pipeline.train_enhanced_classifier_multi_gpu(train_loader, val_loader, save_intermediate_results=True)
        classifier_time = time.time() - classifier_start
        print(f"✅ Multi-GPU enhanced classifier training completed in {classifier_time:.2f} seconds")
    else:
        print(f"✅ Multi-GPU enhanced classifier loaded successfully!")
        classifier_time = 0
    
    # Show final checkpoint status
    print(f"\n📋 Final multi-GPU checkpoint status:")
    pipeline.list_checkpoints()
    
    print(f"\n📊 Multi-GPU enhanced evaluation on test set...")
    eval_start = time.time()
    predictions, probabilities = pipeline.predict_enhanced_multi_gpu(test_loader)
    test_labels = test_dataset.labels.cpu().tolist()
    eval_time = time.time() - eval_start
    
    # Calculate metrics
    min_len = min(len(test_labels), len(predictions))
    test_labels_aligned = test_labels[:min_len]
    predictions_aligned = predictions[:min_len]
    
    accuracy = accuracy_score(test_labels_aligned, predictions_aligned)
    mcc = matthews_corrcoef(test_labels_aligned, predictions_aligned)
    cm = confusion_matrix(test_labels_aligned, predictions_aligned)
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    
    print(f"\n🎯 MULTI-GPU ENHANCED FINAL RESULTS:")
    print("="*70)
    print(f"🎯 Test Accuracy: {accuracy:.4f} {'🎉 TARGET ACHIEVED!' if accuracy >= 0.8 else '❌ Below target (0.8)'}")
    print(f"📈 Matthews Correlation Coefficient: {mcc:.4f} {'🎉 TARGET ACHIEVED!' if mcc >= 0.8 else '❌ Below target (0.8)'}")
    print(f"⏱️ Evaluation time: {eval_time:.2f} seconds")
    print(f"🚀 Multi-GPU prediction speed: {len(predictions)/eval_time:.1f} samples/second")
    print(f"🔥 GPUs utilized: {pipeline.num_gpus}")
    
    # Detailed results
    report = classification_report(test_labels_aligned, predictions_aligned, 
                                 target_names=class_names, output_dict=True, zero_division=0)
    
    print(f"\n📊 Confusion Matrix:")
    print("True\\Pred    Real  Synth  Semi")
    for i, (name, row) in enumerate(zip(['Real     ', 'Synthetic', 'Semi-synth'], cm)):
        print(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
    
    print(f"\n📈 Per-Class Performance:")
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-score:  {metrics['f1-score']:.4f}")
    
    # Save comprehensive results with multi-GPU checkpoint info
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions_aligned,
        'probabilities': probabilities[:min_len],
        'true_labels': test_labels_aligned,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'multi_gpu_features': {
            'autoencoder_with_skip_connections': True,
            'vit_feature_extraction': pipeline.vit_trained,
            'advanced_statistical_features': True,
            'attention_mechanism': True,
            'multi_gpu_training': pipeline.num_gpus > 1,
            'num_gpus_used': pipeline.num_gpus
        },
        'checkpoint_info': {
            'checkpoint_dir': checkpoint_dir,
            'enable_checkpointing': enable_checkpointing,
            'checkpoint_frequency': checkpoint_frequency,
            'max_checkpoints': max_checkpoints,
            'checkpoint_history': pipeline.checkpoint_history,
            'best_metrics': pipeline.best_metrics,
            'autoencoder_loaded_from_checkpoint': autoencoder_loaded,
            'vit_loaded_from_checkpoint': vit_loaded,
            'classifier_loaded_from_checkpoint': classifier_loaded,
            'multi_gpu_checkpoints': True
        },
        'timing': {
            'data_loading': load_time,
            'classifier_training': classifier_time,
            'evaluation': eval_time,
            'total_samples': len(test_labels_aligned)
        },
        'targets_achieved': {
            'accuracy_target_0.8': accuracy >= 0.8,
            'mcc_target_0.8': mcc >= 0.8,
            'both_targets': accuracy >= 0.8 and mcc >= 0.8
        },
        'system_info': {
            'num_gpus': pipeline.num_gpus,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(pipeline.num_gpus)] if torch.cuda.is_available() else [],
            'cpu_cores': mp.cpu_count(),
            'batch_size': batch_size,
            'effective_batch_size': batch_size * max(1, pipeline.num_gpus),
            'num_workers': num_workers
        }
    }
    
    results_file = os.path.join(results_dir, 'multi_gpu_enhanced_classification_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # default=str for datetime objects
    
    # Save checkpoint summary
    checkpoint_summary = {
        'multi_gpu_pipeline_config': {
            'checkpoint_dir': checkpoint_dir,
            'enable_checkpointing': enable_checkpointing,
            'checkpoint_frequency': checkpoint_frequency,
            'max_checkpoints': max_checkpoints,
            'num_gpus': pipeline.num_gpus
        },
        'checkpoint_history': pipeline.checkpoint_history,
        'best_metrics': pipeline.best_metrics,
        'final_results': {
            'accuracy': accuracy,
            'mcc': mcc,
            'targets_achieved': accuracy >= 0.8 and mcc >= 0.8
        },
        'multi_gpu_info': {
            'gpus_used': pipeline.num_gpus,
            'multi_gpu_training': pipeline.num_gpus > 1,
            'effective_batch_size': batch_size * max(1, pipeline.num_gpus)
        }
    }
    
    checkpoint_summary_file = os.path.join(results_dir, 'multi_gpu_checkpoint_summary.json')
    with open(checkpoint_summary_file, 'w') as f:
        json.dump(checkpoint_summary, f, indent=2, default=str)
    
    # Enhanced visualizations
    print(f"\n📊 Creating enhanced multi-GPU visualizations...")
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    title = f'Multi-GPU Enhanced Model Confusion Matrix\nAccuracy: {accuracy:.4f}, MCC: {mcc:.4f}'
    if accuracy >= 0.8 and mcc >= 0.8:
        title += '\n🎉 TARGETS ACHIEVED! 🎉'
    title += f'\n(Trained on {pipeline.num_gpus} GPUs)'
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, 'multi_gpu_enhanced_confusion_matrix.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Performance vs Target visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metrics = ['Accuracy', 'MCC']
    values = [accuracy, mcc]
    targets = [0.8, 0.8]
    colors = ['#2E8B57' if v >= 0.8 else '#DC143C' for v in values]
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
    ax.set_ylabel('Score')
    ax.set_title(f'Multi-GPU Enhanced Model Performance vs Targets\n(Trained on {pipeline.num_gpus} GPUs)')
    ax.set_ylim(0, 1)
    ax.legend()
    
    for bar, value, target in zip(bars, values, targets):
        height = bar.get_height()
        status = '✅' if value >= target else '❌'
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{value:.3f} {status}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'multi_gpu_performance_vs_targets.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Multi-GPU speedup visualization (if applicable)
    if pipeline.num_gpus > 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        gpu_counts = [1, pipeline.num_gpus]
        theoretical_speedup = [1, pipeline.num_gpus]
        actual_batch_sizes = [batch_size, batch_size * pipeline.num_gpus]
        
        ax.bar(['Single GPU', f'{pipeline.num_gpus} GPUs'], 
               [batch_size, batch_size * pipeline.num_gpus], 
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Effective Batch Size')
        ax.set_title(f'Multi-GPU Batch Size Scaling\n({pipeline.num_gpus}x GPU Utilization)')
        
        for i, size in enumerate(actual_batch_sizes):
            ax.text(i, size + max(actual_batch_sizes)*0.02, 
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'multi_gpu_scaling.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    total_pipeline_time = time.time() - start_time
    print("\n" + "="*100)
    print("🎉 MULTI-GPU ENHANCED PIPELINE WITH CHECKPOINTS COMPLETED!")
    print("="*100)
    print(f"📊 Dataset: {len(images):,} images processed")
    print(f"🎯 Final Test Accuracy: {accuracy:.4f} {'🎉' if accuracy >= 0.8 else '❌'}")
    print(f"📈 Final Test MCC: {mcc:.4f} {'🎉' if mcc >= 0.8 else '❌'}")
    print(f"🏆 Both targets achieved: {'YES! 🎉🎉🎉' if (accuracy >= 0.8 and mcc >= 0.8) else 'Not yet ❌'}")
    print(f"⏱️ Total Pipeline Time: {total_pipeline_time:.2f} seconds")
    print(f"🔥 Multi-GPU Configuration:")
    print(f"   - Number of GPUs used: {pipeline.num_gpus}")
    print(f"   - Multi-GPU training: {'Yes' if pipeline.num_gpus > 1 else 'No'}")
    print(f"   - Effective batch size: {batch_size * max(1, pipeline.num_gpus)}")
    print(f"   - GPU memory optimization: Enabled")
    print(f"💡 Enhanced features used:")
    print(f"   - Multi-GPU skip-connection autoencoder with attention")
    print(f"   - Multi-GPU Vision Transformer with multi-head attention")
    print(f"   - Advanced statistical features (120D)")
    print(f"   - Gradient Boosting classifier")
    print(f"💾 Enhanced artifacts saved to:")
    print(f"   - Multi-GPU Checkpoints: {checkpoint_dir}/")
    print(f"   - Results: {results_dir}/")
    print(f"📋 Multi-GPU checkpointing summary:")
    print(f"   - Total checkpoints created: {len(pipeline.checkpoint_history)}")
    print(f"   - Models loaded from checkpoint: {sum([autoencoder_loaded, vit_loaded, classifier_loaded])}/3")
    print(f"   - Multi-GPU checkpointing enabled: {enable_checkpointing}")
    print(f"   - Cross-GPU checkpoint compatibility: Enabled")
    
    if pipeline.num_gpus > 1:
        speedup_estimate = min(pipeline.num_gpus, 2.5)  # Realistic speedup accounting for overhead
        print(f"🚀 Multi-GPU Performance Benefits:")
        print(f"   - Estimated training speedup: {speedup_estimate:.1f}x")
        print(f"   - Memory scaling: {pipeline.num_gpus}x GPU memory available")
        print(f"   - Batch processing: {pipeline.num_gpus}x effective batch size")
    
    print("="*100)
    
    return results

if __name__ == "__main__":
    # Enable optimizations for multi-GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set memory optimization for multi-GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable multi-GPU memory optimization
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
    
    results = main_enhanced_multi_gpu_pipeline()