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
        self.images = images.float()
        self.labels = labels.long()
        self.device = device
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image.to(self.device), label.to(self.device)

class ImprovedDenoiseAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(ImprovedDenoiseAutoencoder, self).__init__()
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(2, 2),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(2, 2),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
        ])
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
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        reconstructed = self.decoder(x)
        return reconstructed

class UltraFastNoiseExtractor:
    def __init__(self, n_jobs=-1, use_gpu_features=True):
        self.scaler = StandardScaler()
        self.fitted = False
        self.n_jobs = n_jobs if n_jobs != -1 else min(mp.cpu_count(), 16)
        self.use_gpu_features = use_gpu_features and torch.cuda.is_available()
        print(f"Noise extractor initialized with {self.n_jobs} CPU cores")
        if self.use_gpu_features:
            print("GPU acceleration enabled for feature extraction")
    
    def extract_noise_features_ultra_fast(self, noise_map: np.ndarray) -> np.ndarray:
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
        features = []
        noise_flat = noise_tensor.flatten()
        features.extend([
            torch.mean(noise_flat).item(),
            torch.std(noise_flat).item(),
            torch.var(noise_flat).item()
        ])
        sorted_noise = torch.sort(noise_flat)[0]
        n = len(sorted_noise)
        indices = [int(n * p) for p in [0.05, 0.25, 0.5, 0.75, 0.95]]
        percentiles = [sorted_noise[min(idx, n-1)].item() for idx in indices]
        features.extend(percentiles)
        features.append(percentiles[-1] - percentiles[0])
        hist = torch.histc(noise_flat, bins=12, min=-1, max=1)
        hist = hist / (torch.sum(hist) + 1e-8)
        features.extend(hist.cpu().tolist())
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
            torch.quantile(grad_flat, 0.9).item()
        ])
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
        # Wavelet features
        coeffs = pywt.wavedec2(noise_tensor.cpu().numpy().squeeze(), 'db1', level=2)
        for coeff in coeffs:
            if isinstance(coeff, np.ndarray):
                features.extend([np.mean(coeff), np.std(coeff)])
            else:
                for c in coeff:
                    features.extend([np.mean(c), np.std(c)])
        target_length = 40
        while len(features) < target_length:
            features.append(0.0)
        return torch.tensor(features[:target_length], dtype=torch.float32)
    
    def _extract_cpu_features_optimized(self, noise_map: np.ndarray) -> np.ndarray:
        features = []
        noise_flat = noise_map.flatten()
        features.extend([
            np.mean(noise_flat),
            np.std(noise_flat),
            np.var(noise_flat)
        ])
        percentiles = np.percentile(noise_flat, [5, 25, 50, 75, 95])
        features.extend(percentiles.tolist())
        features.append(percentiles[4] - percentiles[0])
        hist, _ = np.histogram(noise_flat, bins=12, range=(-1, 1))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist.tolist())
        h, w = noise_map.shape
        if h > 8 and w > 8:
            grad_x = cv2.Sobel(noise_map, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(noise_map, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_flat = grad_mag.flatten()
            features.extend([
                np.mean(grad_flat),
                np.std(grad_flat),
                np.percentile(grad_flat, 90)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
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
        coeffs = pywt.wavedec2(noise_map, 'db1', level=2)
        for coeff in coeffs:
            if isinstance(coeff, np.ndarray):
                features.extend([np.mean(coeff), np.std(coeff)])
            else:
                for c in coeff:
                    features.extend([np.mean(c), np.std(c)])
        target_length = 40
        while len(features) < target_length:
            features.append(0.0)
        return np.array(features[:target_length], dtype=np.float32)
    
    def extract_batch_features_parallel(self, noise_maps_batch: List[np.ndarray]) -> List[np.ndarray]:
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
                    results.append(np.zeros(40, dtype=np.float32))
            return results
    
    def fit_transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        print(f"Ultra-fast feature extraction from {len(noise_maps)} noise maps...")
        batch_size = max(32, len(noise_maps) // (self.n_jobs * 2))
        batches = [noise_maps[i:i + batch_size] for i in range(0, len(noise_maps), batch_size)]
        feature_matrix = []
        for batch in tqdm(batches, desc="Ultra-fast feature extraction"):
            batch_features = self.extract_batch_features_parallel(batch)
            feature_matrix.extend(batch_features)
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.fitted = True
        return feature_matrix
    
    def transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Must call fit_transform first")
        print(f"Transforming {len(noise_maps)} noise maps...")
        batch_size = max(32, len(noise_maps) // (self.n_jobs * 2))
        batches = [noise_maps[i:i + batch_size] for i in range(0, len(noise_maps), batch_size)]
        feature_matrix = []
        for batch in tqdm(batches, desc="Feature transformation"):
            batch_features = self.extract_batch_features_parallel(batch)
            feature_matrix.extend(batch_features)
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.scaler.transform(feature_matrix)

class MultiGPUNoiseClassificationPipeline:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {self.num_gpus}")
        print(f"Primary device: {self.device}")
        if self.num_gpus > 1:
            print("Multi-GPU setup detected!")
            for i in range(self.num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        self.autoencoder = ImprovedDenoiseAutoencoder().to(self.device)
        if self.num_gpus > 1:
            self.autoencoder = nn.DataParallel(self.autoencoder)
        self.noise_extractor = UltraFastNoiseExtractor(n_jobs=-1, use_gpu_features=True)
        self.classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.checkpoint_dir = checkpoint_dir
        self.autoencoder_trained = False
        self.classifier_trained = False
        self.scaler = GradScaler() if torch.cuda.is_available() else None
    
    def train_autoencoder(self, train_loader: DataLoader, val_loader: DataLoader = None,
                         num_epochs: int = 50, learning_rate: float = 1e-3,
                         patience: int = 10, min_delta: float = 1e-4):
        print("\n" + "="*70)
        print("🚀 TRAINING AUTOENCODER")
        print("="*70)
        
        self.autoencoder.train()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_path = os.path.join(self.checkpoint_dir, 'best_autoencoder.pth')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_ssim = 0.0, 0.0
            self.autoencoder.train()
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = images.to(self.device)
                optimizer.zero_grad()
                
                with autocast(enabled=torch.cuda.is_available()):
                    reconstructed = self.autoencoder(images)
                    loss = criterion(reconstructed, images)
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                for i in range(images.size(0)):
                    img_np = images[i].cpu().detach().numpy().transpose(1, 2, 0)
                    recon_np = reconstructed[i].cpu().detach().numpy().transpose(1, 2, 0)
                    train_ssim += ssim(img_np, recon_np, channel_axis=2, data_range=1.0)
            
            train_loss /= len(train_loader.dataset)
            train_ssim /= len(train_loader.dataset)
            
            val_loss, val_ssim = 0.0, 0.0
            if val_loader:
                self.autoencoder.eval()
                with torch.no_grad():
                    for images, _ in val_loader:
                        images = images.to(self.device)
                        with autocast(enabled=torch.cuda.is_available()):
                            reconstructed = self.autoencoder(images)
                            loss = criterion(reconstructed, images)
                        val_loss += loss.item() * images.size(0)
                        for i in range(images.size(0)):
                            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                            recon_np = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
                            val_ssim += ssim(img_np, recon_np, channel_axis=2, data_range=1.0)
                    val_loss /= len(val_loader.dataset)
                    val_ssim /= len(val_loader.dataset)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save({
                        'epoch': epoch,
                        'autoencoder_state_dict': self.autoencoder.module.state_dict() if self.num_gpus > 1 else self.autoencoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_ssim': val_ssim
                    }, best_model_path)
                    print(f"✅ Saved best model at epoch {epoch+1} with val_loss: {val_loss:.6f}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"🛑 Early stopping at epoch {epoch+1}")
                        break
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Train SSIM: {train_ssim:.4f}")
            if val_loader:
                print(f"Val Loss: {val_loss:.6f} | Val SSIM: {val_ssim:.4f} | Time: {epoch_time:.2f}s")
        
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            if self.num_gpus > 1:
                self.autoencoder.module.load_state_dict(checkpoint['autoencoder_state_dict'])
            else:
                self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            print(f"✅ Loaded best autoencoder model with val_loss: {checkpoint['val_loss']:.6f}")
        
        self.autoencoder_trained = True
        print("="*70)
    
    def load_trained_autoencoder(self, checkpoint_name='best_autoencoder'):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth')
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        print(f"🔄 Loading pre-trained autoencoder from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if self.num_gpus > 1:
                self.autoencoder.module.load_state_dict(checkpoint['autoencoder_state_dict'])
            else:
                self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            self.autoencoder_trained = True
            print(f"✅ Autoencoder loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading autoencoder: {e}")
            return False
    
    def load_trained_classifier(self, classifier_path='random_forest_classifier.pkl'):
        classifier_path = os.path.join(self.checkpoint_dir, classifier_path)
        if not os.path.exists(classifier_path):
            print(f"❌ Classifier checkpoint not found: {classifier_path}")
            return False
        print(f"🔄 Loading pre-trained classifier from: {classifier_path}")
        try:
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            self.classifier_trained = True
            print(f"✅ Classifier loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            return False
    
    def load_feature_scaler(self, scaler_path='feature_scaler.pkl'):
        scaler_path = os.path.join(self.checkpoint_dir, scaler_path)
        if not os.path.exists(scaler_path):
            print(f"❌ Scaler not found: {scaler_path}")
            return False
        print(f"🔄 Loading scaler from: {scaler_path}")
        try:
            with open(scaler_path, 'rb') as f:
                self.noise_extractor.scaler = pickle.load(f)
            self.noise_extractor.fitted = True
            print(f"✅ Scaler loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            return False
    
    def extract_noise_maps_multi_gpu(self, images: torch.Tensor, batch_size: int = 128) -> List[np.ndarray]:
        if not self.autoencoder_trained:
            raise ValueError("Autoencoder must be trained or loaded first!")
        self.autoencoder.eval()
        noise_maps = []
        effective_batch_size = batch_size * max(1, self.num_gpus)
        print(f"Extracting noise maps with batch size: {effective_batch_size}")
        with torch.no_grad():
            for i in tqdm(range(0, len(images), effective_batch_size), desc="Extracting noise"):
                batch = images[i:i+effective_batch_size].to(self.device)
                try:
                    with autocast(enabled=torch.cuda.is_available()):
                        reconstructed = self.autoencoder(batch)
                        noise = batch - reconstructed
                    noise_np = noise.cpu().numpy()
                    for j in range(noise_np.shape[0]):
                        noise_map = np.mean(noise_np[j], axis=0).astype(np.float32)
                        noise_maps.append(noise_map)
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️ GPU memory error, reducing batch size...")
                    for k in range(0, len(batch), batch_size // 2):
                        mini_batch = batch[k:k+batch_size//2]
                        with autocast(enabled=torch.cuda.is_available()):
                            reconstructed = self.autoencoder(mini_batch)
                            noise = mini_batch - reconstructed
                        noise_np = noise.cpu().numpy()
                        for j in range(noise_np.shape[0]):
                            noise_map = np.mean(noise_np[j], axis=0).astype(np.float32)
                            noise_maps.append(noise_map)
                if i % (effective_batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        print(f"✅ Extracted {len(noise_maps)} noise maps")
        return noise_maps
    
    def train_classifier_ultra_fast(self, train_loader: DataLoader, val_loader: DataLoader = None):
        if self.classifier_trained:
            print("✅ Classifier already trained")
            return
        if not self.autoencoder_trained:
            raise ValueError("❌ Autoencoder must be trained or loaded first!")
        
        print("\n" + "="*70)
        print("🚀 ULTRA-FAST MULTI-GPU NOISE CLASSIFIER TRAINING")
        print("="*70)
        
        total_start_time = time.time()
        print("\n[1/3] 🔍 Extracting noise maps from training data...")
        extraction_start = time.time()
        
        all_noise_maps = []
        all_labels = []
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing batches")):
            noise_maps = self.extract_noise_maps_multi_gpu(images, batch_size=128)
            all_noise_maps.extend(noise_maps)
            all_labels.extend(labels.cpu().numpy())
            
            # Visualize sample noise maps per class
            if batch_idx == 0:
                for class_idx in range(3):
                    class_noise = [nm for nm, lbl in zip(noise_maps, labels.cpu().numpy()) if lbl == class_idx]
                    if class_noise:
                        plt.figure(figsize=(5, 5))
                        plt.imshow(class_noise[0], cmap='gray')
                        plt.title(f'Noise Map - {["Real", "Synthetic", "Semi-synthetic"][class_idx]}')
                        plt.savefig(os.path.join(self.checkpoint_dir, f'noise_map_class_{class_idx}.png'))
                        plt.close()
        
        extraction_time = time.time() - extraction_start
        print(f"✅ Noise extraction completed in {extraction_time:.2f} seconds")
        print(f"📊 Extracted {len(all_noise_maps)} noise maps")
        
        print(f"\n[2/3] ⚡ Ultra-fast feature extraction...")
        feature_start = time.time()
        feature_matrix = self.noise_extractor.fit_transform(all_noise_maps)
        feature_time = time.time() - feature_start
        print(f"✅ Feature extraction completed in {feature_time:.2f} seconds")
        print(f"📊 Feature matrix shape: {feature_matrix.shape}")
        
        # Save the StandardScaler
        scaler_path = os.path.join(self.checkpoint_dir, 'feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.noise_extractor.scaler, f)
        print(f"✅ Feature scaler saved to: {scaler_path}")
        
        print(f"\n[3/3] 🌳 Training Random Forest classifier with hyperparameter tuning...")
        unique, counts = np.unique(all_labels, return_counts=True)
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        print("📊 Training class distribution:")
        for label, count in zip(unique, counts):
            print(f"   {class_names[label]}: {count:,} samples ({100*count/len(all_labels):.1f}%)")
        
        param_dist = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_search = RandomizedSearchCV(
            self.classifier, param_distributions=param_dist, n_iter=10,
            scoring='accuracy', cv=5, n_jobs=-1, random_state=42
        )
        rf_start = time.time()
        rf_search.fit(feature_matrix, all_labels)
        self.classifier = rf_search.best_estimator_
        print(f"Best parameters: {rf_search.best_params_}")
        rf_time = time.time() - rf_start
        
        print(f"✅ Random Forest training completed in {rf_time:.2f} seconds")
        
        classifier_path = os.path.join(self.checkpoint_dir, 'random_forest_classifier.pkl')
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"✅ Random Forest classifier saved to: {classifier_path}")
        
        train_predictions = self.classifier.predict(feature_matrix)
        train_accuracy = accuracy_score(all_labels, train_predictions)
        train_mcc = matthews_corrcoef(all_labels, train_predictions)
        
        total_time = time.time() - total_start_time
        print(f"\n🎯 TRAINING RESULTS:")
        print(f"   Training Accuracy: {train_accuracy:.4f}")
        print(f"   Training MCC: {train_mcc:.4f}")
        print(f"\n⏱️ TIMING BREAKDOWN:")
        print(f"   Noise Extraction: {extraction_time:.2f}s ({100*extraction_time/total_time:.1f}%)")
        print(f"   Feature Extraction: {feature_time:.2f}s ({100*feature_time/total_time:.1f}%)")
        print(f"   RF Training: {rf_time:.2f}s ({100*rf_time/total_time:.1f}%)")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Speed: {len(all_noise_maps)/total_time:.1f} samples/second")
        
        self.classifier_trained = True
        
        if val_loader:
            print(f"\n🔍 Validating on validation set...")
            val_predictions, _ = self.predict_fast(val_loader)
            val_labels = []
            for _, labels in val_loader:
                val_labels.extend(labels.cpu().numpy())
            val_accuracy = accuracy_score(val_labels[:len(val_predictions)], val_predictions)
            val_mcc = matthews_corrcoef(val_labels[:len(val_predictions)], val_predictions)
            print(f"   Validation Accuracy: {val_accuracy:.4f}")
            print(f"   Validation MCC: {val_mcc:.4f}")
        
        print("\n" + "="*70)
        print("🎉 ULTRA-FAST TRAINING COMPLETED!")
        print("="*70)
    
    def predict_fast(self, test_loader: DataLoader) -> Tuple[List[int], List[float]]:
        if not self.autoencoder_trained or not self.classifier_trained:
            raise ValueError("Both autoencoder and classifier must be ready!")
        if not self.noise_extractor.fitted:
            raise ValueError("Feature scaler must be fitted or loaded!")
        print("🔮 Generating predictions...")
        all_predictions = []
        all_probabilities = []
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Predicting")):
            try:
                noise_maps = self.extract_noise_maps_multi_gpu(images, batch_size=128)
                feature_matrix = self.noise_extractor.transform(noise_maps)
                predictions = self.classifier.predict(feature_matrix)
                probabilities = self.classifier.predict_proba(feature_matrix)
                all_predictions.extend(predictions.tolist())
                all_probabilities.extend(probabilities.tolist())
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                batch_size_actual = len(images)
                all_predictions.extend([0] * batch_size_actual)
                all_probabilities.extend([[1.0, 0.0, 0.0]] * batch_size_actual)
        print(f"✅ Generated predictions for {len(all_predictions)} samples")
        return all_predictions, all_probabilities

def load_pt_data(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                elif images.dim() == 5:
                    images = images.squeeze()
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                else:
                    images = images.float()
                    if images.max() > 1.0:
                        images = images / 255.0
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
        print(f"\nTotal dataset: {combined_images.shape[0]} images")
        print(f"Image shape: {combined_images.shape[1:]}")
        print(f"Label distribution: {torch.bincount(combined_labels)}")
        return combined_images, combined_labels
    else:
        raise ValueError("No data loaded! Check your directory structure and .pt files.")

def create_train_val_test_split(images: torch.Tensor, labels: torch.Tensor, 
                               train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                               random_seed=42) -> Tuple[PTFileDataset, PTFileDataset, PTFileDataset]:
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
    train_dataset = PTFileDataset(images[train_indices], labels[train_indices])
    val_dataset = PTFileDataset(images[val_indices], labels[val_indices])
    test_dataset = PTFileDataset(images[test_indices], labels[test_indices])
    print(f"\nStratified dataset split:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")
    return train_dataset, val_dataset, test_dataset

def main_multi_gpu():
    print("🚀 MULTI-GPU NOISE CLASSIFICATION PIPELINE WITH AUTOENCODER TRAINING")
    print("="*70)
    
    # Configuration
    data_dir = './datasets/train'
    batch_size = 128
    checkpoint_dir = './improved_checkpoints'
    results_dir = './results'
    num_epochs = 50
    learning_rate = 1e-3
    patience = 10
    
    # GPU info
    if torch.cuda.is_available():
        print(f"🔥 CUDA available with {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    print(f"💾 CPU cores available: {mp.cpu_count()}")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Checkpoint directory: {checkpoint_dir}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n📊 Loading dataset...")
    start_time = time.time()
    try:
        images, labels = load_pt_data(data_dir)
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"\n🔄 Creating train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_train_val_test_split(
        images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    num_workers = min(8, mp.cpu_count() // 2)
    pin_memory = torch.cuda.is_available()
    print(f"⚙️ DataLoader settings: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"\n🚀 Initializing Multi-GPU pipeline...")
    pipeline = MultiGPUNoiseClassificationPipeline(checkpoint_dir=checkpoint_dir)
    
    print(f"\n🔄 Checking for pre-trained autoencoder...")
    if not pipeline.load_trained_autoencoder('best_autoencoder'):
        print(f"\n🌟 Training new autoencoder...")
        pipeline.train_autoencoder(train_loader, val_loader, num_epochs=num_epochs,
                                 learning_rate=learning_rate, patience=patience)
    
    print(f"\n🧪 Testing autoencoder with sample batch...")
    try:
        test_batch = next(iter(train_loader))[0][:4]
        test_noise = pipeline.extract_noise_maps_multi_gpu(test_batch, batch_size=4)
        print(f"✅ Autoencoder test successful! Generated {len(test_noise)} noise maps")
        if test_noise:
            noise_stats = [f"Shape: {nm.shape}, Range: [{nm.min():.3f}, {nm.max():.3f}]" for nm in test_noise[:2]]
            for i, stats in enumerate(noise_stats):
                print(f"   Sample {i+1}: {stats}")
            for i, nm in enumerate(test_noise[:2]):
                plt.figure(figsize=(5, 5))
                plt.imshow(nm, cmap='gray')
                plt.title(f'Sample Noise Map {i+1}')
                plt.savefig(os.path.join(results_dir, f'sample_noise_map_{i+1}.png'))
                plt.close()
    except Exception as e:
        print(f"❌ Autoencoder test failed: {e}")
        return
    
    print(f"\n🌳 Training noise classifier...")
    classifier_start = time.time()
    pipeline.train_classifier_ultra_fast(train_loader, val_loader)
    classifier_time = time.time() - classifier_start
    print(f"✅ Classifier training completed in {classifier_time:.2f} seconds")
    
    print(f"\n📊 Evaluating on test set...")
    eval_start = time.time()
    predictions, probabilities = pipeline.predict_fast(test_loader)
    test_labels = test_dataset.labels.cpu().tolist()
    eval_time = time.time() - eval_start
    
    min_len = min(len(test_labels), len(predictions))
    test_labels_aligned = test_labels[:min_len]
    predictions_aligned = predictions[:min_len]
    
    accuracy = accuracy_score(test_labels_aligned, predictions_aligned)
    mcc = matthews_corrcoef(test_labels_aligned, predictions_aligned)
    cm = confusion_matrix(test_labels_aligned, predictions_aligned)
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    
    print(f"\n🎯 FINAL RESULTS:")
    print("="*50)
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    print(f"📈 Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"⏱️ Evaluation time: {eval_time:.2f} seconds")
    print(f"🚀 Prediction speed: {len(predictions)/eval_time:.1f} samples/second")
    
    print(f"\n📊 Confusion Matrix:")
    print("True\\Pred    Real  Synth  Semi")
    for i, (name, row) in enumerate(zip(['Real     ', 'Synthetic', 'Semi-synth'], cm)):
        print(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
    
    report = classification_report(test_labels_aligned, predictions_aligned, 
                                 target_names=class_names, output_dict=True, zero_division=0)
    
    print(f"\n📈 Per-Class Performance:")
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-score:  {metrics['f1-score']:.4f}")
    
    # Confusion matrix analysis
    print("\n📊 Confusion Matrix Analysis:")
    for i, class_name in enumerate(class_names):
        fn = np.sum(cm[i, :]) - cm[i, i]  # False negatives
        fp = np.sum(cm[:, i]) - cm[i, i]  # False positives
        print(f"{class_name}: False Negatives = {fn}, False Positives = {fp}")
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions_aligned,
        'probabilities': probabilities[:min_len],
        'true_labels': test_labels_aligned,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'timing': {
            'data_loading': load_time,
            'classifier_training': classifier_time,
            'evaluation': eval_time,
            'total_samples': len(test_labels_aligned),
            'prediction_speed': len(predictions)/eval_time
        },
        'system_info': {
            'num_gpus': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            'cpu_cores': mp.cpu_count(),
            'batch_size': batch_size,
            'num_workers': num_workers
        }
    }
    
    results_file = os.path.join(results_dir, 'multi_gpu_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Creating result visualizations...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, MCC: {mcc:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(results_dir, 'multi_gpu_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
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
    
    f1_scores = [report[class_name]['f1-score'] for class_name in class_names if class_name in report]
    axes[1].bar(class_names, f1_scores, color=['#FF9F40', '#4BC0C0', '#9966FF'], alpha=0.8)
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('Per-Class F1 Scores')
    axes[1].set_ylim(0, 1)
    for i, score in enumerate(f1_scores):
        axes[1].text(i, score + 0.01, f'{score:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    metrics_path = os.path.join(results_dir, 'multi_gpu_performance.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    total_pipeline_time = time.time() - start_time
    print("\n" + "="*70)
    print("🎉 MULTI-GPU PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"📊 Dataset: {len(images):,} images processed")
    print(f"🎯 Final Test Accuracy: {accuracy:.4f}")
    print(f"📈 Final Test MCC: {mcc:.4f}")
    print(f"⏱️ Total Pipeline Time: {total_pipeline_time:.2f} seconds")
    print(f"🚀 Overall Speed: {len(images)/total_pipeline_time:.1f} images/second")
    print(f"💾 Results saved to: {results_dir}/")
    print(f"🔗 Multi-GPU acceleration: {torch.cuda.device_count()} GPUs used")
    print("="*70)
    
    return results

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    results = main_multi_gpu()