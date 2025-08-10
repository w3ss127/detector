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
import scipy.stats as stats
from scipy import ndimage
import os
from typing import Tuple, List, Dict
from tqdm import tqdm
import pickle
import json
import seaborn as sns

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

class EnhancedNoiseDistributionExtractor:
    """Enhanced noise feature extractor with more sophisticated analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def extract_noise_features(self, noise_map: np.ndarray) -> np.ndarray:
        """Extract comprehensive noise features from noise map"""
        features = []
        
        # Basic statistical features
        features.extend([
            np.mean(noise_map),
            np.std(noise_map),
            np.var(noise_map),
            stats.skew(noise_map.flatten()),
            stats.kurtosis(noise_map.flatten()),
            np.percentile(noise_map, 5),
            np.percentile(noise_map, 25),
            np.percentile(noise_map, 75),
            np.percentile(noise_map, 95),
            np.max(noise_map) - np.min(noise_map),  # Range
        ])
        
        # Histogram features (more bins for better discrimination)
        hist, _ = np.histogram(noise_map, bins=30, range=(-1, 1))
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        features.extend(hist.tolist())
        
        # Local variance analysis
        local_var_3x3 = ndimage.generic_filter(noise_map, np.var, size=3)
        local_var_5x5 = ndimage.generic_filter(noise_map, np.var, size=5)
        features.extend([
            np.mean(local_var_3x3),
            np.std(local_var_3x3),
            np.max(local_var_3x3),
            np.mean(local_var_5x5),
            np.std(local_var_5x5),
            np.max(local_var_5x5)
        ])
        
        # Gradient analysis
        grad_x = cv2.Sobel(noise_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(noise_map, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([
            np.mean(grad_mag),
            np.std(grad_mag),
            np.percentile(grad_mag, 90),
            np.percentile(grad_mag, 95),
        ])
        
        # Frequency domain analysis
        try:
            fft = np.fft.fft2(noise_map)
            fft_mag = np.abs(fft)
            fft_shift = np.fft.fftshift(fft_mag)
            
            # Low, mid, high frequency energy
            h, w = fft_shift.shape
            center_h, center_w = h // 2, w // 2
            
            # Create frequency masks
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_w)**2 + (y - center_h)**2)
            
            low_freq = np.sum(fft_shift[dist <= min(h, w) * 0.1])
            mid_freq = np.sum(fft_shift[(dist > min(h, w) * 0.1) & (dist <= min(h, w) * 0.3)])
            high_freq = np.sum(fft_shift[dist > min(h, w) * 0.3])
            
            total_energy = low_freq + mid_freq + high_freq
            if total_energy > 0:
                features.extend([
                    low_freq / total_energy,
                    mid_freq / total_energy,
                    high_freq / total_energy
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
                
        except Exception as e:
            # Fallback if FFT fails
            features.extend([0.0, 0.0, 0.0])
        
        # Texture analysis using different kernels
        kernels = [
            np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # Vertical edges
            np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),   # Horizontal edges
            np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),   # Diagonal
        ]
        
        for kernel in kernels:
            response = cv2.filter2D(noise_map, -1, kernel)
            features.extend([
                np.mean(np.abs(response)),
                np.std(response)
            ])
        
        # Spatial distribution analysis
        # Divide image into quadrants and analyze distribution
        h, w = noise_map.shape
        quadrants = [
            noise_map[:h//2, :w//2],     # Top-left
            noise_map[:h//2, w//2:],     # Top-right
            noise_map[h//2:, :w//2],     # Bottom-left
            noise_map[h//2:, w//2:]      # Bottom-right
        ]
        
        quad_vars = [np.var(quad) for quad in quadrants]
        features.extend([
            np.mean(quad_vars),
            np.std(quad_vars),
            np.max(quad_vars) - np.min(quad_vars)  # Variance in variance across quadrants
        ])
        
        return np.array(features)
    
    def fit_transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        """Extract features from multiple noise maps and fit scaler"""
        feature_matrix = []
        
        print(f"Extracting features from {len(noise_maps)} noise maps...")
        for noise_map in tqdm(noise_maps, desc="Feature extraction"):
            features = self.extract_noise_features(noise_map)
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Handle any NaN or infinite values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.fitted = True
        
        return feature_matrix
    
    def transform(self, noise_maps: List[np.ndarray]) -> np.ndarray:
        """Transform noise maps to features (after fitting)"""
        if not self.fitted:
            raise ValueError("Must call fit_transform first")
        
        feature_matrix = []
        for noise_map in noise_maps:
            features = self.extract_noise_features(noise_map)
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.scaler.transform(feature_matrix)

class NoiseClassificationPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint_dir='checkpoints'):
        self.device = device
        self.autoencoder = None
        self.noise_extractor = EnhancedNoiseDistributionExtractor()
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.autoencoder_trained = False
        self.classifier_trained = False
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
            autoencoder_state = self.autoencoder.state_dict()
        
        # Save noise extractor
        extractor_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_extractor.pkl')
        if self.noise_extractor.fitted:
            with open(extractor_path, 'wb') as f:
                pickle.dump(self.noise_extractor, f)
        
        # Save classifier
        classifier_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_classifier.pkl')
        if self.classifier_trained:
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.classifier, f)
        
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
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name='latest'):
        """Load training state from checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found.")
            return False, 0
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load autoencoder
        if checkpoint['autoencoder_state_dict'] is not None:
            self.autoencoder = ImprovedDenoiseAutoencoder().to(self.device)
            self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            
        # Load noise extractor
        extractor_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_extractor.pkl')
        if os.path.exists(extractor_path):
            try:
                with open(extractor_path, 'rb') as f:
                    self.noise_extractor = pickle.load(f)
                print("Noise extractor loaded")
            except Exception as e:
                print(f"Warning: Could not load noise extractor: {e}")
        
        # Load classifier
        classifier_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}_classifier.pkl')
        if os.path.exists(classifier_path):
            try:
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                print("Classifier loaded")
            except Exception as e:
                print(f"Warning: Could not load classifier: {e}")
        
        self.autoencoder_trained = checkpoint.get('autoencoder_trained', False)
        self.classifier_trained = checkpoint.get('classifier_trained', False)
        self.training_history = checkpoint.get('training_history', {
            'autoencoder_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
            'best_test_mcc': 0.0,
            'epochs_trained': 0
        })
        
        print(f"Loaded: Autoencoder trained: {self.autoencoder_trained}, Classifier trained: {self.classifier_trained}")
        return True, self.training_history.get('epochs_trained', 0)

    def train_autoencoder(self, train_loader: DataLoader, val_loader: DataLoader = None, 
                         epochs=30, resume_from_epoch=0, save_every=1):
        """Train autoencoder with improved training strategy"""
        if self.autoencoder_trained and resume_from_epoch == 0:
            print("Autoencoder already trained.")
            return
            
        print(f"Training autoencoder for {epochs} epochs")
        
        if self.autoencoder is None:
            self.autoencoder = ImprovedDenoiseAutoencoder().to(self.device)
        
        # Use different optimizers for different phases - FIXED: removed verbose parameter
        initial_lr = 0.001
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=initial_lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        criterion = nn.MSELoss()
        self.autoencoder.train()
        
        start_epoch = resume_from_epoch
        
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (data, _) in enumerate(progress_bar):
                data = data.to(self.device)
                
                # Progressive noise strategy
                if epoch < epochs // 3:
                    # Early training: focus on synthetic images (less noise)
                    noise_strength = 0.05
                elif epoch < 2 * epochs // 3:
                    # Mid training: moderate noise
                    noise_strength = 0.1
                else:
                    # Late training: more challenging noise
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
                reconstructed = self.autoencoder(noisy_data)
                
                # Perceptual loss component
                loss = criterion(reconstructed, data)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
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
            
            print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Regular checkpointing
            if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
                checkpoint_name = f'autoencoder_epoch_{epoch+1}'
                self.save_checkpoint(checkpoint_name)
        
        self.autoencoder_trained = True
        self.save_checkpoint('autoencoder_final')
        print("Autoencoder training completed!")
    
    def _validate_autoencoder(self, val_loader: DataLoader, current_epoch: int, total_epochs: int) -> float:
        """Validate autoencoder performance"""
        self.autoencoder.eval()
        total_loss = 0
        num_batches = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                
                # Use similar noise as training
                if current_epoch < total_epochs // 3:
                    noise_strength = 0.05
                elif current_epoch < 2 * total_epochs // 3:
                    noise_strength = 0.1
                else:
                    noise_strength = 0.15
                
                noise = torch.randn_like(data) * noise_strength
                noisy_data = torch.clamp(data + noise, 0., 1.)
                
                reconstructed = self.autoencoder(noisy_data)
                loss = criterion(reconstructed, data)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.autoencoder.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def extract_noise_from_images(self, images: torch.Tensor, batch_size: int = 32) -> List[np.ndarray]:
        """Extract noise maps using trained autoencoder with batching"""
        if self.autoencoder is None or not self.autoencoder_trained:
            raise ValueError("Autoencoder must be trained first")
        
        self.autoencoder.eval()
        noise_maps = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                reconstructed = self.autoencoder(batch)
                
                noise = batch - reconstructed
                noise_np = noise.cpu().numpy()
                
                for j in range(noise_np.shape[0]):
                    # Average across channels to get single noise map
                    noise_map = np.mean(noise_np[j], axis=0)
                    noise_maps.append(noise_map)
        
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
                print(f"Noise maps saved to {save_path}")
            plt.show()
    
    def train_noise_classifier(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train the noise distribution classifier with detailed progress tracking"""
        if self.classifier_trained:
            print("Noise classifier already trained.")
            return
            
        if not self.autoencoder_trained:
            raise ValueError("Autoencoder must be trained first")
        
        print("\n" + "="*60)
        print("STARTING NOISE CLASSIFIER TRAINING")
        print("="*60)
        
        # Step 1: Extract noise features from training data
        print("\n[1/4] Extracting noise maps from training data...")
        all_noise_maps = []
        all_labels = []
        total_samples = 0
        
        # Get total number of samples for progress tracking
        for _, labels in train_loader:
            total_samples += len(labels)
        print(f"Total training samples to process: {total_samples}")
        
        # Process training data in batches to avoid memory issues
        batch_count = 0
        samples_processed = 0
        
        progress_bar = tqdm(train_loader, desc="Extracting noise maps", unit="batch")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            try:
                noise_maps = self.extract_noise_from_images(images, batch_size=32)
                all_noise_maps.extend(noise_maps)
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1
                samples_processed += len(images)
                
                # Update progress bar with detailed info
                progress_bar.set_postfix({
                    'Samples': f"{samples_processed}/{total_samples}",
                    'Batches': f"{batch_count}",
                    'Maps': f"{len(all_noise_maps)}"
                })
                
                # Print periodic updates
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print(f"\n    Progress: {samples_processed}/{total_samples} samples ({100*samples_processed/total_samples:.1f}%)")
                    
            except Exception as e:
                print(f"\n    Error processing batch {batch_idx}: {e}")
                continue
                
        print(f"\n✓ Successfully processed {batch_count} batches")
        print(f"✓ Extracted {len(all_noise_maps)} noise maps from {samples_processed} samples")
        
        if len(all_noise_maps) == 0:
            raise ValueError("No noise maps extracted from training data")
        
        # Step 2: Extract features
        print(f"\n[2/4] Computing noise distribution features...")
        print("This may take several minutes depending on dataset size...")
        
        try:
            feature_matrix = self.noise_extractor.fit_transform(all_noise_maps)
            print(f"✓ Feature extraction completed")
            print(f"✓ Feature matrix shape: {feature_matrix.shape}")
            print(f"✓ Features per sample: {feature_matrix.shape[1]}")
        except Exception as e:
            print(f"✗ Error extracting features: {e}")
            raise
        
        # Step 3: Train classifier
        print(f"\n[3/4] Training Random Forest classifier...")
        print(f"Configuration:")
        print(f"  - Estimators: {self.classifier.n_estimators}")
        print(f"  - Max depth: {self.classifier.max_depth}")
        print(f"  - Class weight: {self.classifier.class_weight}")
        
        # Show class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"Training class distribution:")
        class_names = ['Real', 'Synthetic', 'Semi-synthetic']
        for label, count in zip(unique, counts):
            print(f"  {class_names[label]}: {count} samples ({100*count/len(all_labels):.1f}%)")
        
        print("Training in progress...")
        self.classifier.fit(feature_matrix, all_labels)
        print("✓ Random Forest training completed")
        
        # Step 4: Evaluate training performance
        print(f"\n[4/4] Evaluating classifier performance...")
        
        print("Computing training metrics...")
        train_predictions = self.classifier.predict(feature_matrix)
        train_accuracy = accuracy_score(all_labels, train_predictions)
        train_mcc = matthews_corrcoef(all_labels, train_predictions)
        
        print(f"\nTraining Results:")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  MCC: {train_mcc:.4f}")
        
        # Training confusion matrix
        train_cm = confusion_matrix(all_labels, train_predictions)
        print(f"\nTraining Confusion Matrix:")
        print("True\\Pred    Real  Synth  Semi")
        for i, row in enumerate(train_cm):
            print(f"{class_names[i]:10s} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
        
        # Feature importance analysis
        if hasattr(self.classifier, 'feature_importances_'):
            feature_importance = self.classifier.feature_importances_
            top_features = np.argsort(feature_importance)[-10:]
            print(f"\nTop 10 Most Important Features:")
            for i, feat_idx in enumerate(reversed(top_features)):
                print(f"  {i+1:2d}. Feature {feat_idx:3d}: {feature_importance[feat_idx]:.4f}")
        
        # Validation evaluation
        if val_loader is not None:
            print(f"\nValidating on validation set...")
            try:
                val_predictions, val_probabilities = self.predict(val_loader)
                val_labels = []
                for _, labels in val_loader:
                    val_labels.extend(labels.cpu().numpy())
                
                val_accuracy = accuracy_score(val_labels, val_predictions)
                val_mcc = matthews_corrcoef(val_labels, val_predictions)
                
                print(f"\nValidation Results:")
                print(f"  Accuracy: {val_accuracy:.4f}")
                print(f"  MCC: {val_mcc:.4f}")
                
                # Validation confusion matrix
                val_cm = confusion_matrix(val_labels, val_predictions)
                print(f"\nValidation Confusion Matrix:")
                print("True\\Pred    Real  Synth  Semi")
                for i, row in enumerate(val_cm):
                    print(f"{class_names[i]:10s} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
                
                # Per-class performance
                val_report = classification_report(val_labels, val_predictions, 
                                                 target_names=class_names, 
                                                 output_dict=True, zero_division=0)
                print(f"\nPer-class Validation Performance:")
                for class_name in class_names:
                    class_key = class_name.lower().replace('-', '_')  # Fixed key mapping
                    if class_key in val_report:
                        metrics = val_report[class_key]
                        print(f"  {class_name}:")
                        print(f"    Precision: {metrics['precision']:.4f}")
                        print(f"    Recall:    {metrics['recall']:.4f}")
                        print(f"    F1-score:  {metrics['f1-score']:.4f}")
                    
            except Exception as e:
                print(f"Error during validation: {e}")
        
        # Save checkpoint
        self.classifier_trained = True
        self.save_checkpoint('classifier_final')
        
        print("\n" + "="*60)
        print("NOISE CLASSIFIER TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"✓ Processed {len(all_noise_maps)} training samples")
        print(f"✓ Extracted {feature_matrix.shape[1]} noise features per sample")
        print(f"✓ Trained Random Forest with {self.classifier.n_estimators} trees")
        print(f"✓ Final training accuracy: {train_accuracy:.4f}")
        if val_loader:
            print(f"✓ Final validation accuracy: {val_accuracy:.4f}")
        print(f"✓ Model saved to checkpoint: 'classifier_final'")
        print("="*60)
    
    def predict(self, test_loader: DataLoader, batch_size: int = 32) -> Tuple[List[int], List[float]]:
        """Predict class labels for test data"""
        if not self.autoencoder_trained or not self.classifier_trained:
            raise ValueError("Both autoencoder and classifier must be trained")
        
        print("Generating predictions...")
        
        all_predictions = []
        all_probabilities = []
        
        self.autoencoder.eval()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Predicting")):
                try:
                    # Extract noise maps
                    noise_maps = self.extract_noise_from_images(images, batch_size=batch_size)
                    
                    # Convert to features
                    feature_matrix = self.noise_extractor.transform(noise_maps)
                    
                    # Predict
                    predictions = self.classifier.predict(feature_matrix)
                    probabilities = self.classifier.predict_proba(feature_matrix)
                    
                    all_predictions.extend(predictions.tolist())
                    all_probabilities.extend(probabilities.tolist())
                    
                except Exception as e:
                    print(f"Error predicting batch {batch_idx}: {e}")
                    # Add dummy predictions to maintain alignment
                    batch_size_actual = len(images)
                    all_predictions.extend([0] * batch_size_actual)  # Default to class 0
                    all_probabilities.extend([[1.0, 0.0, 0.0]] * batch_size_actual)
        
        print(f"Generated predictions for {len(all_predictions)} samples")
        return all_predictions, all_probabilities
    
    def evaluate(self, test_loader: DataLoader, test_labels: List[int], 
                 save_results: bool = True, results_dir: str = 'results') -> Dict:
        """Comprehensive evaluation with visualization"""
        predictions, probabilities = self.predict(test_loader)
        
        # Ensure same length
        min_len = min(len(test_labels), len(predictions))
        test_labels = test_labels[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        
        mcc = matthews_corrcoef(test_labels, predictions)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, 
                                    target_names=['Real', 'Synthetic', 'Semi-synthetic'],
                                    output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
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
            'probabilities': probabilities
        }
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        
        print("\nConfusion Matrix:")
        print("True\\Pred    Real  Synth  Semi")
        class_names = ['Real     ', 'Synthetic', 'Semi-synth']
        for i, (name, row) in enumerate(zip(class_names, cm)):
            print(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
        
        print("\nPer-class Metrics:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
        
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            
            # Save text results
            results_path = os.path.join(results_dir, 'evaluation_results.txt')
            with open(results_path, 'w') as f:
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n\n")
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
                        
            # Save predictions
            predictions_path = os.path.join(results_dir, 'predictions.json')
            with open(predictions_path, 'w') as f:
                json.dump({
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'true_labels': test_labels,
                    'accuracy': accuracy,
                    'mcc': mcc
                }, f, indent=2)
            
            # Visualize confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Synthetic', 'Semi-synthetic'],
                       yticklabels=['Real', 'Synthetic', 'Semi-synthetic'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = os.path.join(results_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"\nResults saved to {results_dir}/")
        
        return results

def main():
    """Main script to run the improved noise classification pipeline"""
    # Configuration
    data_dir = './datasets/train'
    batch_size = 32  # Reduced for memory efficiency
    num_epochs = 30  # Increased for better training
    checkpoint_dir = './checkpoints'
    results_dir = './results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Load and split dataset
    print("Loading dataset...")
    try:
        images, labels = load_pt_data(data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_dataset, val_dataset, test_dataset = create_train_val_test_split(
        images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # Create data loaders with reduced num_workers for stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize pipeline
    pipeline = NoiseClassificationPipeline(device=device, checkpoint_dir=checkpoint_dir)

    # Check for existing checkpoint
    print("\nChecking for existing checkpoints...")
    loaded, resume_epoch = pipeline.load_checkpoint('latest')
    if loaded:
        print(f"Loaded existing checkpoint. Resume from epoch: {resume_epoch}")
    else:
        print("No checkpoint found. Starting fresh training.")
        resume_epoch = 0

    # Train autoencoder if not already trained
    if not pipeline.autoencoder_trained:
        print(f"\nTraining autoencoder...")
        pipeline.train_autoencoder(
            train_loader, val_loader, 
            epochs=num_epochs,
            resume_from_epoch=resume_epoch, 
            save_every=1  # Changed from 10 to 1 for better checkpointing
        )
        pipeline.save_checkpoint('autoencoder_complete')
    else:
        print("Autoencoder already trained.")

    # Visualize noise maps
    print("\nVisualizing noise maps...")
    try:
        # Get samples from each class
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
            os.makedirs(results_dir, exist_ok=True)  # Ensure results dir exists
            pipeline.visualize_noise_maps(
                sample_images, sample_labels, 
                num_samples=len(sample_images),
                save_path=os.path.join(results_dir, 'noise_maps_by_class.png')
            )
    except Exception as e:
        print(f"Error visualizing noise maps: {e}")

    # Train classifier if not already trained
    if not pipeline.classifier_trained:
        print(f"\nTraining noise classifier...")
        pipeline.train_noise_classifier(train_loader, val_loader)
        pipeline.save_checkpoint('complete_pipeline')
    else:
        print("Noise classifier already trained.")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_labels = test_dataset.labels.cpu().tolist()
    results = pipeline.evaluate(
        test_loader, test_labels,
        save_results=True, results_dir=results_dir
    )

    # Plot comprehensive training metrics
    print("\nGenerating training plots...")
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
    
    # Class-wise F1 scores - Fixed key mapping
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    f1_scores = []
    report_keys = ['Real', 'Synthetic', 'Semi-synthetic']  # Use exact keys from report
    
    for key in report_keys:
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
    
    # Class distribution
    class_counts = torch.bincount(test_dataset.labels, minlength=3).numpy()
    axes[1, 1].pie(class_counts, labels=class_names, autopct='%1.1f%%', 
                  colors=['#FF9F40', '#4BC0C0', '#9966FF'])
    axes[1, 1].set_title('Test Set Class Distribution')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'comprehensive_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comprehensive results plot saved to {plot_path}")
    plt.show()

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test MCC: {results['mcc']:.4f}")
    print(f"Best Test MCC: {pipeline.training_history['best_test_mcc']:.4f}")
    
    if pipeline.training_history['autoencoder_losses']:
        initial_loss = pipeline.training_history['autoencoder_losses'][0]
        final_loss = pipeline.training_history['autoencoder_losses'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"Autoencoder Loss Improvement: {improvement:.1f}% ({initial_loss:.6f} → {final_loss:.6f})")
    
    print(f"\nModel files saved in: {checkpoint_dir}")
    print(f"Results saved in: {results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()