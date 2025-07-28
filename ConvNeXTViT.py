import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from timm import create_model
import math
import os
import json
from datetime import datetime
import argparse
from PIL import Image
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter
import glob
import warnings
warnings.filterwarnings("ignore")

# ===== DISTRIBUTED TRAINING UTILITIES =====

def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
        else:
            dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=world_size)
        dist.barrier()
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0

def create_distributed_sampler(dataset, shuffle=True):
    """Create distributed sampler"""
    if dist.is_initialized():
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None

# ===== ATTENTION MECHANISMS =====

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# ===== HYBRID MODEL DEFINITION =====

class AttentionEnhancedConvNeXTViTHybrid(nn.Module):
    def __init__(self, num_classes=3, freeze_convnext=True, freeze_vit=True, 
                 use_gradual_unfreezing=True, dropout_rate=0.3):
        super().__init__()
        self.use_gradual_unfreezing = use_gradual_unfreezing
        self.freeze_convnext = freeze_convnext
        self.freeze_vit = freeze_vit
        self.optimizer_switched = False  # Track if optimizer has been switched
        
        # ConvNeXt backbone for spatial features
        self.convnext = create_model('convnext_base', pretrained=True, num_classes=0)
        convnext_dim = self.convnext.num_features
        
        # Vision Transformer for global context
        self.vit = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        vit_dim = self.vit.num_features
        
        # CBAM attention for ConvNeXt features
        self.cbam = CBAM(convnext_dim)
        
        # Feature fusion layers
        self.convnext_proj = nn.Linear(convnext_dim, 512)
        self.vit_proj = nn.Linear(vit_dim, 512)
        
        # Cross-attention mechanism
        self.cross_attention = MultiHeadSelfAttention(512, num_heads=8, attn_drop=0.1, proj_drop=0.1)
        
        # Final classification layers
        self.fusion_norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize freezing
        if freeze_convnext:
            self._freeze_backbone(self.convnext)
        if freeze_vit:
            self._freeze_backbone(self.vit)
    
    def _freeze_backbone(self, model):
        """Freeze backbone parameters"""
        for param in model.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self, model):
        """Unfreeze backbone parameters"""
        for param in model.parameters():
            param.requires_grad = True
    
    def gradual_unfreezing(self, epoch, total_epochs):
        """Gradually unfreeze layers during training"""
        if not self.use_gradual_unfreezing:
            return False
        
        optimizer_changed = False
        
        # Unfreeze ConvNeXt after 20% of training
        if epoch > total_epochs * 0.2 and self.freeze_convnext:
            self._unfreeze_backbone(self.convnext)
            self.freeze_convnext = False
            self.optimizer_switched = True
            optimizer_changed = True
            print(f"🔓 Unfroze ConvNeXt at epoch {epoch} - Fine-tuning phase started!")
        
        # Unfreeze ViT after 40% of training  
        if epoch > total_epochs * 0.4 and self.freeze_vit:
            self._unfreeze_backbone(self.vit)
            self.freeze_vit = False
            if not self.optimizer_switched:  # Only switch once
                self.optimizer_switched = True
                optimizer_changed = True
            print(f"🔓 Unfroze ViT at epoch {epoch} - Full fine-tuning phase!")
        
        return optimizer_changed
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # ConvNeXt path - spatial features with attention
        conv_features = self.convnext.forward_features(x)  # [B, C, H, W]
        conv_features = self.cbam(conv_features)  # Apply CBAM attention
        conv_features = F.adaptive_avg_pool2d(conv_features, 1).flatten(1)  # [B, C]
        conv_features = self.convnext_proj(conv_features)  # [B, 512]
        
        # ViT path - global context
        vit_features = self.vit.forward_features(x)  # [B, N, C]
        vit_features = vit_features.mean(dim=1)  # Global average pooling [B, C]
        vit_features = self.vit_proj(vit_features)  # [B, 512]
        
        # Cross-attention fusion
        # Prepare features for attention [B, 2, 512]
        fusion_input = torch.stack([conv_features, vit_features], dim=1)
        attended_features = self.cross_attention(fusion_input)  # [B, 2, 512]
        
        # Concatenate attended features
        fused_features = attended_features.flatten(1)  # [B, 1024]
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.dropout(fused_features)
        
        # Final classification
        output = self.classifier(fused_features)
        return output

# ===== OPTIMIZER SWITCHING UTILITIES =====

def create_sgd_optimizer(model, current_lr, weight_decay=0.01):
    """Create SGD optimizer for fine-tuning phase"""
    return optim.SGD(
        model.parameters(),
        lr=current_lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
    )

def create_sgd_scheduler(optimizer, remaining_epochs):
    """Create appropriate scheduler for SGD optimizer"""
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=remaining_epochs,
        eta_min=1e-6
    )

def switch_optimizer_to_sgd(model, current_optimizer, current_epoch, total_epochs):
    """Switch from AdamW to SGD optimizer"""
    current_lr = current_optimizer.param_groups[0]['lr']
    
    # Create new SGD optimizer
    sgd_optimizer = create_sgd_optimizer(model, current_lr * 0.1)  # Reduce LR for SGD
    
    # Create new scheduler for remaining epochs
    remaining_epochs = total_epochs - current_epoch
    sgd_scheduler = create_sgd_scheduler(sgd_optimizer, remaining_epochs)
    
    print(f"🔄 Switched to SGD optimizer at epoch {current_epoch}")
    print(f"   AdamW LR: {current_lr:.2e} → SGD LR: {current_lr * 0.1:.2e}")
    print(f"   Remaining epochs: {remaining_epochs}")
    
    return sgd_optimizer, sgd_scheduler


# ===== DATASET CLASS (from your original code) =====

class TensorImageDataset(Dataset):
    """Dataset for loading images from PyTorch tensor files (.pt)"""
    def __init__(self, tensor_info: List[Tuple[str, int, int]], transform=None, 
                 cache_tensors: bool = False, max_cache_size: int = 50):
        """
        Args:
            tensor_info: List of (tensor_file_path, tensor_index_in_file, label)
            transform: Image transforms to apply
            cache_tensors: Whether to cache loaded tensor files
            max_cache_size: Maximum number of tensor files to cache
        """
        self.tensor_info = tensor_info
        self.transform = transform
        self.cache_tensors = cache_tensors
        self.max_cache_size = max_cache_size
        self.tensor_cache = {}
        
        print(f"Dataset created with {len(tensor_info)} samples")
        if cache_tensors:
            print(f"Tensor caching enabled (max {max_cache_size} files)")
    
    def __len__(self):
        return len(self.tensor_info)
    
    def _load_tensor_file(self, tensor_path):
        """Load tensor file with caching"""
        if self.cache_tensors and tensor_path in self.tensor_cache:
            return self.tensor_cache[tensor_path]
        
        try:
            # Load tensor file
            tensor_data = torch.load(tensor_path, map_location='cpu')
            
            # Handle different possible formats
            if isinstance(tensor_data, dict):
                # If it's a dictionary, look for common keys
                if 'images' in tensor_data:
                    images = tensor_data['images']
                elif 'data' in tensor_data:
                    images = tensor_data['data']
                else:
                    # Take the first tensor value
                    images = list(tensor_data.values())[0]
            else:
                # Assume it's directly the tensor
                images = tensor_data
            
            # Ensure it's a tensor
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images)
            
            # Cache if enabled and cache not full
            if (self.cache_tensors and 
                len(self.tensor_cache) < self.max_cache_size):
                self.tensor_cache[tensor_path] = images
                
            return images
            
        except Exception as e:
            print(f"Error loading tensor file {tensor_path}: {e}")
            # Return a dummy tensor as fallback
            return torch.zeros((5000, 3, 224, 224), dtype=torch.float32)
    
    def __getitem__(self, idx):
        tensor_path, tensor_idx, label = self.tensor_info[idx]
        
        # Load the tensor file
        images_tensor = self._load_tensor_file(tensor_path)
        
        # Extract the specific image
        try:
            if tensor_idx >= images_tensor.shape[0]:
                print(f"Warning: Index {tensor_idx} >= tensor size {images_tensor.shape[0]} for {tensor_path}")
                tensor_idx = 0  # Fallback to first image
            
            image = images_tensor[tensor_idx]
            
            # Ensure image is in correct format [C, H, W] and float
            if image.dim() == 4:  # [1, C, H, W]
                image = image.squeeze(0)
            
            # Normalize to [0, 1] if needed
            if image.max() > 1.0:
                image = image.float() / 255.0
            
            # Apply transforms if provided
            if self.transform:
                # Convert to PIL for torchvision transforms compatibility
                if image.shape[0] == 3:  # [C, H, W]
                    image_pil = transforms.ToPILImage()(image)
                    image = self.transform(image_pil)
                else:
                    # If transforms expect tensor input
                    image = self.transform(image)
            
        except Exception as e:
            print(f"Error processing image {tensor_idx} from {tensor_path}: {e}")
            # Return a dummy image
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        return image, label

# ===== DATA LOADING FUNCTIONS (from your original code) =====

def load_tensor_data_info(data_root: str = "datasets/train/", 
                         total_images: int = 240000,
                         class_mapping: dict = None,
                         images_per_file: int = 5000) -> Tuple[List[Tuple[str, int, int]], List[int]]:
    """
    Load tensor file paths and create index mapping for images.
    
    Args:
        data_root: Root directory containing class subdirectories with .pt files
        total_images: Maximum total images to use
        class_mapping: Dictionary mapping class names to indices
        images_per_file: Number of images per tensor file
    
    Returns:
        tensor_info: List of (tensor_file_path, image_index_in_file, class_label)
        labels: List of labels (for compatibility)
    """
    if class_mapping is None:
        class_mapping = {
            'real': 0,
            'semi-synthetic': 1, 
            'synthetic': 2
        }
    
    tensor_info = []
    labels = []
    
    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(data_root, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue
        
        # Find all .pt files in the class directory
        tensor_files = glob.glob(os.path.join(class_dir, "*.pt"))
        tensor_files.sort()  # Ensure consistent ordering
        
        print(f"Found {len(tensor_files)} tensor files in {class_name}")
        
        class_samples = 0
        max_per_class = total_images // len(class_mapping) if total_images else float('inf')
        
        for tensor_file in tensor_files:
            if class_samples >= max_per_class:
                break
                
            try:
                # Quick check to get tensor shape without loading all data
                tensor_data = torch.load(tensor_file, map_location='cpu')
                
                # Handle different formats
                if isinstance(tensor_data, dict):
                    if 'images' in tensor_data:
                        num_images = tensor_data['images'].shape[0]
                    elif 'data' in tensor_data:
                        num_images = tensor_data['data'].shape[0]
                    else:
                        images = list(tensor_data.values())[0]
                        num_images = images.shape[0] if hasattr(images, 'shape') else images_per_file
                else:
                    num_images = tensor_data.shape[0] if hasattr(tensor_data, 'shape') else images_per_file
                
                # Create entries for each image in this tensor file
                for img_idx in range(min(num_images, max_per_class - class_samples)):
                    tensor_info.append((tensor_file, img_idx, class_idx))
                    labels.append(class_idx)
                    class_samples += 1
                    
                    if class_samples >= max_per_class:
                        break
                        
            except Exception as e:
                print(f"Error processing tensor file {tensor_file}: {e}")
                continue
        
        print(f"Loaded {class_samples} images from class '{class_name}' (label: {class_idx})")
    
    print(f"Total tensor info entries: {len(tensor_info)}")
    return tensor_info, labels

def create_tensor_data_splits(tensor_info: List[Tuple[str, int, int]], labels: List[int], 
                            train_ratio: float = 0.7, 
                            val_ratio: float = 0.15, 
                            test_ratio: float = 0.15,
                            stratify: bool = True,
                            random_state: int = 42) -> Tuple[Tuple[List, List[int]], 
                                                            Tuple[List, List[int]], 
                                                            Tuple[List, List[int]]]:
    """Create train/validation/test splits for tensor data"""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create indices for splitting
    indices = list(range(len(tensor_info)))
    
    # First split: train vs (val + test)
    stratify_labels = labels if stratify else None
    
    train_indices, temp_indices, y_train, y_temp = train_test_split(
        indices, labels,
        test_size=(val_ratio + test_ratio),
        stratify=stratify_labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    stratify_temp = y_temp if stratify else None
    
    val_indices, test_indices, y_val, y_test = train_test_split(
        temp_indices, y_temp,
        test_size=(1 - val_test_ratio),
        stratify=stratify_temp,
        random_state=random_state
    )
    
    # Create tensor info for each split
    train_tensor_info = [tensor_info[i] for i in train_indices]
    val_tensor_info = [tensor_info[i] for i in val_indices]
    test_tensor_info = [tensor_info[i] for i in test_indices]
    
    # Print split information
    total_samples = len(tensor_info)
    print(f"Tensor data splits created:")
    print(f"  Train: {len(train_tensor_info)} samples ({len(train_tensor_info)/total_samples*100:.1f}%)")
    print(f"  Val:   {len(val_tensor_info)} samples ({len(val_tensor_info)/total_samples*100:.1f}%)")
    print(f"  Test:  {len(test_tensor_info)} samples ({len(test_tensor_info)/total_samples*100:.1f}%)")
    
    # Print class distribution for each split
    for split_name, split_labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        class_counts = Counter(split_labels)
        print(f"  {split_name} class distribution: {dict(class_counts)}")
    
    return (train_tensor_info, y_train), (val_tensor_info, y_val), (test_tensor_info, y_test)

def create_tensor_data_loaders(train_data, val_data, test_data, 
                             train_transform, val_test_transform,
                             batch_size=32, num_workers=4, 
                             world_size=1, cache_tensors=True):
    """Create data loaders for tensor data with caching"""
    
    train_tensor_info, y_train = train_data
    val_tensor_info, y_val = val_data  
    test_tensor_info, y_test = test_data
    
    # Create datasets with tensor caching strategy
    train_dataset = TensorImageDataset(
        train_tensor_info, 
        transform=train_transform,
        cache_tensors=False,  # Don't cache training data (changes with augmentations)
        max_cache_size=0
    )
    
    val_dataset = TensorImageDataset(
        val_tensor_info,
        transform=val_test_transform,
        cache_tensors=cache_tensors,  # Cache validation tensor files
        max_cache_size=50  # Cache up to 50 tensor files
    )
    
    test_dataset = TensorImageDataset(
        test_tensor_info,
        transform=val_test_transform, 
        cache_tensors=cache_tensors,  # Cache test tensor files
        max_cache_size=50  # Cache up to 50 tensor files
    )
    
    # Create distributed samplers
    train_sampler = create_distributed_sampler(train_dataset, shuffle=True)
    val_sampler = create_distributed_sampler(val_dataset, shuffle=False)  
    test_sampler = create_distributed_sampler(test_dataset, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
        shuffle=(train_sampler is None)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    return train_loader, val_loader, test_loader

def save_tensor_data_splits(train_data, val_data, test_data, save_path="tensor_data_splits.pkl"):
    """Save tensor data splits"""
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"Tensor data splits saved to {save_path}")

def load_tensor_data_splits(save_path="tensor_data_splits.pkl"):
    """Load previously saved tensor data splits"""
    try:
        with open(save_path, 'rb') as f:
            splits = pickle.load(f)
        
        print(f"Tensor data splits loaded from {save_path}")
        return splits['train'], splits['val'], splits['test']
    
    except FileNotFoundError:
        print(f"No saved tensor splits found at {save_path}")
        return None, None, None

def create_dummy_tensor_data(data_root='datasets/train', samples_per_class=1000, samples_per_file=100):
    """Create dummy tensor data for testing the training pipeline"""
    import random
    
    print("Creating dummy tensor data...")
    
    class_names = ['real', 'semi-synthetic', 'synthetic']
    
    # Create directory structure
    os.makedirs(data_root, exist_ok=True)
    
    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Creating {samples_per_class} samples for class '{class_name}'...")
        
        # Calculate number of files needed
        num_files = (samples_per_class + samples_per_file - 1) // samples_per_file
        
        for file_idx in range(num_files):
            # Calculate samples for this file
            remaining_samples = samples_per_class - (file_idx * samples_per_file)
            current_file_samples = min(samples_per_file, remaining_samples)
            
            if current_file_samples <= 0:
                break
            
            # Create dummy images (3, 224, 224) in range [0, 1]
            dummy_images = torch.rand(current_file_samples, 3, 224, 224, dtype=torch.float32)
            
            # Add some class-specific patterns to make classes distinguishable
            if class_name == 'real':
                # Add some noise to make it look more "natural"
                dummy_images += torch.randn_like(dummy_images) * 0.1
                dummy_images = torch.clamp(dummy_images, 0, 1)
            elif class_name == 'semi-synthetic':
                # Add some structured patterns
                dummy_images[:, 0, :, :] *= 1.2  # Enhance red channel
                dummy_images = torch.clamp(dummy_images, 0, 1)
            elif class_name == 'synthetic':
                # Add geometric patterns
                for i in range(current_file_samples):
                    # Add some rectangular patterns
                    h_start, w_start = random.randint(0, 150), random.randint(0, 150)
                    h_end, w_end = h_start + 50, w_start + 50
                    dummy_images[i, :, h_start:h_end, w_start:w_end] = 0.8
            
            # Save tensor file
            file_path = os.path.join(class_dir, f'batch_{file_idx}.pt')
            torch.save(dummy_images, file_path)
            
            print(f"  Created {file_path} with {current_file_samples} samples")
    
    print(f"Dummy data creation completed!")
    print(f"Created {len(class_names)} classes with {samples_per_class} samples each")
    print(f"Data saved in: {data_root}")

def analyze_tensor_files(data_root: str, class_mapping: dict = None):
    """Analyze tensor files to understand their structure"""
    if class_mapping is None:
        class_mapping = {
            'real': 0,
            'semi-synthetic': 1, 
            'synthetic': 2
        }
    
    print("=== TENSOR FILE ANALYSIS ===")
    
    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(data_root, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist")
            continue
        
        tensor_files = glob.glob(os.path.join(class_dir, "*.pt"))
        print(f"\nClass: {class_name}")
        print(f"Number of .pt files: {len(tensor_files)}")
        
        if tensor_files:
            # Analyze first file
            sample_file = tensor_files[0]
            try:
                sample_data = torch.load(sample_file, map_location='cpu')
                
                print(f"Sample file: {os.path.basename(sample_file)}")
                print(f"Data type: {type(sample_data)}")
                
                if isinstance(sample_data, dict):
                    print(f"Dictionary keys: {list(sample_data.keys())}")
                    for key, value in sample_data.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: {value.shape}, dtype: {value.dtype}")
                        else:
                            print(f"  {key}: {type(value)}")
                elif hasattr(sample_data, 'shape'):
                    print(f"Tensor shape: {sample_data.shape}")
                    print(f"Tensor dtype: {sample_data.dtype}")
                    print(f"Value range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
                
            except Exception as e:
                print(f"Error analyzing {sample_file}: {e}")
    
    print("\n=== END ANALYSIS ===")

# ===== TRAINING UTILITIES =====

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def compute_class_weights(labels, num_classes=3):
    """Compute class weights for imbalanced datasets"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)

# ===== MODIFIED TRAINING FUNCTION =====
def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}', disable=not is_main_process())
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if is_main_process():
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def train_epoch_with_optimizer_switch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, total_epochs):
    """Train for one epoch with potential optimizer switching"""
    
    # Check if we need to switch optimizer due to unfreezing
    optimizer_changed = False
    if hasattr(model, 'module'):
        optimizer_changed = model.module.gradual_unfreezing(epoch, total_epochs)
    else:
        optimizer_changed = model.gradual_unfreezing(epoch, total_epochs)
    
    # Switch to SGD if fine-tuning phase started
    if optimizer_changed:
        new_optimizer, new_scheduler = switch_optimizer_to_sgd(model, optimizer, epoch, total_epochs)
        return train_epoch_with_optimizer_switch(model, train_loader, criterion, new_optimizer, scaler, device, epoch), new_optimizer, new_scheduler
    
    # Regular training with existing optimizer
    train_loss, train_acc = train_epoch_with_optimizer_switch(model, train_loader, criterion, optimizer, scaler, device, epoch)
    return (train_loss, train_acc), optimizer, scheduler
    
def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', disable=not is_main_process())
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if is_main_process():
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(all_targets, all_preds)
    
    return avg_loss, accuracy, mcc, all_preds, all_targets

def save_checkpoint(model, optimizer, scaler, epoch, val_loss, val_acc, mcc, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'mcc': mcc,
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, filepath)
    if is_main_process():
        print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, scaler, filepath, device):
    """Load training checkpoint"""
    try:
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler state if available
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        if is_main_process():
            print(f"Checkpoint loaded from {filepath}")
            print(f"Resuming from epoch {start_epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")
        
        return start_epoch, best_val_loss
        
    except FileNotFoundError:
        if is_main_process():
            print(f"No checkpoint found at {filepath}")
        return 0, float('inf')

def plot_training_history(train_losses, train_accs, val_losses, val_accs, mccs, save_path):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # MCC plot
    ax3.plot(epochs, mccs, 'g-', label='Matthews Correlation Coefficient')
    ax3.set_title('Matthews Correlation Coefficient')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MCC')
    ax3.legend()
    ax3.grid(True)
    
    # Combined metrics
    ax4.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, mccs, 'g-', label='MCC')
    ax4.set_title('Validation Metrics')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)', color='r')
    ax4_twin.set_ylabel('MCC', color='g')
    ax4.tick_params(axis='y', labelcolor='r')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, device, class_names=['real', 'semi-synthetic', 'synthetic']):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', disable=not is_main_process())
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_targets))
    mcc = matthews_corrcoef(all_targets, all_preds)
    
    # Classification report
    if is_main_process():
        print("\n=== FINAL EVALUATION RESULTS ===")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return accuracy, mcc, all_preds, all_targets, all_probs

# ===== COMPLETE MAIN FUNCTION =====

# ===== CORRECTED TRAINING FUNCTION =====

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}', disable=not is_main_process())
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if is_main_process():
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def train_epoch_with_optimizer_switch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, total_epochs):
    """Train for one epoch with potential optimizer switching"""
    
    # Check if we need to switch optimizer due to unfreezing
    optimizer_changed = False
    if hasattr(model, 'module'):
        optimizer_changed = model.module.gradual_unfreezing(epoch, total_epochs)
    else:
        optimizer_changed = model.gradual_unfreezing(epoch, total_epochs)
    
    # Switch to SGD if fine-tuning phase started
    if optimizer_changed:
        new_optimizer, new_scheduler = switch_optimizer_to_sgd(model, optimizer, epoch, total_epochs)
        # Train with new optimizer
        train_loss, train_acc = train_epoch(model, train_loader, criterion, new_optimizer, scaler, device, epoch)
        return (train_loss, train_acc), new_optimizer, new_scheduler
    
    # Regular training with existing optimizer
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
    return (train_loss, train_acc), optimizer, scheduler

# ===== COMPLETE CORRECTED MAIN FUNCTION =====

def complete_tensor_main_function_with_sgd_switch():
    """Complete main function with AdamW to SGD switching during fine-tuning"""
    parser = argparse.ArgumentParser(description='Multi-GPU Training with AdamW→SGD Switch')
    parser.add_argument('--data_root', type=str, default='datasets/train')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--total_images', type=int, default=240000)
    parser.add_argument('--sgd_lr_factor', type=float, default=0.1, 
                       help='LR reduction factor when switching to SGD (default: 0.1)')
    parser.add_argument('--checkpoint_dir', type=str, default='tensor_checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cache_tensors', action='store_true')
    parser.add_argument('--save_splits', type=str, default='tensor_data_splits.pkl')
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--analyze_tensors', action='store_true')
    parser.add_argument('--images_per_file', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--create_dummy_data', action='store_true')
    parser.add_argument('--dummy_samples_per_class', type=int, default=1000)
    parser.add_argument('--dummy_samples_per_file', type=int, default=100)
    parser.add_argument('--demo_mode', action='store_true')
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    
    # Adjust parameters for demo/quick test modes
    if args.demo_mode:
        args.num_epochs = min(args.num_epochs, 10)
        args.total_images = min(args.total_images, 3000)
        args.batch_size = min(args.batch_size, 16)
        if is_main_process():
            print("🚀 Demo mode activated: reduced epochs and dataset size for quick testing")
    
    if args.quick_test:
        args.num_epochs = min(args.num_epochs, 3)
        args.total_images = min(args.total_images, 300)
        args.batch_size = min(args.batch_size, 8)
        args.patience = 2
        if is_main_process():
            print("⚡ Quick test mode: minimal configuration for rapid testing")
    
    # Analyze tensor files if requested
    if args.analyze_tensors:
        if not os.path.exists(args.data_root):
            print(f"Error: Data root directory '{args.data_root}' does not exist!")
            print("Please create the directory or specify a different path with --data_root")
            return
        analyze_tensor_files(args.data_root)
        return
    
    # Check if data directory exists or create dummy data
    if not os.path.exists(args.data_root):
        if is_main_process():
            print(f"📁 Data directory '{args.data_root}' does not exist.")
            print("🤖 Creating dummy data automatically for demonstration...")
            
            if args.quick_test:
                samples_per_class = 100
                samples_per_file = 50
            elif args.demo_mode:
                samples_per_class = 300
                samples_per_file = 100
            else:
                samples_per_class = 1000
                samples_per_file = 200
            
            create_dummy_tensor_data(
                args.data_root, 
                samples_per_class, 
                samples_per_file
            )
            print("✅ Dummy data created successfully!")
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device('cpu')
        if is_main_process():
            print("CUDA not available, using CPU")
    
    if is_main_process():
        print("=== TRAINING WITH ADAMW → SGD SWITCH ===")
        print(f"📊 Data root: {args.data_root}")
        print(f"📊 Initial optimizer: AdamW (LR: {args.lr})")
        print(f"🔄 Switch to SGD at fine-tuning (LR factor: {args.sgd_lr_factor})")
        print(f"🌐 World size: {world_size}")
        print(f"📦 Batch size per GPU: {args.batch_size}")
        print(f"💾 Cache tensors: {args.cache_tensors}")
        print(f"⚡ Mixed precision: {args.mixed_precision}")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load or create tensor data splits
    if is_main_process():
        print("Loading tensor data information...")
        
        # Try to load existing splits first
        train_data, val_data, test_data = load_tensor_data_splits(args.save_splits)
        
        if train_data is None:
            # Create new splits
            tensor_info, labels = load_tensor_data_info(
                data_root=args.data_root,
                total_images=args.total_images,
                images_per_file=args.images_per_file
            )
            
            train_data, val_data, test_data = create_tensor_data_splits(
                tensor_info, labels,
                train_ratio=0.7,
                val_ratio=0.15, 
                test_ratio=0.15,
                stratify=True
            )
            
            # Save splits for future use
            save_tensor_data_splits(train_data, val_data, test_data, args.save_splits)
    else:
        # Non-main processes wait for data to be prepared
        train_data, val_data, test_data = None, None, None
    
    # Broadcast data to all processes
    if world_size > 1:
        data_to_broadcast = [train_data, val_data, test_data]
        dist.broadcast_object_list(data_to_broadcast, src=0)
        train_data, val_data, test_data = data_to_broadcast
    
    # Enhanced transforms for tensor data
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create tensor data loaders
    if is_main_process():
        print("Creating tensor data loaders...")
    
    train_loader, val_loader, test_loader = create_tensor_data_loaders(
        train_data, val_data, test_data,
        train_transform, val_test_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        cache_tensors=args.cache_tensors
    )
    
    # Initialize model
    if is_main_process():
        print("Initializing attention-enhanced hybrid model...")
    
    model = AttentionEnhancedConvNeXTViTHybrid(
        num_classes=3,
        freeze_convnext=True,
        freeze_vit=True,
        use_gradual_unfreezing=True,
        dropout_rate=0.3
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Setup loss function with class weights if requested
    if args.use_class_weights:
        train_labels = train_data[1]
        class_weights = compute_class_weights(train_labels, num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if is_main_process():
            print(f"Using class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Initial AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Initial scheduler for AdamW
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # Setup early stopping 
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scaler, args.resume, device
        )
    
    # Training variables
    train_losses, train_accs = [], []
    val_losses, val_accs, mccs = [], [], []
    optimizer_switch_epoch = None
    current_optimizer_type = "AdamW"
    
    if is_main_process():
        print("🚀 Starting training with AdamW...")
        print(f"🔄 Will switch to SGD when fine-tuning begins")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop with optimizer switching
    for epoch in range(start_epoch, args.num_epochs):
        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.num_epochs} [{current_optimizer_type}]")
            print("-" * 60)
        
        # Training with potential optimizer switching
        result, optimizer, scheduler = train_epoch_with_optimizer_switch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, args.num_epochs
        )
        train_loss, train_acc = result
        
        # Check if optimizer was switched
        if current_optimizer_type == "AdamW" and isinstance(optimizer, optim.SGD):
            current_optimizer_type = "SGD"
            optimizer_switch_epoch = epoch
            if is_main_process():
                print(f"✅ Successfully switched to SGD optimizer!")
        
        # Validation phase
        val_loss, val_acc, mcc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        mccs.append(mcc)
        
        if is_main_process():
            print(f"📈 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"📊 Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, MCC={mcc:.4f}")
            print(f"🎯 Optimizer: {current_optimizer_type}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    model, optimizer, scaler, epoch, val_loss, val_acc, mcc, best_model_path
                )
        
        # Save regular checkpoint
        if is_main_process() and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(
                model, optimizer, scaler, epoch, val_loss, val_acc, mcc, checkpoint_path
            )
        
        # Early stopping check
        if early_stopping(val_loss, model):
            if is_main_process():
                print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Plot training history
    if is_main_process():
        print("Plotting training history...")
        plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
        plot_training_history(
            train_losses, train_accs, val_losses, val_accs, mccs, plot_path
        )
    
    # Final evaluation on test set
    if is_main_process():
        print("Evaluating on test set...")
        
        # Load best model for evaluation
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            load_checkpoint(model, optimizer, scaler, best_model_path, device)
        
        test_acc, test_mcc, test_preds, test_targets, test_probs = evaluate_model(
            model, test_loader, device
        )
        
        # Save final results
        results = {
            'test_accuracy': test_acc,
            'test_mcc': test_mcc,
            'optimizer_switch_epoch': optimizer_switch_epoch,
            'training_history': {
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'mccs': mccs
            },
            'test_predictions': test_preds,
            'test_targets': test_targets,
            'test_probabilities': test_probs
        }
        
        results_path = os.path.join(args.checkpoint_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        
        print(f"Final results saved to {results_path}")
    
    # Final evaluation and results
    if is_main_process():
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED!")
        print("="*60)
        print(f"📊 Final Test Accuracy: {test_acc:.2f}%")
        print(f"📈 Final MCC Score: {test_mcc:.4f}")
        print(f"📊 Optimizer Timeline:")
        print(f"   • AdamW: Epochs 1-{optimizer_switch_epoch or args.num_epochs}")
        if optimizer_switch_epoch:
            print(f"   • SGD: Epochs {optimizer_switch_epoch+1}-{args.num_epochs}")
        print(f"🔄 Optimizer switch occurred at epoch: {optimizer_switch_epoch or 'Never'}")
        print(f"💾 Results saved in: {args.checkpoint_dir}/")
        print(f"🏆 Best model: {args.checkpoint_dir}/best_model.pth")
        print(f"📋 Full results: {args.checkpoint_dir}/final_results.json")
        print(f"📊 Training plots: {args.checkpoint_dir}/training_history.png")
        print("="*60)
        
        # Quick start message for next time
        if args.demo_mode or args.quick_test:
            print("\n💡 For full training, run:")
            print(f"   python script.py --data_root {args.data_root}")
        
        print("\n🔄 To resume or continue training:")
        print(f"   python script.py --resume {args.checkpoint_dir}/best_model.pth")
        print("\n✨ Happy training! ✨")
    
    # Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    complete_tensor_main_function_with_sgd_switch()