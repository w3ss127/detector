import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import timm
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from tqdm import tqdm
import random
import warnings
import logging
import argparse
from pathlib import Path
import gc
import socket
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import signal
import threading

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedConfig:
    """Configuration for deepfake detection training with progressive unfreezing"""
    def __init__(self):
        self.MODEL_TYPE = "enhanced_convnext_vit"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 1024
        self.DROPOUT_RATE = 0.3
        self.FREEZE_BACKBONES = True  # Start with frozen backbones
        self.ATTENTION_DROPOUT = 0.1
        self.USE_SPECTRAL_NORM = True
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DISTRIBUTED = torch.cuda.device_count() > 1
        self.BACKEND = "nccl"
        self.MASTER_ADDR = "localhost"
        self.MASTER_PORT = "12355"
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.USE_AMP = True
        self.TRAIN_PATH = "datasets/train"
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = 4
        
        # Progressive unfreezing configuration
        self.INITIAL_EPOCHS = 10  # Train with frozen backbones
        self.UNFREEZE_EPOCHS = [15, 25, 35]  # Epochs to unfreeze layers progressively
        self.FINE_TUNE_START_EPOCH = 15  # When to switch to SGD
        
        # Optimizer configurations
        self.ADAMW_LR = 1e-3
        self.SGD_LR = 1e-4  # Lower learning rate for fine-tuning
        self.SGD_MOMENTUM = 0.9
        self.WEIGHT_DECAY = 1e-2
        
        # Loss configuration
        self.FOCAL_ALPHA = [1.0, 2.0, 2.0]
        self.FOCAL_GAMMA = 2.0
        self.LABEL_SMOOTHING = 0.1
        
        # Checkpointing
        self.CHECKPOINT_DIR = "multi_checkpoints"
        self.CHECKPOINT_EVERY_N_EPOCHS = 5
        
        # MCC-based model selection
        self.USE_MCC_FOR_BEST_MODEL = True

    def validate(self):
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0, "Batch size must be positive"
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0, "Epochs must be positive"
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES, "Class names must match NUM_CLASSES"
        assert self.CONVNEXT_BACKBONE in ["convnext_tiny", "convnext_small"], "Unsupported backbone"
        assert self.FINE_TUNE_START_EPOCH < self.EPOCHS, "Fine-tune start epoch must be less than total epochs"
        assert all(epoch < self.EPOCHS for epoch in self.UNFREEZE_EPOCHS), "Unfreeze epochs must be within total epochs"

class ImprovedConfig(EnhancedConfig):
    """Enhanced configuration specifically for better semi-synthetic detection"""
    def __init__(self):
        super().__init__()
        
        # Adjusted focal loss - give more weight to semi-synthetic class
        self.FOCAL_ALPHA = [1.0, 4.0, 2.0]  # Increased weight for semi-synthetic
        self.FOCAL_GAMMA = 3.0  # Increased gamma for harder examples
        
        # More aggressive data augmentation for forensic artifact preservation
        self.USE_FORENSIC_AUGMENTATION = True
        
        # Class-specific learning adjustments
        self.CLASS_WEIGHTS = [1.0, 2.5, 1.5]  # Higher weight for semi-synthetic
        
        # Progressive unfreezing adjusted for better feature learning
        self.UNFREEZE_EPOCHS = [10, 20, 30, 40]  # Earlier and more gradual
        self.FINE_TUNE_START_EPOCH = 10  # Start fine-tuning earlier
        
        # Learning rate schedule adjustments
        self.ADAMW_LR = 5e-4  # Lower initial LR for more stable learning
        self.SGD_LR = 1e-5    # Much lower for fine-tuning
        
        # Regularization to prevent overfitting to easy examples
        self.DROPOUT_RATE = 0.4  # Increased dropout
        self.ATTENTION_DROPOUT = 0.2  # Increased attention dropout
        
        # Loss combination strategy
        self.USE_COMBINED_LOSS = True
        self.CONTRASTIVE_WEIGHT = 0.3  # Add contrastive loss component

class AdvancedLoss(nn.Module):
    """Loss function with focal loss"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.focal_alpha = torch.tensor(config.FOCAL_ALPHA, device=config.DEVICE)
        self.focal_gamma = config.FOCAL_GAMMA
        self.label_smoothing = config.LABEL_SMOOTHING

    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha[targets]
        return (alpha_t * (1 - pt) ** self.focal_gamma * ce_loss).mean()

    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)

class FixedAttentionModule(nn.Module):
    """Fixed attention module that properly handles feature dimensions"""
    def __init__(self, in_features, config=None):
        super().__init__()
        self.config = config or EnhancedConfig()
        self.in_features = in_features
        
        # Channel attention using global average pooling
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )
        
        # Apply spectral normalization if enabled
        if self.config.USE_SPECTRAL_NORM:
            self.channel_attention[0] = nn.utils.spectral_norm(self.channel_attention[0])
            self.channel_attention[3] = nn.utils.spectral_norm(self.channel_attention[3])

    def forward(self, x):
        # x should be [batch_size, features]
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        
        # Channel attention
        attention_weights = self.channel_attention(x)
        attended_features = x * attention_weights
        
        return attended_features

class CombinedLoss(nn.Module):
    """Combined loss function for better semi-synthetic detection"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.focal_loss = AdvancedLoss(config)
        self.class_weights = torch.tensor(config.CLASS_WEIGHTS, device=config.DEVICE)
        
    def contrastive_loss(self, features, labels):
        """Contrastive loss to better separate semi-synthetic from real"""
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.t())
        
        # Create label similarity matrix
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        # Positive pairs (same class)
        pos_mask = label_matrix.float()
        pos_loss = (1 - similarity_matrix) * pos_mask
        
        # Negative pairs (different class)
        neg_mask = ~label_matrix
        neg_loss = torch.clamp(similarity_matrix - 0.5, min=0) * neg_mask.float()
        
        return (pos_loss.sum() + neg_loss.sum()) / (batch_size * batch_size)
    
    def forward(self, logits, targets, features=None):
        # Primary focal loss
        focal_loss = self.focal_loss(logits, targets)
        
        # Weighted cross-entropy for class imbalance
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        total_loss = 0.7 * focal_loss + 0.3 * ce_loss
        
        # Add contrastive loss if features provided
        if features is not None and self.config.USE_COMBINED_LOSS:
            cont_loss = self.contrastive_loss(features, targets)
            total_loss += self.config.CONTRASTIVE_WEIGHT * cont_loss
        
        return total_loss

class EnhancedConvNextViTModel(nn.Module):
    """Hybrid ConvNeXt and ViT model with progressive unfreezing capability"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize backbones
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        else:
            raise ValueError(f"Unsupported ConvNeXt backbone: {config.CONVNEXT_BACKBONE}")
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Get feature dimensions
        convnext_features = 768  # ConvNeXt tiny features
        vit_features = self.vit.num_features
        
        # Initially freeze backbones if specified
        if config.FREEZE_BACKBONES:
            self.freeze_backbones()
        
        # Fixed fusion layers
        total_features = convnext_features + vit_features
        self.attention_module = FixedAttentionModule(total_features, config)
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        )
        
        if config.USE_SPECTRAL_NORM:
            self.fusion[0] = nn.utils.spectral_norm(self.fusion[0])
            self.fusion[3] = nn.utils.spectral_norm(self.fusion[3])

    def freeze_backbones(self):
        """Freeze backbone parameters"""
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        logger.info("Frozen backbone parameters")

    def unfreeze_convnext_layers(self, num_layers=None):
        """Progressively unfreeze ConvNeXt layers"""
        if num_layers is None:
            # Unfreeze all ConvNeXt layers
            for param in self.convnext.parameters():
                param.requires_grad = True
            logger.info("Unfrozen all ConvNeXt layers")
        else:
            # Unfreeze specific number of layers from the end
            layers = list(self.convnext.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True
            logger.info(f"Unfrozen last {num_layers} ConvNeXt layers")

    def unfreeze_vit_layers(self, num_layers=None):
        """Progressively unfreeze ViT layers"""
        if num_layers is None:
            # Unfreeze all ViT layers
            for param in self.vit.parameters():
                param.requires_grad = True
            logger.info("Unfrozen all ViT layers")
        else:
            # Unfreeze specific number of layers from the end
            layers = list(self.vit.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True
            logger.info(f"Unfrozen last {num_layers} ViT layers")

    def get_trainable_params(self):
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # ConvNeXt features
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        
        # ViT features
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]  # CLS token
        
        # Fuse features
        fused_features = torch.cat([convnext_feats, vit_feats], dim=1)
        attended_features = self.attention_module(fused_features)
        logits = self.fusion(attended_features)
        
        return logits

class ImprovedConvNextViTModel(EnhancedConvNextViTModel):
    """Enhanced model with better feature extraction for semi-synthetic detection"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Add additional forensic-aware layers
        self.forensic_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Update fusion layer to include forensic features
        convnext_features = 768  # ConvNeXt tiny features
        vit_features = self.vit.num_features
        forensic_features = 256
        
        total_features = convnext_features + vit_features + forensic_features
        
        self.attention_module = FixedAttentionModule(total_features, config)
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )
        
        if config.USE_SPECTRAL_NORM:
            self.fusion[0] = nn.utils.spectral_norm(self.fusion[0])
            self.fusion[3] = nn.utils.spectral_norm(self.fusion[3])
            self.fusion[6] = nn.utils.spectral_norm(self.fusion[6])
    
    def forward(self, x):
        # Original ConvNeXt and ViT features
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]
        
        # Additional forensic features from raw input
        forensic_feats = self.forensic_features(x)
        
        # Combine all features
        fused_features = torch.cat([convnext_feats, vit_feats, forensic_feats], dim=1)
        attended_features = self.attention_module(fused_features)
        
        logits = self.fusion(attended_features)
        
        # Return both logits and features for contrastive loss
        return logits, fused_features

class OptimizedCustomDatasetPT(Dataset):
    """Memory-efficient dataset for loading .pt files"""
    def __init__(self, root_dir, config, transform=None):
        self.root_dir = Path(root_dir)
        self.config = config
        # Ensure transform is always valid
        if transform is None:
            raise ValueError("Transform cannot be None. Provide a valid albumentations transform.")
        self.transform = transform
        self.class_names = config.CLASS_NAMES
        self.file_indices = []
        self.labels = []
        self._load_file_mapping()

    def _load_file_mapping(self):
        """Create mapping without loading actual image data"""
        logger.info(f"Creating file mapping from {self.root_dir}...")
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist")
                continue
            
            pt_files = list(class_dir.glob('*.pt'))
            logger.info(f"Found {len(pt_files)} .pt files in {class_name}")
            
            for pt_file in pt_files:
                try:
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    if isinstance(tensor_data, dict):
                        tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                    
                    num_images = tensor_data.shape[0]
                    for i in range(num_images):
                        self.file_indices.append((str(pt_file), i))
                        self.labels.append(class_idx)
                    
                    del tensor_data
                    
                except Exception as e:
                    logger.error(f"Error checking {pt_file}: {e}")
        
        logger.info(f"Total images found: {len(self.file_indices)}")
        if len(self.file_indices) == 0:
            raise ValueError(f"No valid data found in {self.root_dir}")

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        try:
            file_path, image_idx = self.file_indices[idx]
            label = self.labels[idx]
            
            tensor_data = torch.load(file_path, map_location='cpu')
            if isinstance(tensor_data, dict):
                tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
            
            image_tensor = tensor_data[image_idx]
            
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            if self.transform and hasattr(self.transform, '__call__'):
                # Convert to numpy for albumentations
                image_np = image_tensor.permute(1, 2, 0).numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                # Apply albumentations transform
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            else:
                # If no transform, ensure correct format
                logger.warning(f"No valid transform at index {idx}, returning raw tensor")
            
            return image_tensor, label
            
        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 0

class EnhancedDataAugmentation:
    """Data augmentation preserving forensic artifacts"""
    def __init__(self, config, is_training=True):
        self.config = config
        self.is_training = is_training

    def get_train_transforms(self):
        if not self.is_training:
            return A.Compose([
                A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        return A.Compose([
            A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class ForensicAwareAugmentation(EnhancedDataAugmentation):
    """Enhanced augmentation that preserves forensic artifacts while improving robustness"""
    
    def get_train_transforms(self):
        if not self.is_training:
            return A.Compose([
                A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        return A.Compose([
            A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            
            # Careful geometric augmentations that preserve artifacts
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3
            ),
            
            # Forensic-aware augmentations
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.3),
            
            # Compression artifacts simulation
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
            
            # Subtle color/brightness changes
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3
            ),
            
            # Final normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def calculate_mcc(y_true, y_pred, num_classes):
    """Calculate Matthews Correlation Coefficient for multiclass"""
    if num_classes == 2:
        return matthews_corrcoef(y_true, y_pred)
    else:
        # For multiclass, calculate MCC using confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate MCC for multiclass
        cov_ytyp = np.trace(cm)
        cov_ypyp = np.sum(cm)
        cov_ytyt = np.sum(cm)
        
        sum_yt_yp = np.sum(np.sum(cm, axis=1) * np.sum(cm, axis=0))
        sum_yt2 = np.sum(np.sum(cm, axis=1) ** 2)
        sum_yp2 = np.sum(np.sum(cm, axis=0) ** 2)
        
        numerator = cov_ytyp * cov_ypyp - sum_yt_yp
        denominator = np.sqrt((cov_ypyp**2 - sum_yp2) * (cov_ytyt**2 - sum_yt2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator

def create_enhanced_data_loaders(config, local_rank=-1):
    """Create data loaders with proper distributed handling"""
    try:
        augmentation_class = ForensicAwareAugmentation if config.USE_FORENSIC_AUGMENTATION else EnhancedDataAugmentation
        
        # Create base dataset
        full_dataset = OptimizedCustomDatasetPT(
            root_dir=config.TRAIN_PATH,
            config=config,
            transform=augmentation_class(config, is_training=True).get_train_transforms()
        )
        
        # Ensure dataset has data before proceeding
        if len(full_dataset) == 0:
            raise ValueError(f"No data found in {config.TRAIN_PATH}")
        
        logger.info(f"Rank {local_rank}: Found {len(full_dataset)} total samples")
        
        # Use deterministic split with generator
        generator = torch.Generator().manual_seed(42)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        # Get indices for splits
        indices = list(range(len(full_dataset)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        logger.info(f"Rank {local_rank}: Split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Create separate dataset instances with appropriate transforms
        train_transform = augmentation_class(config, is_training=True).get_train_transforms()
        val_test_transform = augmentation_class(config, is_training=False).get_train_transforms()
        
        # Create subset datasets
        from torch.utils.data import Subset
        
        train_dataset = Subset(
            OptimizedCustomDatasetPT(config.TRAIN_PATH, config, train_transform), 
            train_indices
        )
        val_dataset = Subset(
            OptimizedCustomDatasetPT(config.TRAIN_PATH, config, val_test_transform), 
            val_indices
        )
        test_dataset = Subset(
            OptimizedCustomDatasetPT(config.TRAIN_PATH, config, val_test_transform), 
            test_indices
        )
        
        # Create distributed sampler only for training
        train_sampler = None
        if config.DISTRIBUTED and local_rank != -1:
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=torch.cuda.device_count(), 
                rank=local_rank, 
                shuffle=True,
                seed=42
            )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=(train_sampler is None),
            sampler=train_sampler, 
            num_workers=config.NUM_WORKERS, 
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if config.NUM_WORKERS > 0 else False
        )
        
        # Validation and test loaders (no distributed sampling needed)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=config.NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=True if config.NUM_WORKERS > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=config.NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=True if config.NUM_WORKERS > 0 else False
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Error creating data loaders for rank {local_rank}: {e}")
        raise

def find_free_port(start_port=12355, max_attempts=100):
    """Find an available port starting from start_port"""
    port = start_port
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError as e:
            if e.errno == 98:  # Address already in use
                port += 1
                continue
            raise
    raise RuntimeError(f"No free port found after {max_attempts} attempts")

def setup_distributed(local_rank, world_size, backend='nccl', master_addr='localhost', master_port='12355'):
    """Initialize distributed training with proper error handling"""
    try:
        # Use the same port for all processes
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(local_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Set CUDA device before initializing process group
        torch.cuda.set_device(local_rank)
        
        # Initialize process group with timeout
        dist.init_process_group(
            backend=backend, 
            rank=local_rank, 
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout
        )
        
        logger.info(f"Distributed process group initialized for rank {local_rank}")
        return True
    except Exception as e:
        logger.error(f"Failed to setup distributed training for rank {local_rank}: {e}")
        return False

def cleanup_distributed():
    """Clean up distributed process group with proper error handling"""
    try:
        if dist.is_initialized():
            # Wait for all processes to reach this point
            dist.barrier()
            # Destroy the process group
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")
    except Exception as e:
        logger.warning(f"Error during distributed cleanup: {e}")
    
    # Clean up CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def cleanup_memory():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def evaluate_model(model, data_loader, criterion, config, device):
    """Evaluate model and return metrics including MCC"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with autocast(enabled=config.USE_AMP):
                # Handle both model types
                if hasattr(model, 'module'):
                    model_output = model.module(data)
                else:
                    model_output = model(data)
                
                # Handle different return types
                if isinstance(model_output, tuple):
                    output, features = model_output
                    loss = criterion(output, target, features)
                else:
                    output = model_output
                    loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    mcc = calculate_mcc(all_targets, all_predictions, config.NUM_CLASSES)
    
    return avg_loss, accuracy, mcc, all_predictions, all_targets

def create_optimizer(model, config, epoch):
    """Create optimizer based on training phase"""
    if epoch >= config.FINE_TUNE_START_EPOCH:
        # Use SGD for fine-tuning
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.SGD_LR,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        logger.info(f"Switched to SGD optimizer (lr={config.SGD_LR})")
    else:
        # Use AdamW for initial training
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.ADAMW_LR,
            weight_decay=config.WEIGHT_DECAY
        )
    return optimizer

def progressive_unfreeze(model, epoch, config):
    """Progressive unfreezing based on epoch"""
    if epoch in config.UNFREEZE_EPOCHS:
        unfreeze_stage = config.UNFREEZE_EPOCHS.index(epoch) + 1
        
        if unfreeze_stage == 1:
            # Unfreeze last few layers of both backbones
            model.unfreeze_convnext_layers(20)  # Unfreeze last 20 layers
            model.unfreeze_vit_layers(20)
            
        elif unfreeze_stage == 2:
            # Unfreeze more layers
            model.unfreeze_convnext_layers(50)
            model.unfreeze_vit_layers(50)
            
        elif unfreeze_stage == 3:
            # Unfreeze all layers
            model.unfreeze_convnext_layers()
            model.unfreeze_vit_layers()
        
        trainable_params = model.get_trainable_params()
        logger.info(f"Progressive unfreezing stage {unfreeze_stage}: {trainable_params:,} trainable parameters")
        
        return True  # Indicates that unfreezing occurred
    return False

def save_checkpoint(model, optimizer, scaler, epoch, val_mcc, config, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_mcc': val_mcc,
        'config': config
    }
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, scaler, filename, config):
    """Load model checkpoint"""
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_mcc = checkpoint['val_mcc']
    logger.info(f"Checkpoint loaded: {filename}, resuming from epoch {start_epoch}")
    return start_epoch, best_mcc

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    cleanup_distributed()
    cleanup_memory()
    exit(0)

def train_worker(local_rank, config, master_port):
    """Worker function for distributed training with proper error handling"""
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        logger.info(f"Starting train_worker for rank {local_rank}")
        
        # Set CUDA device immediately
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            config.DEVICE = torch.device(f'cuda:{local_rank}')
        
        # Set up distributed training
        if config.DISTRIBUTED:
            success = setup_distributed(local_rank, torch.cuda.device_count(), config.BACKEND, 
                                     config.MASTER_ADDR, master_port)
            if not success:
                logger.error(f"Failed to setup distributed training for rank {local_rank}")
                return
        
        logger.info(f"Distributed setup complete for rank {local_rank}")

        # Set seeds for reproducibility
        torch.manual_seed(42 + local_rank)
        np.random.seed(42 + local_rank)
        random.seed(42 + local_rank)
        
        # Create checkpoint directory
        if local_rank in [-1, 0]:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        # Add barrier to ensure all processes reach this point
        if config.DISTRIBUTED:
            dist.barrier()
        
        best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model_mcc.pth")
        
        # Create data loaders
        logger.info(f"Creating data loaders for rank {local_rank}")
        train_loader, val_loader, test_loader = create_enhanced_data_loaders(config, local_rank)
        logger.info(f"Data loaders created for rank {local_rank}")
        
        # Add another barrier after data loading
        if config.DISTRIBUTED:
            dist.barrier()
        
        # Initialize model
        logger.info(f"Initializing model for rank {local_rank}")
        model = ImprovedConvNextViTModel(config).to(config.DEVICE)
        logger.info(f"Model initialized for rank {local_rank}")
        
        if config.DISTRIBUTED:
            # Use find_unused_parameters=False for better performance and stability
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
        # Initialize loss, scaler
        criterion = CombinedLoss(config).to(config.DEVICE)
        scaler = GradScaler(enabled=config.USE_AMP)
        
        # Initialize tracking variables
        best_mcc = -1.0
        training_history = {
            'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_mcc': [],
            'optimizer_switches': [], 'unfreeze_epochs': []
        }
        
        # Log initial trainable parameters
        if local_rank in [-1, 0]:
            initial_trainable = model.module.get_trainable_params() if config.DISTRIBUTED else model.get_trainable_params()
            logger.info(f"Starting training with {initial_trainable:,} trainable parameters")
        
        current_optimizer = None
        
        # Add final barrier before training starts
        if config.DISTRIBUTED:
            dist.barrier()
        
        logger.info(f"Starting training loop for rank {local_rank}")
        
        for epoch in range(config.EPOCHS):
            start_time = time.time()
            
            # Progressive unfreezing
            unfroze_this_epoch = False
            model_for_unfreeze = model.module if config.DISTRIBUTED else model
            unfroze_this_epoch = progressive_unfreeze(model_for_unfreeze, epoch + 1, config)
            
            if unfroze_this_epoch and local_rank in [-1, 0]:
                training_history['unfreeze_epochs'].append(epoch + 1)
            
            # Create or recreate optimizer if needed
            if current_optimizer is None or unfroze_this_epoch or epoch == config.FINE_TUNE_START_EPOCH:
                current_optimizer = create_optimizer(model, config, epoch + 1)
                if epoch == config.FINE_TUNE_START_EPOCH and local_rank in [-1, 0]:
                    training_history['optimizer_switches'].append(epoch + 1)
            
            # Training phase
            model.train()
            train_loss = 0
            train_batches = 0
            
            # Set epoch for distributed sampler
            if config.DISTRIBUTED and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_loader, desc=f'Rank {local_rank} - Epoch {epoch+1}/{config.EPOCHS}', disable=local_rank not in [-1, 0])
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                try:
                    data, target = data.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
                    current_optimizer.zero_grad()
                    
                    with autocast(enabled=config.USE_AMP):
                        # Handle different model outputs
                        model_output = model(data)
                        if isinstance(model_output, tuple):
                            logits, features = model_output
                            loss = criterion(logits, target, features)
                        else:
                            logits = model_output
                            loss = criterion(logits, target)
                    
                    scaler.scale(loss).backward()
                    scaler.step(current_optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # Update progress bar
                    if local_rank in [-1, 0]:
                        progress_bar.set_postfix({
                            'Loss': f'{loss.item():.4f}',
                            'Avg Loss': f'{train_loss/train_batches:.4f}'
                        })
                    
                    # Memory cleanup every 100 batches
                    if batch_idx % 100 == 0:
                        cleanup_memory()
                        
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx} for rank {local_rank}: {e}")
                    continue
            
            # Synchronize before validation
            if config.DISTRIBUTED:
                dist.barrier()
            
            avg_train_loss = train_loss / max(train_batches, 1)  # Avoid division by zero
            if local_rank in [-1, 0]:
                training_history['train_loss'].append(avg_train_loss)
            
            # Validation phase (only on main process)
            if local_rank in [-1, 0]:
                try:
                    val_loss, val_acc, val_mcc, val_preds, val_targets = evaluate_model(
                        model, val_loader, criterion, config, config.DEVICE
                    )
                    
                    training_history['val_loss'].append(val_loss)
                    training_history['val_acc'].append(val_acc)
                    training_history['val_mcc'].append(val_mcc)
                    
                    epoch_time = time.time() - start_time
                    
                    # Log epoch results
                    logger.info(f"Epoch {epoch+1}/{config.EPOCHS} completed in {epoch_time:.2f}s")
                    logger.info(f"Train Loss: {avg_train_loss:.4f}")
                    logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MCC: {val_mcc:.4f}")
                    
                    # Save best model based on MCC
                    if config.USE_MCC_FOR_BEST_MODEL and val_mcc > best_mcc:
                        best_mcc = val_mcc
                        save_checkpoint(model, current_optimizer, scaler, epoch, val_mcc, config, best_model_path)
                        logger.info(f"New best model saved with MCC: {best_mcc:.4f}")
                    
                    # Save regular checkpoints
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
                    save_checkpoint(model, current_optimizer, scaler, epoch, val_mcc, config, checkpoint_path)
                        
                except Exception as e:
                    logger.error(f"Error during validation at epoch {epoch+1}: {e}")
            
            # Synchronize distributed processes
            if config.DISTRIBUTED:
                dist.barrier()
            
            # Memory cleanup after each epoch
            cleanup_memory()
        
        # Final evaluation on test set (only on main process)
        if local_rank in [-1, 0]:
            logger.info("\n" + "="*50)
            logger.info("TRAINING COMPLETED - FINAL EVALUATION")
            logger.info("="*50)
            
            # Load best model for final evaluation
            if os.path.exists(best_model_path):
                logger.info("Loading best model for final evaluation...")
                checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
                if config.DISTRIBUTED:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Best model loaded (MCC: {checkpoint['val_mcc']:.4f})")
            
            # Final test evaluation
            test_loss, test_acc, test_mcc, test_preds, test_targets = evaluate_model(
                model, test_loader, criterion, config, config.DEVICE
            )
            
            logger.info(f"\nFINAL TEST RESULTS:")
            logger.info(f"Test Loss: {test_loss:.4f}")
            logger.info(f"Test Accuracy: {test_acc:.4f}")
            logger.info(f"Test MCC: {test_mcc:.4f}")
            
            # Generate detailed classification report
            logger.info("\nDETAILED CLASSIFICATION REPORT:")
            class_report = classification_report(
                test_targets, test_preds, 
                target_names=config.CLASS_NAMES,
                digits=4
            )
            logger.info(f"\n{class_report}")
            
            # Generate confusion matrix
            cm = confusion_matrix(test_targets, test_preds)
            logger.info(f"\nCONFUSION MATRIX:")
            logger.info(f"Classes: {config.CLASS_NAMES}")
            logger.info(f"\n{cm}")
            
            # Training summary
            logger.info(f"\nTRAINING SUMMARY:")
            logger.info(f"Total epochs: {config.EPOCHS}")
            logger.info(f"Best validation MCC: {best_mcc:.4f}")
            logger.info(f"Final test MCC: {test_mcc:.4f}")
            logger.info(f"Optimizer switches at epochs: {training_history['optimizer_switches']}")
            logger.info(f"Progressive unfreezing at epochs: {training_history['unfreeze_epochs']}")
            
            # Save training history
            history_path = os.path.join(config.CHECKPOINT_DIR, "training_history.pt")
            torch.save(training_history, history_path)
            logger.info(f"Training history saved: {history_path}")
        
        # Cleanup
        if config.DISTRIBUTED:
            cleanup_distributed()
        cleanup_memory()
        
        if local_rank in [-1, 0]:
            logger.info("Training completed successfully!")
        
        return training_history
    
    except Exception as e:
        logger.error(f"Error in train_worker for rank {local_rank}: {e}")
        if config.DISTRIBUTED:
            cleanup_distributed()
        cleanup_memory()
        raise

def train_with_improved_loss(config):
    """Main training function with progressive unfreezing and MCC tracking"""
    return train_worker(-1, config, config.MASTER_PORT)

def get_training_recommendations():
    """Specific recommendations for improving semi-synthetic detection"""
    recommendations = {
        "data_strategy": [
            "Ensure balanced sampling - equal numbers of each class per batch",
            "Add more semi-synthetic examples if possible",
            "Use stratified validation split to maintain class balance",
            "Consider oversampling semi-synthetic class during training"
        ],
        
        "model_strategy": [
            "Train longer with lower learning rates",
            "Use earlier progressive unfreezing (epoch 10 vs 15)",
            "Add forensic-aware preprocessing layers",
            "Implement attention mechanisms focused on artifact detection"
        ],
        
        "loss_strategy": [
            "Increase focal loss weight for semi-synthetic class (alpha=3.0)",
            "Use higher focal gamma (3.0) to focus on hard examples",
            "Add contrastive loss to better separate classes",
            "Consider using ArcFace or similar metric learning losses"
        ],
        
        "evaluation_strategy": [
            "Monitor per-class precision/recall during training",
            "Use stratified K-fold validation",
            "Implement early stopping based on semi-synthetic class F1-score",
            "Analyze confusion patterns to identify specific failure modes"
        ]
    }
    return recommendations

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Deepfake Detection Training')
    parser.add_argument('--train_path', type=str, default='datasets/train', 
                      help='Path to training dataset')
    parser.add_argument('--batch_size', type=int, default=24, 
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=80, 
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, 
                      help='Initial learning rate for AdamW')
    parser.add_argument('--sgd_lr', type=float, default=1e-5, 
                      help='Learning rate for SGD fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=1e-2, 
                      help='Weight decay for regularization')
    parser.add_argument('--backbone', type=str, default='convnext_tiny', 
                      choices=['convnext_tiny', 'convnext_small'],
                      help='ConvNeXt backbone architecture')
    parser.add_argument('--image_size', type=int, default=224, 
                      help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, 
                      help='Number of data loader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='multi_checkpoints', 
                      help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, 
                      help='Path to checkpoint to resume from')
    parser.add_argument('--no_amp', action='store_true', 
                      help='Disable automatic mixed precision')
    parser.add_argument('--no_spectral_norm', action='store_true', 
                      help='Disable spectral normalization')
    parser.add_argument('--fine_tune_start', type=int, default=10, 
                      help='Epoch to start fine-tuning with SGD')
    parser.add_argument('--unfreeze_epochs', type=int, nargs='+', default=[10, 20, 30, 40], 
                      help='Epochs for progressive unfreezing')
    parser.add_argument('--focal_gamma', type=float, default=3.0, 
                      help='Gamma parameter for focal loss')
    parser.add_argument('--label_smoothing', type=float, default=0.1, 
                      help='Label smoothing factor')
    parser.add_argument('--no_distributed', action='store_true',
                      help='Force single GPU training even with multiple GPUs')
    
    args = parser.parse_args()
    
    # Create and validate configuration
    config = ImprovedConfig()

    # Update config with command line arguments
    config.TRAIN_PATH = args.train_path
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.ADAMW_LR = args.lr
    config.SGD_LR = args.sgd_lr
    config.WEIGHT_DECAY = args.weight_decay
    config.CONVNEXT_BACKBONE = args.backbone
    config.IMAGE_SIZE = args.image_size
    config.NUM_WORKERS = args.num_workers
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.USE_AMP = not args.no_amp
    config.USE_SPECTRAL_NORM = not args.no_spectral_norm
    config.FINE_TUNE_START_EPOCH = args.fine_tune_start
    config.UNFREEZE_EPOCHS = args.unfreeze_epochs
    config.FOCAL_GAMMA = args.focal_gamma
    config.LABEL_SMOOTHING = args.label_smoothing
    
    # Handle distributed training flag
    if args.no_distributed:
        config.DISTRIBUTED = False
    
    # Validate configuration
    try:
        config.validate()
        logger.info("Configuration validated successfully")
    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    # Log configuration
    logger.info("="*50)
    logger.info("ENHANCED DEEPFAKE DETECTION TRAINING")
    logger.info("="*50)
    logger.info(f"Model Type: {config.MODEL_TYPE}")
    logger.info(f"Backbone: {config.CONVNEXT_BACKBONE}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Distributed: {config.DISTRIBUTED}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Image Size: {config.IMAGE_SIZE}")
    logger.info(f"Use AMP: {config.USE_AMP}")
    logger.info(f"Use Spectral Norm: {config.USE_SPECTRAL_NORM}")
    logger.info(f"Progressive Unfreezing: {config.UNFREEZE_EPOCHS}")
    logger.info(f"Fine-tuning Start: {config.FINE_TUNE_START_EPOCH}")
    logger.info("="*50)
    
    # Start training
    try:
        if config.DISTRIBUTED and torch.cuda.device_count() > 1:
            logger.info("Starting distributed training...")
            # Find a base port that's available
            base_port = find_free_port(int(config.MASTER_PORT))
            logger.info(f"Using port: {base_port}")
            
            world_size = torch.cuda.device_count()
            
            # Use mp.spawn instead of manual process creation
            mp.spawn(
                train_worker,
                args=(config, str(base_port)),
                nprocs=world_size,
                join=True
            )
                
        else:
            logger.info("Starting single-GPU training...")
            train_with_improved_loss(config)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        cleanup_distributed()
        cleanup_memory()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        cleanup_distributed()
        cleanup_memory()
        raise

if __name__ == "__main__":
    main()