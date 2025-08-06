import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import timm
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, precision_recall_fscore_support
from tqdm import tqdm
import random
import warnings
import logging
import argparse
from pathlib import Path
import gc
import socket
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import signal
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
from PIL import Image
import psutil

warnings.filterwarnings('ignore')

def find_free_port():
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return str(port)

def setup_logging():
    """Setup logging for distributed training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [Process %(process)d] - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def get_optimal_batch_size(device_memory_gb, num_gpus):
    """Calculate optimal batch size based on available GPU memory"""
    # Conservative memory allocation
    if device_memory_gb >= 24:  # RTX 4090, A100, etc.
        base_batch_size = 32
    elif device_memory_gb >= 16:  # RTX 4080, etc.
        base_batch_size = 24
    elif device_memory_gb >= 12:  # RTX 4070 Ti, etc.
        base_batch_size = 16
    elif device_memory_gb >= 8:   # RTX 4060 Ti, etc.
        base_batch_size = 12
    else:
        base_batch_size = 8
    
    # Scale down for multiple GPUs to prevent memory issues
    if num_gpus >= 8:
        base_batch_size = max(8, base_batch_size // 2)
    elif num_gpus >= 4:
        base_batch_size = max(12, int(base_batch_size * 0.75))
    
    return base_batch_size

class SuperiorConfig:
    def __init__(self):
        self.MODEL_TYPE = "superior_forensics_model"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        
        # Binary configuration settings
        self.USE_BINARY_MODE = True
        self.ORDINAL_REGRESSION = True
        self.NUM_CLASSES = 3
        self.REALNESS_SCORE_MODE = True
        
        self.HIDDEN_DIM = 1024  # Reduced from 1536
        self.DROPOUT_RATE = 0.3  # Reduced from 0.4
        self.FREEZE_BACKBONES = True
        self.ATTENTION_DROPOUT = 0.2
        self.USE_SPECTRAL_NORM = False  # Disabled to save memory
        
        # FIXED: Better distributed settings
        self.DISTRIBUTED = torch.cuda.device_count() > 1
        self.BACKEND = "nccl"
        self.MASTER_ADDR = "127.0.0.1"
        self.MASTER_PORT = find_free_port()
        self.WORLD_SIZE = torch.cuda.device_count()
        
        # OPTIMIZED: Dynamic batch size based on GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 16
        # self.BATCH_SIZE = get_optimal_batch_size(gpu_memory, self.WORLD_SIZE)
        self.BATCH_SIZE = 24
        
        self.EPOCHS = 50
        self.USE_AMP = True
        self.TRAIN_PATH = "datasets/train"
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["synthetic", "semi-synthetic", "real"]
        
        # FIXED: Conservative worker settings to prevent crashes
        self.NUM_WORKERS = min(2, max(1, psutil.cpu_count() // max(1, self.WORLD_SIZE)))
        self.PREFETCH_FACTOR = 2
        self.PERSISTENT_WORKERS = False  # Disabled to prevent worker issues
        
        self.UNFREEZE_EPOCHS = [1, 6, 16, 26, 36]
        self.FINE_TUNE_START_EPOCH = 26
        self.EARLY_STOPPING_PATIENCE = 40
        
        # OPTIMIZED: Conservative learning rates
        self.ADAMW_LR = 5e-4  # Reduced
        self.SGD_LR = 1e-5    # Reduced
        self.SGD_MOMENTUM = 0.9
        self.WEIGHT_DECAY = 1e-3  # Reduced
        
        # Ordinal regression specific settings
        self.ORDINAL_WEIGHT = 0.7
        self.BINARY_WEIGHT = 0.3
        self.CUTPOINT_REGULARIZATION = 0.01
        self.TEMPERATURE = 1.0
        
        # Loss weights
        self.FOCAL_ALPHA = [1.0, 2.0, 1.5]
        self.FOCAL_GAMMA = 2.0
        self.LABEL_SMOOTHING = 0.1
        self.CLASS_WEIGHTS = [1.0, 2.0, 1.0]
        
        self.USE_FORENSICS_MODULE = True
        self.USE_UNCERTAINTY_ESTIMATION = False  # Disabled to save memory
        self.USE_MIXUP = False
        self.MIXUP_ALPHA = 0.3
        self.USE_CUTMIX = False
        self.CUTMIX_ALPHA = 1.0
        self.USE_ENSEMBLE = False
        self.ENSEMBLE_SIZE = 3
        self.CONTRASTIVE_WEIGHT = 0.1  # Reduced
        self.EVIDENTIAL_WEIGHT = 0.1   # Reduced
        self.BOUNDARY_WEIGHT = 0.1
        self.TRIPLET_WEIGHT = 0.05     # Reduced
        self.CHECKPOINT_DIR = "ordinal_checkpoints"
        self.CHECKPOINT_EVERY_N_EPOCHS = 5
        self.USE_MCC_FOR_BEST_MODEL = True
        self.SAVE_TOP_K_MODELS = 3
        
        # FIXED: Memory optimization settings
        self.SYNC_BN = True
        self.GRADIENT_CLIP_VAL = 1.0
        self.GRADIENT_ACCUMULATION_STEPS = 1
        self.COMPILE_MODEL = False  # Disabled due to memory issues
        self.CHANNELS_LAST = False  # Disabled due to compatibility issues
        
        # Memory management settings
        self.MEMORY_FRACTION = 0.85  # Use 85% of GPU memory per process
        self.EMPTY_CACHE_STEPS = 10  # Clear cache every N steps
        self.MAX_SPLIT_SIZE_MB = 512  # Limit memory fragmentation

    def validate(self):
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0, "Batch size must be positive"
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0, "Epochs must be positive"
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES, "Class names must match NUM_CLASSES"
        assert self.CONVNEXT_BACKBONE in ["convnext_tiny", "convnext_small"], "Unsupported backbone"
        assert self.FINE_TUNE_START_EPOCH < self.EPOCHS, "Fine-tune start epoch must be less than total epochs"
        assert all(epoch <= self.EPOCHS for epoch in self.UNFREEZE_EPOCHS), "Unfreeze epochs must be within total epochs"

class ForensicsAwareModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Simplified and memory-efficient forensics module
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=8),  # Reduced channels
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Reduced channels
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # Smaller output
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),  # Reduced size
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.noise_analyzer = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),   # Reduced channels
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Reduced channels
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Smaller output
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),   # Reduced size
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.forensics_fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),  # Simplified fusion
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64)  # Final output size
        )
    
    def forward(self, x):
        dct_feats = self.dct_analyzer(x)
        noise_feats = self.noise_analyzer(x)
        combined_feats = torch.cat([dct_feats, noise_feats], dim=1)
        forensics_output = self.forensics_fusion(combined_feats)
        return forensics_output

class OrdinalRegressionLoss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.device = device
        self.ordinal_weight = config.ORDINAL_WEIGHT
        self.binary_weight = config.BINARY_WEIGHT
        self.cutpoint_reg = config.CUTPOINT_REGULARIZATION
        self.temperature = config.TEMPERATURE
        
    def ordinal_loss(self, scores, cutpoints, targets):
        """Ordinal regression loss using learnable cutpoints"""
        batch_size = scores.size(0)
        num_cutpoints = cutpoints.size(0)
        
        # Sort cutpoints for stability
        cutpoints_sorted = torch.sort(cutpoints)[0]
        
        # Expand dimensions for broadcasting
        scores_expanded = scores.expand(-1, num_cutpoints)
        cutpoints_expanded = cutpoints_sorted.unsqueeze(0).expand(batch_size, -1)
        
        # Calculate probabilities for each ordinal level
        ordinal_probs = torch.sigmoid((scores_expanded - cutpoints_expanded) / self.temperature)
        
        # Convert to class probabilities
        class_probs = torch.zeros(batch_size, self.num_classes, device=self.device)
        
        class_probs[:, 0] = 1 - ordinal_probs[:, 0]
        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = ordinal_probs[:, k-1] - ordinal_probs[:, k]
        class_probs[:, -1] = ordinal_probs[:, -1]
        
        # Add small epsilon for numerical stability
        class_probs = torch.clamp(class_probs, min=1e-7, max=1-1e-7)
        
        # Cross-entropy loss
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        ce_loss = -torch.sum(targets_one_hot * torch.log(class_probs), dim=1)
        
        return ce_loss.mean(), class_probs
    
    def binary_loss(self, scores, targets):
        """Binary loss treating real vs synthetic"""
        binary_targets = targets.float() / (self.num_classes - 1)
        scores_sigmoid = torch.sigmoid(scores.squeeze())
        mse_loss = F.mse_loss(scores_sigmoid, binary_targets)
        return mse_loss
    
    def cutpoint_regularization(self, cutpoints):
        """Regularization to ensure cutpoints are ordered"""
        if len(cutpoints) < 2:
            return torch.tensor(0.0, device=self.device)
        
        cutpoints_sorted = torch.sort(cutpoints)[0]
        ordering_penalty = torch.sum(F.relu(cutpoints[:-1] - cutpoints[1:]))
        min_distance = 0.5
        distance_penalty = torch.sum(F.relu(min_distance - (cutpoints_sorted[1:] - cutpoints_sorted[:-1])))
        
        return ordering_penalty + distance_penalty
    
    def forward(self, scores, cutpoints, targets):
        ordinal_loss, class_probs = self.ordinal_loss(scores, cutpoints, targets)
        binary_loss = self.binary_loss(scores, targets)
        reg_loss = self.cutpoint_regularization(cutpoints)
        
        total_loss = (self.ordinal_weight * ordinal_loss + 
                    self.binary_weight * binary_loss + 
                    self.cutpoint_reg * reg_loss)
        
        return total_loss, class_probs, {
            'ordinal_loss': ordinal_loss.item(),
            'binary_loss': binary_loss.item(),
            'reg_loss': reg_loss.item()
        }

class SuperiorLoss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.device = device
        
        self.focal_alpha = torch.tensor(config.FOCAL_ALPHA, device=device, dtype=torch.float32)
        self.focal_gamma = config.FOCAL_GAMMA
        self.class_weights = torch.tensor(config.CLASS_WEIGHTS, device=device, dtype=torch.float32)
        
        self.contrastive_weight = config.CONTRASTIVE_WEIGHT
        self.boundary_weight = config.BOUNDARY_WEIGHT
        self.triplet_weight = config.TRIPLET_WEIGHT
        self.label_smoothing = config.LABEL_SMOOTHING
        self.triplet_margin = 1.0
        
        self.ordinal_loss = OrdinalRegressionLoss(config, device)
        self.use_ordinal = config.ORDINAL_REGRESSION
    
    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def contrastive_loss(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features, features.t())
        labels_expanded = labels.unsqueeze(1)
        same_class_mask = (labels_expanded == labels_expanded.t()).float()
        diff_class_mask = 1 - same_class_mask
        
        pos_loss = same_class_mask * (1 - similarity_matrix) ** 2
        neg_loss = diff_class_mask * torch.clamp(similarity_matrix - 0.3, min=0) ** 2
        
        return (pos_loss.sum() + neg_loss.sum()) / (labels.size(0) ** 2)
    
    def boundary_loss(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        real_mask = (targets == 2).float()
        synthetic_mask = (targets == 0).float()
        semi_mask = (targets == 1).float()
        
        real_boundary = real_mask * probs[:, 1]
        synthetic_boundary = synthetic_mask * probs[:, 1]
        semi_boundary = semi_mask * (1 - probs[:, 1])
        
        return (real_boundary.sum() + synthetic_boundary.sum() + 2.0 * semi_boundary.sum()) / targets.size(0)
    
    def forward(self, model_output, targets, features=None, epoch=1):
        if self.use_ordinal and len(model_output) >= 3:
            logits, realness_scores, cutpoints = model_output[:3]
            
            ordinal_loss, ordinal_probs, loss_dict = self.ordinal_loss(realness_scores, cutpoints, targets)
            focal_loss = self.focal_loss(logits, targets) * 0.3
            ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights) * 0.2
            
            total_loss = ordinal_loss + focal_loss + ce_loss
            
            if features is not None:
                contrastive_loss = self.contrastive_loss(features, targets)
                boundary_loss = self.boundary_loss(logits, targets)
                total_loss += (self.contrastive_weight * contrastive_loss + 
                            self.boundary_weight * boundary_loss)
            
            return total_loss
        else:
            logits = model_output[0] if isinstance(model_output, tuple) else model_output
            focal_loss = self.focal_loss(logits, targets)
            ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
            boundary_loss = self.boundary_loss(logits, targets)
            total_loss = 0.5 * focal_loss + 0.3 * ce_loss + self.boundary_weight * boundary_loss
            
            if features is not None:
                contrastive_loss = self.contrastive_loss(features, targets)
                total_loss += self.contrastive_weight * contrastive_loss
            
            return total_loss

class SuperiorAttentionModule(nn.Module):
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        
        # Simplified attention module
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 8),  # Reduced bottleneck
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() != 2:
            x = x.view(batch_size, -1)
        
        channel_weights = self.channel_attention(x)
        attended_features = x * channel_weights
        return attended_features

class SuperiorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize backbones with memory optimization
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        
        # Use smaller ViT model for memory efficiency
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        
        if config.FREEZE_BACKBONES:
            self.freeze_backbones()
        
        # Calculate feature dimensions
        convnext_features = 768
        vit_features = self.vit.num_features
        forensics_features = 64 if config.USE_FORENSICS_MODULE else 0
        total_features = convnext_features + vit_features + forensics_features
        
        # Initialize modules
        if config.USE_FORENSICS_MODULE:
            self.forensics_module = ForensicsAwareModule(config)
        
        self.attention_module = SuperiorAttentionModule(total_features, config)
        
        # Simplified fusion network
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
        # Traditional classifier
        self.classifier = nn.Linear(config.HIDDEN_DIM // 4, config.NUM_CLASSES)
        
        # Ordinal regression components
        if config.ORDINAL_REGRESSION:
            self.realness_predictor = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM // 4, config.HIDDEN_DIM // 8),
                nn.GELU(),
                nn.Dropout(config.DROPOUT_RATE * 0.5),
                nn.Linear(config.HIDDEN_DIM // 8, 1)
            )
            
            # Initialize cutpoints
            initial_cutpoints = torch.linspace(-1.0, 1.0, config.NUM_CLASSES - 1)
            self.cutpoints = nn.Parameter(initial_cutpoints.clone())
    
    def freeze_backbones(self):
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze backbone networks for fine-tuning"""
        for param in self.convnext.parameters():
            param.requires_grad = True
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features from backbones
        convnext_features = self.convnext.features(x)
        convnext_features = self.convnext.avgpool(convnext_features).flatten(1)
        
        vit_features = self.vit(x)
        
        # Combine features
        features_list = [convnext_features, vit_features]
        
        # Add forensics features if enabled
        if self.config.USE_FORENSICS_MODULE:
            forensics_features = self.forensics_module(x)
            features_list.append(forensics_features)
        
        # Concatenate all features
        combined_features = torch.cat(features_list, dim=1)
        
        # Apply attention
        attended_features = self.attention_module(combined_features)
        
        # Feature fusion
        fused_features = self.fusion(attended_features)
        
        # Traditional classification
        logits = self.classifier(fused_features)
        
        outputs = [logits]
        
        # Ordinal regression if enabled
        if self.config.ORDINAL_REGRESSION:
            realness_scores = self.realness_predictor(fused_features)
            outputs.extend([realness_scores, self.cutpoints])
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

class OptimizedForensicsDataset(Dataset):
    """Memory-optimized dataset with improved error handling"""
    def __init__(self, data_path, config, transform=None, is_train=True):
        self.data_path = Path(data_path)
        self.config = config
        self.transform = transform
        self.is_train = is_train
        
        # Build dataset from .pt files
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
        
        # Scan for .pt files in each class directory
        for class_name in config.CLASS_NAMES:
            class_path = self.data_path / class_name
            if class_path.exists():
                pt_files = list(class_path.glob("*.pt"))
                print(f"Found {len(pt_files)} .pt files in {class_path}")
                
                for pt_file in pt_files:
                    try:
                        # Quick check of tensor file structure
                        tensor_data = torch.load(pt_file, map_location='cpu')
                        
                        if isinstance(tensor_data, torch.Tensor):
                            num_images = tensor_data.size(0)
                        elif isinstance(tensor_data, (list, tuple)):
                            num_images = len(tensor_data)
                        elif isinstance(tensor_data, dict) and 'images' in tensor_data:
                            if isinstance(tensor_data['images'], torch.Tensor):
                                num_images = tensor_data['images'].size(0)
                            else:
                                num_images = len(tensor_data['images'])
                        else:
                            print(f"Unknown tensor format in {pt_file}, skipping...")
                            del tensor_data
                            continue
                        
                        # Add each image in the tensor file as a separate sample
                        for img_idx in range(num_images):
                            self.samples.append((str(pt_file), img_idx, self.class_to_idx[class_name]))
                        
                        print(f"Added {num_images} images from {pt_file}")
                        del tensor_data  # Free memory immediately
                        
                    except Exception as e:
                        print(f"Error processing {pt_file}: {e}")
                        continue
        
        print(f"Total samples found: {len(self.samples)}")
        
        # Class distribution
        class_counts = defaultdict(int)
        for _, _, label in self.samples:
            class_counts[label] += 1
        
        for class_idx, count in class_counts.items():
            print(f"Class {config.CLASS_NAMES[class_idx]}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pt_file_path, img_idx, label = self.samples[idx]
        
        try:
            # Load tensor file
            tensor_data = torch.load(pt_file_path, map_location='cpu')
            
            # Extract the specific image tensor
            if isinstance(tensor_data, torch.Tensor):
                if tensor_data.dim() == 4:
                    image_tensor = tensor_data[img_idx]
                else:
                    raise ValueError("Unexpected tensor dimensions")
                    
            elif isinstance(tensor_data, (list, tuple)):
                image_tensor = tensor_data[img_idx]
                
            elif isinstance(tensor_data, dict):
                if 'images' in tensor_data:
                    if isinstance(tensor_data['images'], torch.Tensor):
                        image_tensor = tensor_data['images'][img_idx]
                    else:
                        image_tensor = tensor_data['images'][img_idx]
                elif 'data' in tensor_data:
                    if isinstance(tensor_data['data'], torch.Tensor):
                        image_tensor = tensor_data['data'][img_idx]
                    else:
                        image_tensor = tensor_data['data'][img_idx]
                else:
                    raise ValueError("Unknown dict structure")
            else:
                raise ValueError("Unknown tensor format")
            
            # Clean up
            del tensor_data
            
            # Ensure tensor is float and in correct range
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            
            # Normalize to [0, 1] if needed
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            # Ensure correct shape [C, H, W]
            if image_tensor.dim() == 3:
                if image_tensor.shape[0] not in [1, 3]:
                    if image_tensor.shape[-1] in [1, 3]:
                        image_tensor = image_tensor.permute(2, 0, 1)
            elif image_tensor.dim() == 2:
                image_tensor = image_tensor.unsqueeze(0)
                if image_tensor.shape[0] == 1:
                    image_tensor = image_tensor.repeat(3, 1, 1)
            
            # Resize if needed
            if image_tensor.shape[-2:] != (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE):
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0), 
                    size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Apply transforms
            if self.transform:
                image_np = image_tensor.permute(1, 2, 0).numpy()
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            else:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                image_tensor = normalize(image_tensor)
            
            return image_tensor, label
            
        except Exception as e:
            print(f"Error processing image {img_idx} from {pt_file_path}: {e}")
            # Return random noise as fallback
            image_tensor = torch.randn(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE) * 0.1
            
            if self.transform:
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_np = np.clip(image_np, 0, 255)
                transformed = self.transform(image=image_np)
                return transformed['image'], label
            else:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                return normalize(image_tensor), label

def get_transforms(config, is_train=True):
    """Memory-efficient transforms"""
    if is_train:
        transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.4),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            ], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform

def setup_distributed(rank, world_size, config):
    """Improved distributed setup with proper memory management"""
    try:
        # Set CUDA device for this process BEFORE initializing process group
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        # Set memory fraction per process
        torch.cuda.set_per_process_memory_fraction(config.MEMORY_FRACTION, device=rank)
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = config.MASTER_ADDR
        os.environ['MASTER_PORT'] = config.MASTER_PORT
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        
        # Initialize process group with timeout
        timeout = torch.distributed.default_pg_timeout * 2
        
        dist.init_process_group(
            backend=config.BACKEND, 
            init_method=f'tcp://{config.MASTER_ADDR}:{config.MASTER_PORT}',
            world_size=world_size, 
            rank=rank,
            timeout=timeout
        )
        
        # Verify the device is set correctly
        assert torch.cuda.current_device() == rank, f"CUDA device mismatch: expected {rank}, got {torch.cuda.current_device()}"
        
        # Synchronize all processes
        dist.barrier()
        
        print(f"Successfully initialized distributed training on rank {rank}, device cuda:{rank}")
        
    except Exception as e:
        print(f"Failed to setup distributed training on rank {rank}: {e}")
        raise

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
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
        if hasattr(model, 'module'):
            self.best_weights = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
        else:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

def calculate_metrics(y_true, y_pred, y_scores=None, config=None):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Basic metrics
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)
    metrics['classification_report'] = report
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    metrics['mcc'] = mcc
    
    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics['per_class_precision'] = precision
    metrics['per_class_recall'] = recall
    metrics['per_class_f1'] = f1
    metrics['per_class_support'] = support
    
    # Overall metrics
    metrics['macro_f1'] = f1.mean()
    metrics['weighted_f1'] = report['weighted avg']['f1-score']
    metrics['accuracy'] = report['accuracy']
    
    # Ordinal-specific metrics if scores provided
    if y_scores is not None:
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = mae
        
        ordinal_acc = np.mean(np.abs(y_true - y_pred) <= 1)
        metrics['ordinal_accuracy'] = ordinal_acc
    
    return metrics

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes for distributed training"""
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def train_epoch(model, dataloader, criterion, optimizer, scaler, config, epoch, rank=0, world_size=1):
    """Memory-optimized training function"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_realness_scores = []
    
    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    else:
        progress_bar = dataloader
    
    accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.cuda(rank, non_blocking=True)
        targets = targets.cuda(rank, non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast(enabled=config.USE_AMP):
            output = model(images)
            
            # Handle different output formats
            if isinstance(output, tuple) and len(output) >= 3:
                logits, realness_scores, cutpoints = output[:3]
                
                loss = criterion((logits, realness_scores, cutpoints), targets)
                
                # Get predictions from realness scores using cutpoints
                with torch.no_grad():
                    cutpoints_sorted = torch.sort(cutpoints)[0]
                    scores_expanded = realness_scores.expand(-1, len(cutpoints_sorted))
                    cutpoints_expanded = cutpoints_sorted.unsqueeze(0).expand(len(realness_scores), -1)
                    ordinal_probs = torch.sigmoid(scores_expanded - cutpoints_expanded)
                    
                    class_probs = torch.zeros(len(realness_scores), config.NUM_CLASSES, device=images.device)
                    class_probs[:, 0] = 1 - ordinal_probs[:, 0]
                    for k in range(1, config.NUM_CLASSES - 1):
                        class_probs[:, k] = ordinal_probs[:, k-1] - ordinal_probs[:, k]
                    class_probs[:, -1] = ordinal_probs[:, -1]
                    
                    predictions = torch.argmax(class_probs, dim=1)
                    
                    all_realness_scores.extend(torch.sigmoid(realness_scores.squeeze()).cpu().numpy())
            else:
                logits = output if not isinstance(output, tuple) else output[0]
                loss = criterion(logits, targets)
                predictions = torch.argmax(logits, dim=1)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
        
        # Backward pass
        if config.USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation and clipping
        if (batch_idx + 1) % accumulation_steps == 0:
            if config.USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Collect metrics
        total_loss += loss.item() * accumulation_steps
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            progress_bar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        # Memory cleanup
        if batch_idx % config.EMPTY_CACHE_STEPS == 0:
            torch.cuda.empty_cache()
    
    # Handle any remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        if config.USE_AMP:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    
    # Synchronize metrics across all processes
    if world_size > 1:
        # Convert lists to tensors for reduction
        all_preds_tensor = torch.tensor(all_predictions, dtype=torch.float32).cuda(rank)
        all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32).cuda(rank)
        
        # Gather all predictions and targets from all processes
        gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(all_targets_tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered_preds, all_preds_tensor)
        dist.all_gather(gathered_targets, all_targets_tensor)
        
        # Flatten and convert back to numpy (only on rank 0)
        if rank == 0:
            all_predictions = torch.cat(gathered_preds).cpu().numpy().astype(int)
            all_targets = torch.cat(gathered_targets).cpu().numpy().astype(int)
        
        # Reduce loss
        avg_loss_tensor = torch.tensor(avg_loss).cuda(rank)
        reduced_loss = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = reduced_loss.item()
    
    # Calculate metrics (only on rank 0)
    if rank == 0:
        metrics = calculate_metrics(all_targets, all_predictions, 
                                all_realness_scores if all_realness_scores else None, config)
    else:
        metrics = {}
    
    return avg_loss, metrics

def validate_epoch(model, dataloader, criterion, config, epoch, rank=0, world_size=1):
    """Validate for one epoch with memory optimization"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_realness_scores = []
    
    with torch.no_grad():
        if rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Validation")
        else:
            progress_bar = dataloader
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.cuda(rank, non_blocking=True)
            targets = targets.cuda(rank, non_blocking=True)
            
            with autocast(enabled=config.USE_AMP):
                output = model(images)
                
                if isinstance(output, tuple) and len(output) >= 3:
                    logits, realness_scores, cutpoints = output[:3]
                    
                    loss = criterion((logits, realness_scores, cutpoints), targets)
                    
                    cutpoints_sorted = torch.sort(cutpoints)[0]
                    scores_expanded = realness_scores.expand(-1, len(cutpoints_sorted))
                    cutpoints_expanded = cutpoints_sorted.unsqueeze(0).expand(len(realness_scores), -1)
                    ordinal_probs = torch.sigmoid(scores_expanded - cutpoints_expanded)
                    
                    class_probs = torch.zeros(len(realness_scores), config.NUM_CLASSES, device=images.device)
                    class_probs[:, 0] = 1 - ordinal_probs[:, 0]
                    for k in range(1, config.NUM_CLASSES - 1):
                        class_probs[:, k] = ordinal_probs[:, k-1] - ordinal_probs[:, k]
                    class_probs[:, -1] = ordinal_probs[:, -1]
                    
                    predictions = torch.argmax(class_probs, dim=1)
                    
                    all_realness_scores.extend(torch.sigmoid(realness_scores.squeeze()).cpu().numpy())
                else:
                    logits = output if not isinstance(output, tuple) else output[0]
                    loss = criterion(logits, targets)
                    predictions = torch.argmax(logits, dim=1)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if rank == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Memory cleanup
            if batch_idx % config.EMPTY_CACHE_STEPS == 0:
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    
    # Synchronize metrics across all processes
    if world_size > 1:
        all_preds_tensor = torch.tensor(all_predictions, dtype=torch.float32).cuda(rank)
        all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32).cuda(rank)
        
        gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(all_targets_tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered_preds, all_preds_tensor)
        dist.all_gather(gathered_targets, all_targets_tensor)
        
        if rank == 0:
            all_predictions = torch.cat(gathered_preds).cpu().numpy().astype(int)
            all_targets = torch.cat(gathered_targets).cpu().numpy().astype(int)
        
        avg_loss_tensor = torch.tensor(avg_loss).cuda(rank)
        reduced_loss = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = reduced_loss.item()
    
    # Calculate metrics (only on rank 0)
    if rank == 0:
        metrics = calculate_metrics(all_targets, all_predictions,
                                all_realness_scores if all_realness_scores else None, config)
    else:
        metrics = {}
    
    return avg_loss, metrics

def save_checkpoint(model, optimizer, scaler, epoch, metrics, config, is_best=False, filename=None, rank=0):
    """Save model checkpoint (only on rank 0)"""
    if rank != 0:
        return
        
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    
    if filename is None:
        filename = f"model_epoch_{epoch+1}.pt"
    
    # Handle DDP models
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'config': vars(config)
    }
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")
    
    print(f"Saved checkpoint to {checkpoint_path}")

def main_worker(rank, world_size, config):
    """Main worker function with memory optimization"""
    logger = setup_logging()
    
    try:
        # Setup distributed training
        if world_size > 1:
            setup_distributed(rank, world_size, config)
        
        # Set device and optimize CUDA settings
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        # Memory management settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{config.MAX_SPLIT_SIZE_MB},expandable_segments:True'
        
        # Performance settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        logger.info(f"Process {rank}/{world_size} starting on device {device}")
        
        # Create datasets
        train_transform = get_transforms(config, is_train=True)
        val_transform = get_transforms(config, is_train=False)
        
        full_dataset = OptimizedForensicsDataset(config.TRAIN_PATH, config, train_transform, is_train=True)
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform
        
        # Create samplers
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        
        # DataLoaders with memory optimization
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0,
            prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0,
            prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
        )
        
        # Create model
        model = SuperiorModel(config).to(device)
        
        # Convert to SyncBatchNorm for distributed training
        if world_size > 1 and config.SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # Wrap model with DDP
        if world_size > 1:
            model = DDP(
                model, 
                device_ids=[rank], 
                output_device=rank, 
                find_unused_parameters=False,
                broadcast_buffers=True,
                gradient_as_bucket_view=True
            )
        
        # Loss and optimizer
        criterion = SuperiorLoss(config, device)
        
        # Scale learning rate based on effective batch size
        effective_batch_size = config.BATCH_SIZE * world_size
        base_lr = config.ADAMW_LR
        scaled_lr = base_lr * (effective_batch_size / 32)  # Scale from base batch size of 32
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=scaled_lr, 
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.EPOCHS
        warmup_steps = min(500, total_steps // 20)  # 5% warmup
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scaled_lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )
        
        scaler = GradScaler() if config.USE_AMP else None
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
        
        best_metric = 0.0
        
        if rank == 0:
            logger.info(f"=== MEMORY-OPTIMIZED TRAINING CONFIGURATION ===")
            logger.info(f"GPUs: {world_size}")
            logger.info(f"Batch size per GPU: {config.BATCH_SIZE}")
            logger.info(f"Effective batch size: {effective_batch_size}")
            logger.info(f"Base learning rate: {base_lr}")
            logger.info(f"Scaled learning rate: {scaled_lr}")
            logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            logger.info(f"Workers per GPU: {config.NUM_WORKERS}")
            logger.info(f"Memory fraction: {config.MEMORY_FRACTION}")
            logger.info(f"Max split size MB: {config.MAX_SPLIT_SIZE_MB}")
        
        # Training loop
        for epoch in range(config.EPOCHS):
            epoch_start_time = time.time()
            
            if world_size > 1:
                train_sampler.set_epoch(epoch)
            
            # Unfreeze layers at specified epochs
            if epoch in config.UNFREEZE_EPOCHS:
                if hasattr(model, 'module'):
                    model.module.unfreeze_backbones()
                else:
                    model.unfreeze_backbones()
                if rank == 0:
                    logger.info(f"Unfroze backbones at epoch {epoch+1}")
            
            # Switch to SGD for fine-tuning
            if epoch == config.FINE_TUNE_START_EPOCH:
                scaled_sgd_lr = config.SGD_LR * (effective_batch_size / 32)
                optimizer = optim.SGD(
                    model.parameters(), 
                    lr=scaled_sgd_lr, 
                    momentum=config.SGD_MOMENTUM, 
                    weight_decay=config.WEIGHT_DECAY,
                    nesterov=True
                )
                
                remaining_steps = (config.EPOCHS - config.FINE_TUNE_START_EPOCH) * len(train_loader)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=remaining_steps,
                    eta_min=scaled_sgd_lr * 0.01
                )
                
                if rank == 0:
                    logger.info(f"Switched to SGD optimizer with LR {scaled_sgd_lr} for fine-tuning")
            
            # Train and validate
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, config, epoch, rank, world_size)
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, config, epoch, rank, world_size)
            
            # Update learning rate
            if epoch < config.FINE_TUNE_START_EPOCH:
                scheduler.step()
            else:
                scheduler.step()
            
            # Main process handles checkpoints and logging
            if rank == 0:
                main_metric = val_metrics.get('mcc', 0) if config.USE_MCC_FOR_BEST_MODEL else val_metrics.get('accuracy', 0)
                
                # Save regular checkpoint
                if (epoch + 1) % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                    save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config, rank=rank)
                
                # Save best model
                if main_metric > best_metric:
                    best_metric = main_metric
                    save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config, is_best=True, rank=rank)
                
                # Early stopping
                if early_stopping(val_loss, model):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{config.EPOCHS} ({epoch_time:.1f}s):")
                logger.info(f"  Learning Rate: {current_lr:.2e}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val Accuracy: {val_metrics.get('accuracy', 0):.4f}")
                logger.info(f"  Val MCC: {val_metrics.get('mcc', 0):.4f}")
                logger.info(f"  Val Macro F1: {val_metrics.get('macro_f1', 0):.4f}")
                
                if 'mae' in val_metrics:
                    logger.info(f"  Val MAE: {val_metrics['mae']:.4f}")
                    logger.info(f"  Val Ordinal Acc: {val_metrics['ordinal_accuracy']:.4f}")
                
                # Log GPU memory stats
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(rank) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(rank) / 1024**3
                    logger.info(f"  GPU {rank} Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
            
            # Synchronize all processes before next epoch
            if world_size > 1:
                dist.barrier()
            
            # Memory cleanup every few epochs
            if (epoch + 1) % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Final cleanup
        if rank == 0:
            logger.info("Training completed successfully!")
            
            # Final memory stats
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(rank) / 1024**3
                memory_peak = torch.cuda.max_memory_allocated(rank) / 1024**3
                logger.info(f"Final GPU memory - Allocated: {memory_allocated:.2f}GB, Peak: {memory_peak:.2f}GB")
            
    except Exception as e:
        logger.error(f"Error in main_worker on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if world_size > 1:
            cleanup_distributed()

def main():
    """Main function with memory optimization"""
    # Set optimal environment variables
    os.environ.update({
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:512',
        'CUDA_LAUNCH_BLOCKING': '0',
        'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
        'NCCL_DEBUG': 'WARN',
        'NCCL_TREE_THRESHOLD': '0',
        'NCCL_IB_DISABLE': '1',
        'NCCL_P2P_DISABLE': '1',
        'CUDA_VISIBLE_DEVICES': ','.join(str(i) for i in range(torch.cuda.device_count())),
    })
    
    parser = argparse.ArgumentParser(description='Memory-Optimized Multi-GPU Forensics Model Training')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--train_path', type=str, default='datasets/train', help='Path to training data')
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, help='Number of data loading workers per GPU')
    parser.add_argument('--checkpoint_dir', type=str, default='ordinal_checkpoints', help='Checkpoint directory')
    parser.add_argument('--memory_fraction', type=float, default=0.85, help='GPU memory fraction per process')
    args = parser.parse_args()
    
    # Create config
    config = SuperiorConfig()
    
    # Override config with command line arguments
    if args.train_path:
        config.TRAIN_PATH = args.train_path
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.lr:
        config.ADAMW_LR = args.lr
    if args.workers:
        config.NUM_WORKERS = args.workers
    if args.checkpoint_dir:
        config.CHECKPOINT_DIR = args.checkpoint_dir
    if args.memory_fraction:
        config.MEMORY_FRACTION = args.memory_fraction
    
    # Validate config
    config.validate()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    # Determine world size and optimize settings
    world_size = torch.cuda.device_count()
    config.WORLD_SIZE = world_size
    
    # Get GPU memory info
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Dynamic optimization based on GPU count and memory
    if world_size >= 8:
        config.BATCH_SIZE = max(8, min(16, config.BATCH_SIZE))
        config.NUM_WORKERS = 1
        config.GRADIENT_ACCUMULATION_STEPS = 2
        config.MEMORY_FRACTION = 0.8
    elif world_size >= 4:
        config.BATCH_SIZE = max(12, min(24, config.BATCH_SIZE))
        config.NUM_WORKERS = 2
        config.GRADIENT_ACCUMULATION_STEPS = 1
        config.MEMORY_FRACTION = 0.85
    else:
        config.NUM_WORKERS = min(4, config.NUM_WORKERS)
        config.GRADIENT_ACCUMULATION_STEPS = 1
        config.MEMORY_FRACTION = 0.9
    
    # Adjust batch size based on GPU memory
    if gpu_memory_gb < 12:
        config.BATCH_SIZE = min(config.BATCH_SIZE, 8)
        config.MEMORY_FRACTION = 0.8
    elif gpu_memory_gb < 16:
        config.BATCH_SIZE = min(config.BATCH_SIZE, 16)
        config.MEMORY_FRACTION = 0.85
    
    # Create checkpoint directory
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    
    # Save config
    config_path = Path(config.CHECKPOINT_DIR) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)
    
    print("=== MEMORY-OPTIMIZED MULTI-GPU TRAINING CONFIGURATION ===")
    print(f"Available GPUs: {world_size}")
    print(f"GPU Memory per device: {gpu_memory_gb:.1f} GB")
    print(f"Memory fraction per GPU: {config.MEMORY_FRACTION}")
    print(f"Batch size per GPU: {config.BATCH_SIZE}")
    print(f"Total effective batch size: {config.BATCH_SIZE * world_size}")
    print(f"Workers per GPU: {config.NUM_WORKERS}")
    print(f"Gradient accumulation steps: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Base learning rate: {config.ADAMW_LR}")
    print(f"Scaled learning rate: {config.ADAMW_LR * (config.BATCH_SIZE * world_size / 32)}")
    print(f"Mixed precision: {config.USE_AMP}")
    print(f"Max split size MB: {config.MAX_SPLIT_SIZE_MB}")
    print(f"Empty cache every N steps: {config.EMPTY_CACHE_STEPS}")
    print(f"Checkpoint directory: {config.CHECKPOINT_DIR}")
    
    # Set multiprocessing start method
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    
    # Setup distributed training or single GPU
    if world_size > 1:
        print(f"Starting memory-optimized distributed training with {world_size} GPUs")
        print(f"Master address: {config.MASTER_ADDR}:{config.MASTER_PORT}")
        
        try:
            mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)
        except Exception as e:
            print(f"Distributed training failed: {e}")
            print("Falling back to single GPU training...")
            world_size = 1
            config.DISTRIBUTED = False
            config.SYNC_BN = False
            main_worker(0, world_size, config)
    else:
        print("Starting single GPU training")
        config.DISTRIBUTED = False
        config.SYNC_BN = False
        main_worker(0, world_size, config)

if __name__ == "__main__":
    # Handle signals gracefully
    def signal_handler(sig, frame):
        print("Received interrupt signal. Cleaning up...")
        cleanup_distributed()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set memory optimization flags
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    main()