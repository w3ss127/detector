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
from torchvision.models import convnext_tiny, convnext_small, convnext_base, efficientnet_b4
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
import time
import signal
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import math
import cv2
from sklearn.model_selection import StratifiedKFold
import pickle
from PIL import Image

warnings.filterwarnings('ignore')

def set_multiprocessing_start_method():
    """Set the multiprocessing start method to 'spawn' for CUDA compatibility."""
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError as e:
        if "context has already been set" in str(e):
            print("Multiprocessing context already set")
        else:
            raise e

class RankFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'rank'):
            record.rank = 0
        return super().format(record)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = RankFormatter('%(asctime)s - %(levelname)s - [Rank %(rank)d] - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

class UltimateDeepfakeConfig:
    def __init__(self):
        # MODEL ARCHITECTURE
        self.MODEL_TYPE = "ultimate_forensics_ensemble"
        self.BACKBONES = ["convnext_small", "efficientnet_b4"]
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 512  # Reduced for better memory efficiency
        self.DROPOUT_RATE = 0.3
        self.FREEZE_BACKBONES = True
        self.ATTENTION_DROPOUT = 0.2
        self.USE_SPECTRAL_NORM = True
        
        # DEVICE AND DISTRIBUTION
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DISTRIBUTED = torch.cuda.device_count() > 1
        self.BACKEND = "nccl"
        self.MASTER_ADDR = "localhost"
        self.MASTER_PORT = "12356"
        
        # TRAINING PARAMETERS
        self.BATCH_SIZE = 4  # Conservative for memory
        self.EPOCHS = 20  # Default epochs
        self.USE_AMP = True
        self.TRAIN_PATH = "datasets/train"
        self.IMAGE_SIZE = (224, 224)
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = 2
        
        # PROGRESSIVE UNFREEZING
        self.UNFREEZE_EPOCHS = [3, 8, 15]
        self.FINE_TUNE_START_EPOCH = 8
        self.EARLY_STOPPING_PATIENCE = 15
        
        # LEARNING RATES
        self.ADAMW_LR = 5e-5
        self.WEIGHT_DECAY = 1e-2
        
        # CLASS BALANCING
        self.FOCAL_ALPHA = torch.tensor([1.0, 3.0, 1.0])
        self.FOCAL_GAMMA = 2.0
        self.LABEL_SMOOTHING = 0.05
        self.CLASS_WEIGHTS = torch.tensor([1.0, 2.5, 1.0])
        
        # FEATURES
        self.USE_FORENSICS_MODULE = True
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = 0.2
        
        # LOSS WEIGHTS
        self.BOUNDARY_WEIGHT = 0.2
        
        # CHECKPOINTING
        self.CHECKPOINT_DIR = "ultimate_deepfake_checkpoints"
        self.CHECKPOINT_EVERY_N_EPOCHS = 5
        self.USE_MCC_FOR_BEST_MODEL = True
        
        # REGULARIZATION
        self.USE_GRADIENT_CLIPPING = True
        self.GRADIENT_CLIP_VALUE = 1.0
        
        # LEARNING RATE SCHEDULING
        self.USE_COSINE_ANNEALING = True
        self.USE_WARMUP = True
        self.WARMUP_EPOCHS = 3
        
        # FORENSICS
        self.USE_FREQUENCY_ANALYSIS = True

    def adjust_for_epochs(self, epochs):
        """Adjust epoch-dependent parameters based on total epochs"""
        self.EPOCHS = epochs
        
        if epochs <= 20:
            self.UNFREEZE_EPOCHS = [3, 8]
            self.FINE_TUNE_START_EPOCH = 8
            self.WARMUP_EPOCHS = 2
        elif epochs <= 50:
            self.UNFREEZE_EPOCHS = [3, 8, 15, 25]
            self.FINE_TUNE_START_EPOCH = 15
            self.WARMUP_EPOCHS = 5
        else:
            self.UNFREEZE_EPOCHS = [3, 8, 15, 25, 40]
            self.FINE_TUNE_START_EPOCH = 25
            self.WARMUP_EPOCHS = 8
        
        self.EARLY_STOPPING_PATIENCE = min(15, epochs // 2)

    def validate(self):
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES
        assert self.FINE_TUNE_START_EPOCH < self.EPOCHS

class UltimateForensicsModule(nn.Module):
    """Simplified forensics analysis module"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # DCT analyzer
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Noise pattern analyzer
        self.noise_analyzer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        
        # Fusion network
        self.forensics_fusion = nn.Sequential(
            nn.Linear(96, 64),  # 64 + 32
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        device = x.device
        
        dct_feats = self.dct_analyzer(x)
        noise_feats = self.noise_analyzer(x)
        
        combined_feats = torch.cat([dct_feats, noise_feats], dim=1)
        forensics_output = self.forensics_fusion(combined_feats)
        
        return forensics_output

class UltimateAttentionModule(nn.Module):
    """Simplified attention mechanism"""
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if x.dim() != 2:
            x = x.view(batch_size, -1)
        
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        return attended_features

class UltimateAugmentation:
    """Standard PyTorch transforms for augmentation"""
    def __init__(self, config):
        self.config = config
        height, width = config.IMAGE_SIZE
        
        self.train_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class UltimateMixupCutmix:
    """Simplified mixup"""
    def __init__(self, config):
        self.config = config
        self.mixup_alpha = config.MIXUP_ALPHA if config.USE_MIXUP else 0
    
    def mixup_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def __call__(self, x, y):
        if self.mixup_alpha > 0:
            return self.mixup_data(x, y, self.mixup_alpha)
        else:
            return x, y, y, 1.0

class UltimateLoss(nn.Module):
    """Simplified loss function"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.focal_gamma = config.FOCAL_GAMMA
        self.label_smoothing = config.LABEL_SMOOTHING
        self.boundary_weight = config.BOUNDARY_WEIGHT
        
        self.focal_alpha = None
        self.class_weights = None
        
    def focal_loss(self, inputs, targets, device):
        if self.focal_alpha is None or self.focal_alpha.device != device:
            self.focal_alpha = self.config.FOCAL_ALPHA.to(device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def boundary_loss(self, logits, targets, device):
        probs = F.softmax(logits, dim=1)
        semi_mask = (targets == 1).float()
        semi_confidence_loss = semi_mask * (1 - probs[:, 1])
        return semi_confidence_loss.mean()
    
    def forward(self, logits, targets, features=None, alpha=None, epoch=1, class_f1_scores=None):
        device = logits.device
        
        if self.class_weights is None or self.class_weights.device != device:
            self.class_weights = self.config.CLASS_WEIGHTS.to(device)
        
        focal_loss = self.focal_loss(logits, targets, device)
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        boundary_loss = self.boundary_loss(logits, targets, device)
        
        total_loss = 0.4 * focal_loss + 0.4 * ce_loss + self.boundary_weight * boundary_loss
        
        return total_loss

class UltimateDataset(Dataset):
    """Dataset that handles .pt files with 5000 image tensors each"""
    def __init__(self, root_dir, config, transform_type='train', is_training=True):
        self.root_dir = Path(root_dir)
        self.config = config
        self.is_training = is_training
        self.class_names = config.CLASS_NAMES
        self.file_indices = []  # (file_path, tensor_index) pairs
        self.labels = []
        
        # Initialize augmentation
        self.augmentation = UltimateAugmentation(config)
        
        self._validate_dataset_path()
        self._load_file_mapping()
    
    def _validate_dataset_path(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.root_dir} does not exist")
    
    def _load_file_mapping(self):
        logger.info("Loading .pt file mapping...", extra={'rank': 0})
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist", extra={'rank': 0})
                continue
            
            # Find all .pt files in the class directory
            pt_files = list(class_dir.glob('*.pt'))
            logger.info(f"Found {len(pt_files)} .pt files for class {class_name}", extra={'rank': 0})
            
            total_images_in_class = 0
            for pt_file in pt_files:
                try:
                    # Load tensor data to check shape
                    tensor_data = torch.load(pt_file, map_location='cpu', weights_only=True)
                    
                    # Handle different possible formats
                    if isinstance(tensor_data, dict):
                        # If it's a dictionary, try common keys
                        if 'images' in tensor_data:
                            images = tensor_data['images']
                        elif 'data' in tensor_data:
                            images = tensor_data['data']
                        elif 'tensors' in tensor_data:
                            images = tensor_data['tensors']
                        else:
                            # Take the first tensor value
                            images = list(tensor_data.values())[0]
                    else:
                        # Direct tensor
                        images = tensor_data
                    
                    if images is None or not isinstance(images, torch.Tensor):
                        logger.warning(f"Invalid tensor format in {pt_file}", extra={'rank': 0})
                        continue
                    
                    # Validate tensor shape
                    if len(images.shape) != 4:  # Should be (N, C, H, W)
                        logger.warning(f"Invalid tensor shape {images.shape} in {pt_file}, expected 4D tensor", extra={'rank': 0})
                        continue
                    
                    num_images = images.shape[0]
                    logger.info(f"File {pt_file.name}: {num_images} images", extra={'rank': 0})
                    
                    # Add all images from this file
                    for i in range(num_images):
                        self.file_indices.append((str(pt_file), i))
                        self.labels.append(class_idx)
                        total_images_in_class += 1
                
                except Exception as e:
                    logger.warning(f"Error loading {pt_file}: {e}", extra={'rank': 0})
                    continue
            
            logger.info(f"Total images loaded for class {class_name}: {total_images_in_class}", extra={'rank': 0})
        
        logger.info(f"Total dataset samples: {len(self.file_indices)}", extra={'rank': 0})
        
        if len(self.file_indices) == 0:
            raise ValueError("No valid .pt files found in dataset. Please check your data path and file format.")
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        try:
            file_path, tensor_idx = self.file_indices[idx]
            label = self.labels[idx]
            
            # Load the .pt file
            tensor_data = torch.load(file_path, map_location='cpu', weights_only=True)
            
            # Handle different possible formats
            if isinstance(tensor_data, dict):
                if 'images' in tensor_data:
                    images = tensor_data['images']
                elif 'data' in tensor_data:
                    images = tensor_data['data']
                elif 'tensors' in tensor_data:
                    images = tensor_data['tensors']
                else:
                    images = list(tensor_data.values())[0]
            else:
                images = tensor_data
            
            # Get the specific image tensor
            image_tensor = images[tensor_idx]
            
            # Ensure proper format (C, H, W) and dtype
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            
            # Normalize to [0, 1] if needed
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            # Clamp values to valid range
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # Convert to PIL Image for augmentation
            # Tensor should be (C, H, W), convert to (H, W, C) for PIL
            if image_tensor.shape[0] == 3:  # (C, H, W)
                image_np = image_tensor.permute(1, 2, 0).numpy()
            else:
                logger.warning(f"Unexpected tensor shape: {image_tensor.shape}", extra={'rank': 0})
                image_np = image_tensor.numpy()
                if len(image_np.shape) == 2:  # Grayscale
                    image_np = np.stack([image_np] * 3, axis=-1)
            
            # Convert to uint8 for PIL
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            # Create PIL Image
            image = Image.fromarray(image_np)
            
            # Apply transformations
            if self.is_training:
                image_tensor = self.augmentation.train_transform(image)
            else:
                image_tensor = self.augmentation.val_transform(image)
            
            return image_tensor, label
            
        except Exception as e:
            logger.warning(f"Error loading tensor at index {idx} from {file_path}: {e}", extra={'rank': 0})
            # Return dummy tensor on error
            dummy_tensor = torch.zeros(3, self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1])
            return dummy_tensor, self.labels[idx] if idx < len(self.labels) else 0

class UltimateModel(nn.Module):
    """Simplified model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize backbones
        self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        convnext_features = 768
        
        self.efficientnet = efficientnet_b4(weights=config.PRETRAINED_WEIGHTS)
        efficientnet_features = 1792
        
        if config.FREEZE_BACKBONES:
            self.freeze_backbones()
        
        # Calculate total features
        forensics_features = 32 if config.USE_FORENSICS_MODULE else 0
        total_features = convnext_features + efficientnet_features + forensics_features
        
        # Forensics module
        if config.USE_FORENSICS_MODULE:
            self.forensics_module = UltimateForensicsModule(config)
        
        # Attention module
        self.attention_module = UltimateAttentionModule(total_features, config)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.BatchNorm1d(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE // 2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(config.HIDDEN_DIM // 4, config.NUM_CLASSES)
        )
    
    def freeze_backbones(self):
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        logger.info("Frozen backbone parameters", extra={'rank': 0})
    
    def unfreeze_backbones(self):
        for param in self.convnext.parameters():
            param.requires_grad = True
        for param in self.efficientnet.parameters():
            param.requires_grad = True
        logger.info("Unfrozen backbone parameters", extra={'rank': 0})
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        device = x.device
        
        # ConvNeXt features
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        
        # EfficientNet features
        efficientnet_feats = self.efficientnet.features(x)
        efficientnet_feats = self.efficientnet.avgpool(efficientnet_feats)
        efficientnet_feats = torch.flatten(efficientnet_feats, 1)
        
        # Combine features
        features_list = [convnext_feats, efficientnet_feats]
        
        # Add forensics features if enabled
        if self.config.USE_FORENSICS_MODULE:
            forensics_feats = self.forensics_module(x)
            features_list.append(forensics_feats)
        
        # Fuse all features
        fused_features = torch.cat(features_list, dim=1)
        
        # Apply attention
        attended_features = self.attention_module(fused_features)
        
        # Process through fusion network
        processed_features = self.fusion(attended_features)
        
        # Get final logits
        logits = self.classifier(processed_features)
        
        return logits, processed_features

def create_data_loaders(config, local_rank=-1):
    """Create simplified data loaders"""
    dataset = UltimateDataset(
        root_dir=config.TRAIN_PATH, 
        config=config, 
        is_training=True
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check your data path and format.")
    
    # Simple split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset, rank=local_rank, shuffle=True
    ) if config.DISTRIBUTED and local_rank != -1 else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=(train_sampler is None),
        sampler=train_sampler, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def calculate_metrics(y_true, y_pred):
    """Calculate basic metrics"""
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0
    
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        macro_f1 = np.mean(f1)
        semi_synthetic_f1 = f1[1] if len(f1) > 1 else 0
    except:
        macro_f1 = 0.0
        semi_synthetic_f1 = 0.0
        f1 = [0.0, 0.0, 0.0]
    
    # Ultimate score
    ultimate_score = 0.4 * semi_synthetic_f1 + 0.3 * mcc + 0.3 * accuracy
    
    metrics = {
        'accuracy': accuracy,
        'mcc': mcc,
        'macro_f1': macro_f1,
        'semi_synthetic_f1': semi_synthetic_f1,
        'ultimate_score': ultimate_score,
        'per_class_f1': f1.tolist()
    }
    
    return metrics

def evaluate_model(model, data_loader, criterion, config, device, epoch=1, local_rank=0):
    """Simplified model evaluation"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating", leave=False, disable=local_rank not in [-1, 0]):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with autocast(enabled=config.USE_AMP):
                logits, features = model(data)
                loss = criterion(logits, target, features, None, epoch)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    metrics = calculate_metrics(all_targets, all_predictions)
    
    return avg_loss, metrics, all_predictions, all_targets

def create_optimizer_scheduler(model, config, epoch):
    """Create simplified optimizer"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.ADAMW_LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    if config.USE_COSINE_ANNEALING:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    
    return optimizer, scheduler

def progressive_unfreeze(model, epoch, config):
    """Simplified progressive unfreezing"""
    if epoch in config.UNFREEZE_EPOCHS:
        model.unfreeze_backbones()
        return True
    return False

def enhanced_mixup_criterion(criterion, model_output, target_a, target_b, lam, epoch):
    """Simplified mixup criterion"""
    logits, features = model_output
    loss_a = criterion(logits, target_a, features, None, epoch)
    loss_b = criterion(logits, target_b, features, None, epoch)
    return lam * loss_a + (1 - lam) * loss_b

def find_free_port(start_port=12355, max_attempts=100):
    port = start_port
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                s.listen(1)
                return port
        except OSError:
            port += 1
            continue
    raise RuntimeError(f"No free port found after {max_attempts} attempts")

def setup_distributed(local_rank, world_size, backend='nccl', master_addr='localhost', master_port='12355'):
    try:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = str(local_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)
        logger.info(f"Distributed process group initialized for rank {local_rank}", extra={'rank': local_rank})
        return True
    except Exception as e:
        logger.error(f"Failed to setup distributed training for rank {local_rank}: {e}", extra={'rank': local_rank})
        return False

def cleanup_distributed():
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed", extra={'rank': 0})
    except Exception as e:
        logger.warning(f"Error during distributed cleanup: {e}", extra={'rank': 0})
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, initiating graceful shutdown...", extra={'rank': 0})
    cleanup_distributed()
    cleanup_memory()
    exit(0)

def save_checkpoint(model, optimizer, scaler, scheduler, epoch, metrics, config, filename, local_rank=0):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config.__dict__
    }
    torch.save(checkpoint, filename)
    if local_rank in [-1, 0]:
        logger.info(f"Checkpoint saved: {filename}", extra={'rank': local_rank})

def plot_metrics(history, save_path):
    """Plot training metrics"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Ultimate Score
        if 'ultimate_score' in history:
            axes[0, 1].plot(history['ultimate_score'], label='Ultimate Score', color='green')
            axes[0, 1].set_title('Ultimate Performance Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # MCC
        if 'val_mcc' in history:
            axes[1, 0].plot(history['val_mcc'], label='Val MCC', color='purple')
            axes[1, 0].set_title('Matthews Correlation Coefficient')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MCC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Accuracy
        if 'val_accuracy' in history:
            axes[1, 1].plot(history['val_accuracy'], label='Val Accuracy', color='orange')
            axes[1, 1].set_title('Validation Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Could not save plot: {e}", extra={'rank': 0})

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Could not save confusion matrix: {e}", extra={'rank': 0})

def train_worker(local_rank, config, master_port):
    """Main training worker with proper device handling and GPU optimization"""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Setup device handling
        if config.DISTRIBUTED:
            success = setup_distributed(local_rank, torch.cuda.device_count(), 
                                      config.BACKEND, config.MASTER_ADDR, master_port)
            if not success:
                logger.error(f"Failed to setup distributed training for rank {local_rank}", 
                           extra={'rank': local_rank})
                return
            
            device = torch.device(f'cuda:{local_rank}')
            device_id = local_rank
        else:
            if torch.cuda.is_available():
                device_id = 0
                device = torch.device(f'cuda:{device_id}')
            else:
                device = torch.device('cpu')
                device_id = None
            local_rank = 0
        
        config.DEVICE = device
        
        # GPU optimizations
        if device.type == 'cuda' and device_id is not None:
            torch.cuda.set_device(device_id)
            torch.backends.cudnn.benchmark = True
        
        # Move config tensors to correct device
        config.FOCAL_ALPHA = config.FOCAL_ALPHA.to(device)
        config.CLASS_WEIGHTS = config.CLASS_WEIGHTS.to(device)
        
        logger.info(f"Training setup complete for rank {local_rank} on device {device}", 
                   extra={'rank': local_rank})
        if device.type == 'cuda':
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB", 
                       extra={'rank': local_rank})
        
        # Set seeds for reproducibility
        torch.manual_seed(42 + local_rank)
        np.random.seed(42 + local_rank)
        random.seed(42 + local_rank)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(42 + local_rank)
        
        if local_rank in [-1, 0]:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(config, local_rank)
        
        # Initialize model
        model = UltimateModel(config).to(device)
        if config.DISTRIBUTED:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
        # Log model info
        trainable_params = model.module.get_trainable_params() if config.DISTRIBUTED else model.get_trainable_params()
        logger.info(f"Model has {trainable_params:,} trainable parameters", extra={'rank': local_rank})
        
        # Initialize loss, scaler, and augmentation
        criterion = UltimateLoss(config).to(device)
        scaler = GradScaler(enabled=config.USE_AMP)
        mixup_cutmix = UltimateMixupCutmix(config)
        
        # Training state
        best_metrics = {'ultimate_score': -1.0}
        epochs_no_improve = 0
        training_history = defaultdict(list)
        
        # Training loop
        for epoch in range(config.EPOCHS):
            epoch_start_time = time.time()
            
            # Progressive unfreezing
            model_for_unfreeze = model.module if config.DISTRIBUTED else model
            unfroze_this_epoch = progressive_unfreeze(model_for_unfreeze, epoch + 1, config)
            if unfroze_this_epoch:
                logger.info(f"Unfroze backbones at epoch {epoch + 1}", extra={'rank': local_rank})
            
            # Create optimizer and scheduler
            optimizer, scheduler = create_optimizer_scheduler(model, config, epoch + 1)
            
            # Training phase
            model.train()
            train_loss = 0
            train_batches = 0
            
            if config.DISTRIBUTED:
                train_loader.sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS}', 
                              disable=local_rank not in [-1, 0])
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                # Apply mixup if enabled
                use_mixup = False
                if config.USE_MIXUP and np.random.random() < 0.5:
                    data, target_a, target_b, lam = mixup_cutmix(data, target)
                    use_mixup = True
                
                optimizer.zero_grad()
                
                with autocast(enabled=config.USE_AMP):
                    model_output = model(data)
                    
                    if use_mixup:
                        loss = enhanced_mixup_criterion(criterion, model_output, 
                                                      target_a, target_b, lam, epoch + 1)
                    else:
                        logits, features = model_output
                        loss = criterion(logits, target, features, None, epoch + 1)
                
                scaler.scale(loss).backward()
                
                if config.USE_GRADIENT_CLIPPING:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_batches += 1
                
                if local_rank in [-1, 0]:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg': f'{train_loss/train_batches:.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                        'GPU': f'{torch.cuda.memory_reserved(device)/1024**3:.1f}GB' if device.type == 'cuda' else 'CPU'
                    })
                
                # Memory cleanup every 20 batches
                if batch_idx % 20 == 0:
                    cleanup_memory()
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            
            # Update scheduler
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            # Evaluation on main process
            if local_rank in [-1, 0]:
                training_history['train_loss'].append(avg_train_loss)
                training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
                
                val_loss, val_metrics, val_preds, val_targets = evaluate_model(
                    model, val_loader, criterion, config, device, epoch + 1, local_rank
                )
                
                # Update scheduler with validation metric
                if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['ultimate_score'])
                
                # Record metrics
                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_metrics['accuracy'])
                training_history['val_mcc'].append(val_metrics['mcc'])
                training_history['macro_f1'].append(val_metrics['macro_f1'])
                training_history['semi_f1'].append(val_metrics['semi_synthetic_f1'])
                training_history['ultimate_score'].append(val_metrics['ultimate_score'])
                
                epoch_time = time.time() - epoch_start_time
                
                # Logging
                logger.info(f"Epoch {epoch+1}/{config.EPOCHS} completed in {epoch_time:.2f}s", 
                           extra={'rank': local_rank})
                logger.info(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}", 
                           extra={'rank': local_rank})
                logger.info(f"Accuracy: {val_metrics['accuracy']:.6f}, MCC: {val_metrics['mcc']:.6f}", 
                           extra={'rank': local_rank})
                logger.info(f"Semi F1: {val_metrics['semi_synthetic_f1']:.6f}, Ultimate Score: {val_metrics['ultimate_score']:.6f}", 
                           extra={'rank': local_rank})
                
                # Save best model
                current_score = val_metrics['ultimate_score']
                if current_score > best_metrics['ultimate_score']:
                    best_metrics = val_metrics.copy()
                    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
                    save_checkpoint(model, optimizer, scaler, scheduler, 
                                  epoch, val_metrics, config, best_model_path, local_rank)
                    epochs_no_improve = 0
                    logger.info(f"New best model saved! Score: {current_score:.6f}", 
                               extra={'rank': local_rank})
                else:
                    epochs_no_improve += 1
                
                # Periodic checkpoint saving
                if (epoch + 1) % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
                    save_checkpoint(model, optimizer, scaler, scheduler,
                                  epoch, val_metrics, config, checkpoint_path, local_rank)
                
                # Plot metrics every 5 epochs
                if (epoch + 1) % 5 == 0:
                    plot_metrics(training_history, 
                               os.path.join(config.CHECKPOINT_DIR, f'metrics_epoch_{epoch+1}.png'))
                
                # Early stopping
                if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs", 
                               extra={'rank': local_rank})
                    break
            
            # Synchronization barrier for distributed training
            if config.DISTRIBUTED:
                dist.barrier()
            cleanup_memory()
        
        # Final evaluation
        if local_rank in [-1, 0]:
            logger.info("Training completed - Final evaluation", extra={'rank': local_rank})
            
            # Load best model for final evaluation
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
                if config.DISTRIBUTED:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                # Final test evaluation
                test_loss, test_metrics, test_preds, test_targets = evaluate_model(
                    model, test_loader, criterion, config, device, 999, local_rank
                )
                
                # Print final results
                logger.info("=" * 80, extra={'rank': local_rank})
                logger.info("FINAL TEST RESULTS", extra={'rank': local_rank})
                logger.info("=" * 80, extra={'rank': local_rank})
                logger.info(f"Ultimate Score: {test_metrics['ultimate_score']:.6f}", extra={'rank': local_rank})
                logger.info(f"Test Accuracy: {test_metrics['accuracy']:.6f}", extra={'rank': local_rank})
                logger.info(f"MCC: {test_metrics['mcc']:.6f}", extra={'rank': local_rank})
                logger.info(f"Semi-synthetic F1: {test_metrics['semi_synthetic_f1']:.6f}", extra={'rank': local_rank})
                logger.info(f"Macro F1: {test_metrics['macro_f1']:.6f}", extra={'rank': local_rank})
                logger.info(f"Per-class F1: {test_metrics['per_class_f1']}", extra={'rank': local_rank})
                
                # Save final results
                final_results = {
                    'test_metrics': test_metrics,
                    'test_loss': test_loss,
                    'best_val_metrics': best_metrics,
                    'training_history': dict(training_history),
                    'config': config.__dict__
                }
                
                results_path = os.path.join(config.CHECKPOINT_DIR, "final_results.json")
                with open(results_path, 'w') as f:
                    json.dump(final_results, f, indent=2, default=str)
                
                # Final confusion matrix
                if len(test_targets) > 0:
                    cm = confusion_matrix(test_targets, test_preds)
                    plot_confusion_matrix(cm, config.CLASS_NAMES, 
                                        os.path.join(config.CHECKPOINT_DIR, 'final_confusion_matrix.png'))
                
                # Final metrics plot
                plot_metrics(training_history, 
                           os.path.join(config.CHECKPOINT_DIR, 'final_training_metrics.png'))
                
                logger.info(f"Results saved to {results_path}", extra={'rank': local_rank})
                logger.info("Training pipeline completed successfully!", extra={'rank': local_rank})
    
    except Exception as e:
        logger.error(f"Training error on rank {local_rank}: {e}", extra={'rank': local_rank})
        import traceback
        traceback.print_exc()
        raise e
    finally:
        cleanup_distributed()
        cleanup_memory()

def main():
    """Main function to run the deepfake detection training"""
    # Memory optimization settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    set_multiprocessing_start_method()
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Training')
    parser.add_argument('--train_path', type=str, default='datasets/train', 
                       help='Path to training dataset')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, 
                       help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--distributed', action='store_true', 
                       help='Enable distributed training')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    args = parser.parse_args()
    
    # Initialize configuration
    config = UltimateDeepfakeConfig()
    
    # Override config with command line arguments
    config.TRAIN_PATH = args.train_path
    config.BATCH_SIZE = args.batch_size
    config.ADAMW_LR = args.lr
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.DISTRIBUTED = args.distributed and torch.cuda.device_count() > 1
    config.NUM_WORKERS = args.num_workers
    
    # Adjust configuration for epochs
    config.adjust_for_epochs(args.epochs)
    
    # Validate configuration
    config.validate()
    
    logger.info("Starting Deepfake Detection Training Pipeline", extra={'rank': 0})
    logger.info(f"Available GPUs: {torch.cuda.device_count()}", extra={'rank': 0})
    logger.info(f"Distributed training: {config.DISTRIBUTED}", extra={'rank': 0})
    logger.info(f"Training for {config.EPOCHS} epochs", extra={'rank': 0})
    logger.info(f"Batch size: {config.BATCH_SIZE}", extra={'rank': 0})
    logger.info(f"Learning rate: {config.ADAMW_LR}", extra={'rank': 0})
    logger.info(f"Unfreeze schedule: {config.UNFREEZE_EPOCHS}", extra={'rank': 0})
    
    # GPU memory info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name} - {gpu_props.total_memory / 1024**3:.1f} GB", 
                       extra={'rank': 0})
    
    # Launch training
    if config.DISTRIBUTED:
        master_port = str(find_free_port())
        config.MASTER_PORT = master_port
        logger.info(f"Using master port {master_port} for distributed training", extra={'rank': 0})
        
        world_size = torch.cuda.device_count()
        logger.info(f"Launching training on {world_size} GPUs", extra={'rank': 0})
        mp.spawn(
            train_worker,
            args=(config, master_port),
            nprocs=world_size,
            join=True
        )
    else:
        logger.info("Launching single-device training", extra={'rank': 0})
        train_worker(-1, config, config.MASTER_PORT)
    
    logger.info("Training pipeline completed", extra={'rank': 0})

if __name__ == "__main__":
    main()