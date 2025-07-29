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
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
from tqdm import tqdm
import json
import random
from collections import Counter
import warnings
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from pathlib import Path
import gc
import psutil
from contextlib import contextmanager
import time
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import sys

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Optimized configuration for 30 epochs with 120K images"""
    def __init__(self):
        # Dataset configuration
        self.TOTAL_IMAGES = 120000
        self.TRAIN_PATH = "datasets/train"
        self.IMAGE_SIZE = 224
        self.BATCH_SIZE = 64
        self.NUM_WORKERS = 8
        self.CLASS_NAMES = ['real', 'semi-synthetic', 'synthetic']
        
        # Multi-GPU configuration
        self.WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.DISTRIBUTED = self.WORLD_SIZE > 1
        self.BACKEND = 'nccl'
        self.MASTER_ADDR = 'localhost'
        self.MASTER_PORT = '12355'
        
        # Model configuration
        self.NUM_CLASSES = 3
        self.MODEL_TYPE = "convnext_vit"
        self.CNN_TYPE = "efficientnet_b3"
        
        # Training configuration
        self.EPOCHS = 30
        self.INITIAL_LR = 2e-4
        self.WEIGHT_DECAY = 1e-4
        self.PATIENCE = 8
        self.GRADIENT_CLIP_VAL = 1.0
        self.ACCUMULATE_GRAD_BATCHES = 1
        
        # Optimizer switching configuration
        self.OPTIMIZER_SWITCH_EPOCH = 20
        self.ADAMW_LR = 2e-4
        self.SGD_LR = 1e-5
        self.SGD_MOMENTUM = 0.9
        self.SGD_NESTEROV = True
        self.WARMUP_EPOCHS = 3
        
        # Fine-tuning configuration
        self.FINE_TUNE_START_EPOCH = 15
        self.BACKBONE_LR_SCALE = 0.1
        
        # Checkpoint configuration
        self.SAVE_EVERY_EPOCH = True
        self.SAVE_BEST_ONLY = False
        self.KEEP_LAST_N_CHECKPOINTS = 5
        self.RESUME_FROM_CHECKPOINT = None
        
        # Mixed precision training
        self.USE_AMP = True
        
        # Data augmentation
        self.USE_MINIMAL_AUGMENTATION = False
        self.CUTMIX_PROB = 0.1
        self.MIXUP_ALPHA = 0.1
        
        # Progressive unfreezing
        self.UNFREEZE_EPOCHS = [8, 15, 22]
        
        # Monitoring and logging
        self.LOG_EVERY_N_STEPS = 50
        self.VAL_CHECK_INTERVAL = 1.0
        self.USE_TENSORBOARD = True
        self.EXPERIMENT_NAME = "deepfake_detection_30epochs"
        
        # Device configuration
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_yaml(self, yaml_path: str):
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def print_training_stats(self):
        train_images = int(0.7 * self.TOTAL_IMAGES)
        batches_per_epoch = train_images // self.BATCH_SIZE
        total_batches = batches_per_epoch * self.EPOCHS
        
        print(f"\nTraining Configuration Statistics:")
        print(f"{'='*50}")
        print(f"Training images: {train_images:,}")
        print(f"Batch size: {self.BATCH_SIZE}")
        print(f"Batches per epoch: {batches_per_epoch:,}")
        print(f"Total epochs: {self.EPOCHS}")
        print(f"Total training batches: {total_batches:,}")
        print(f"Optimizer switch at epoch: {self.OPTIMIZER_SWITCH_EPOCH}")
        print(f"Fine-tuning starts at epoch: {self.FINE_TUNE_START_EPOCH}")
        print(f"{'='*50}\n")

    def validate(self):
        """Validate configuration parameters"""
        if self.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be positive")
        if self.EPOCHS <= 0:
            raise ValueError("EPOCHS must be positive")
        if self.NUM_CLASSES != len(self.CLASS_NAMES):
            raise ValueError("NUM_CLASSES must match length of CLASS_NAMES")
        if self.OPTIMIZER_SWITCH_EPOCH >= self.EPOCHS:
            logger.warning(f"Optimizer switch epoch adjusted to {self.EPOCHS - 5}")
            self.OPTIMIZER_SWITCH_EPOCH = max(1, self.EPOCHS - 5)
        if self.FINE_TUNE_START_EPOCH >= self.EPOCHS:
            logger.warning(f"Fine-tune start epoch adjusted to {self.EPOCHS // 2}")
            self.FINE_TUNE_START_EPOCH = max(1, self.EPOCHS // 2)

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            ce_loss = -(targets_one_hot * torch.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedDataAugmentation:
    @staticmethod
    def get_train_transforms():
        if Config.USE_MINIMAL_AUGMENTATION:
            return transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
                transforms.RandomCrop((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    @staticmethod
    def get_val_transforms():
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_test_transforms():
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class CustomDatasetPT(Dataset):
    """Dataset for loading .pt files containing image tensors"""
    def __init__(self, root_dir, transform=None, total_images=None, preload_all=True, class_names=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.preload_all = preload_all
        self.class_names = class_names or ['real', 'semi-synthetic', 'synthetic']
        
        self.images = []
        self.labels = []
        self.file_mapping = []  # [(file_path, tensor_index), ...]
        
        self._load_dataset(total_images)
        
        if self.preload_all:
            self._preload_all_tensors()
    
    def _load_dataset(self, total_images):
        logger.info("Loading dataset from .pt files...")
        
        total_loaded = 0
        target_per_class = total_images // len(self.class_names) if total_images else None
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist")
                continue
            
            pt_files = list(class_dir.glob('*.pt'))
            if not pt_files:
                logger.warning(f"No .pt files found in {class_dir}")
                continue
            
            logger.info(f"Found {len(pt_files)} .pt files for class '{class_name}'")
            
            class_images_loaded = 0
            
            for pt_file in pt_files:
                try:
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    
                    if isinstance(tensor_data, dict):
                        if 'images' in tensor_data:
                            images_tensor = tensor_data['images']
                        elif 'data' in tensor_data:
                            images_tensor = tensor_data['data']
                        else:
                            images_tensor = list(tensor_data.values())[0]
                    elif isinstance(tensor_data, (list, tuple)):
                        images_tensor = tensor_data[0]
                    else:
                        images_tensor = tensor_data
                    
                    if not isinstance(images_tensor, torch.Tensor):
                        logger.error(f"Expected tensor in {pt_file}, got {type(images_tensor)}")
                        continue
                    
                    if len(images_tensor.shape) != 4:
                        logger.error(f"Expected 4D tensor in {pt_file}, got shape {images_tensor.shape}")
                        continue
                    
                    num_images = images_tensor.shape[0]
                    logger.info(f"Loaded {num_images} images from {pt_file.name}")
                    
                    for i in range(num_images):
                        if target_per_class and class_images_loaded >= target_per_class:
                            break
                        
                        self.labels.append(class_idx)
                        self.file_mapping.append((str(pt_file), i))
                        
                        if self.preload_all:
                            self.images.append(images_tensor[i])
                        else:
                            self.images.append(None)
                        
                        class_images_loaded += 1
                        total_loaded += 1
                    
                    if target_per_class and class_images_loaded >= target_per_class:
                        break
                        
                except Exception as e:
                    logger.error(f"Error loading {pt_file}: {e}")
                    continue
            
            logger.info(f"Loaded {class_images_loaded} images for class '{class_name}'")
        
        logger.info(f"Total images loaded: {total_loaded}")
        logger.info(f"Class distribution: {Counter(self.labels)}")
        
        if total_loaded == 0:
            raise ValueError("No images were loaded! Please check your .pt files.")
    
    def _load_single_file(self, file_path):
        try:
            tensor_data = torch.load(file_path, map_location='cpu')
            if isinstance(tensor_data, dict):
                if 'images' in tensor_data:
                    return tensor_data['images']
                elif 'data' in tensor_data:
                    return tensor_data['data']
                else:
                    return list(tensor_data.values())[0]
            elif isinstance(tensor_data, (list, tuple)):
                return tensor_data[0]
            return tensor_data
        except Exception as e:
            logger.error(f"Error preloading {file_path}: {e}")
            return None

    def _preload_all_tensors(self):
        logger.info("Preloading all tensors into memory...")
        
        file_tensors = {}
        unique_files = set(file_path for file_path, _ in self.file_mapping)
        
        with ProcessPoolExecutor(max_workers=min(self.NUM_WORKERS, 4)) as executor:
            future_to_file = {executor.submit(self._load_single_file, file_path): file_path 
                            for file_path in unique_files}
            for future in tqdm(future_to_file, desc="Preloading .pt files"):
                file_path = future_to_file[future]
                result = future.result()
                if result is not None:
                    file_tensors[file_path] = result
        
        for idx, (file_path, tensor_idx) in enumerate(self.file_mapping):
            if file_path in file_tensors:
                self.images[idx] = file_tensors[file_path][tensor_idx]
            else:
                self.images[idx] = torch.zeros(3, 224, 224)
        
        logger.info("All tensors preloaded successfully!")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            label = self.labels[idx]
            image_tensor = self.images[idx]
            
            # Ensure tensor is float32
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            
            # Validate and normalize tensor range
            max_val = image_tensor.max()
            if max_val > 1.0 and max_val <= 255.0:
                image_tensor = image_tensor / 255.0
            elif max_val > 1.0:
                logger.warning(f"Unexpected tensor range at index {idx}: max={max_val}")
            
            # Apply transforms
            if self.transform:
                import torchvision.transforms.functional as TF
                pil_image = TF.to_pil_image(image_tensor.clamp(0, 1))
                image_tensor = self.transform(pil_image)
            
            return image_tensor, label
            
        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, 0

class DatasetWrapper:
    """Wrapper to apply transforms to subset datasets"""
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset_dataset)
    
    def __getitem__(self, idx):
        image_tensor, label = self.subset_dataset[idx]
        
        if self.transform:
            import torchvision.transforms.functional as TF
            
            if image_tensor.max() > 1.0 and image_tensor.max() <= 255.0:
                image_tensor = image_tensor / 255.0
            elif image_tensor.max() > 1.0:
                logger.warning(f"Unexpected tensor range at index {idx}: max={image_tensor.max()}")
            
            pil_image = TF.to_pil_image(image_tensor.clamp(0, 1))
            image_tensor = self.transform(pil_image)
        
        return image_tensor, label

class EfficientAttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(EfficientAttentionModule, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_weight = self.spatial_attention(spatial_input)
        x = x * sa_weight
        
        return x

class ConvNextViTModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(ConvNextViTModel, self).__init__()
        
        self.convnext = convnext_small(pretrained=True)
        convnext_features = self.convnext.classifier.in_features
        self.convnext.classifier = nn.Identity()
        
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        vit_features = self.vit.head.in_features
        self.vit.head = nn.Identity()
        
        self.convnext_attention = EfficientAttentionModule(convnext_features)
        self.vit_attention = EfficientAttentionModule(vit_features)
        
        combined_features = convnext_features + vit_features
        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, num_classes)
        )
        
        self.convnext_proj = nn.Linear(convnext_features, 512)
        self.vit_proj = nn.Linear(vit_features, 512)
        
        self.final_fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_parameter_groups(self, config):
        backbone_params = []
        head_params = []
        
        backbone_params.extend(list(self.convnext.parameters()))
        backbone_params.extend(list(self.vit.parameters()))
        
        head_params.extend(list(self.fusion.parameters()))
        head_params.extend(list(self.convnext_proj.parameters()))
        head_params.extend(list(self.vit_proj.parameters()))
        head_params.extend(list(self.final_fusion.parameters()))
        
        return [
            {'params': backbone_params, 'lr': config.INITIAL_LR * config.BACKBONE_LR_SCALE},
            {'params': head_params, 'lr': config.INITIAL_LR}
        ]
    
    def forward(self, x):
        conv_features = self.convnext(x)
        vit_features = self.vit(x)
        
        conv_proj = self.convnext_proj(conv_features)
        vit_proj = self.vit_proj(vit_features)
        
        combined = torch.cat([conv_proj, vit_proj], dim=1)
        output = self.final_fusion(combined)
        
        return output

class ViTCNNModel(nn.Module):
    def __init__(self, num_classes=3, cnn_type='efficientnet_b3', dropout_rate=0.3):
        super(ViTCNNModel, self).__init__()
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit_features = self.vit.head.in_features
        self.vit.head = nn.Identity()
        
        cnn_configs = {
            'efficientnet_b3': {'model': 'efficientnet_b3', 'features': 1536},
            'resnet50': {'model': 'resnet50', 'features': 2048},
            'convnext_small': {'model': 'convnext_small', 'features': 768},
            'swin_small_patch4_window7_224': {'model': 'swin_small_patch4_window7_224', 'features': 768}
        }
        
        self.cnn = timm.create_model(cnn_configs[cnn_type]['model'], pretrained=True)
        cnn_features = cnn_configs[cnn_type]['features']
        
        if hasattr(self.cnn, 'classifier'):
            self.cnn.classifier = nn.Identity()
        elif hasattr(self.cnn, 'head'):
            self.cnn.head = nn.Identity()
        elif hasattr(self.cnn, 'fc'):
            self.cnn.fc = nn.Identity()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=min(vit_features, cnn_features), 
            num_heads=8, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        combined_features = vit_features + cnn_features
        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"ViT-CNN Model created with {cnn_type}")
    
    def get_parameter_groups(self, config):
        backbone_params = []
        head_params = []
        
        backbone_params.extend(list(self.vit.parameters()))
        backbone_params.extend(list(self.cnn.parameters()))
        
        head_params.extend(list(self.fusion.parameters()))
        head_params.extend(list(self.cross_attention.parameters()))
        
        return [
            {'params': backbone_params, 'lr': config.INITIAL_LR * config.BACKBONE_LR_SCALE},
            {'params': head_params, 'lr': config.INITIAL_LR}
        ]
    
    def forward(self, x):
        vit_features = self.vit(x)
        cnn_features = self.cnn(x)
        
        combined = torch.cat([vit_features, cnn_features], dim=1)
        output = self.fusion(combined)
        
        return output

class MetricsTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions, targets, loss):
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute_metrics(self):
        if not self.predictions:
            return {}
        
        accuracy = np.mean(np.array(self.predictions) == np.array(self.targets))
        avg_loss = np.mean(self.losses)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.targets, self.predictions, average=None, zero_division=0
        )
        
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support
        }
    
    def plot_confusion_matrix(self, class_names, output_path):
        """Generate and save confusion matrix plot"""
        if not self.predictions:
            return
        
        cm = confusion_matrix(self.targets, self.predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_path)
        plt.close()

class CheckpointManager:
    def __init__(self, config, global_rank=0):
        self.config = config
        self.global_rank = global_rank
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.saved_checkpoints = []
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, scaler, metrics, is_best=False):
        if self.global_rank != 0:
            return
        
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': self.config.__dict__,
            'metrics': metrics,
            'model_architecture': str(model),
        }
        
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        if self.config.SAVE_EVERY_EPOCH:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            self.saved_checkpoints.append((epoch, checkpoint_path))
            self._cleanup_checkpoints()
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
        
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
    
    def _cleanup_checkpoints(self):
        if len(self.saved_checkpoints) > self.config.KEEP_LAST_N_CHECKPOINTS:
            self.saved_checkpoints.sort(key=lambda x: x[0])
            
            while len(self.saved_checkpoints) > self.config.KEEP_LAST_N_CHECKPOINTS:
                epoch, checkpoint_path = self.saved_checkpoints.pop(0)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            epoch = checkpoint.get('epoch', 0)
            metrics = checkpoint.get('metrics', {})
            
            logger.info(f"Checkpoint loaded successfully from epoch {epoch}")
            return epoch, metrics
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0, {}

class OptimizerManager:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.current_optimizer = None
        self.current_scheduler = None
        
        self._create_adamw_optimizer()
    
    def _create_adamw_optimizer(self):
        if hasattr(self.model, 'get_parameter_groups'):
            param_groups = self.model.get_parameter_groups(self.config)
        else:
            param_groups = [{'params': self.model.parameters(), 'lr': self.config.ADAMW_LR}]
        
        self.current_optimizer = optim.AdamW(
            param_groups,
            lr=self.config.ADAMW_LR,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.current_scheduler = optim.lr_scheduler.OneCycleLR(
            self.current_optimizer,
            max_lr=self.config.ADAMW_LR * 8,
            epochs=self.config.OPTIMIZER_SWITCH_EPOCH,
            steps_per_epoch=1,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        logger.info("Initialized AdamW optimizer")
    
    def _create_sgd_optimizer(self):
        if hasattr(self.model, 'get_parameter_groups'):
            param_groups = self.model.get_parameter_groups(self.config)
            for group in param_groups:
                if group['lr'] == self.config.INITIAL_LR * self.config.BACKBONE_LR_SCALE:
                    group['lr'] = self.config.SGD_LR * self.config.BACKBONE_LR_SCALE
                else:
                    group['lr'] = self.config.SGD_LR
        else:
            param_groups = [{'params': self.model.parameters(), 'lr': self.config.SGD_LR}]
        
        self.current_optimizer = optim.SGD(
            param_groups,
            lr=self.config.SGD_LR,
            momentum=self.config.SGD_MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY,
            nesterov=self.config.SGD_NESTEROV
        )
        
        remaining_epochs = self.config.EPOCHS - self.config.OPTIMIZER_SWITCH_EPOCH
        self.current_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.current_optimizer,
            T_max=remaining_epochs,
            eta_min=self.config.SGD_LR * 0.01
        )
        
        logger.info("Switched to SGD optimizer for fine-tuning")
    
    def maybe_switch_optimizer(self, epoch):
        if epoch >= self.config.OPTIMIZER_SWITCH_EPOCH and isinstance(self.current_optimizer, optim.AdamW):
            logger.info(f"Switching from AdamW to SGD at epoch {epoch}")
            self._create_sgd_optimizer()
            return True
        return False
    
    def update_scheduler_steps(self, steps_per_epoch):
        if isinstance(self.current_scheduler, optim.lr_scheduler.OneCycleLR):
            if isinstance(self.current_optimizer, optim.AdamW):
                self.current_scheduler = optim.lr_scheduler.OneCycleLR(
                    self.current_optimizer,
                    max_lr=self.config.ADAMW_LR * 8,
                    epochs=self.config.OPTIMIZER_SWITCH_EPOCH,
                    steps_per_epoch=steps_per_epoch,
                    pct_start=0.1,
                    anneal_strategy='cos'
                )
    
    def get_optimizer(self):
        return self.current_optimizer
    
    def get_scheduler(self):
        return self.current_scheduler

class DistributedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config, local_rank):
        self.config = config
        self.local_rank = local_rank
        self.global_rank = dist.get_rank() if config.DISTRIBUTED else 0
        self.world_size = dist.get_world_size() if config.DISTRIBUTED else 1
        
        if config.DISTRIBUTED:
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = config.DEVICE
        
        self.model = model.to(self.device)
        if config.DISTRIBUTED:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        class_weights = torch.tensor([1.0, 2.0, 1.0]).to(self.device)
        self.criterion = FocalLoss(gamma=2.0, alpha=class_weights, label_smoothing=0.1)
        
        model_for_optimizer = self.model.module if config.DISTRIBUTED else self.model
        self.optimizer_manager = OptimizerManager(model_for_optimizer, config)
        
        self.optimizer_manager.update_scheduler_steps(len(train_loader))
        
        self.scaler = GradScaler() if config.USE_AMP else None
        
        self.train_metrics = MetricsTracker(config.NUM_CLASSES)
        self.val_metrics = MetricsTracker(config.NUM_CLASSES)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0
        self.patience_counter = 0
        
        self.checkpoint_manager = CheckpointManager(config, self.global_rank)
        
        if self.global_rank == 0 and config.USE_TENSORBOARD:
            self.writer = SummaryWriter(f'runs/{config.EXPERIMENT_NAME}')
        else:
            self.writer = None
        
        if config.RESUME_FROM_CHECKPOINT:
            self.load_checkpoint(config.RESUME_FROM_CHECKPOINT)
    
    def train_epoch(self):
        self.model.train()
        self.train_metrics.reset()
        
        if self.config.DISTRIBUTED:
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        optimizer_switched = self.optimizer_manager.maybe_switch_optimizer(self.current_epoch)
        if optimizer_switched and self.global_rank == 0:
            logger.info(f"Optimizer switched at epoch {self.current_epoch}")
        
        optimizer = self.optimizer_manager.get_optimizer()
        scheduler = self.optimizer_manager.get_scheduler()
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch}', 
                   disable=self.global_rank != 0)
        
        accumulated_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            if self.config.USE_AMP:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target) / self.config.ACCUMULATE_GRAD_BATCHES
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.ACCUMULATE_GRAD_BATCHES == 0 or (batch_idx + 1) == len(self.train_loader):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_VAL)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                    if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                        scheduler.step()
                    
                    optimizer.zero_grad()
            else:
                output = self.model(data)
                loss = self.criterion(output, target) / self.config.ACCUMULATE_GRAD_BATCHES
                loss.backward()
                
                if (batch_idx + 1) % self.config.ACCUMULATE_GRAD_BATCHES == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_VAL)
                    optimizer.step()
                    
                    if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                        scheduler.step()
                    
                    optimizer.zero_grad()
            
            accumulated_loss += loss.item() * self.config.ACCUMULATE_GRAD_BATCHES
            pred = output.argmax(dim=1)
            self.train_metrics.update(pred, target, loss.item() * self.config.ACCUMULATE_GRAD_BATCHES)
            
            if self.global_rank == 0 and batch_idx % self.config.LOG_EVERY_N_STEPS == 0:
                current_lr = optimizer.param_groups[0]['lr']
                optimizer_type = type(optimizer).__name__
                pbar.set_postfix({
                    'Loss': f'{accumulated_loss:.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Opt': optimizer_type,
                    'GPU': f'{torch.cuda.memory_allocated(self.device) / 1e9:.1f}GB'
                })
                
                if self.writer:
                    self.writer.add_scalar('Train/BatchLoss', accumulated_loss, self.global_step)
                    self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
                    self.writer.add_scalar('Train/GPUMemory', 
                                         torch.cuda.memory_allocated(self.device) / 1e9, self.global_step)
            
            self.global_step += 1
            
            # Periodic memory cleanup
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        return self.train_metrics.compute_metrics()
    
    def validate(self):
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation', disable=self.global_rank != 0):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if self.config.USE_AMP:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                pred = output.argmax(dim=1)
                self.val_metrics.update(pred, target, loss.item())
        
        if self.config.DISTRIBUTED:
            self._sync_metrics(self.val_metrics)
        
        # Generate confusion matrix plot
        if self.global_rank == 0:
            cm_path = Path(f'runs/{self.config.EXPERIMENT_NAME}/confusion_matrix_epoch_{self.current_epoch}.png')
            self.val_metrics.plot_confusion_matrix(self.config.CLASS_NAMES, cm_path)
            if self.writer:
                self.writer.add_image('Validation/ConfusionMatrix', 
                                    plt.imread(cm_path), self.current_epoch, dataformats='HWC')
        
        return self.val_metrics.compute_metrics()
    
    def test(self):
        if not self.test_loader:
            return {}
        
        self.model.eval()
        test_metrics = MetricsTracker(self.config.NUM_CLASSES)
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing', disable=self.global_rank != 0):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                outputs = []
                for _ in range(3):
                    if self.config.USE_AMP:
                        with autocast():
                            output = self.model(data)
                    else:
                        output = self.model(data)
                    outputs.append(output)
                
                output = torch.stack(outputs).mean(dim=0)
                loss = self.criterion(output, target)
                
                pred = output.argmax(dim=1)
                test_metrics.update(pred, target, loss.item())
        
        if self.config.DISTRIBUTED:
            self._sync_metrics(test_metrics)
        
        if self.global_rank == 0:
            cm_path = Path(f'runs/{self.config.EXPERIMENT_NAME}/confusion_matrix_test.png')
            test_metrics.plot_confusion_matrix(self.config.CLASS_NAMES, cm_path)
            if self.writer:
                self.writer.add_image('Test/ConfusionMatrix', 
                                    plt.imread(cm_path), self.current_epoch, dataformats='HWC')
        
        return test_metrics.compute_metrics()
    
    def _sync_metrics(self, metrics_tracker):
        if not self.config.DISTRIBUTED:
            return
        
        predictions_tensor = torch.tensor(metrics_tracker.predictions, dtype=torch.long, device=self.device)
        targets_tensor = torch.tensor(metrics_tracker.targets, dtype=torch.long, device=self.device)
        losses_tensor = torch.tensor(metrics_tracker.losses, device=self.device)
        
        gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(self.world_size)]
        gathered_targets = [torch.zeros_like(targets_tensor) for _ in range(self.world_size)]
        gathered_losses = [torch.zeros_like(losses_tensor) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_predictions, predictions_tensor)
        dist.all_gather(gathered_targets, targets_tensor)
        dist.all_gather(gathered_losses, losses_tensor)
        
        metrics_tracker.predictions = torch.cat(gathered_predictions).cpu().numpy().tolist()
        metrics_tracker.targets = torch.cat(gathered_targets).cpu().numpy().tolist()
        metrics_tracker.losses = torch.cat(gathered_losses).cpu().numpy().tolist()
    
    def load_checkpoint(self, checkpoint_path):
        epoch, metrics = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, 
            self.model,
            self.optimizer_manager.get_optimizer(),
            self.optimizer_manager.get_scheduler(),
            self.scaler
        )
        
        self.current_epoch = epoch + 1
        if 'validation' in metrics and 'accuracy' in metrics['validation']:
            self.best_val_acc = metrics['validation']['accuracy']
    
    def train(self):
        logger.info(f"Starting training on {self.device}")
        logger.info(f"World size: {self.world_size}, Global rank: {self.global_rank}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.global_rank == 0:
            self.config.print_training_stats()
            self.config.to_yaml('training_config_30epochs.yaml')
        
        for epoch in range(self.current_epoch, self.config.EPOCHS):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            if self.global_rank == 0:
                optimizer_type = type(self.optimizer_manager.get_optimizer()).__name__
                current_lr = self.optimizer_manager.get_optimizer().param_groups[0]['lr']
                
                logger.info(f'\nEpoch {epoch+1}/{self.config.EPOCHS} (Optimizer: {optimizer_type}, LR: {current_lr:.2e})')
                logger.info('-' * 80)
                logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                           f"Acc: {train_metrics['accuracy']:.4f}, "
                           f"F1: {train_metrics['macro_f1']:.4f}")
                logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                           f"Acc: {val_metrics['accuracy']:.4f}, "
                           f"F1: {val_metrics['macro_f1']:.4f}")
                
                for i, class_name in enumerate(self.config.CLASS_NAMES):
                    logger.info(f"Class {class_name}: "
                               f"Precision: {val_metrics['per_class_precision'][i]:.3f}, "
                               f"Recall: {val_metrics['per_class_recall'][i]:.3f}, "
                               f"F1: {val_metrics['per_class_f1'][i]:.3f}")
                
                if self.writer:
                    self.writer.add_scalars('Loss', {
                        'Train': train_metrics['loss'],
                        'Validation': val_metrics['loss']
                    }, epoch)
                    
                    self.writer.add_scalars('Accuracy', {
                        'Train': train_metrics['accuracy'],
                        'Validation': val_metrics['accuracy']
                    }, epoch)
                    
                    self.writer.add_scalars('F1_Score', {
                        'Train': train_metrics['macro_f1'],
                        'Validation': val_metrics['macro_f1']
                    }, epoch)
                    
                    self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
                    
                    for i, class_name in enumerate(self.config.CLASS_NAMES):
                        self.writer.add_scalar(f'Val_F1/{class_name}', 
                                             val_metrics['per_class_f1'][i], epoch)
                
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                self.checkpoint_manager.save_checkpoint(
                    epoch,
                    self.model,
                    self.optimizer_manager.get_optimizer(),
                    self.optimizer_manager.get_scheduler(),
                    self.scaler,
                    {
                        'train': train_metrics,
                        'validation': val_metrics
                    },
                    is_best
                )
                
                if self.patience_counter >= self.config.PATIENCE:
                    logger.info(f"Early stopping triggered after {self.config.PATIENCE} epochs without improvement")
                    break
        
        if self.test_loader:
            logger.info("Running final test evaluation...")
            test_metrics = self.test()
            if self.global_rank == 0:
                logger.info(f"Test Results - Acc: {test_metrics['accuracy']:.4f}, "
                           f"F1: {test_metrics['macro_f1']:.4f}")
                
                if self.writer:
                    self.writer.add_scalar('Test/Accuracy', test_metrics['accuracy'])
                    self.writer.add_scalar('Test/F1_Score', test_metrics['macro_f1'])
        
        if self.writer:
            self.writer.close()
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")

def setup_distributed(rank, world_size, backend='nccl', master_addr='localhost', master_port='12355'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def create_data_loaders_pt(config, local_rank=-1):
    """Create distributed data loaders for .pt files"""
    train_transform = AdvancedDataAugmentation.get_train_transforms()
    val_transform = AdvancedDataAugmentation.get_val_transforms()
    test_transform = AdvancedDataAugmentation.get_test_transforms()
    
    with torch_distributed_zero_first(local_rank):
        logger.info("Creating dataset from .pt files...")
        
        full_dataset = CustomDatasetPT(
            root_dir=config.TRAIN_PATH,
            transform=None,
            total_images=config.TOTAL_IMAGES,
            preload_all=True,
            class_names=config.CLASS_NAMES
        )
        
        logger.info(f"Total images in dataset: {len(full_dataset)}")
        
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        
        logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    train_dataset_with_transform = DatasetWrapper(train_dataset, train_transform)
    val_dataset_with_transform = DatasetWrapper(val_dataset, val_transform)
    test_dataset_with_transform = DatasetWrapper(test_dataset, test_transform) if test_size > 0 else None
    
    train_sampler = DistributedSampler(train_dataset_with_transform) if config.DISTRIBUTED else None
    val_sampler = DistributedSampler(val_dataset_with_transform, shuffle=False) if config.DISTRIBUTED else None
    test_sampler = DistributedSampler(test_dataset_with_transform, shuffle=False) if config.DISTRIBUTED and test_dataset_with_transform else None
    
    train_loader = DataLoader(
        train_dataset_with_transform,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset_with_transform,
        batch_size=config.BATCH_SIZE,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset_with_transform,
        batch_size=config.BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    ) if test_dataset_with_transform else None
    
    return train_loader, val_loader, test_loader

def create_model(config):
    if config.MODEL_TYPE == "convnext_vit":
        model = ConvNextViTModel(num_classes=config.NUM_CLASSES)
    else:
        model = ViTCNNModel(num_classes=config.NUM_CLASSES, cnn_type=config.CNN_TYPE)
    
    return model

def train_worker(local_rank, config):
    try:
        if config.DISTRIBUTED:
            setup_distributed(local_rank, config.WORLD_SIZE, config.BACKEND, 
                           config.MASTER_ADDR, config.MASTER_PORT)
        
        torch.manual_seed(42 + local_rank)
        np.random.seed(42 + local_rank)
        random.seed(42 + local_rank)
        
        train_loader, val_loader, test_loader = create_data_loaders_pt(config, local_rank)
        
        model = create_model(config)
        
        trainer = DistributedTrainer(model, train_loader, val_loader, test_loader, config, local_rank)
        
        trainer.train()
        
    except Exception as e:
        logger.error(f"Error in training worker {local_rank}: {e}")
        raise
    finally:
        cleanup_distributed()

def inspect_pt_files(data_dir, max_files_per_class=3, class_names=None):
    """Inspect the structure of .pt files in the dataset"""
    data_dir = Path(data_dir)
    class_names = class_names or ['real', 'semi-synthetic', 'synthetic']
    
    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"❌ Class directory {class_dir} does not exist")
            continue
        
        pt_files = list(class_dir.glob('*.pt'))
        print(f"\n📁 Class '{class_name}': {len(pt_files)} .pt files found")
        
        for i, pt_file in enumerate(pt_files[:max_files_per_class]):
            try:
                print(f"\n  📄 File: {pt_file.name}")
                tensor_data = torch.load(pt_file, map_location='cpu')
                
                print(f"     Type: {type(tensor_data)}")
                
                if isinstance(tensor_data, dict):
                    print(f"     Dict keys: {list(tensor_data.keys())}")
                    for key, value in tensor_data.items():
                        if isinstance(value, torch.Tensor):
                            print(f"       {key}: {value.shape}, dtype: {value.dtype}")
                        else:
                            print(f"       {key}: {type(value)}")
                            
                elif isinstance(tensor_data, torch.Tensor):
                    print(f"     Shape: {tensor_data.shape}")
                    print(f"     Dtype: {tensor_data.dtype}")
                    print(f"     Min: {tensor_data.min():.3f}, Max: {tensor_data.max():.3f}")
                    
                elif isinstance(tensor_data, (list, tuple)):
                    print(f"     Length: {len(tensor_data)}")
                    for j, item in enumerate(tensor_data):
                        if isinstance(item, torch.Tensor):
                            print(f"       [{j}]: {item.shape}, dtype: {item.dtype}")
                        else:
                            print(f"       [{j}]: {type(item)}")
                
            except Exception as e:
                print(f"     ❌ Error loading {pt_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Deep Learning Training - .pt Files')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment-name', type=str, default='deepfake_detection_30epochs', help='Experiment name')
    parser.add_argument('--model-type', type=str, default='convnext_vit', 
                       choices=['convnext_vit', 'vit_cnn'], help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--adamw-lr', type=float, default=2e-4, help='AdamW learning rate')
    parser.add_argument('--sgd-lr', type=float, default=1e-5, help='SGD learning rate')
    parser.add_argument('--optimizer-switch-epoch', type=int, default=20, help='Epoch to switch from AdamW to SGD')
    parser.add_argument('--fine-tune-start-epoch', type=int, default=15, help='Epoch to start fine-tuning')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--minimal-aug', action='store_true', help='Use minimal augmentation')
    parser.add_argument('--train-path', type=str, default='datasets/train', help='Path to training data')
    parser.add_argument('--class-names', type=str, nargs='+', default=['real', 'semi-synthetic', 'synthetic'], 
                       help='Class names for dataset')
    parser.add_argument('--master-addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master-port', type=str, default='12355', help='Master port for distributed training')
    
    args = parser.parse_args()
    
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    config.RESUME_FROM_CHECKPOINT = args.resume or config.RESUME_FROM_CHECKPOINT
    config.EXPERIMENT_NAME = args.experiment_name or config.EXPERIMENT_NAME
    config.MODEL_TYPE = args.model_type or config.MODEL_TYPE
    config.BATCH_SIZE = args.batch_size or config.BATCH_SIZE
    config.EPOCHS = args.epochs or config.EPOCHS
    config.ADAMW_LR = args.adamw_lr or config.ADAMW_LR
    config.INITIAL_LR = config.ADAMW_LR
    config.SGD_LR = args.sgd_lr or config.SGD_LR
    config.OPTIMIZER_SWITCH_EPOCH = args.optimizer_switch_epoch or config.OPTIMIZER_SWITCH_EPOCH
    config.FINE_TUNE_START_EPOCH = args.fine_tune_start_epoch or config.FINE_TUNE_START_EPOCH
    config.PATIENCE = args.patience or config.PATIENCE
    config.USE_AMP = not args.no_amp
    config.USE_MINIMAL_AUGMENTATION = args.minimal_aug or config.USE_MINIMAL_AUGMENTATION
    config.TRAIN_PATH = args.train_path or config.TRAIN_PATH
    config.CLASS_NAMES = args.class_names or config.CLASS_NAMES
    config.MASTER_ADDR = args.master_addr or config.MASTER_ADDR
    config.MASTER_PORT = args.master_port or config.MASTER_PORT
    config.NUM_CLASSES = len(config.CLASS_NAMES)
    
    # Validate configuration
    config.validate()
    
    config.UNFREEZE_EPOCHS = [epoch for epoch in config.UNFREEZE_EPOCHS if epoch < config.EPOCHS]
    if not config.UNFREEZE_EPOCHS:
        config.UNFREEZE_EPOCHS = [config.EPOCHS // 4, config.EPOCHS // 2, 3 * config.EPOCHS // 4]
        config.UNFREEZE_EPOCHS = [epoch for epoch in config.UNFREEZE_EPOCHS if epoch > 0 and epoch < config.EPOCHS]
    
    # Print configuration summary
    logger.info("=== Training Configuration Summary ===")
    logger.info(f"Experiment: {config.EXPERIMENT_NAME}")
    logger.info(f"Model: {config.MODEL_TYPE}")
    logger.info(f"Total epochs: {config.EPOCHS}")
    logger.info(f"Batch size per GPU: {config.BATCH_SIZE}")
    logger.info(f"Total GPUs: {config.WORLD_SIZE}")
    logger.info(f"Effective batch size: {config.BATCH_SIZE * config.WORLD_SIZE}")
    logger.info(f"AdamW LR: {config.ADAMW_LR}, SGD LR: {config.SGD_LR}")
    logger.info(f"Optimizer switch epoch: {config.OPTIMIZER_SWITCH_EPOCH}")
    logger.info(f"Data path: {config.TRAIN_PATH}")
    logger.info(f"Class names: {config.CLASS_NAMES}")
    logger.info("=====================================")
    
    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path(f'runs/{config.EXPERIMENT_NAME}').mkdir(parents=True, exist_ok=True)
    
    # Inspect dataset structure before training
    logger.info("Inspecting dataset structure...")
    inspect_pt_files(config.TRAIN_PATH, class_names=config.CLASS_NAMES)
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            logger.info(f"GPU {i}: {props.name}, Memory: {memory_gb:.1f} GB")
    else:
        logger.warning("CUDA not available, training will use CPU")
        config.DISTRIBUTED = False
        config.WORLD_SIZE = 1
    
    # Memory optimization for large datasets
    if config.TOTAL_IMAGES > 100000:
        logger.info("Large dataset detected, enabling memory optimizations")
        torch.backends.cudnn.benchmark = True
        if psutil.virtual_memory().available < 32e9:  # Less than 32GB RAM
            config.NUM_WORKERS = min(config.NUM_WORKERS, 4)
            logger.info(f"Reduced num_workers to {config.NUM_WORKERS} due to memory constraints")
    
    # Set memory management settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    try:
        if config.DISTRIBUTED and config.WORLD_SIZE > 1:
            logger.info(f"Starting distributed training with {config.WORLD_SIZE} GPUs")
            mp.spawn(
                train_worker,
                args=(config,),
                nprocs=config.WORLD_SIZE,
                join=True
            )
        else:
            logger.info("Starting single GPU/CPU training")
            train_worker(0, config)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        cleanup_distributed()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        cleanup_distributed()
        raise
    
    logger.info("Training script completed successfully!")

def memory_cleanup():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_lr_schedule_info(config):
    """Print learning rate schedule information"""
    print(f"\n📊 Learning Rate Schedule:")
    print(f"{'='*50}")
    print(f"Phase 1 (Epochs 1-{config.OPTIMIZER_SWITCH_EPOCH}): AdamW")
    print(f"  - Initial LR: {config.ADAMW_LR:.2e}")
    print(f"  - Max LR: {config.ADAMW_LR * 8:.2e} (OneCycleLR)")
    print(f"  - Warmup epochs: {config.WARMUP_EPOCHS}")
    
    print(f"\nPhase 2 (Epochs {config.OPTIMIZER_SWITCH_EPOCH + 1}-{config.EPOCHS}): SGD")
    print(f"  - Initial LR: {config.SGD_LR:.2e}")
    print(f"  - Min LR: {config.SGD_LR * 0.01:.2e} (CosineAnnealingLR)")
    print(f"  - Momentum: {config.SGD_MOMENTUM}")
    print(f"  - Nesterov: {config.SGD_NESTEROV}")
    
    print(f"\nFine-tuning starts at epoch: {config.FINE_TUNE_START_EPOCH}")
    print(f"Backbone LR scale: {config.BACKBONE_LR_SCALE}")
    print(f"Progressive unfreezing epochs: {config.UNFREEZE_EPOCHS}")
    print(f"{'='*50}\n")

def validate_dataset_structure(data_path, class_names):
    """Validate that the dataset has the correct structure"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path {data_path} does not exist")
    
    missing_classes = []
    
    for class_name in class_names:
        class_dir = data_path / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            continue
        
        pt_files = list(class_dir.glob('*.pt'))
        if not pt_files:
            missing_classes.append(f"{class_name} (no .pt files)")
    
    if missing_classes:
        raise ValueError(f"Missing or empty class directories: {missing_classes}")
    
    logger.info("✅ Dataset structure validation passed")

def estimate_training_time(config):
    """Estimate total training time based on configuration"""
    images_per_second = {
        'V100': 200,
        'A100': 400,
        'RTX3090': 150,
        'RTX4090': 250,
        'default': 100
    }
    
    train_images = int(0.7 * config.TOTAL_IMAGES)
    val_images = int(0.2 * config.TOTAL_IMAGES)
    
    processing_speed = images_per_second['default'] * config.WORLD_SIZE
    
    train_time_per_epoch = train_images / processing_speed
    val_time_per_epoch = val_images / processing_speed
    total_time_per_epoch = train_time_per_epoch + val_time_per_epoch
    
    total_estimated_time = total_time_per_epoch * config.EPOCHS / 60
    
    print(f"\n⏱️  Training Time Estimate:")
    print(f"{'='*40}")
    print(f"Training images: {train_images:,}")
    print(f"Validation images: {val_images:,}")
    print(f"Processing speed: ~{processing_speed} images/sec")
    print(f"Time per epoch: ~{total_time_per_epoch/60:.1f} minutes")
    print(f"Total estimated time: ~{total_estimated_time:.1f} minutes ({total_estimated_time/60:.1f} hours)")
    print(f"{'='*40}\n")

def setup_logging(experiment_name):
    """Setup comprehensive logging"""
    log_dir = Path(f'logs/{experiment_name}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")

def print_system_info():
    """Print system information for debugging"""
    print(f"\n🖥️  System Information:")
    print(f"{'='*50}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("GPUs: None (CUDA not available)")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Setup logging first
    setup_logging("deepfake_training")
    
    # Print system information
    print_system_info()
    
    # Run main function with error handling
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        memory_cleanup()