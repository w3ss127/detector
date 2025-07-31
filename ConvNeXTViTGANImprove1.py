import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import timm
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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
from functools import lru_cache
import PIL.Image
import io

warnings.filterwarnings('ignore')

# Custom formatter for logging with rank
class RankFormatter(logging.Formatter):
    """Formatter to include rank in distributed training logs."""
    def format(self, record):
        if not hasattr(record, 'rank'):
            record.rank = 0
        return super().format(record)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = RankFormatter('%(asctime)s - %(levelname)s - [Rank %(rank)d] - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

class ImprovedConfig:
    """Configuration class for the forensics model hyperparameters."""
    def __init__(self, args):
        self.MODEL_TYPE = "improved_forensics_model"
        self.CONVNEXT_BACKBONE = args.convnext_backbone
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        self.NUM_CLASSES = args.num_classes
        self.HIDDEN_DIM = args.hidden_dim
        self.DROPOUT_RATE = args.dropout_rate
        self.FREEZE_BACKBONES = args.freeze_backbones
        self.ATTENTION_DROPOUT = args.attention_dropout
        self.USE_SPECTRAL_NORM = args.use_spectral_norm
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DISTRIBUTED = torch.cuda.device_count() > 1 and args.world_size > 1
        self.BACKEND = "nccl" if torch.cuda.is_available() else "gloo"
        self.MASTER_ADDR = args.master_addr
        self.MASTER_PORT = args.master_port
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.USE_AMP = args.use_amp
        self.TRAIN_PATH = args.train_path
        self.IMAGE_SIZE = args.image_size
        self.CLASS_NAMES = args.class_names.split(",") if args.class_names else ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = args.num_workers
        self.UNFREEZE_EPOCHS = args.unfreeze_epochs
        self.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
        self.ADAMW_LR = args.adamw_lr
        self.SGD_LR = args.sgd_lr
        self.WEIGHT_DECAY = args.weight_decay
        self.SGD_MOMENTUM = args.sgd_momentum
        self.FOCAL_ALPHA = torch.tensor(args.focal_alpha).to(self.DEVICE)
        self.FOCAL_GAMMA = args.focal_gamma
        self.CLASS_WEIGHTS = torch.tensor(args.class_weights).to(self.DEVICE)
        self.USE_FORENSICS_MODULE = args.use_forensics_module
        self.USE_UNCERTAINTY_ESTIMATION = args.use_uncertainty_estimation
        self.USE_MIXUP = args.use_mixup
        self.USE_CUTMIX = args.use_cutmix
        self.MIXUP_ALPHA = args.mixup_alpha
        self.CUTMIX_ALPHA = args.cutmix_alpha
        self.CHECKPOINT_DIR = args.checkpoint_dir
        self.CHECKPOINT_EVERY_N_EPOCHS = args.checkpoint_every_n_epochs
        self.USE_MCC_FOR_BEST_MODEL = args.use_mcc_for_best_model
        self.SAVE_TOP_K_MODELS = args.save_top_k_models
        self.FORENSICS_LOSS_WEIGHT = args.forensics_loss_weight
        self.CONSISTENCY_LOSS_WEIGHT = args.consistency_loss_weight
        self.MAX_CHECKPOINTS = args.max_checkpoints
        self.MAX_WEIGHT_FACTOR = args.max_weight_factor
        self.USE_SGD_FINE_TUNE = args.use_sgd_fine_tune
        self.BACKBONE_LR_FACTOR = args.backbone_lr_factor
        self.LR_SCHEDULE = args.lr_schedule
        self.T_MAX = args.t_max

    def validate(self):
        """Validate configuration parameters."""
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0, "Batch size must be positive"
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0, "Epochs must be positive"
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES, "Class names must match NUM_CLASSES"
        assert self.CONVNEXT_BACKBONE in ["convnext_tiny", "convnext_small"], "Unsupported backbone"
        assert len(self.FOCAL_ALPHA) == self.NUM_CLASSES, "FOCAL_ALPHA must match NUM_CLASSES"
        assert len(self.CLASS_WEIGHTS) == self.NUM_CLASSES, "CLASS_WEIGHTS must match NUM_CLASSES"
        assert self.MIXUP_ALPHA > 0, "MIXUP_ALPHA must be positive"
        assert self.CUTMIX_ALPHA > 0, "CUTMIX_ALPHA must be positive"
        assert self.SGD_LR > 0, "SGD_LR must be positive"
        assert 0 <= self.SGD_MOMENTUM < 1, "SGD_MOMENTUM must be in [0, 1)"
        assert self.BACKBONE_LR_FACTOR > 0, "BACKBONE_LR_FACTOR must be positive"
        assert self.T_MAX > 0, "T_MAX must be positive"
        logger.info("Configuration validated successfully", extra={'rank': 0})

class ForensicsAwareModule(nn.Module):
    """Module for extracting forensics-specific features (DCT, noise, edges, frequency)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.noise_analyzer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.edge_analyzer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.freq_analyzer = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 96),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.forensics_fusion = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 96, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def extract_edge_inconsistencies(self, x):
        """Extract edge-based features from grayscale input."""
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edge_feats = self.edge_analyzer(gray)
        return edge_feats

    def forward(self, x):
        """Forward pass combining DCT, noise, edge, and frequency features."""
        dct_feats = self.dct_analyzer(x)
        noise_feats = self.noise_analyzer(x)
        edge_feats = self.extract_edge_inconsistencies(x)
        freq_feats = self.freq_analyzer(x)
        combined_feats = torch.cat([dct_feats, noise_feats, edge_feats, freq_feats], dim=1)
        forensics_output = self.forensics_fusion(combined_feats)
        return forensics_output

class UncertaintyModule(nn.Module):
    """Module for estimating epistemic and aleatoric uncertainty."""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes),
            nn.Softplus()
        )
        self.aleatoric_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, num_classes),
            nn.Softplus()
        )

    def forward(self, x):
        """Compute probabilities and uncertainties using evidential deep learning."""
        evidence = self.evidence_layer(x)
        aleatoric = self.aleatoric_layer(x)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        epistemic_uncertainty = self.num_classes / S
        aleatoric_uncertainty = aleatoric
        return probs, epistemic_uncertainty, aleatoric_uncertainty, alpha

class SimplifiedLoss(nn.Module):
    """Custom loss combining focal loss, cross-entropy, and evidential loss."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.focal_alpha = config.FOCAL_ALPHA
        self.focal_gamma = config.FOCAL_GAMMA
        self.class_weights = config.CLASS_WEIGHTS
        self.forensics_weight = config.FORENSICS_LOSS_WEIGHT
        self.consistency_weight = config.CONSISTENCY_LOSS_WEIGHT

    def focal_loss(self, inputs, targets):
        """Compute focal loss with class-specific weights."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def forensics_consistency_loss(self, features, targets):
        """Encourage feature consistency within and across classes."""
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        targets_expanded = targets.unsqueeze(1)
        same_class_mask = (targets_expanded == targets_expanded.t()).float()
        positive_loss = same_class_mask * (1 - similarity_matrix)
        different_class_mask = 1 - same_class_mask
        negative_loss = different_class_mask * torch.clamp(similarity_matrix - 0.2, min=0)
        return (positive_loss.sum() + negative_loss.sum()) / (targets.size(0) ** 2)

    def evidential_loss(self, alpha, targets):
        """Compute evidential loss for uncertainty estimation."""
        targets_one_hot = F.one_hot(targets, num_classes=self.config.NUM_CLASSES).float()
        S = torch.sum(alpha, dim=1, keepdim=True)
        likelihood_loss = torch.sum(targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        return likelihood_loss.mean()

    def forward(self, logits, targets, features=None, alpha=None, epoch=1):
        """Combine focal, cross-entropy, consistency, and evidential losses."""
        focal_loss = self.focal_loss(logits, targets)
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        total_loss = 0.6 * focal_loss + 0.4 * ce_loss
        if features is not None:
            consistency_loss = self.forensics_consistency_loss(features, targets)
            total_loss += self.consistency_weight * consistency_loss
        if alpha is not None:
            evidential_loss = self.evidential_loss(alpha, targets)
            total_loss += self.forensics_weight * evidential_loss
        return total_loss

class SuperiorAttentionModule(nn.Module):
    """Attention module combining multi-head and channel attention."""
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.forensics_attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=8,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )
        if config.USE_SPECTRAL_NORM:
            self.channel_attention[0] = nn.utils.spectral_norm(self.channel_attention[0])
            self.channel_attention[3] = nn.utils.spectral_norm(self.channel_attention[3])

    def forward(self, x):
        """Apply attention to input features."""
        batch_size = x.size(0)
        if x.dim() != 2:
            x = x.view(batch_size, -1)
        x_reshaped = x.unsqueeze(1)
        attn_output, _ = self.forensics_attention(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.squeeze(1)
        channel_weights = self.channel_attention(x)
        attended_features = x * channel_weights + attn_output
        return attended_features

class ImprovedModel(nn.Module):
    """Main model combining ConvNeXt, ViT, forensics, and attention modules."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        if config.FREEZE_BACKBONES:
            self.freeze_backbones()

        convnext_features = 768
        vit_features = self.vit.num_features
        forensics_features = 128 if config.USE_FORENSICS_MODULE else 0
        total_features = convnext_features + vit_features + forensics_features

        if config.USE_FORENSICS_MODULE:
            self.forensics_module = ForensicsAwareModule(config)

        self.attention_module = SuperiorAttentionModule(total_features, config)

        if config.USE_UNCERTAINTY_ESTIMATION:
            self.uncertainty_module = UncertaintyModule(config.HIDDEN_DIM // 4, config.NUM_CLASSES)

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
        self.classifier = nn.Linear(config.HIDDEN_DIM // 4, config.NUM_CLASSES)

        if config.USE_SPECTRAL_NORM:
            self.fusion[0] = nn.utils.spectral_norm(self.fusion[0])
            self.fusion[3] = nn.utils.spectral_norm(self.fusion[3])
            self.fusion[6] = nn.utils.spectral_norm(self.fusion[6])
            self.classifier = nn.utils.spectral_norm(self.classifier)

    def freeze_backbones(self):
        """Freeze ConvNeXt and ViT parameters."""
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        logger.info("Frozen backbone parameters", extra={'rank': 0})

    def unfreeze_convnext_layers(self):
        """Unfreeze ConvNeXt layers for fine-tuning."""
        for param in self.convnext.parameters():
            param.requires_grad = True
        logger.info("Unfrozen ConvNeXt layers", extra={'rank': 0})

    def unfreeze_vit_layers(self):
        """Unfreeze ViT layers for fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = True
        logger.info("Unfrozen ViT layers", extra={'rank': 0})

    def unfreeze_forensics_and_attention(self):
        """Unfreeze forensics and attention modules."""
        if self.config.USE_FORENSICS_MODULE:
            for param in self.forensics_module.parameters():
                param.requires_grad = True
        for param in self.attention_module.parameters():
            param.requires_grad = True
        for param in self.fusion.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        logger.info("Unfrozen forensics module, attention module, fusion, and classifier layers", extra={'rank': 0})

    def get_parameter_groups(self):
        """Return parameter groups with differential learning rates."""
        backbone_params = list(self.convnext.parameters()) + list(self.vit.parameters())
        task_specific_params = (
            (list(self.forensics_module.parameters()) if self.config.USE_FORENSICS_MODULE else []) +
            list(self.attention_module.parameters()) +
            list(self.fusion.parameters()) +
            list(self.classifier.parameters()) +
            (list(self.uncertainty_module.parameters()) if self.config.USE_UNCERTAINTY_ESTIMATION else [])
        )
        return [
            {'params': backbone_params, 'lr': self.config.BACKBONE_LR_FACTOR},
            {'params': task_specific_params, 'lr': 1.0}
        ]

    def forward(self, x):
        """Forward pass through the entire model."""
        convnext_features = self.convnext(x)
        vit_features = self.vit(x)
        combined_features = [convnext_features, vit_features]
        if self.config.USE_FORENSICS_MODULE:
            forensics_features = self.forensics_module(x)
            combined_features.append(forensics_features)
        combined_features = torch.cat(combined_features, dim=1)
        attended_features = self.attention_module(combined_features)
        fused_features = self.fusion(attended_features)
        logits = self.classifier(fused_features)
        if self.config.USE_UNCERTAINTY_ESTIMATION:
            probs, epistemic_uncertainty, aleatoric_uncertainty, alpha = self.uncertainty_module(fused_features)
            return logits, attended_features, probs, epistemic_uncertainty, aleatoric_uncertainty, alpha
        return logits, attended_features

class ForensicsDataset(Dataset):
    """Dataset for loading .pt files or raw images with labels."""
    def __init__(self, data_path, config, transform=None):
        self.data_path = data_path
        self.config = config
        self.transform = transform
        self.classes = config.CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        """Load data from .pt files or raw images."""
        if os.path.isdir(self.data_path):
            for class_name in self.classes:
                class_path = os.path.join(self.data_path, class_name)
                if os.path.exists(class_path):
                    for file in os.listdir(class_path):
                        if file.endswith(('.jpg', '.jpeg', '.png', '.pt')):
                            self.data.append(os.path.join(class_path, file))
                            self.labels.append(self.class_to_idx[class_name])
        else:
            for file in os.listdir(self.data_path):
                if file.endswith('.pt'):
                    self.data.append(os.path.join(self.data_path, file))
                    label = os.path.basename(file).split('_')[0]
                    if label in self.class_to_idx:
                        self.labels.append(self.class_to_idx[label])
        logger.info(f"Loaded {len(self.data)} samples", extra={'rank': 0})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        if data_path.endswith('.pt'):
            data = torch.load(data_path)
        else:
            image = PIL.Image.open(data_path).convert('RGB')
            image = np.array(image)
            if self.transform:
                augmented = self.transform(image=image)
                data = augmented['image']
        return data, label

def get_transforms(config):
    """Define data augmentation and transformation pipeline."""
    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.RandomResizedCrop(config.IMAGE_SIZE, config.IMAGE_SIZE, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform

def create_improved_data_loaders(config, rank=-1):
    """Create data loaders with weighted sampling and distributed support."""
    train_transform, val_transform = get_transforms(config)
    dataset = ForensicsDataset(config.TRAIN_PATH, config, transform=train_transform)
    labels = dataset.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = np.clip(sample_weights, 0, config.MAX_WEIGHT_FACTOR * np.mean(sample_weights))
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) if rank == -1 else DistributedSampler(dataset, num_replicas=config.world_size, rank=rank)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler if rank != -1 else None,
        shuffle=(sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=True
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

def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=0.2, device='cuda'):
    """Apply cutmix augmentation to a batch."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for cutmix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def save_checkpoint(model, optimizer, scheduler, epoch, mcc, config, rank, filename):
    """Save model checkpoint."""
    if rank == 0:
        checkpoint = {
            'model_state_dict': model.module.state_dict() if config.DISTRIBUTED else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'mcc': mcc
        }
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}", extra={'rank': rank})

def load_checkpoint(model, optimizer, scheduler, config, rank, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    if config.DISTRIBUTED:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    mcc = checkpoint['mcc']
    logger.info(f"Loaded checkpoint: {checkpoint_path}, Epoch: {epoch}, MCC: {mcc}", extra={'rank': rank})
    return epoch, mcc

def setup_distributed(rank, world_size, config):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    dist.init_process_group(backend=config.BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info(f"Initialized distributed training on rank {rank}", extra={'rank': rank})

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()
    logger.info("Cleaned up distributed training", extra={'rank': 0})

def train_epoch(model, data_loader, criterion, optimizer, scaler, config, epoch, rank):
    """Train one epoch with mixed precision and augmentations."""
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc=f"Rank {rank} Training Epoch {epoch+1}", disable=rank != 0)):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        if config.USE_MIXUP and random.random() < 0.5:
            data, target_a, target_b, lam = mixup_data(data, target, config.MIXUP_ALPHA, config.DEVICE)
        elif config.USE_CUTMIX and random.random() < 0.5:
            data, target_a, target_b, lam = cutmix_data(data, target, config.CUTMIX_ALPHA, config.DEVICE)
        else:
            target_a, target_b, lam = target, target, 1.0

        optimizer.zero_grad()
        with autocast(enabled=config.USE_AMP):
            outputs = model(data)
            if config.USE_UNCERTAINTY_ESTIMATION:
                logits, features, probs, epistemic_uncertainty, aleatoric_uncertainty, alpha = outputs
                loss = lam * criterion(logits, target_a, features, alpha, epoch) + (1 - lam) * criterion(logits, target_b, features, alpha, epoch)
            else:
                logits, features = outputs
                loss = lam * criterion(logits, target_a, features, None, epoch) + (1 - lam) * criterion(logits, target_b, features, None, epoch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    if config.DISTRIBUTED:
        loss_tensor = torch.tensor(avg_loss).to(config.DEVICE)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / config.world_size
    return avg_loss

def evaluate_model(model, data_loader, criterion, config, rank, phase="Validation"):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_epistemic = []
    all_aleatoric = []
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc=f"Rank {rank} {phase}", disable=rank != 0):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            with autocast(enabled=config.USE_AMP):
                outputs = model(data)
                if config.USE_UNCERTAINTY_ESTIMATION:
                    logits, features, probs, epistemic_uncertainty, aleatoric_uncertainty, alpha = outputs
                    loss = criterion(logits, target, features, alpha)
                    all_epistemic.append(epistemic_uncertainty.cpu())
                    all_aleatoric.append(aleatoric_uncertainty.cpu())
                else:
                    logits, features = outputs
                    loss = criterion(logits, target, features, None)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / len(data_loader)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    if config.DISTRIBUTED:
        loss_tensor = torch.tensor(avg_loss).to(config.DEVICE)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / config.world_size
        gathered_preds = [torch.zeros_like(torch.tensor(all_preds)) for _ in range(config.world_size)]
        gathered_targets = [torch.zeros_like(torch.tensor(all_targets)) for _ in range(config.world_size)]
        dist.all_gather(gathered_preds, torch.tensor(all_preds).to(config.DEVICE))
        dist.all_gather(gathered_targets, torch.tensor(all_targets).to(config.DEVICE))
        all_preds = np.concatenate([p.numpy() for p in gathered_preds])
        all_targets = np.concatenate([t.numpy() for t in gathered_targets])

    mcc = matthews_corrcoef(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    if config.USE_UNCERTAINTY_ESTIMATION:
        all_epistemic = torch.cat(all_epistemic).numpy()
        all_aleatoric = torch.cat(all_aleatoric).numpy()
    else:
        all_epistemic, all_aleatoric = None, None
    return avg_loss, mcc, precision, recall, f1

def train_model(rank, world_size, config):
    """Main training loop with unfreezing and optimizer switching."""
    if config.DISTRIBUTED:
        setup_distributed(rank, world_size, config)

    model = ImprovedModel(config).to(config.DEVICE)
    if config.DISTRIBUTED:
        model = DDP(model, device_ids=[rank] if config.DEVICE.type == 'cuda' else None)

    optimizer = optim.AdamW(model.get_parameter_groups(), lr=config.ADAMW_LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX) if config.LR_SCHEDULE else None
    criterion = SimplifiedLoss(config)
    scaler = GradScaler(enabled=config.USE_AMP)

    train_loader, val_loader, test_loader = create_improved_data_loaders(config, rank if config.DISTRIBUTED else -1)

    best_mcc = -1.0
    patience_counter = 0
    top_k_checkpoints = []

    for epoch in range(config.EPOCHS):
        if config.DISTRIBUTED:
            train_loader.sampler.set_epoch(epoch)

        # Handle unfreezing and optimizer switching
        if epoch in config.UNFREEZE_EPOCHS or epoch == 16:
            if epoch == config.UNFREEZE_EPOCHS[0]:
                model.module.unfreeze_forensics_and_attention() if config.DISTRIBUTED else model.unfreeze_forensics_and_attention()
                logger.info(f"Unfreezing forensics, attention, fusion, and classifier at epoch {epoch+1}", extra={'rank': rank})
            elif epoch == config.UNFREEZE_EPOCHS[1]:
                model.module.unfreeze_convnext_layers() if config.DISTRIBUTED else model.unfreeze_convnext_layers()
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX) if config.LR_SCHEDULE else None
                logger.info(f"Unfreezing ConvNeXt and resetting scheduler at epoch {epoch+1}", extra={'rank': rank})
            elif epoch == config.UNFREEZE_EPOCHS[2]:
                model.module.unfreeze_vit_layers() if config.DISTRIBUTED else model.unfreeze_vit_layers()
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX) if config.LR_SCHEDULE else None
                logger.info(f"Unfreezing ViT and resetting scheduler at epoch {epoch+1}", extra={'rank': rank})
            elif epoch == 16:
                optimizer = optim.SGD(model.get_parameter_groups(), lr=config.SGD_LR, momentum=config.SGD_MOMENTUM, weight_decay=config.WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX) if config.LR_SCHEDULE else None
                logger.info(f"Switched to SGD optimizer for fine-tuning at epoch {epoch+1}", extra={'rank': rank})

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, config, epoch, rank)
        val_loss, val_mcc, _, _, _ = evaluate_model(model, val_loader, criterion, config, rank, phase="Validation")

        if scheduler:
            scheduler.step()

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}", extra={'rank': rank})
            if scheduler:
                lrs = [group['lr'] for group in optimizer.param_groups]
                logger.info(f"Learning rates: {lrs}", extra={'rank': rank})

            if (epoch + 1) % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                filename = f"checkpoint_epoch_{epoch+1}_mcc_{val_mcc:.4f}.pth"
                save_checkpoint(model, optimizer, scheduler, epoch + 1, val_mcc, config, rank, filename)
                top_k_checkpoints.append((val_mcc, filename))
                top_k_checkpoints = sorted(top_k_checkpoints, key=lambda x: x[0], reverse=True)[:config.SAVE_TOP_K_MODELS]

            if config.USE_MCC_FOR_BEST_MODEL and val_mcc > best_mcc:
                best_mcc = val_mcc
                save_checkpoint(model, optimizer, scheduler, epoch + 1, val_mcc, config, rank, "best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement", extra={'rank': rank})
                break

        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0:
        logger.info("Evaluating on test set...", extra={'rank': rank})
        _, test_mcc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, config, rank, phase="Test")
        logger.info(f"Test MCC: {test_mcc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}", extra={'rank': rank})

    if config.DISTRIBUTED:
        cleanup_distributed()

def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Improved Forensics Model Training")
    parser.add_argument('--convnext-backbone', type=str, default='convnext_tiny', choices=['convnext_tiny', 'convnext_small'])
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--dropout-rate', type=float, default=0.5)
    parser.add_argument('--freeze-backbones', action='store_true', default=True)
    parser.add_argument('--attention-dropout', type=float, default=0.3)
    parser.add_argument('--use-spectral-norm', action='store_true', default=False)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--master-addr', type=str, default='localhost')
    parser.add_argument('--master-port', type=str, default='12355')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--train-path', type=str, default='./dataset')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--class-names', type=str, default='real,semi-synthetic,synthetic')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--unfreeze-epochs', type=int, nargs='+', default=[1, 6, 11])
    parser.add_argument('--early-stopping-patience', type=int, default=8)
    parser.add_argument('--adamw-lr', type=float, default=3e-4)
    parser.add_argument('--sgd-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=2e-2)
    parser.add_argument('--sgd-momentum', type=float, default=0.9)
    parser.add_argument('--focal-alpha', type=float, nargs='+', default=[0.25, 0.5, 0.75])
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--class-weights', type=float, nargs='+', default=[1.0, 1.0, 1.0])
    parser.add_argument('--use-forensics-module', action='store_true', default=True)
    parser.add_argument('--use-uncertainty-estimation', action='store_true', default=False)
    parser.add_argument('--use-mixup', action='store_true', default=True)
    parser.add_argument('--use-cutmix', action='store_true', default=True)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--cutmix-alpha', type=float, default=0.2)
    parser.add_argument('--checkpoint-dir', type=str, default='improve1_checkpoints')
    parser.add_argument('--checkpoint-every-n-epochs', type=int, default=5)
    parser.add_argument('--use-mcc-for-best-model', action='store_true', default=True)
    parser.add_argument('--save-top-k-models', type=int, default=3)
    parser.add_argument('--forensics-loss-weight', type=float, default=0.5)
    parser.add_argument('--consistency-loss-weight', type=float, default=0.1)
    parser.add_argument('--max-checkpoints', type=int, default=10)
    parser.add_argument('--max-weight-factor', type=float, default=10.0)
    parser.add_argument('--use-sgd-fine-tune', action='store_true', default=True)
    parser.add_argument('--backbone-lr-factor', type=float, default=0.1)
    parser.add_argument('--lr-schedule', action='store_true', default=True)
    parser.add_argument('--t-max', type=int, default=10)

    args = parser.parse_args()
    config = ImprovedConfig(args)
    config.validate()

    if config.DISTRIBUTED:
        mp.spawn(train_model, args=(config.world_size, config), nprocs=config.world_size, join=True)
    else:
        train_model(0, 1, config)

if __name__ == "__main__":
    main()