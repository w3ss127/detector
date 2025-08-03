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
from torchvision.models import convnext_tiny, convnext_small, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights
import timm
import os
from datetime import timedelta
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import cv2
from PIL import Image
import sys
import glob

warnings.filterwarnings('ignore')

# Enhanced Custom formatter for logging with rank
class RankFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'rank'):
            record.rank = 0
        return super().format(record)

# Configure enhanced logging
def setup_logging(rank):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.handlers.clear()  # Clear existing handlers
    
    handler = logging.StreamHandler()
    formatter = RankFormatter(f'%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    return logger

class EnhancedQualityRobustConfig:
    def __init__(self):
        # Model Configuration
        self.MODEL_TYPE = "quality_robust_forensics_model"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 1536
        self.DROPOUT_RATE = 0.5
        self.FREEZE_BACKBONES = True
        self.ATTENTION_DROPOUT = 0.3
        self.USE_SPECTRAL_NORM = True
        
        # Distributed Training
        self.DISTRIBUTED = torch.cuda.device_count() > 1
        self.BACKEND = "nccl" if torch.cuda.is_available() else "gloo"
        self.WORLD_SIZE = torch.cuda.device_count() if self.DISTRIBUTED else 1
        
        # Training Configuration - OPTIMIZED FOR QUALITY ROBUSTNESS
        base_batch_size = 16  # Reduced from 32 to prevent memory issues
        self.BATCH_SIZE = base_batch_size // self.WORLD_SIZE if self.DISTRIBUTED else base_batch_size
        self.EPOCHS = 80  # Increased for quality robustness
        self.USE_AMP = True
        self.TRAIN_PATH = "datasets/train"
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = min(4, os.cpu_count() // self.WORLD_SIZE) if self.DISTRIBUTED else 4
        
        # Progressive Unfreezing Strategy - Modified for quality adaptation
        self.UNFREEZE_EPOCHS = [3, 10, 20, 35, 50, 65]
        self.FINE_TUNE_START_EPOCH = 40
        self.EARLY_STOPPING_PATIENCE = 60
        
        # Optimizer Configuration
        self.ADAMW_LR = 1.5e-4 * max(1, self.WORLD_SIZE ** 0.5)
        self.SGD_LR = 2e-6 * max(1, self.WORLD_SIZE ** 0.5)
        self.SGD_MOMENTUM = 0.95
        self.WEIGHT_DECAY = 4e-2
        
        # Enhanced Loss Configuration - QUALITY FOCUSED
        self.FOCAL_ALPHA = torch.tensor([1.0, 5.0, 3.5])  # Higher weight for semi-synthetic
        self.FOCAL_GAMMA = 5.0  # Increased focus on hard examples
        self.LABEL_SMOOTHING = 0.15
        self.CLASS_WEIGHTS = torch.tensor([1.0, 6.0, 3.0])  # Strong emphasis on semi-synthetic
        
        # Quality-Specific Module Configuration
        self.USE_QUALITY_ROBUST_FORENSICS = True
        self.USE_QUALITY_INVARIANT_LOSS = True
        self.USE_MULTI_QUALITY_TRAINING = True
        self.USE_UNCERTAINTY_ESTIMATION = True
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = 0.3
        self.USE_CUTMIX = True
        self.CUTMIX_ALPHA = 1.0
        
        # Enhanced Loss Weights - QUALITY FOCUSED
        self.CONTRASTIVE_WEIGHT = 0.6
        self.EVIDENTIAL_WEIGHT = 0.4
        self.BOUNDARY_WEIGHT = 0.4
        self.TRIPLET_WEIGHT = 0.4
        self.QUALITY_INVARIANT_WEIGHT = 0.8
        self.FORENSICS_CONSISTENCY_WEIGHT = 0.3
        
        # Quality-Specific Augmentation Parameters
        self.QUALITY_DEGRADATION_PROB = 0.8
        self.JPEG_COMPRESSION_PROB = 0.7
        self.NOISE_AUGMENTATION_PROB = 0.8
        self.BLUR_AUGMENTATION_PROB = 0.6
        self.COLOR_DISTORTION_PROB = 0.9
        
        # Multi-Quality Training Strategy
        self.QUALITY_LEVELS = ['high', 'medium', 'low']
        self.QUALITY_DISTRIBUTION = [0.2, 0.3, 0.5]
        
        # Checkpoint Configuration
        self.CHECKPOINT_DIR = "quality_robust_checkpoints"
        self.CHECKPOINT_EVERY_N_EPOCHS = 3
        self.USE_MCC_FOR_BEST_MODEL = True
        self.SAVE_TOP_K_MODELS = 7
        
        # Enhanced Regularization
        self.USE_GRADIENT_CLIPPING = True
        self.GRADIENT_CLIP_VALUE = 0.8
        self.USE_STOCHASTIC_DEPTH = True
        self.STOCHASTIC_DEPTH_PROB = 0.25
        self.USE_COSINE_ANNEALING = True
        self.USE_WARMUP = True
        self.WARMUP_EPOCHS = 8
        
        # Quality-Robustness Features
        self.USE_FREQUENCY_DOMAIN_LOSS = True
        self.USE_ARTIFACT_CONSISTENCY_LOSS = True
        self.USE_CROSS_QUALITY_CONTRASTIVE = True
        
        # Logging and Monitoring
        self.USE_WANDB = False
        self.WANDB_PROJECT = "quality-robust-deepfake-detection"
        self.LOG_EVERY_N_STEPS = 25

    def validate(self):
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0, "Batch size must be positive"
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0, "Epochs must be positive"
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES, "Class names must match NUM_CLASSES"
        assert self.CONVNEXT_BACKBONE in ["convnext_tiny", "convnext_small"], "Unsupported backbone"
        assert self.FINE_TUNE_START_EPOCH < self.EPOCHS, "Fine-tune start epoch must be less than total epochs"
        assert all(epoch <= self.EPOCHS for epoch in self.UNFREEZE_EPOCHS), "Unfreeze epochs must be within total epochs"
        if self.DISTRIBUTED:
            assert torch.cuda.device_count() > 1, "Multiple GPUs required for distributed training"

class QualityAgnosticForensicsModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.noise_analyzer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        self.freq_analyzer = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 96)
        )
        
        self.artifact_detector = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, 64)
        )
        
        self.texture_analyzer = nn.ModuleList([
            self._make_texture_branch(scale) for scale in [1, 2, 4]
        ])
        
        total_features = 256 + 128 + 96 + 64 + (32 * 3)
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=total_features, num_heads=8, dropout=0.2, batch_first=True
        )
        
        self.forensics_fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
    
    def _make_texture_branch(self, scale):
        return nn.Sequential(
            nn.AvgPool2d(scale) if scale > 1 else nn.Identity(),
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
    
    def extract_high_freq_features(self, x):
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                             dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 1, 1, 1).to(x.device)
        high_freq = F.conv2d(x, kernel, padding=1, groups=3)
        return torch.abs(high_freq)
    
    def forward(self, x):
        dct_feats = self.dct_analyzer(x)
        noise_feats = self.noise_analyzer(x)
        freq_feats = self.freq_analyzer(x)
        artifact_feats = self.artifact_detector(x)
        
        texture_feats = []
        for texture_branch in self.texture_analyzer:
            texture_feats.append(texture_branch(x))
        texture_combined = torch.cat(texture_feats, dim=1)
        
        all_features = torch.cat([
            dct_feats, noise_feats, freq_feats, 
            artifact_feats, texture_combined
        ], dim=1)
        
        all_features = all_features.unsqueeze(1)
        attended_features, _ = self.fusion_attention(
            all_features, all_features, all_features
        )
        attended_features = attended_features.squeeze(1)
        
        forensics_output = self.forensics_fusion(attended_features)
        return forensics_output

class EnhancedUncertaintyModule(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 4, num_classes),
            nn.Softplus()
        )
        self.aleatoric_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 3),
            nn.BatchNorm1d(input_dim // 3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 3, num_classes),
            nn.Softplus()
        )
    
    def forward(self, x):
        evidence = self.evidence_layer(x) + 1e-8
        aleatoric = self.aleatoric_layer(x) + 1e-8
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True) + 1e-8
        probs = alpha / S
        epistemic_uncertainty = self.num_classes / (S + 1e-8)
        aleatoric_uncertainty = aleatoric
        return probs, epistemic_uncertainty, aleatoric_uncertainty, alpha

class QualityRobustLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.focal_alpha = config.FOCAL_ALPHA
        self.focal_gamma = config.FOCAL_GAMMA
        self.class_weights = config.CLASS_WEIGHTS
        self.evidential_weight = config.EVIDENTIAL_WEIGHT
        self.contrastive_weight = config.CONTRASTIVE_WEIGHT
        self.boundary_weight = config.BOUNDARY_WEIGHT
        self.triplet_weight = config.TRIPLET_WEIGHT
        self.quality_invariant_weight = config.QUALITY_INVARIANT_WEIGHT
        self.label_smoothing = config.LABEL_SMOOTHING
        self.triplet_margin = 1.5
    
    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha.to(targets.device)[targets]  # Move to same device as targets
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def evidential_loss(self, alpha, targets, epoch=1):
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        S = torch.sum(alpha, dim=1, keepdim=True) + 1e-8
        likelihood_loss = torch.sum(targets_one_hot * (torch.digamma(S) - torch.digamma(alpha + 1e-8)), dim=1)
        alpha_tilde = targets_one_hot + (1 - targets_one_hot) * alpha
        alpha_tilde_sum = torch.sum(alpha_tilde, dim=1) + 1e-8
        kl_div = torch.lgamma(alpha_tilde_sum) - torch.sum(torch.lgamma(alpha_tilde + 1e-8), dim=1) + \
                 torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde + 1e-8) - torch.digamma(alpha_tilde_sum.unsqueeze(1) + 1e-8)), dim=1)
        annealing_coef = min(1.0, (epoch / 25.0) ** 2)
        return likelihood_loss.mean() + annealing_coef * 0.2 * kl_div.mean()
    
    def quality_invariant_loss(self, features, targets, epoch=1):
        features_norm = F.normalize(features, p=2, dim=1)
        batch_size = features.size(0)
        quality_invariant_loss = 0.0
        num_pairs = 0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if targets[i] == targets[j]:
                    similarity = F.cosine_similarity(
                        features_norm[i:i+1], features_norm[j:j+1]
                    )
                    quality_invariant_loss += (1 - similarity) ** 2
                    num_pairs += 1
        
        if num_pairs > 0:
            quality_invariant_loss /= num_pairs
            synthetic_mask = (targets == 2).float()
            if synthetic_mask.sum() > 0:
                synthetic_weight = 2.5 * synthetic_mask.mean()
                quality_invariant_loss *= (1.0 + synthetic_weight)
        
        return quality_invariant_loss
    
    def enhanced_contrastive_loss(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features, features.t())
        labels_expanded = labels.unsqueeze(1)
        same_class_mask = (labels_expanded == labels_expanded.t()).float()
        diff_class_mask = 1 - same_class_mask
        
        semi_synthetic_mask = (labels == 1).float().unsqueeze(1)
        semi_synthetic_pairs = semi_synthetic_mask * semi_synthetic_mask.t()
        
        pos_loss = same_class_mask * (1 - similarity_matrix) ** 2
        neg_loss = diff_class_mask * torch.clamp(similarity_matrix - 0.2, min=0) ** 2
        semi_loss = semi_synthetic_pairs * (1 - similarity_matrix) ** 2
        
        total_loss = pos_loss.sum() + neg_loss.sum() + 4.0 * semi_loss.sum()
        batch_size_squared = labels.size(0) ** 2
        return total_loss / (batch_size_squared + 1e-8)
    
    def enhanced_boundary_loss(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        real_mask = (targets == 0).float()
        synthetic_mask = (targets == 2).float()
        semi_mask = (targets == 1).float()
        
        real_boundary = real_mask * (probs[:, 1] + 0.6 * probs[:, 2])
        synthetic_boundary = synthetic_mask * (probs[:, 1] + 0.4 * probs[:, 0])
        semi_boundary = semi_mask * (1 - probs[:, 1] + 0.3 * (probs[:, 0] + probs[:, 2]))
        
        total_samples = targets.size(0) + 1e-8
        return (real_boundary.sum() + synthetic_boundary.sum() + 4.5 * semi_boundary.sum()) / total_samples
    
    def enhanced_triplet_loss(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        triplet_losses = []
        
        for i in range(features.size(0)):
            anchor_label = labels[i]
            anchor_feat = features[i]
            pos_mask = (labels == anchor_label) & (torch.arange(features.size(0)).to(features.device) != i)
            if pos_mask.sum() == 0:
                continue
            neg_mask = (labels != anchor_label)
            if neg_mask.sum() == 0:
                continue
            
            pos_distances = torch.norm(features[pos_mask] - anchor_feat.unsqueeze(0), p=2, dim=1)
            neg_distances = torch.norm(features[neg_mask] - anchor_feat.unsqueeze(0), p=2, dim=1)
            hardest_pos_dist = pos_distances.max()
            hardest_neg_dist = neg_distances.min()
            
            margin = self.triplet_margin * (2.5 if anchor_label == 1 else 1.0)
            triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
            triplet_losses.append(triplet_loss)
        
        return torch.stack(triplet_losses).mean() if triplet_losses else torch.tensor(0.0, device=features.device)
    
    def forward(self, logits, targets, features=None, alpha=None, epoch=1):
        focal_loss = self.focal_loss(logits, targets)
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights.to(logits.device))  # Move to same device
        boundary_loss = self.enhanced_boundary_loss(logits, targets)
        
        total_loss = 0.4 * focal_loss + 0.3 * ce_loss + self.boundary_weight * boundary_loss
        
        if features is not None:
            contrastive_loss = self.enhanced_contrastive_loss(features, targets)
            triplet_loss = self.enhanced_triplet_loss(features, targets)
            quality_loss = self.quality_invariant_loss(features, targets, epoch)
            
            total_loss += self.contrastive_weight * contrastive_loss
            total_loss += self.triplet_weight * triplet_loss
            total_loss += self.quality_invariant_weight * quality_loss
        
        if alpha is not None:
            evidential_loss = self.evidential_loss(alpha, targets, epoch)
            total_loss += self.evidential_weight * evidential_loss
        
        return total_loss

class EnhancedAttentionModule(nn.Module):
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        
        preferred_heads = [16, 12, 8, 4, 2, 1]
        self.num_heads = 1
        for heads in preferred_heads:
            if in_features % heads == 0:
                self.num_heads = heads
                break
        
        if in_features % self.num_heads != 0:
            target_heads = 8
            padded_features = ((in_features + target_heads - 1) // target_heads) * target_heads
            self.feature_projection = nn.Linear(in_features, padded_features)
            self.attention_features = padded_features
            self.num_heads = target_heads
        else:
            self.feature_projection = nn.Identity()
            self.attention_features = in_features
        
        self.forensics_attention = nn.MultiheadAttention(
            embed_dim=self.attention_features,
            num_heads=self.num_heads,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True
        )
        
        self.channel_attention = nn.Sequential(
            nn.Linear(self.attention_features, max(self.attention_features // 12, 64)),
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(max(self.attention_features // 12, 64), max(self.attention_features // 4, 128)),
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(max(self.attention_features // 4, 128), self.attention_features),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(self.attention_features, max(self.attention_features // 8, 64)),
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(max(self.attention_features // 8, 64), self.attention_features),
            nn.Sigmoid()
        )
        
        if self.attention_features != in_features:
            self.output_projection = nn.Linear(self.attention_features, in_features)
        else:
            self.output_projection = nn.Identity()
        
        if config.USE_SPECTRAL_NORM:
            if isinstance(self.channel_attention[0], nn.Linear):
                self.channel_attention[0] = nn.utils.spectral_norm(self.channel_attention[0])
            if isinstance(self.channel_attention[3], nn.Linear):
                self.channel_attention[3] = nn.utils.spectral_norm(self.channel_attention[3])
            if isinstance(self.channel_attention[6], nn.Linear):
                self.channel_attention[6] = nn.utils.spectral_norm(self.channel_attention[6])
            
            if isinstance(self.spatial_attention[0], nn.Linear):
                self.spatial_attention[0] = nn.utils.spectral_norm(self.spatial_attention[0])
            if isinstance(self.spatial_attention[3], nn.Linear):
                self.spatial_attention[3] = nn.utils.spectral_norm(self.spatial_attention[3])
    
    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() != 2:
            x = x.view(batch_size, -1)
        
        x_projected = self.feature_projection(x)
        x_reshaped = x_projected.unsqueeze(1)
        attn_output, _ = self.forensics_attention(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.squeeze(1)
        
        channel_weights = self.channel_attention(x_projected)
        channel_attended = x_projected * channel_weights
        
        spatial_weights = self.spatial_attention(x_projected)
        spatial_attended = x_projected * spatial_weights
        
        attended_features = x_projected + 0.3 * attn_output + 0.4 * channel_attended + 0.3 * spatial_attended
        output = self.output_projection(attended_features)
        return output

class QualityRobustModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        
        self.convnext.classifier = nn.Identity()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        if config.FREEZE_BACKBONES:
            self.freeze_backbones()
        
        convnext_features = 768
        vit_features = self.vit.num_features
        forensics_features = 128 if config.USE_QUALITY_ROBUST_FORENSICS else 0
        total_features = convnext_features + vit_features + forensics_features
        
        if config.USE_QUALITY_ROBUST_FORENSICS:
            self.forensics_module = QualityAgnosticForensicsModule(config)
        
        self.attention_module = EnhancedAttentionModule(total_features, config)
        
        if config.USE_UNCERTAINTY_ESTIMATION:
            self.uncertainty_module = EnhancedUncertaintyModule(config.HIDDEN_DIM // 4, config.NUM_CLASSES)
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.BatchNorm1d(config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.BatchNorm1d(config.HIDDEN_DIM // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE * 0.8)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM // 4, config.HIDDEN_DIM // 6),
            nn.BatchNorm1d(config.HIDDEN_DIM // 6),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(config.HIDDEN_DIM // 6, config.HIDDEN_DIM // 8),
            nn.BatchNorm1d(config.HIDDEN_DIM // 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.HIDDEN_DIM // 8, config.NUM_CLASSES)
        )
        
        if config.USE_SPECTRAL_NORM:
            if isinstance(self.fusion[0], nn.Linear):
                self.fusion[0] = nn.utils.spectral_norm(self.fusion[0])
            if isinstance(self.fusion[4], nn.Linear):
                self.fusion[4] = nn.utils.spectral_norm(self.fusion[4])
            if isinstance(self.fusion[8], nn.Linear):
                self.fusion[8] = nn.utils.spectral_norm(self.fusion[8])
            
            if isinstance(self.classifier[0], nn.Linear):
                self.classifier[0] = nn.utils.spectral_norm(self.classifier[0])
            if isinstance(self.classifier[4], nn.Linear):
                self.classifier[4] = nn.utils.spectral_norm(self.classifier[4])
            if isinstance(self.classifier[8], nn.Linear):
                self.classifier[8] = nn.utils.spectral_norm(self.classifier[8])
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbones(self):
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_convnext_layers(self, num_layers=None):
        if num_layers is None:
            for param in self.convnext.parameters():
                param.requires_grad = True
        else:
            layers = list(self.convnext.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True
    
    def unfreeze_vit_layers(self, num_layers=None):
        if num_layers is None:
            for param in self.vit.parameters():
                param.requires_grad = True
        else:
            layers = list(self.vit.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True
    
    def unfreeze_forensics_and_attention(self):
        if self.config.USE_QUALITY_ROBUST_FORENSICS:
            for param in self.forensics_module.parameters():
                param.requires_grad = True
        for param in self.attention_module.parameters():
            param.requires_grad = True
    
    def unfreeze_classifier(self):
        for param in self.fusion.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        convnext_features = self.convnext(x)
        if convnext_features.dim() == 4:
            convnext_features = convnext_features.mean(dim=[2, 3])
        
        vit_features = self.vit(x)
        
        combined_features = [convnext_features, vit_features]
        
        if self.config.USE_QUALITY_ROBUST_FORENSICS:
            forensics_features = self.forensics_module(x)
            combined_features.append(forensics_features)
        
        features = torch.cat(combined_features, dim=1)
        
        attended_features = self.attention_module(features)
        
        fused_features = self.fusion(attended_features)
        
        logits = self.classifier(fused_features)
        
        if self.config.USE_UNCERTAINTY_ESTIMATION:
            probs, epistemic_unc, aleatoric_unc, alpha = self.uncertainty_module(fused_features)
            return logits, fused_features, (probs, epistemic_unc, aleatoric_unc, alpha)
        else:
            return logits, fused_features

class QualityRobustDataset(Dataset):
    def __init__(self, data_path, transform=None, class_names=None, cache_tensors=False, quality_level='mixed'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.class_names = class_names or ["real", "semi-synthetic", "synthetic"]
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.cache_tensors = cache_tensors
        self.tensor_cache = {}
        self.quality_level = quality_level
        self._load_tensor_samples()
    
    def _load_tensor_samples(self):
        total_samples = 0
        for class_name in self.class_names:
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist")
                continue
            pt_files = list(class_dir.glob("*.pt"))
            if not pt_files:
                print(f"Warning: No .pt files found in {class_name}")
                continue
            for pt_file in pt_files:
                try:
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    if isinstance(tensor_data, dict):
                        if 'images' in tensor_data:
                            tensors = tensor_data['images']
                        elif 'data' in tensor_data:
                            tensors = tensor_data['data']
                        elif 'tensors' in tensor_data:
                            tensors = tensor_data['tensors']
                        else:
                            tensors = next(iter(tensor_data.values()))
                    else:
                        tensors = tensor_data
                    
                    if not isinstance(tensors, torch.Tensor):
                        tensors = torch.tensor(tensors)
                    
                    if tensors.dim() == 4:
                        if tensors.shape[0] != 5000:
                            print(f"Warning: Expected 5000 images, got {tensors.shape[0]} in {pt_file}")
                        if tensors.shape[-1] == 3 and tensors.shape[1] == 224:
                            tensors = tensors.permute(0, 3, 1, 2)
                        if tensors.shape[1:] != (3, 224, 224):
                            print(f"Warning: Unexpected tensor shape: {tensors.shape} in {pt_file}")
                    else:
                        print(f"Error: Invalid tensor dimensions: {tensors.shape} in {pt_file}")
                        continue
                    
                    for tensor_idx in range(tensors.shape[0]):
                        self.samples.append((str(pt_file), tensor_idx, self.class_to_idx[class_name]))
                        total_samples += 1
                except Exception as e:
                    print(f"Error loading {pt_file}: {e}")
                    continue
            print(f"Total samples in {class_name}: {len(pt_files) * 5000}")
        print(f"Total dataset samples: {total_samples}")
    
    def _load_tensor_file(self, pt_file_path):
        if self.cache_tensors and pt_file_path in self.tensor_cache:
            return self.tensor_cache[pt_file_path]
        
        try:
            tensor_data = torch.load(pt_file_path, map_location='cpu')
            if isinstance(tensor_data, dict):
                if 'images' in tensor_data:
                    tensors = tensor_data['images']
                elif 'data' in tensor_data:
                    tensors = tensor_data['data']
                elif 'tensors' in tensor_data:
                    tensors = tensor_data['tensors']
                else:
                    tensors = next(iter(tensor_data.values()))
            else:
                tensors = tensor_data
            
            if not isinstance(tensors, torch.Tensor):
                tensors = torch.tensor(tensors)
            
            if tensors.dim() == 4:
                if tensors.shape[0] != 5000:
                    print(f"Warning: Expected 5000 images, got {tensors.shape[0]} in {pt_file_path}")
                if tensors.shape[-1] == 3 and tensors.shape[1] == 224:
                    tensors = tensors.permute(0, 3, 1, 2)
                if tensors.shape[1:] != (3, 224, 224):
                    print(f"Warning: Unexpected tensor shape: {tensors.shape} in {pt_file_path}")
            else:
                print(f"Error: Invalid tensor dimensions: {tensors.shape} in {pt_file_path}")
                return None
            
            if tensors.max() > 1.0:
                tensors = tensors.float() / 255.0
            else:
                tensors = tensors.float()
            
            if self.cache_tensors:
                self.tensor_cache[pt_file_path] = tensors
            
            return tensors
        except Exception as e:
            print(f"Error loading tensor file {pt_file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pt_file_path, tensor_idx, label = self.samples[idx]
        tensors = self._load_tensor_file(pt_file_path)
        
        if tensors is None:
            print(f"Warning: Failed to load tensor from {pt_file_path}, returning dummy tensor")
            image = torch.zeros(3, 224, 224, dtype=torch.float32)
            return image, label
        
        if tensor_idx >= tensors.shape[0]:
            print(f"Warning: Tensor index {tensor_idx} out of range for {pt_file_path}")
            tensor_idx = 0
        
        image = tensors[tensor_idx].clone()
        
        if self.transform:
            try:
                image_np = image.permute(1, 2, 0).numpy()
                image_np = (image_np * 255).astype(np.uint8)
                augmented = self.transform(image=image_np)
                image = augmented['image']
            except Exception as e:
                print(f"Warning: Error applying transform: {e}, using original tensor")
                if image.max() <= 1.0:
                    pass
                else:
                    image = image / 255.0
        
        return image, label

def create_quality_robust_augmentations(config):
    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=10, p=0.4),
        A.OneOf([
            A.ImageCompression(quality_lower=15, quality_upper=75, p=0.5),
            A.Downscale(scale_min=0.25, scale_max=0.75, interpolation=cv2.INTER_LINEAR, p=0.3),
            A.Blur(blur_limit=7, p=0.2),
        ], p=config.QUALITY_DEGRADATION_PROB),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 45.0), p=0.4),
            A.ISONoise(color_shift=(0.01, 0.15), intensity=(0.1, 0.9), p=0.3),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], p=0.3),
        ], p=config.NOISE_AUGMENTATION_PROB),
        A.OneOf([
            A.Blur(blur_limit=8, p=0.4),
            A.GaussianBlur(blur_limit=8, p=0.4),
            A.MotionBlur(blur_limit=8, p=0.2),
        ], p=config.BLUR_AUGMENTATION_PROB),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.4),
            A.HueSaturationValue(hue_shift_limit=35, sat_shift_limit=60, val_shift_limit=35, p=0.4),
            A.CLAHE(clip_limit=6, p=0.25),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.35),
            A.ChannelShuffle(p=0.1),
        ], p=config.COLOR_DISTORTION_PROB),
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.9), lightness=(0.4, 1.6), p=0.3),
            A.Emboss(alpha=(0.1, 0.6), strength=(0.1, 0.8), p=0.2),
            A.Posterize(num_bits=3, p=0.15),
            A.Equalize(p=0.2),
            A.Solarize(threshold=128, p=0.15),
        ], p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.2),
        ], p=0.3),
        A.OneOf([
            A.CoarseDropout(max_holes=20, max_height=40, max_width=40, 
                           min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.4),
            A.GridDropout(ratio=0.4, random_offset=True, p=0.3),
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
        ], p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.15),
        ], p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def create_safe_quality_robust_augmentations(config):
    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=10, p=0.4),
        A.OneOf([
            A.Blur(blur_limit=8, p=0.4),
            A.GaussianBlur(blur_limit=8, p=0.4),
            A.MotionBlur(blur_limit=8, p=0.2),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 45.0), p=0.5),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], p=0.3),
        ], p=0.8),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.4),
            A.HueSaturationValue(hue_shift_limit=35, sat_shift_limit=60, val_shift_limit=35, p=0.4),
            A.CLAHE(clip_limit=6, p=0.3),
        ], p=0.9),
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.9), lightness=(0.4, 1.6), p=0.3),
            A.Emboss(alpha=(0.1, 0.6), strength=(0.1, 0.8), p=0.2),
            A.Posterize(num_bits=3, p=0.2),
            A.Equalize(p=0.3),
        ], p=0.5),
        A.OneOf([
            A.CoarseDropout(max_holes=15, max_height=30, max_width=30, 
                           min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.5),
        ], p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def create_compatible_augmentations(config):
    available_transforms = []
    try:
        A.ImageCompression(quality_lower=50, quality_upper=100, p=1.0)
        available_transforms.append('ImageCompression')
    except:
        pass
    try:
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)
        available_transforms.append('ISONoise')
    except:
        pass
    try:
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        available_transforms.append('ColorJitter')
    except:
        pass
    try:
        A.GridDropout(ratio=0.2, p=1.0)
        available_transforms.append('GridDropout')
    except:
        pass
    try:
        A.ChannelDropout(channel_drop_range=(1, 1), p=1.0)
        available_transforms.append('ChannelDropout')
    except:
        pass
    
    transforms_list = [
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=10, p=0.4),
    ]
    
    quality_augs = [A.Blur(blur_limit=8, p=0.4), A.GaussianBlur(blur_limit=8, p=0.4)]
    if 'ImageCompression' in available_transforms:
        quality_augs.append(A.ImageCompression(quality_lower=15, quality_upper=75, p=0.2))
    transforms_list.append(A.OneOf(quality_augs, p=0.7))
    
    noise_augs = [A.GaussNoise(var_limit=(5.0, 45.0), p=0.5)]
    if 'ISONoise' in available_transforms:
        noise_augs.append(A.ISONoise(color_shift=(0.01, 0.15), intensity=(0.1, 0.9), p=0.3))
    transforms_list.append(A.OneOf(noise_augs, p=0.8))
    
    color_augs = [
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.4),
        A.HueSaturationValue(hue_shift_limit=35, sat_shift_limit=60, val_shift_limit=35, p=0.4),
        A.CLAHE(clip_limit=6, p=0.3),
    ]
    if 'ColorJitter' in available_transforms:
        color_augs.append(A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.3))
    transforms_list.append(A.OneOf(color_augs, p=0.9))
    
    transforms_list.extend([
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.9), lightness=(0.4, 1.6), p=0.3),
            A.Emboss(alpha=(0.1, 0.6), strength=(0.1, 0.8), p=0.2),
            A.Posterize(num_bits=3, p=0.2),
            A.Equalize(p=0.3),
        ], p=0.5),
    ])
    
    dropout_augs = [
        A.CoarseDropout(max_holes=15, max_height=30, max_width=30, 
                       min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.5)
    ]
    if 'GridDropout' in available_transforms:
        dropout_augs.append(A.GridDropout(ratio=0.4, random_offset=True, p=0.3))
    if 'ChannelDropout' in available_transforms:
        dropout_augs.append(A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2))
    
    transforms_list.append(A.OneOf(dropout_augs, p=0.4))
    
    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    train_transform = A.Compose(transforms_list)
    
    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    print(f"Created augmentation pipeline with available transforms: {available_transforms}")
    
    return train_transform, val_transform

def enhanced_progressive_unfreeze(model, epoch, config, logger):
    unfroze_this_epoch = False
    if epoch in config.UNFREEZE_EPOCHS:
        unfreeze_stage = config.UNFREEZE_EPOCHS.index(epoch) + 1
        if unfreeze_stage == 1:
            model.unfreeze_forensics_and_attention()
            unfroze_this_epoch = True
        elif unfreeze_stage == 2:
            model.unfreeze_classifier()
            unfroze_this_epoch = True
        elif unfreeze_stage == 3:
            model.unfreeze_convnext_layers(num_layers=60)
            unfroze_this_epoch = True
        elif unfreeze_stage == 4:
            model.unfreeze_vit_layers(num_layers=30)
            unfroze_this_epoch = True
        elif unfreeze_stage == 5:
            model.unfreeze_convnext_layers()
            model.unfreeze_vit_layers(num_layers=60)
            unfroze_this_epoch = True
        elif unfreeze_stage == 6:
            model.unfreeze_vit_layers()
            model.unfreeze_convnext_layers()
            unfroze_this_epoch = True
        
        if unfroze_this_epoch:
            trainable_params = model.get_trainable_params()
            logger.info(f"Progressive unfreezing stage {unfreeze_stage}: {trainable_params:,} trainable parameters")
    
    return unfroze_this_epoch

def save_enhanced_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, config, filename, is_best=False, logger=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'config': config.__dict__,
        'is_best': is_best,
        'trainable_params': model.module.get_trainable_params() if hasattr(model, 'module') else model.get_trainable_params()
    }
    torch.save(checkpoint, filename)
    if logger:
        logger.info(f"Checkpoint saved: {filename}")
        if is_best:
            logger.info(f"New best model - MCC: {metrics.get('mcc', 0):.4f}, Semi-synthetic F1: {metrics.get('semi_synthetic_f1', 0):.4f}")

def plot_enhanced_metrics(history, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    if 'val_mcc' in history:
        axes[0, 1].plot(history['val_mcc'], label='Overall MCC', color='green', linewidth=2)
        if 'semi_mcc' in history:
            axes[0, 1].plot(history['semi_mcc'], label='Semi-synthetic MCC', color='purple', linewidth=2)
        axes[0, 1].set_title('Matthews Correlation Coefficient', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MCC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'real_mcc' in history and 'semi_mcc' in history and 'synthetic_mcc' in history:
        axes[0, 2].plot(history['real_mcc'], label='Real MCC', color='blue', linewidth=2)
        axes[0, 2].plot(history['semi_mcc'], label='Semi-synthetic MCC', color='orange', linewidth=2)
        axes[0, 2].plot(history['synthetic_mcc'], label='Synthetic MCC', color='red', linewidth=2)
        axes[0, 2].set_title('Per-Class MCC', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MCC')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    if 'val_accuracy' in history:
        axes[1, 0].plot(history['val_accuracy'], label='Overall Accuracy', color='cyan', linewidth=2)
        if 'real_acc' in history:
            axes[1, 0].plot(history['real_acc'], label='Real Accuracy', color='blue', linewidth=1)
        if 'semi_acc' in history:
            axes[1, 0].plot(history['semi_acc'], label='Semi-synthetic Accuracy', color='orange', linewidth=1)
        if 'synthetic_acc' in history:
            axes[1, 0].plot(history['synthetic_acc'], label='Synthetic Accuracy', color='red', linewidth=1)
        axes[1, 0].set_title('Accuracy Metrics', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'macro_f1' in history and 'weighted_f1' in history:
        axes[1, 1].plot(history['macro_f1'], label='Macro F1', color='magenta', linewidth=2)
        axes[1, 1].plot(history['weighted_f1'], label='Weighted F1', color='yellow', linewidth=2)
        if 'semi_f1' in history:
            axes[1, 1].plot(history['semi_f1'], label='Semi-synthetic F1', color='purple', linewidth=2)
        axes[1, 1].set_title('F1 Scores', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    if 'semi_precision' in history and 'semi_recall' in history and 'semi_f1' in history:
        axes[1, 2].plot(history['semi_precision'], label='Precision', color='blue', linewidth=2)
        axes[1, 2].plot(history['semi_recall'], label='Recall', color='red', linewidth=2)
        axes[1, 2].plot(history['semi_f1'], label='F1-Score', color='green', linewidth=2)
        axes[1, 2].set_title('Semi-synthetic Class Metrics', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    if 'learning_rate' in history:
        axes[2, 0].plot(history['learning_rate'], label='Learning Rate', color='orange', linewidth=2)
        axes[2, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Learning Rate')
        axes[2, 0].set_yscale('log')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    if 'semi_confusion_rate' in history:
        axes[2, 1].plot(history['semi_confusion_rate'], label='Semi-synthetic Confusion Rate', color='red', linewidth=2)
        axes[2, 1].set_title('Semi-synthetic Confusion Rate', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Confusion Rate')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    if 'mean_uncertainty' in history:
        axes[2, 2].plot(history['mean_uncertainty'], label='Mean Uncertainty', color='purple', linewidth=2)
        if 'uncertainty_std' in history:
            axes[2, 2].fill_between(range(len(history['mean_uncertainty'])), 
                                   np.array(history['mean_uncertainty']) - np.array(history['uncertainty_std']),
                                   np.array(history['mean_uncertainty']) + np.array(history['uncertainty_std']),
                                   alpha=0.3, color='purple')
        axes[2, 2].set_title('Uncertainty Estimation', fontsize=12, fontweight='bold')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Uncertainty')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_enhanced_confusion_matrix(cm, class_names, save_path, per_class_mcc=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Quality-Robust Model Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    total = np.sum(cm)
    if total > 0:
        accuracy = np.trace(cm) / total
        plt.figtext(0.15, 0.02, f'Overall Accuracy: {accuracy:.4f}', fontsize=12, fontweight='bold')
        
        y_pos = 0.08
        for i, class_name in enumerate(class_names):
            if i < len(cm):
                class_acc = cm[i, i] / (cm[i].sum() + 1e-8)
                mcc_text = f', MCC: {per_class_mcc[i]:.4f}' if per_class_mcc and i < len(per_class_mcc) else ''
                plt.figtext(0.15, y_pos - i*0.02, 
                           f'{class_name} - Acc: {class_acc:.4f}{mcc_text}', 
                           fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def log_epoch_results(epoch, train_loss, val_loss, val_metrics, learning_rates, logger):
    logger.info(f"=" * 80)
    logger.info(f"EPOCH {epoch} RESULTS - QUALITY ROBUST MODEL")
    logger.info(f"=" * 80)
    logger.info(f"Train Loss: {train_loss:.6f}")
    logger.info(f"Val Loss: {val_loss:.6f}")
    logger.info(f"Val Accuracy: {val_metrics['accuracy']:.6f}")
    logger.info(f"Overall MCC: {val_metrics['mcc']:.6f}")
    logger.info(f"Per-class MCC:")
    class_names = ['Real', 'Semi-synthetic', 'Synthetic']
    per_class_mcc = val_metrics.get('per_class_mcc', [0, 0, 0])
    for i, name in enumerate(class_names):
        mcc_val = per_class_mcc[i] if i < len(per_class_mcc) else 0.0
        logger.info(f"  {name}: {mcc_val:.6f}")
    
    logger.info(f"Macro F1: {val_metrics.get('macro_f1', 0.0):.6f}")
    logger.info(f"Weighted F1: {val_metrics.get('weighted_f1', 0.0):.6f}")
    
    semi_metrics = val_metrics.get('per_class_metrics', [{}, {}, {}])
    if len(semi_metrics) > 1:
        semi = semi_metrics[1]
        logger.info(f"Semi-synthetic Performance:")
        logger.info(f"  Precision: {semi.get('precision', 0.0):.6f}")
        logger.info(f"  Recall: {semi.get('recall', 0.0):.6f}")
        logger.info(f"  F1-Score: {semi.get('f1', 0.0):.6f}")
        logger.info(f"  MCC: {semi.get('mcc', 0.0):.6f}")
    
    if learning_rates:
        logger.info(f"Learning Rates:")
        for param_group_idx, lr in enumerate(learning_rates):
            logger.info(f"  Group {param_group_idx}: {lr:.8f}")
    
    if 'mean_uncertainty' in val_metrics:
        logger.info(f"Mean Uncertainty: {val_metrics['mean_uncertainty']:.6f}")
        logger.info(f"Uncertainty Std: {val_metrics['uncertainty_std']:.6f}")
    
    logger.info(f"=" * 80)

def calculate_enhanced_metrics(y_true, y_pred, class_names, y_probs=None, uncertainties=None):
    metrics = {}
    
    metrics['accuracy'] = np.mean(y_true == y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_class_metrics = []
    per_class_mcc = []
    
    for i in range(len(class_names)):
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)
        class_mcc = matthews_corrcoef(binary_true, binary_pred)
        per_class_mcc.append(class_mcc)
        
        per_class_metrics.append({
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'mcc': class_mcc,
            'support': support[i]
        })
    
    metrics['per_class_mcc'] = per_class_mcc
    metrics['per_class_metrics'] = per_class_metrics
    
    metrics['macro_f1'] = np.mean(f1)
    metrics['weighted_f1'] = np.average(f1, weights=support)
    
    if len(per_class_metrics) > 1:
        semi_metrics = per_class_metrics[1]
        metrics['semi_synthetic_precision'] = semi_metrics['precision']
        metrics['semi_synthetic_recall'] = semi_metrics['recall']
        metrics['semi_synthetic_f1'] = semi_metrics['f1']
        metrics['semi_synthetic_mcc'] = semi_metrics['mcc']
        
        semi_true_mask = (y_true == 1)
        semi_predictions = y_pred[semi_true_mask]
        if len(semi_predictions) > 0:
            confused_with_real = np.sum(semi_predictions == 0) / len(semi_predictions)
            confused_with_synthetic = np.sum(semi_predictions == 2) / len(semi_predictions)
            metrics['semi_confused_with_real'] = confused_with_real
            metrics['semi_confused_with_synthetic'] = confused_with_synthetic
            metrics['semi_confusion_rate'] = 1 - semi_metrics['recall']
    
    if uncertainties is not None:
        metrics['mean_uncertainty'] = np.mean(uncertainties)
        metrics['uncertainty_std'] = np.std(uncertainties)
        
        correct_predictions = y_true == y_pred
        if len(uncertainties) == len(correct_predictions):
            correct_uncertainty = uncertainties[correct_predictions]
            incorrect_uncertainty = uncertainties[~correct_predictions]
            if len(correct_uncertainty) > 0 and len(incorrect_uncertainty) > 0:
                metrics['correct_mean_uncertainty'] = np.mean(correct_uncertainty)
                metrics['incorrect_mean_uncertainty'] = np.mean(incorrect_uncertainty)
    
    if y_probs is not None:
        from sklearn.metrics import log_loss, roc_auc_score
        try:
            metrics['cross_entropy'] = log_loss(y_true, y_probs)
            if len(np.unique(y_true)) > 2:
                auc_scores = []
                for i in range(len(class_names)):
                    binary_true = (y_true == i).astype(int)
                    if len(np.unique(binary_true)) > 1:
                        auc = roc_auc_score(binary_true, y_probs[:, i])
                        auc_scores.append(auc)
                if auc_scores:
                    metrics['macro_auc'] = np.mean(auc_scores)
                    metrics['per_class_auc'] = auc_scores
        except Exception as e:
            print(f"Warning: Could not calculate probability-based metrics: {e}")
    
    return metrics

def train_epoch(model, dataloader, criterion, optimizer, scaler, config, epoch, logger):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast(enabled=config.USE_AMP):
            if config.USE_MIXUP and np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, targets, config.MIXUP_ALPHA)
                
            if config.USE_UNCERTAINTY_ESTIMATION:
                logits, features, uncertainty_outputs = model(images)
                probs, epistemic_unc, aleatoric_unc, alpha = uncertainty_outputs
                
                if config.USE_MIXUP and 'targets_a' in locals():
                    loss = lam * criterion(logits, targets_a, features, alpha, epoch) + \
                           (1 - lam) * criterion(logits, targets_b, features, alpha, epoch)
                else:
                    loss = criterion(logits, targets, features, alpha, epoch)
            else:
                logits, features = model(images)
                
                if config.USE_MIXUP and 'targets_a' in locals():
                    loss = lam * criterion(logits, targets_a, features, None, epoch) + \
                           (1 - lam) * criterion(logits, targets_b, features, None, epoch)
                else:
                    loss = criterion(logits, targets, features, None, epoch)
        
        scaler.scale(loss).backward()
        
        if config.USE_GRADIENT_CLIPPING:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % config.LOG_EVERY_N_STEPS == 0:
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}',
                'LR': f'{current_lr:.2e}'
            })
    
    return total_loss / num_batches

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def validate_epoch(model, dataloader, criterion, config, epoch, logger):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probs = []
    all_uncertainties = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch}")
        
        for images, targets in progress_bar:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            with autocast(enabled=config.USE_AMP):
                if config.USE_UNCERTAINTY_ESTIMATION:
                    logits, features, uncertainty_outputs = model(images)
                    probs, epistemic_unc, aleatoric_unc, alpha = uncertainty_outputs
                    loss = criterion(logits, targets, features, alpha, epoch)
                    
                    total_uncertainty = epistemic_unc.squeeze() + aleatoric_unc.mean(dim=1)
                    all_uncertainties.extend(total_uncertainty.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                else:
                    logits, features = model(images)
                    loss = criterion(logits, targets, features, None, epoch)
                    all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs) if all_probs else None
    all_uncertainties = np.array(all_uncertainties) if all_uncertainties else None
    
    metrics = calculate_enhanced_metrics(
        all_targets, all_predictions, config.CLASS_NAMES, 
        all_probs, all_uncertainties
    )
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics

def find_free_port():
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a random free port
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def main(rank=0, world_size=1):
    try:
        # Initialize logger first to ensure it's available for error handling
        logger = setup_logging(rank)
        
        # Setup distributed training
        if world_size > 1:
            # Find a free port dynamically
            if rank == 0:
                port = find_free_port()
                # Share the port with other processes (e.g., via environment variable or file)
                os.environ['MASTER_PORT'] = str(port)
            else:
                # Wait for rank 0 to set the port
                while 'MASTER_PORT' not in os.environ:
                    time.sleep(1)
            
            os.environ['MASTER_ADDR'] = 'localhost'
            logger.info(f"Rank {rank}: Using port {os.environ['MASTER_PORT']} for distributed training")
            
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://localhost:{os.environ['MASTER_PORT']}",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=1800)
            )
            dist.barrier()
            torch.cuda.set_device(rank)

        # Initialize configuration
        config = EnhancedQualityRobustConfig()
        config.validate()
        
        if rank == 0:
            logger.info(f"Starting training with {world_size} GPUs")
            logger.info(f"Configuration: {config.__dict__}")

        # Create augmentations
        try:
            train_transform, val_transform = create_quality_robust_augmentations(config)
        except Exception as e:
            logger.warning(f"Failed to create full augmentations: {e}. Falling back to safe augmentations")
            train_transform, val_transform = create_safe_quality_robust_augmentations(config)

        # Initialize dataset
        dataset = QualityRobustDataset(
            data_path=config.TRAIN_PATH,
            transform=train_transform,
            class_names=config.CLASS_NAMES,
            cache_tensors=True,
            quality_level='mixed'
        )

        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform

        # Create samplers
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        # Create dataloaders
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
            sampler=val_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        # Initialize model
        model = QualityRobustModel(config)
        model = model.cuda()
        
        if world_size > 1:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        # Initialize optimizer, scheduler, and scaler
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.ADAMW_LR,
            weight_decay=config.WEIGHT_DECAY
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.ADAMW_LR,
            total_steps=config.EPOCHS * len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )

        scaler = GradScaler(enabled=config.USE_AMP)

        # Initialize loss function
        criterion = QualityRobustLoss(config).cuda()

        # Training history and best model tracking
        history = defaultdict(list)
        best_mcc = -1.0
        best_semi_f1 = 0.0
        early_stopping_counter = 0
        best_checkpoint_path = None
        top_k_checkpoints = []

        # Create checkpoint directory
        if rank == 0:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # Training loop
        for epoch in range(config.EPOCHS):
            if world_size > 1 and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Progressive unfreezing
            unfroze = enhanced_progressive_unfreeze(model, epoch, config, logger)

            if unfroze and world_size > 1:
                dist.barrier()

            # Train for one epoch
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scaler, config, epoch, logger
            )

            # Validate
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, config, epoch, logger)

            # Update scheduler
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
            else:
                scheduler.step(val_loss)

            # Log results
            if rank == 0:
                learning_rates = [param_group['lr'] for param_group in optimizer.param_groups]
                log_epoch_results(epoch, train_loss, val_loss, val_metrics, learning_rates, logger)

                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_mcc'].append(val_metrics['mcc'])
                history['macro_f1'].append(val_metrics['macro_f1'])
                history['weighted_f1'].append(val_metrics['weighted_f1'])
                
                if 'semi_synthetic_f1' in val_metrics:
                    history['semi_f1'].append(val_metrics['semi_synthetic_f1'])
                    history['semi_precision'].append(val_metrics['semi_synthetic_precision'])
                    history['semi_recall'].append(val_metrics['semi_synthetic_recall'])
                    history['semi_mcc'].append(val_metrics['semi_synthetic_mcc'])
                
                if 'semi_confusion_rate' in val_metrics:
                    history['semi_confusion_rate'].append(val_metrics['semi_confusion_rate'])
                
                history['learning_rate'].append(learning_rates[0])
                
                if 'mean_uncertainty' in val_metrics:
                    history['mean_uncertainty'].append(val_metrics['mean_uncertainty'])
                    history['uncertainty_std'].append(val_metrics['uncertainty_std'])

                # Save checkpoint
                checkpoint_path = os.path.join(
                    config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'
                )
                
                current_mcc = val_metrics['mcc']
                current_semi_f1 = val_metrics.get('semi_synthetic_f1', 0.0)
                
                is_best = current_mcc > best_mcc or (
                    abs(current_mcc - best_mcc) < 0.01 and current_semi_f1 > best_semi_f1
                )

                if is_best:
                    best_mcc = current_mcc
                    best_semi_f1 = current_semi_f1
                    best_checkpoint_path = os.path.join(
                        config.CHECKPOINT_DIR, 'best_model.pth'
                    )
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if rank == 0 and (epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0 or is_best):
                    save_enhanced_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, 
                        val_metrics, config, checkpoint_path, is_best, logger
                    )
                    if is_best:
                        save_enhanced_checkpoint(
                            model, optimizer, scheduler, scaler, epoch, 
                            val_metrics, config, best_checkpoint_path, True, logger
                        )
                    
                    # Manage top-k checkpoints
                    top_k_checkpoints.append((current_mcc, checkpoint_path))
                    top_k_checkpoints.sort(reverse=True)
                    if len(top_k_checkpoints) > config.SAVE_TOP_K_MODELS:
                        _, old_path = top_k_checkpoints.pop()
                        if os.path.exists(old_path) and old_path != best_checkpoint_path:
                            os.remove(old_path)

                # Plot metrics and confusion matrix
                if rank == 0:
                    plot_enhanced_metrics(
                        history, os.path.join(config.CHECKPOINT_DIR, f'metrics_epoch_{epoch}.png')
                    )
                    plot_enhanced_confusion_matrix(
                        val_metrics['confusion_matrix'],
                        config.CLASS_NAMES,
                        os.path.join(config.CHECKPOINT_DIR, f'cm_epoch_{epoch}.png'),
                        val_metrics['per_class_mcc']
                    )

                # Early stopping
                if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {early_stopping_counter} epochs without improvement")
                    break

            if world_size > 1:
                dist.barrier()

        # Final cleanup
        if rank == 0:
            logger.info(f"Training completed. Best MCC: {best_mcc:.4f}, Best Semi-synthetic F1: {best_semi_f1:.4f}")
            logger.info(f"Best model saved at: {best_checkpoint_path}")

    except Exception as e:
        logger.error(f"Error in main process (rank {rank}): {str(e)}")
        raise

    finally:
        if world_size > 1:
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"Error during process group cleanup (rank {rank}): {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main()