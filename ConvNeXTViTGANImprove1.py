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
import cv2
import shutil

warnings.filterwarnings('ignore')

# CRITICAL FIX: Set multiprocessing start method to 'spawn' for CUDA compatibility
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

# Custom formatter for logging with rank
class RankFormatter(logging.Formatter):
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
    """Enhanced configuration with synthetic style expansion"""
    def __init__(self):
        self.MODEL_TYPE = "improved_forensics_model"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 1536
        self.DROPOUT_RATE = 0.4
        self.FREEZE_BACKBONES = True
        self.ATTENTION_DROPOUT = 0.2
        self.USE_SPECTRAL_NORM = True
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DISTRIBUTED = torch.cuda.device_count() > 1
        self.BACKEND = "nccl"
        self.MASTER_ADDR = "localhost"
        self.MASTER_PORT = "12355"
        self.BATCH_SIZE = 24  # Reduced due to expanded dataset
        self.EPOCHS = 60     # Increased for diverse training
        self.USE_AMP = True
        self.TRAIN_PATH = "datasets/train"
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = 4
        self.UNFREEZE_EPOCHS = [1, 8, 18, 28, 38, 48]
        self.FINE_TUNE_START_EPOCH = 28
        self.EARLY_STOPPING_PATIENCE = 50
        self.ADAMW_LR = 2e-4  # Slightly reduced for stable training
        self.SGD_LR = 3e-6    # Reduced for fine-tuning
        self.SGD_MOMENTUM = 0.95
        self.WEIGHT_DECAY = 2e-2
        
        # Adjusted for expanded synthetic data
        self.FOCAL_ALPHA = torch.tensor([1.0, 3.5, 2.0])
        self.FOCAL_GAMMA = 3.0
        self.LABEL_SMOOTHING = 0.1
        self.CLASS_WEIGHTS = torch.tensor([1.0, 3.5, 1.8])
        
        self.USE_FORENSICS_MODULE = True
        self.USE_UNCERTAINTY_ESTIMATION = True
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = 0.25
        self.USE_CUTMIX = True
        self.CUTMIX_ALPHA = 0.8
        self.USE_ENSEMBLE = True
        self.ENSEMBLE_SIZE = 3
        
        # Enhanced loss weights for better class separation
        self.CONTRASTIVE_WEIGHT = 0.5
        self.EVIDENTIAL_WEIGHT = 0.25
        self.BOUNDARY_WEIGHT = 0.3
        self.TRIPLET_WEIGHT = 0.3
        self.CLASS_SEPARATION_WEIGHT = 0.25
        
        self.CHECKPOINT_DIR = "improve1_checkpoints"
        self.CHECKPOINT_EVERY_N_EPOCHS = 3
        self.USE_MCC_FOR_BEST_MODEL = True
        self.SAVE_TOP_K_MODELS = 3
        
        # Synthetic style expansion parameters
        self.EXPAND_SYNTHETIC_STYLES = True
        self.SYNTHETIC_STYLE_MULTIPLIER = 3  # 3x more synthetic samples
        self.BRIGHT_STYLE_PROB = 0.6
        self.OVERLAY_STYLE_PROB = 0.3
        self.PRESERVE_FORENSIC_ARTIFACTS = True

    def validate(self):
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0, "Batch size must be positive"
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0, "Epochs must be positive"
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES, "Class names must match NUM_CLASSES"
        assert self.CONVNEXT_BACKBONE in ["convnext_tiny", "convnext_small"], "Unsupported backbone"
        assert self.FINE_TUNE_START_EPOCH < self.EPOCHS, "Fine-tune start epoch must be less than total epochs"
        assert all(epoch <= self.EPOCHS for epoch in self.UNFREEZE_EPOCHS), "Unfreeze epochs must be within total epochs"

class SyntheticStyleGenerator:
    """Generate diverse synthetic styles to bridge domain gap"""
    
    def __init__(self, config):
        self.config = config
        
    def create_bright_saturated_style(self, image):
        """Convert dark cinematic style to bright saturated style (server domain)"""
        # Ensure input is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 1. Increase brightness and contrast significantly
        alpha = np.random.uniform(1.3, 1.6)  # High contrast for "pop" effect
        beta = np.random.uniform(25, 45)     # Bright appearance
        bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # 2. Boost saturation dramatically (key characteristic of server domain)
        hsv = cv2.cvtColor(bright_image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Increase saturation significantly
        s = cv2.multiply(s, np.random.uniform(1.5, 2.0))
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # Boost value for vibrant look
        v = cv2.multiply(v, np.random.uniform(1.1, 1.4))
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        enhanced = cv2.merge([h, s, v])
        result = cv2.cvtColor(enhanced, cv2.COLOR_HSV2RGB)
        
        # 3. Add digital sharpening (common in processed images)
        if np.random.random() < 0.6:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(result, -1, kernel * 0.5)  # Moderate sharpening
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 4. Apply color grading effects
        if np.random.random() < 0.5:
            color_effects = ['warm', 'cool', 'vibrant', 'instagram']
            effect = np.random.choice(color_effects)
            
            if effect == 'warm':
                # Warm tone (more red/yellow)
                result[:,:,0] = np.clip(result[:,:,0] * 1.15, 0, 255)  # Red boost
                result[:,:,1] = np.clip(result[:,:,1] * 1.08, 0, 255)  # Green slight boost
                result[:,:,2] = np.clip(result[:,:,2] * 0.92, 0, 255)  # Blue reduction
            elif effect == 'cool':
                # Cool tone (more blue/cyan)
                result[:,:,2] = np.clip(result[:,:,2] * 1.15, 0, 255)  # Blue boost
                result[:,:,1] = np.clip(result[:,:,1] * 1.05, 0, 255)  # Green slight boost
                result[:,:,0] = np.clip(result[:,:,0] * 0.92, 0, 255)  # Red reduction
            elif effect == 'vibrant':
                # Enhanced vibrance
                for channel in range(3):
                    result[:,:,channel] = np.clip(result[:,:,channel] * 1.1, 0, 255)
            elif effect == 'instagram':
                # Instagram-like filter
                result[:,:,0] = np.clip(result[:,:,0] * 1.05 + 10, 0, 255)  # Slight warm + brightness
                result[:,:,1] = np.clip(result[:,:,1] * 1.02 + 5, 0, 255)
                result[:,:,2] = np.clip(result[:,:,2] * 0.98, 0, 255)
        
        return result.astype(np.uint8)
    
    def create_overlay_effect(self, image):
        """Add overlay effects similar to social media or processed images"""
        if np.random.random() < self.config.OVERLAY_STYLE_PROB:
            # Create subtle overlay effects
            overlay = image.copy().astype(np.float32)
            h, w = image.shape[:2]
            
            # Add gradient overlay (top-bottom or radial)
            if np.random.random() < 0.5:
                # Linear gradient
                gradient = np.linspace(0, 0.2, h).reshape(-1, 1)
                gradient = np.repeat(gradient, w, axis=1)
            else:
                # Radial gradient from center
                center_x, center_y = w // 2, h // 2
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                gradient = (distance / max_distance) * 0.15
            
            # Apply overlay effect
            overlay_color = np.random.randint(0, 100, 3)  # Dark overlay
            for channel in range(3):
                overlay[:,:,channel] = overlay[:,:,channel] * (1 - gradient) + overlay_color[channel] * gradient
            
            result = overlay.astype(np.uint8)
            return result
        
        return image
    
    def preserve_compression_artifacts(self, image, quality_factor=None):
        """Ensure compression artifacts are preserved during style transfer"""
        if quality_factor is None:
            quality_factor = np.random.randint(75, 95)
        
        # Encode and decode with JPEG to maintain realistic compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        return cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

class ForensicsAwareModule(nn.Module):
    """Enhanced forensics module with better artifact detection"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # DCT coefficient analyzer
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
        
        # Enhanced noise pattern analyzer
        self.noise_analyzer = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Edge inconsistency detector
        self.edge_analyzer = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(40 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Frequency domain analyzer
        self.freq_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 96),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Style-invariant feature fusion
        self.forensics_fusion = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 96, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
    
    def extract_edge_inconsistencies(self, x):
        """Extract edge inconsistency features"""
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edge_feats = self.edge_analyzer(gray)
        return edge_feats
    
    def forward(self, x):
        dct_feats = self.dct_analyzer(x)
        noise_feats = self.noise_analyzer(x)
        edge_feats = self.extract_edge_inconsistencies(x)
        freq_feats = self.freq_analyzer(x)
        
        combined_feats = torch.cat([dct_feats, noise_feats, edge_feats, freq_feats], dim=1)
        forensics_output = self.forensics_fusion(combined_feats)
        return forensics_output

class UncertaintyModule(nn.Module):
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
        evidence = self.evidence_layer(x)
        aleatoric = self.aleatoric_layer(x)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        epistemic_uncertainty = self.num_classes / S
        aleatoric_uncertainty = aleatoric
        return probs, epistemic_uncertainty, aleatoric_uncertainty, alpha

class EnhancedLoss(nn.Module):
    """Enhanced loss with better class separation for real vs semi-synthetic"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.device = config.DEVICE
        self.focal_alpha = config.FOCAL_ALPHA.to(config.DEVICE) if hasattr(config.FOCAL_ALPHA, 'to') else config.FOCAL_ALPHA
        self.focal_gamma = config.FOCAL_GAMMA
        self.class_weights = config.CLASS_WEIGHTS.to(config.DEVICE) if hasattr(config.CLASS_WEIGHTS, 'to') else config.CLASS_WEIGHTS
        self.evidential_weight = config.EVIDENTIAL_WEIGHT
        self.contrastive_weight = config.CONTRASTIVE_WEIGHT
        self.boundary_weight = config.BOUNDARY_WEIGHT
        self.triplet_weight = config.TRIPLET_WEIGHT
        self.class_separation_weight = config.CLASS_SEPARATION_WEIGHT
        self.label_smoothing = config.LABEL_SMOOTHING
        self.triplet_margin = 1.2  # Increased margin
    
    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def evidential_loss(self, alpha, targets, epoch=1):
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        S = torch.sum(alpha, dim=1, keepdim=True)
        likelihood_loss = torch.sum(targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        alpha_tilde = targets_one_hot + (1 - targets_one_hot) * alpha
        kl_div = torch.lgamma(torch.sum(alpha_tilde, dim=1)) - torch.sum(torch.lgamma(alpha_tilde), dim=1) + \
                 torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(torch.sum(alpha_tilde, dim=1, keepdim=True))), dim=1)
        annealing_coef = min(1.0, epoch / 20.0)
        return likelihood_loss.mean() + annealing_coef * 0.1 * kl_div.mean()
    
    def enhanced_contrastive_loss(self, features, labels):
        """Enhanced contrastive loss with specific focus on real vs semi-synthetic separation"""
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features, features.t())
        labels_expanded = labels.unsqueeze(1)
        same_class_mask = (labels_expanded == labels_expanded.t()).float()
        diff_class_mask = 1 - same_class_mask
        
        # Special handling for real vs semi-synthetic confusion
        real_mask = (labels == 0).float().unsqueeze(1)
        semi_mask = (labels == 1).float().unsqueeze(1)
        real_semi_pairs = real_mask * semi_mask.t() + semi_mask * real_mask.t()
        
        # Standard contrastive terms
        pos_loss = same_class_mask * (1 - similarity_matrix) ** 2
        neg_loss = diff_class_mask * torch.clamp(similarity_matrix - 0.2, min=0) ** 2
        
        # Enhanced penalty for real-semi confusion
        real_semi_penalty = real_semi_pairs * torch.clamp(similarity_matrix - 0.1, min=0) ** 2
        
        return (pos_loss.sum() + neg_loss.sum() + 3.0 * real_semi_penalty.sum()) / (labels.size(0) ** 2)
    
    def class_separation_loss(self, features, labels):
        """Force larger margins between class centroids, especially real vs semi-synthetic"""
        unique_labels = torch.unique(labels)
        centroids = {}
        
        # Calculate class centroids
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                centroids[label.item()] = features[mask].mean(dim=0)
        
        separation_loss = 0
        count = 0
        
        # Calculate pairwise centroid distances
        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                if i < j:
                    centroid_i = centroids[label_i.item()]
                    centroid_j = centroids[label_j.item()]
                    distance = torch.norm(centroid_i - centroid_j, p=2)
                    
                    # Larger margin required for real vs semi-synthetic
                    if (label_i.item() == 0 and label_j.item() == 1) or (label_i.item() == 1 and label_j.item() == 0):
                        margin = 2.0  # Larger margin for real vs semi-synthetic
                    else:
                        margin = 1.5  # Standard margin for other pairs
                    
                    separation_loss += torch.clamp(margin - distance, min=0) ** 2
                    count += 1
        
        return separation_loss / count if count > 0 else torch.tensor(0.0).to(self.device)
    
    def boundary_loss(self, logits, targets):
        """Enhanced boundary loss for better class separation"""
        probs = F.softmax(logits, dim=1)
        real_mask = (targets == 0).float()
        synthetic_mask = (targets == 2).float()
        semi_mask = (targets == 1).float()
        
        # Penalize real images that have high semi-synthetic probability
        real_semi_penalty = real_mask * probs[:, 1]
        
        # Penalize synthetic images that have high semi-synthetic probability
        synthetic_semi_penalty = synthetic_mask * probs[:, 1]
        
        # Penalize semi-synthetic images that don't have high semi-synthetic probability
        semi_confidence_penalty = semi_mask * (1 - probs[:, 1])
        
        return (2.0 * real_semi_penalty.sum() + synthetic_semi_penalty.sum() + 1.5 * semi_confidence_penalty.sum()) / targets.size(0)
    
    def triplet_loss(self, features, labels):
        """Enhanced triplet loss with hard negative mining"""
        features = F.normalize(features, p=2, dim=1)
        triplet_losses = []
        
        for i in range(features.size(0)):
            anchor_label = labels[i]
            anchor_feat = features[i]
            
            # Find positive samples (same class, different sample)
            pos_mask = (labels == anchor_label) & (torch.arange(features.size(0)).to(self.device) != i)
            if pos_mask.sum() == 0:
                continue
            
            # Find negative samples (different class)
            neg_mask = (labels != anchor_label)
            if neg_mask.sum() == 0:
                continue
            
            pos_distances = torch.norm(features[pos_mask] - anchor_feat.unsqueeze(0), p=2, dim=1)
            neg_distances = torch.norm(features[neg_mask] - anchor_feat.unsqueeze(0), p=2, dim=1)
            
            # Hard positive and hard negative mining
            hardest_pos_dist = pos_distances.max()
            hardest_neg_dist = neg_distances.min()
            
            # Adaptive margin based on class pair
            if anchor_label.item() == 0:  # Real anchor
                neg_labels = labels[neg_mask]
                if 1 in neg_labels:  # Semi-synthetic negative exists
                    margin = self.triplet_margin * 1.5  # Larger margin for real vs semi
                else:
                    margin = self.triplet_margin
            else:
                margin = self.triplet_margin
            
            triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
            triplet_losses.append(triplet_loss)
        
        return torch.stack(triplet_losses).mean() if triplet_losses else torch.tensor(0.0).to(self.device)
    
    def forward(self, logits, targets, features=None, alpha=None, epoch=1):
        # Primary losses
        focal_loss = self.focal_loss(logits, targets)
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        boundary_loss = self.boundary_loss(logits, targets)
        
        total_loss = 0.4 * focal_loss + 0.3 * ce_loss + self.boundary_weight * boundary_loss
        
        if features is not None:
            contrastive_loss = self.enhanced_contrastive_loss(features, targets)
            triplet_loss = self.triplet_loss(features, targets)
            class_separation_loss = self.class_separation_loss(features, targets)
            
            total_loss += self.contrastive_weight * contrastive_loss
            total_loss += self.triplet_weight * triplet_loss
            total_loss += self.class_separation_weight * class_separation_loss
        
        if alpha is not None:
            evidential_loss = self.evidential_loss(alpha, targets, epoch)
            total_loss += self.evidential_weight * evidential_loss
        
        return total_loss

class SuperiorAttentionModule(nn.Module):
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
    """Improved model with enhanced forensics and style robustness"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone networks
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        if config.FREEZE_BACKBONES:
            self.freeze_backbones()
        
        # Feature dimensions
        convnext_features = 768
        vit_features = self.vit.num_features
        forensics_features = 128 if config.USE_FORENSICS_MODULE else 0
        total_features = convnext_features + vit_features + forensics_features
        
        # Enhanced forensics module
        if config.USE_FORENSICS_MODULE:
            self.forensics_module = ForensicsAwareModule(config)
        
        # Attention module
        self.attention_module = SuperiorAttentionModule(total_features, config)
        
        # Uncertainty estimation
        if config.USE_UNCERTAINTY_ESTIMATION:
            self.uncertainty_module = UncertaintyModule(config.HIDDEN_DIM // 4, config.NUM_CLASSES)
        
        # Enhanced feature fusion
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
        
        # Style-invariant feature extractor
        self.style_invariant_features = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM // 4, config.HIDDEN_DIM // 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.HIDDEN_DIM // 8, config.HIDDEN_DIM // 16),
            nn.ReLU()
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM // 4 + config.HIDDEN_DIM // 16, config.HIDDEN_DIM // 8),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 8, config.NUM_CLASSES)
        )
        
        if config.USE_SPECTRAL_NORM:
            self.fusion[0] = nn.utils.spectral_norm(self.fusion[0])
            self.fusion[3] = nn.utils.spectral_norm(self.fusion[3])
            self.fusion[6] = nn.utils.spectral_norm(self.fusion[6])
            self.classifier[0] = nn.utils.spectral_norm(self.classifier[0])
            self.classifier[3] = nn.utils.spectral_norm(self.classifier[3])
    
    def freeze_backbones(self):
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        logger.info("Frozen backbone parameters", extra={'rank': 0})
    
    def unfreeze_convnext_layers(self, num_layers=None):
        if num_layers is None:
            for param in self.convnext.parameters():
                param.requires_grad = True
            logger.info("Unfrozen all ConvNeXt layers", extra={'rank': 0})
        else:
            layers = list(self.convnext.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True
            logger.info(f"Unfrozen last {num_layers} ConvNeXt layers", extra={'rank': 0})
    
    def unfreeze_vit_layers(self, num_layers=None):
        if num_layers is None:
            for param in self.vit.parameters():
                param.requires_grad = True
            logger.info("Unfrozen all ViT layers", extra={'rank': 0})
        else:
            layers = list(self.vit.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True
            logger.info(f"Unfrozen last {num_layers} ViT layers", extra={'rank': 0})
    
    def unfreeze_forensics_and_attention(self):
        if self.config.USE_FORENSICS_MODULE:
            for param in self.forensics_module.parameters():
                param.requires_grad = True
        for param in self.attention_module.parameters():
            param.requires_grad = True
        for param in self.fusion.parameters():
            param.requires_grad = True
        logger.info("Unfrozen forensics module, attention module, and fusion layers", extra={'rank': 0})
    
    def unfreeze_classifier(self):
        for param in self.fusion.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.style_invariant_features.parameters():
            param.requires_grad = True
        logger.info("Unfrozen classifier, fusion, and style-invariant layers", extra={'rank': 0})
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        # Extract backbone features
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]
        
        features_list = [convnext_feats, vit_feats]
        
        # Add forensics features
        if self.config.USE_FORENSICS_MODULE:
            forensics_feats = self.forensics_module(x)
            features_list.append(forensics_feats)
        
        # Fuse all features
        fused_features = torch.cat(features_list, dim=1)
        attended_features = self.attention_module(fused_features)
        processed_features = self.fusion(attended_features)
        
        # Extract style-invariant features
        style_invariant_feats = self.style_invariant_features(processed_features)
        
        # Combine for final classification
        combined_features = torch.cat([processed_features, style_invariant_feats], dim=1)
        logits = self.classifier(combined_features)
        
        if self.config.USE_UNCERTAINTY_ESTIMATION and hasattr(self, 'uncertainty_module'):
            probs, epistemic_unc, aleatoric_unc, alpha = self.uncertainty_module(processed_features)
            return logits, processed_features, (probs, epistemic_unc, aleatoric_unc, alpha)
        
        return logits, processed_features

class ExpandedSyntheticDataset(Dataset):
    """Dataset with expanded synthetic styles to bridge domain gap"""
    
    def __init__(self, root_dir, config, transform=None, is_training=True):
        self.root_dir = Path(root_dir)
        self.config = config
        self.is_training = is_training
        self.class_names = config.CLASS_NAMES
        self.file_indices = []
        self.labels = []
        self.style_generator = SyntheticStyleGenerator(config)
        self.tensor_cache = {}
        
        # Define transforms for each class
        self._setup_transforms()
        
        if transform is not None:
            self.val_transform = transform
        
        self._validate_dataset_path()
        self._load_and_expand_dataset()
    
    def _setup_transforms(self):
        """Setup class-specific transforms"""
        # Real images: minimal augmentation to preserve authentic characteristics
        self.real_transform = A.Compose([
            A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.2), p=0.3),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Semi-synthetic: edge-preserving augmentations
        self.semi_synthetic_transform = A.Compose([
            A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            A.OneOf([
                A.MedianBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Synthetic: minimal augmentation to preserve style variations we create
        self.synthetic_transform = A.Compose([
            A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation transform
        self.val_transform = A.Compose([
            A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _validate_dataset_path(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.root_dir} does not exist")
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist", extra={'rank': 0})
    
    def _load_and_expand_dataset(self):
        """Load dataset and expand synthetic samples with multiple styles"""
        logger.info("Loading and expanding dataset with synthetic style variations...", extra={'rank': 0})
        
        metadata_file = self.root_dir / "expanded_metadata.json"
        
        if metadata_file.exists() and not self.config.EXPAND_SYNTHETIC_STYLES:
            # Load existing metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.file_indices = metadata['file_indices']
                self.labels = metadata['labels']
                logger.info(f"Loaded expanded metadata from {metadata_file}", extra={'rank': 0})
        else:
            # Create expanded dataset
            self._create_expanded_dataset()
            # Save metadata
            metadata = {
                'file_indices': self.file_indices,
                'labels': self.labels,
                'expansion_config': {
                    'synthetic_multiplier': self.config.SYNTHETIC_STYLE_MULTIPLIER,
                    'bright_prob': self.config.BRIGHT_STYLE_PROB,
                    'overlay_prob': self.config.OVERLAY_STYLE_PROB
                }
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            logger.info(f"Saved expanded metadata to {metadata_file}", extra={'rank': 0})
        
        # Log class distribution
        class_counts = np.bincount(self.labels)
        logger.info("Dataset composition after expansion:", extra={'rank': 0})
        for i, count in enumerate(class_counts):
            if i < len(self.class_names):
                logger.info(f"  {self.class_names[i]}: {count} samples", extra={'rank': 0})
    
    def _create_expanded_dataset(self):
        """Create expanded dataset with multiple synthetic styles"""
        original_indices = []
        original_labels = []
        
        # Load original dataset
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            pt_files = list(class_dir.glob('*.pt'))
            for pt_file in pt_files:
                try:
                    # Load tensor data for size checking
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    if isinstance(tensor_data, dict):
                        tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                    
                    # Cache the tensor data
                    self.tensor_cache[str(pt_file)] = tensor_data
                    
                    num_images = tensor_data.shape[0]
                    for i in range(num_images):
                        original_indices.append((str(pt_file), i))
                        original_labels.append(class_idx)
                        
                except Exception as e:
                    logger.warning(f"Skipping corrupted file {pt_file}: {e}", extra={'rank': 0})
        
        # Start with original dataset
        self.file_indices = original_indices.copy()
        self.labels = original_labels.copy()
        
        if not self.config.EXPAND_SYNTHETIC_STYLES or not self.is_training:
            return
        
        # Expand synthetic class (label=2) with style variations
        synthetic_samples = [(idx, file_idx, label) for idx, (file_idx, label) in 
                           enumerate(zip(original_indices, original_labels)) if label == 2]
        
        logger.info(f"Expanding {len(synthetic_samples)} synthetic samples with style variations...", extra={'rank': 0})
        
        for _, (file_path, img_idx), label in synthetic_samples:
            # Add bright saturated style variation
            self.file_indices.append((file_path, img_idx, 'bright_saturated'))
            self.labels.append(2)
            
            # Add overlay style variation
            self.file_indices.append((file_path, img_idx, 'overlay_effect'))
            self.labels.append(2)
        
        logger.info(f"Dataset expanded from {len(original_indices)} to {len(self.file_indices)} samples", extra={'rank': 0})
        logger.info(f"Synthetic class expanded by {self.config.SYNTHETIC_STYLE_MULTIPLIER}x", extra={'rank': 0})
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        try:
            # Parse file index with potential style information
            if len(self.file_indices[idx]) == 3:
                file_path, image_idx, style_type = self.file_indices[idx]
            else:
                file_path, image_idx = self.file_indices[idx]
                style_type = 'original'
            
            label = self.labels[idx]
            
            # Load tensor data
            tensor_data = self.tensor_cache.get(file_path)
            if tensor_data is None:
                tensor_data = torch.load(file_path, map_location='cpu')
                if isinstance(tensor_data, dict):
                    tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                self.tensor_cache[file_path] = tensor_data
            
            image_tensor = tensor_data[image_idx]
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # Convert to numpy for processing
            image_np = image_tensor.permute(1, 2, 0).numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # Apply style transformations for synthetic images during training
            if self.is_training and label == 2 and style_type != 'original':
                if style_type == 'bright_saturated':
                    image_np = self.style_generator.create_bright_saturated_style(image_np)
                elif style_type == 'overlay_effect':
                    image_np = self.style_generator.create_bright_saturated_style(image_np)
                    image_np = self.style_generator.create_overlay_effect(image_np)
                
                # Preserve compression artifacts
                if self.config.PRESERVE_FORENSIC_ARTIFACTS:
                    image_np = self.style_generator.preserve_compression_artifacts(image_np)
            
            # Apply class-specific transforms
            if self.is_training:
                if label == 0:  # real
                    transformed = self.real_transform(image=image_np)
                elif label == 1:  # semi-synthetic
                    transformed = self.semi_synthetic_transform(image=image_np)
                else:  # synthetic
                    transformed = self.synthetic_transform(image=image_np)
            else:
                transformed = self.val_transform(image=image_np)
            
            image_tensor = transformed['image']
            return image_tensor, label
            
        except Exception as e:
            logger.warning(f"Error loading image at index {idx}: {e}", extra={'rank': 0})
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 0

class MixupCutmixAugmentation:
    def __init__(self, config):
        self.config = config
        self.mixup_alpha = config.MIXUP_ALPHA if config.USE_MIXUP else 0
        self.cutmix_alpha = config.CUTMIX_ALPHA if config.USE_CUTMIX else 0
    
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
    
    def cutmix_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, x, y):
        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            if np.random.random() < 0.5:
                return self.mixup_data(x, y, self.mixup_alpha)
            else:
                return self.cutmix_data(x, y, self.cutmix_alpha)
        elif self.mixup_alpha > 0:
            return self.mixup_data(x, y, self.mixup_alpha)
        elif self.cutmix_alpha > 0:
            return self.cutmix_data(x, y, self.cutmix_alpha)
        else:
            return x, y, y, 1.0

def create_improved_data_loaders(config, local_rank=-1):
    """Create improved data loaders with expanded synthetic styles"""
    train_dataset = ExpandedSyntheticDataset(root_dir=config.TRAIN_PATH, config=config, is_training=True)
    
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.is_training = False
    test_dataset.dataset.is_training = False
    
    train_sampler = DistributedSampler(train_dataset, rank=local_rank, shuffle=True) if config.DISTRIBUTED and local_rank != -1 else None
    num_workers = 0 if config.DISTRIBUTED or torch.cuda.is_available() else config.NUM_WORKERS
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def calculate_improved_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """Calculate comprehensive metrics with focus on real vs semi-synthetic confusion"""
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    mcc = matthews_corrcoef(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    
    # Detailed semi-synthetic metrics
    semi_synthetic_precision = precision[1] if len(precision) > 1 else 0
    semi_synthetic_recall = recall[1] if len(recall) > 1 else 0
    semi_synthetic_f1 = f1[1] if len(f1) > 1 else 0
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_true, y_pred)
    
    # Real vs Semi-synthetic confusion rate
    real_as_semi = cm[0, 1] if len(cm) >= 2 else 0  # Real classified as semi-synthetic
    semi_as_real = cm[1, 0] if len(cm) >= 2 else 0  # Semi-synthetic classified as real
    total_real = cm[0].sum() if len(cm) >= 1 else 0
    total_semi = cm[1].sum() if len(cm) >= 2 else 0
    
    real_semi_confusion_rate = (real_as_semi + semi_as_real) / (total_real + total_semi) if (total_real + total_semi) > 0 else 0
    
    # Synthetic classification accuracy (should remain high)
    synthetic_accuracy = cm[2, 2] / cm[2].sum() if len(cm) >= 3 and cm[2].sum() > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'mcc': mcc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'semi_synthetic_precision': semi_synthetic_precision,
        'semi_synthetic_recall': semi_synthetic_recall,
        'semi_synthetic_f1': semi_synthetic_f1,
        'real_semi_confusion_rate': real_semi_confusion_rate,
        'synthetic_accuracy': synthetic_accuracy,
        'confusion_matrix': cm.tolist(),
        'real_as_semi_count': int(real_as_semi),
        'semi_as_real_count': int(semi_as_real)
    }
    
    return metrics

def evaluate_improved_model(model, data_loader, criterion, config, device, epoch=1, local_rank=0):
    """Enhanced model evaluation with detailed metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_uncertainties = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating", leave=False, disable=local_rank not in [-1, 0]):
            data, target = data.to(device), target.to(device)
            
            with autocast(enabled=config.USE_AMP):
                model_output = model.module(data) if hasattr(model, 'module') else model(data)
                
                if len(model_output) == 3:
                    logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
                    loss = criterion(logits, target, features, alpha, epoch)
                    total_uncertainty = epistemic_unc.squeeze() + aleatoric_unc.mean(dim=1)
                    all_uncertainties.extend(total_uncertainty.cpu().numpy())
                    all_probabilities.extend(probs.cpu().numpy())
                else:
                    logits, features = model_output
                    loss = criterion(logits, target, features, None, epoch)
                    all_probabilities.extend(F.softmax(logits, dim=1).cpu().numpy())
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    metrics = calculate_improved_metrics(all_targets, all_predictions, all_probabilities, config.CLASS_NAMES)
    
    if all_uncertainties:
        metrics['mean_uncertainty'] = np.mean(all_uncertainties)
        metrics['uncertainty_std'] = np.std(all_uncertainties)
    
    return avg_loss, metrics, all_predictions, all_targets, all_probabilities

def mixup_criterion(criterion, model_output, target_a, target_b, lam, epoch):
    """Mixed loss computation for mixup/cutmix"""
    if len(model_output) == 3:
        logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
        loss_a = criterion(logits, target_a, features, alpha, epoch)
        loss_b = criterion(logits, target_b, features, alpha, epoch)
    else:
        logits, features = model_output
        loss_a = criterion(logits, target_a, features, None, epoch)
        loss_b = criterion(logits, target_b, features, None, epoch)
    return lam * loss_a + (1 - lam) * loss_b

def create_improved_optimizer(model, config, epoch):
    """Create optimizer with adaptive learning rates"""
    backbone_params = []
    forensics_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'convnext' in name or 'vit' in name:
            backbone_params.append(param)
        elif 'forensics' in name or 'attention' in name or 'style_invariant' in name:
            forensics_params.append(param)
        else:
            classifier_params.append(param)
    
    if epoch >= config.FINE_TUNE_START_EPOCH:
        optimizer = optim.SGD([
            {'params': backbone_params, 'lr': config.SGD_LR * 0.1},
            {'params': forensics_params, 'lr': config.SGD_LR},
            {'params': classifier_params, 'lr': config.SGD_LR * 2}
        ], momentum=config.SGD_MOMENTUM, weight_decay=config.WEIGHT_DECAY)
        logger.info(f"Fine-tuning SGD optimizer (epoch {epoch})", extra={'rank': 0})
    else:
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': config.ADAMW_LR * 0.5},
            {'params': forensics_params, 'lr': config.ADAMW_LR},
            {'params': classifier_params, 'lr': config.ADAMW_LR * 1.5}
        ], weight_decay=config.WEIGHT_DECAY)
        logger.info(f"AdamW optimizer (epoch {epoch})", extra={'rank': 0})
    
    return optimizer

def progressive_unfreeze_improved(model, epoch, config):
    """Improved progressive unfreezing strategy"""
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
            model.unfreeze_convnext_layers(num_layers=20)
            unfroze_this_epoch = True
        elif unfreeze_stage == 4:
            model.unfreeze_convnext_layers()
            unfroze_this_epoch = True
        elif unfreeze_stage == 5:
            model.unfreeze_vit_layers(num_layers=20)
            unfroze_this_epoch = True
        elif unfreeze_stage == 6:
            model.unfreeze_vit_layers()
            unfroze_this_epoch = True
        
        trainable_params = model.get_trainable_params()
        logger.info(f"Progressive unfreezing stage {unfreeze_stage}: {trainable_params:,} trainable parameters", extra={'rank': 0})
    
    return unfroze_this_epoch

def save_improved_checkpoint(model, optimizer, scaler, epoch, metrics, config, filename, is_best=False, local_rank=0):
    """Save model checkpoint with enhanced metadata"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'config': config.__dict__,
        'is_best': is_best,
        'model_type': 'improved_forensics_model_v2'
    }
    torch.save(checkpoint, filename)
    
    if local_rank in [-1, 0]:
        logger.info(f"Checkpoint saved: {filename}", extra={'rank': local_rank})
        if is_best:
            logger.info(f"New best model - Semi-synthetic F1: {metrics.get('semi_synthetic_f1', 0):.4f}, "
                       f"Real-Semi confusion: {metrics.get('real_semi_confusion_rate', 0):.4f}", extra={'rank': local_rank})

def plot_improved_metrics(history, save_path):
    """Enhanced metrics plotting with real vs semi-synthetic focus"""
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # MCC and Accuracy
    if 'val_mcc' in history and 'val_accuracy' in history:
        axes[0, 1].plot(history['val_mcc'], label='Val MCC', color='green')
        axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', color='orange')
        axes[0, 1].set_title('MCC and Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Semi-synthetic metrics
    if 'semi_f1' in history and 'semi_precision' in history and 'semi_recall' in history:
        axes[0, 2].plot(history['semi_f1'], label='Semi-synthetic F1', color='purple')
        axes[0, 2].plot(history['semi_precision'], label='Semi-synthetic Precision', color='orange')
        axes[0, 2].plot(history['semi_recall'], label='Semi-synthetic Recall', color='brown')
        axes[0, 2].set_title('Semi-synthetic Class Metrics')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # Real vs Semi confusion rate (KEY METRIC)
    if 'real_semi_confusion_rate' in history:
        axes[1, 0].plot(history['real_semi_confusion_rate'], label='Real-Semi Confusion Rate', color='red', linewidth=2)
        axes[1, 0].set_title('Real vs Semi-synthetic Confusion Rate (Lower is Better)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Confusion Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Synthetic accuracy (should remain high)
    if 'synthetic_accuracy' in history:
        axes[1, 1].plot(history['synthetic_accuracy'], label='Synthetic Accuracy', color='cyan', linewidth=2)
        axes[1, 1].set_title('Synthetic Class Accuracy (Should Stay High)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # F1 scores comparison
    if 'macro_f1' in history and 'weighted_f1' in history:
        axes[1, 2].plot(history['macro_f1'], label='Macro F1', color='magenta')
        axes[1, 2].plot(history['weighted_f1'], label='Weighted F1', color='yellow')
        axes[1, 2].set_title('F1 Scores')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    # Real as Semi count
    if 'real_as_semi_count' in history:
        axes[2, 0].plot(history['real_as_semi_count'], label='Real → Semi Errors', color='red')
        axes[2, 0].set_title('Real Images Classified as Semi-synthetic')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
    
    # Semi as Real count
    if 'semi_as_real_count' in history:
        axes[2, 1].plot(history['semi_as_real_count'], label='Semi → Real Errors', color='blue')
        axes[2, 1].set_title('Semi-synthetic Images Classified as Real')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
    
    # Uncertainty if available
    if 'mean_uncertainty' in history:
        axes[2, 2].plot(history['mean_uncertainty'], label='Mean Uncertainty', color='gray')
        axes[2, 2].set_title('Model Uncertainty')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Uncertainty')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    """Enhanced confusion matrix visualization"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Improved Forensics Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Calculate and display metrics
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    
    # Highlight real vs semi-synthetic confusion
    real_as_semi = cm[0, 1] if len(cm) >= 2 else 0
    semi_as_real = cm[1, 0] if len(cm) >= 2 else 0
    
    plt.figtext(0.15, 0.02, f'Overall Accuracy: {accuracy:.4f}')
    plt.figtext(0.15, 0.06, f'Real→Semi Errors: {real_as_semi}')
    plt.figtext(0.15, 0.10, f'Semi→Real Errors: {semi_as_real}')
    
    for i, class_name in enumerate(class_names):
        if i < len(cm):
            class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            plt.figtext(0.60, 0.10 - i*0.04, f'{class_name} Accuracy: {class_acc:.4f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def find_free_port(start_port=12355, max_attempts=100):
    """Find a free port for distributed training"""
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
    """Setup distributed training"""
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
    """Cleanup distributed training"""
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
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...", extra={'rank': 0})
    cleanup_distributed()
    cleanup_memory()
    exit(0)

def improved_train_worker(local_rank, config, master_port):
    """Main training worker with all improvements"""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if config.DISTRIBUTED:
            success = setup_distributed(local_rank, torch.cuda.device_count(), config.BACKEND, config.MASTER_ADDR, master_port)
            if not success:
                logger.error(f"Failed to setup distributed training for rank {local_rank}", extra={'rank': local_rank})
                return
            config.DEVICE = torch.device(f'cuda:{local_rank}')
            # Move tensors to correct device after distributed setup
            config.FOCAL_ALPHA = config.FOCAL_ALPHA.to(config.DEVICE)
            config.CLASS_WEIGHTS = config.CLASS_WEIGHTS.to(config.DEVICE)
        
        logger.info(f"Improved training setup complete for rank {local_rank}", extra={'rank': local_rank})
        
        # Set seeds for reproducibility
        torch.manual_seed(42 + local_rank)
        np.random.seed(42 + local_rank)
        random.seed(42 + local_rank)
        
        if local_rank in [-1, 0]:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            logger.info(f"Created checkpoint directory: {config.CHECKPOINT_DIR}", extra={'rank': local_rank})
        
        # Create improved data loaders
        train_loader, val_loader, test_loader = create_improved_data_loaders(config, local_rank)
        
        # Create improved model
        model = ImprovedModel(config).to(config.DEVICE)
        if config.DISTRIBUTED:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
        # Create enhanced loss and other components
        criterion = EnhancedLoss(config).to(config.DEVICE)
        scaler = GradScaler(enabled=config.USE_AMP)
        mixup_cutmix = MixupCutmixAugmentation(config)
        
        # Training state variables
        best_metrics = {'semi_synthetic_f1': -1.0, 'real_semi_confusion_rate': 1.0, 'mcc': -1.0}
        epochs_no_improve = 0
        training_history = defaultdict(list)
        current_optimizer = None
        
        start_epoch = 0
        
        # Check for existing checkpoint to resume training
        resume_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
        if os.path.exists(resume_checkpoint_path) and local_rank in [-1, 0]:
            try:
                checkpoint = torch.load(resume_checkpoint_path, map_location=config.DEVICE)
                if config.DISTRIBUTED:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_metrics = checkpoint.get('best_metrics', best_metrics)
                training_history = defaultdict(list, checkpoint.get('training_history', {}))
                epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                logger.info(f"Resumed training from epoch {start_epoch}", extra={'rank': local_rank})
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.", extra={'rank': local_rank})
                start_epoch = 0
        
        if local_rank in [-1, 0]:
            initial_trainable = model.module.get_trainable_params() if config.DISTRIBUTED else model.get_trainable_params()
            logger.info(f"Starting improved training with {initial_trainable:,} trainable parameters", extra={'rank': local_rank})
            logger.info(f"Dataset expanded with {config.SYNTHETIC_STYLE_MULTIPLIER}x synthetic styles", extra={'rank': local_rank})
        
        # Main training loop
        for epoch in range(start_epoch, config.EPOCHS):
            epoch_start_time = time.time()
            model_for_unfreeze = model.module if config.DISTRIBUTED else model
            unfroze_this_epoch = progressive_unfreeze_improved(model_for_unfreeze, epoch + 1, config)
            
            # Create/update optimizer
            if current_optimizer is None or unfroze_this_epoch or (epoch + 1) == config.FINE_TUNE_START_EPOCH:
                current_optimizer = create_improved_optimizer(model, config, epoch + 1)
            
            # Training phase
            model.train()
            train_loss = 0
            train_batches = 0
            
            if config.DISTRIBUTED:
                train_loader.sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS}', disable=local_rank not in [-1, 0])
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                
                # Apply mixup/cutmix augmentation
                if config.USE_MIXUP or config.USE_CUTMIX:
                    data, target_a, target_b, lam = mixup_cutmix(data, target)
                    use_mixup = True
                else:
                    use_mixup = False
                
                current_optimizer.zero_grad()
                
                with autocast(enabled=config.USE_AMP):
                    model_output = model(data)
                    if use_mixup:
                        loss = mixup_criterion(criterion, model_output, target_a, target_b, lam, epoch + 1)
                    else:
                        if len(model_output) == 3:
                            logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
                            loss = criterion(logits, target, features, alpha, epoch + 1)
                        else:
                            logits, features = model_output
                            loss = criterion(logits, target, features, None, epoch + 1)
                
                scaler.scale(loss).backward()
                scaler.step(current_optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_batches += 1
                
                if local_rank in [-1, 0]:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}', 
                        'Avg Loss': f'{train_loss/train_batches:.4f}'
                    })
                
                # Periodic memory cleanup
                if batch_idx % 100 == 0:
                    cleanup_memory()
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            if local_rank in [-1, 0]:
                training_history['train_loss'].append(avg_train_loss)
                val_loss, val_metrics, val_preds, val_targets, val_probs = evaluate_improved_model(
                    model, val_loader, criterion, config, config.DEVICE, epoch + 1, local_rank
                )
                
                # Record all metrics
                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_metrics['accuracy'])
                training_history['val_mcc'].append(val_metrics['mcc'])
                training_history['macro_f1'].append(val_metrics['macro_f1'])
                training_history['weighted_f1'].append(val_metrics['weighted_f1'])
                training_history['semi_f1'].append(val_metrics['semi_synthetic_f1'])
                training_history['semi_precision'].append(val_metrics['semi_synthetic_precision'])
                training_history['semi_recall'].append(val_metrics['semi_synthetic_recall'])
                training_history['real_semi_confusion_rate'].append(val_metrics['real_semi_confusion_rate'])
                training_history['synthetic_accuracy'].append(val_metrics['synthetic_accuracy'])
                training_history['real_as_semi_count'].append(val_metrics['real_as_semi_count'])
                training_history['semi_as_real_count'].append(val_metrics['semi_as_real_count'])
                
                if 'mean_uncertainty' in val_metrics:
                    training_history['mean_uncertainty'].append(val_metrics['mean_uncertainty'])
                
                epoch_time = time.time() - epoch_start_time
                
                # Log detailed results
                logger.info(f"Epoch {epoch+1}/{config.EPOCHS} completed in {epoch_time:.2f}s", extra={'rank': local_rank})
                logger.info(f"Train Loss: {avg_train_loss:.4f}", extra={'rank': local_rank})
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val MCC: {val_metrics['mcc']:.4f}", extra={'rank': local_rank})
                logger.info(f"Semi-synthetic - Precision: {val_metrics['semi_synthetic_precision']:.4f}, "
                           f"Recall: {val_metrics['semi_synthetic_recall']:.4f}, F1: {val_metrics['semi_synthetic_f1']:.4f}", extra={'rank': local_rank})
                logger.info(f"Real-Semi Confusion Rate: {val_metrics['real_semi_confusion_rate']:.4f} "
                           f"(Real→Semi: {val_metrics['real_as_semi_count']}, Semi→Real: {val_metrics['semi_as_real_count']})", extra={'rank': local_rank})
                logger.info(f"Synthetic Accuracy: {val_metrics['synthetic_accuracy']:.4f}", extra={'rank': local_rank})
                
                # Improved best model selection (prioritize low confusion rate)
                current_score = (val_metrics['semi_synthetic_f1'] * 0.4 + 
                               val_metrics['mcc'] * 0.3 + 
                               (1 - val_metrics['real_semi_confusion_rate']) * 0.3)
                best_score = (best_metrics['semi_synthetic_f1'] * 0.4 + 
                             best_metrics['mcc'] * 0.3 + 
                             (1 - best_metrics['real_semi_confusion_rate']) * 0.3)
                
                if current_score > best_score:
                    best_metrics = val_metrics.copy()
                    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_improved_model.pth")
                    save_improved_checkpoint(model, current_optimizer, scaler, epoch, val_metrics, config, best_model_path, is_best=True, local_rank=local_rank)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                # Save latest checkpoint
                latest_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': current_optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_metrics': best_metrics,
                    'training_history': dict(training_history),
                    'epochs_no_improve': epochs_no_improve,
                    'config': config.__dict__
                }
                torch.save(latest_checkpoint, os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth"))
                
                # Periodic checkpoints
                if (epoch + 1) % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"improved_checkpoint_epoch_{epoch+1}.pth")
                    save_improved_checkpoint(model, current_optimizer, scaler, epoch, val_metrics, config, checkpoint_path, local_rank=local_rank)
                
                # Generate plots
                plot_improved_metrics(training_history, os.path.join(config.CHECKPOINT_DIR, f'improved_metrics_epoch_{epoch+1}.png'))
                cm = confusion_matrix(val_targets, val_preds)
                plot_confusion_matrix(cm, config.CLASS_NAMES, os.path.join(config.CHECKPOINT_DIR, f'improved_cm_epoch_{epoch+1}.png'))
                
                # Early stopping
                if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs", extra={'rank': local_rank})
                    break
            
            if config.DISTRIBUTED:
                dist.barrier()
            cleanup_memory()
        
        # Final evaluation
        if local_rank in [-1, 0]:
            logger.info("IMPROVED TRAINING COMPLETED - FINAL EVALUATION", extra={'rank': local_rank})
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_improved_model.pth")
            if os.path.exists(best_model_path):
                logger.info("Loading best model for final evaluation...", extra={'rank': local_rank})
                checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
                if config.DISTRIBUTED:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Best model loaded - Semi-synthetic F1: {checkpoint['metrics']['semi_synthetic_f1']:.4f}, "
                           f"Real-Semi confusion: {checkpoint['metrics']['real_semi_confusion_rate']:.4f}", extra={'rank': local_rank})
            
            test_loss, test_metrics, test_preds, test_targets, test_probs = evaluate_improved_model(
                model, test_loader, criterion, config, config.DEVICE, epoch + 1, local_rank
            )
            
            # Detailed final results
            logger.info(f"FINAL TEST RESULTS:", extra={'rank': local_rank})
            logger.info(f"Test Loss: {test_loss:.4f}", extra={'rank': local_rank})
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}", extra={'rank': local_rank})
            logger.info(f"Test MCC: {test_metrics['mcc']:.4f}", extra={'rank': local_rank})
            logger.info(f"Semi-synthetic - Precision: {test_metrics['semi_synthetic_precision']:.4f}, "
                       f"Recall: {test_metrics['semi_synthetic_recall']:.4f}, F1: {test_metrics['semi_synthetic_f1']:.4f}", extra={'rank': local_rank})
            logger.info(f"Real-Semi Confusion Rate: {test_metrics['real_semi_confusion_rate']:.4f}", extra={'rank': local_rank})
            logger.info(f"Synthetic Accuracy: {test_metrics['synthetic_accuracy']:.4f}", extra={'rank': local_rank})
            
            class_report = classification_report(test_targets, test_preds, target_names=config.CLASS_NAMES, digits=4)
            logger.info(f"\nDETAILED CLASSIFICATION REPORT:\n{class_report}", extra={'rank': local_rank})
            
            cm = test_metrics['confusion_matrix']
            logger.info(f"CONFUSION MATRIX:", extra={'rank': local_rank})
            logger.info(f"Classes: {config.CLASS_NAMES}", extra={'rank': local_rank})
            for i, row in enumerate(cm):
                logger.info(f"{config.CLASS_NAMES[i]}: {row}", extra={'rank': local_rank})
            
            # Save final results
            final_results = {
                'training_history': dict(training_history),
                'best_val_metrics': best_metrics,
                'final_test_metrics': test_metrics,
                'config': config.__dict__,
                'model_improvements': {
                    'synthetic_style_expansion': True,
                    'enhanced_class_separation': True,
                    'forensic_aware_augmentation': True,
                    'real_semi_confusion_focus': True
                }
            }
            
            results_path = os.path.join(config.CHECKPOINT_DIR, "improved_final_results.json")
            with open(results_path, 'w') as f:
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {key: convert_numpy(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                json.dump(convert_numpy(final_results), f, indent=2)
            
            logger.info(f"Final results saved: {results_path}", extra={'rank': local_rank})
        
        if config.DISTRIBUTED:
            cleanup_distributed()
        cleanup_memory()
        
        if local_rank in [-1, 0]:
            logger.info("Improved training completed successfully!", extra={'rank': local_rank})
        
        return training_history
        
    except Exception as e:
        logger.error(f"Error in improved_train_worker for rank {local_rank}: {e}", extra={'rank': local_rank})
        if config.DISTRIBUTED:
            cleanup_distributed()
        cleanup_memory()
        raise

def improved_train_single_gpu(config):
    """Single GPU training wrapper"""
    return improved_train_worker(-1, config, config.MASTER_PORT)

def generate_expanded_synthetic_dataset_offline(original_dataset_path, output_path, config):
    """
    Offline generation of expanded synthetic dataset
    Use this to pre-generate all style variations before training
    """
    original_path = Path(original_dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    style_generator = SyntheticStyleGenerator(config)
    
    logger.info("Generating expanded synthetic dataset offline...", extra={'rank': 0})
    
    # Copy real and semi-synthetic classes as-is
    for class_name in ["real", "semi-synthetic"]:
        src_dir = original_path / class_name
        dst_dir = output_path / class_name
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            logger.info(f"Copied {class_name} class to expanded dataset", extra={'rank': 0})
    
    # Process synthetic class with style expansion
    synthetic_dir = original_path / "synthetic"
    output_synthetic_dir = output_path / "synthetic"
    output_synthetic_dir.mkdir(exist_ok=True)
    
    if not synthetic_dir.exists():
        logger.error("Synthetic directory not found!", extra={'rank': 0})
        return
    
    pt_files = list(synthetic_dir.glob('*.pt'))
    total_original_images = 0
    total_generated_images = 0
    
    for pt_file in tqdm(pt_files, desc="Processing synthetic files"):
        logger.info(f"Processing {pt_file.name}...", extra={'rank': 0})
        
        try:
            # Load original tensor data
            tensor_data = torch.load(pt_file, map_location='cpu')
            if isinstance(tensor_data, dict):
                tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
            
            # Create style variations
            original_images = []
            bright_images = []
            overlay_images = []
            
            for i in range(tensor_data.shape[0]):
                image_tensor = tensor_data[i]
                if image_tensor.dtype != torch.float32:
                    image_tensor = image_tensor.float()
                if image_tensor.max() > 1.0:
                    image_tensor = image_tensor / 255.0
                image_tensor = torch.clamp(image_tensor, 0, 1)
                
                # Convert to numpy for processing
                image_np = image_tensor.permute(1, 2, 0).numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                # Keep original (dark cinematic style)
                original_images.append(image_tensor)
                
                # Create bright saturated version (server domain style)
                bright_np = style_generator.create_bright_saturated_style(image_np)
                if config.PRESERVE_FORENSIC_ARTIFACTS:
                    bright_np = style_generator.preserve_compression_artifacts(bright_np)
                bright_tensor = torch.from_numpy(bright_np).permute(2, 0, 1).float() / 255.0
                bright_images.append(bright_tensor)
                
                # Create overlay version (social media style)
                overlay_np = style_generator.create_overlay_effect(bright_np)
                if config.PRESERVE_FORENSIC_ARTIFACTS:
                    overlay_np = style_generator.preserve_compression_artifacts(overlay_np)
                overlay_tensor = torch.from_numpy(overlay_np).permute(2, 0, 1).float() / 255.0
                overlay_images.append(overlay_tensor)
                
                total_original_images += 1
                total_generated_images += 2  # bright + overlay
            
            # Save expanded versions
            base_name = pt_file.stem
            
            # Original style (dark cinematic) - keep original filename pattern
            torch.save(torch.stack(original_images), output_synthetic_dir / f"{base_name}_original.pt")
            
            # Bright saturated style (server domain)
            torch.save(torch.stack(bright_images), output_synthetic_dir / f"{base_name}_bright.pt")
            
            # Overlay style (social media processed)
            torch.save(torch.stack(overlay_images), output_synthetic_dir / f"{base_name}_overlay.pt")
            
        except Exception as e:
            logger.error(f"Error processing {pt_file}: {e}", extra={'rank': 0})
    
    logger.info(f"Offline dataset expansion completed:", extra={'rank': 0})
    logger.info(f"  Original synthetic images: {total_original_images}", extra={'rank': 0})
    logger.info(f"  Generated style variations: {total_generated_images}", extra={'rank': 0})
    logger.info(f"  Total synthetic images: {total_original_images + total_generated_images}", extra={'rank': 0})
    logger.info(f"  Expansion factor: {(total_original_images + total_generated_images) / total_original_images:.1f}x", extra={'rank': 0})
    logger.info(f"Expanded dataset saved to: {output_path}", extra={'rank': 0})

def main():
    """Main function with improved training pipeline"""
    # CRITICAL FIX: Set multiprocessing start method before any CUDA operations
    set_multiprocessing_start_method()
    
    # Environment setup for distributed training
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(torch.cuda.device_count()))
    
    parser = argparse.ArgumentParser(description='Improved Deepfake Detection Training with Synthetic Style Expansion')
    parser.add_argument('--train_path', type=str, default='datasets/train', help='Path to training data')
    parser.add_argument('--expanded_train_path', type=str, default='datasets/train_expanded', help='Path to expanded training data')
    parser.add_argument('--generate_expanded_dataset', action='store_true', help='Generate expanded dataset offline first')
    parser.add_argument('--use_expanded_dataset', action='store_true', help='Use pre-generated expanded dataset')
    
    # Model parameters
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size (reduced for expanded dataset)')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--backbone', type=str, default='convnext_tiny', choices=['convnext_tiny', 'convnext_small'], help='Backbone architecture')
    parser.add_argument('--hidden_dim', type=int, default=1536, help='Hidden dimension size')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--sgd_lr', type=float, default=3e-6, help='SGD learning rate for fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=2e-2, help='Weight decay')
    parser.add_argument('--fine_tune_start', type=int, default=28, help='Epoch to start fine-tuning')
    parser.add_argument('--unfreeze_epochs', type=int, nargs='+', default=[1, 8, 18, 28, 38, 48], help='Epochs for progressive unfreezing')
    parser.add_argument('--early_stopping_patience', type=int, default=50, help='Early stopping patience')
    
    # Loss and augmentation parameters
    parser.add_argument('--focal_gamma', type=float, default=3.0, help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--contrastive_weight', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--class_separation_weight', type=float, default=0.25, help='Class separation loss weight')
    parser.add_argument('--no_mixup', action='store_true', help='Disable mixup augmentation')
    parser.add_argument('--no_cutmix', action='store_true', help='Disable cutmix augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.25, help='Mixup alpha parameter')
    parser.add_argument('--cutmix_alpha', type=float, default=0.8, help='Cutmix alpha parameter')
    
    # Synthetic style expansion parameters
    parser.add_argument('--no_style_expansion', action='store_true', help='Disable synthetic style expansion')
    parser.add_argument('--synthetic_multiplier', type=int, default=3, help='Synthetic style multiplier')
    parser.add_argument('--bright_style_prob', type=float, default=0.6, help='Probability of bright style transformation')
    parser.add_argument('--overlay_style_prob', type=float, default=0.3, help='Probability of overlay effect')
    parser.add_argument('--no_forensic_preservation', action='store_true', help='Disable forensic artifact preservation')
    
    # System parameters
    parser.add_argument('--checkpoint_dir', type=str, default='improve1_checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=3, help='Save checkpoint every N epochs')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port for distributed training')
    
    # Feature flags
    parser.add_argument('--no_forensics', action='store_true', help='Disable forensics module')
    parser.add_argument('--no_uncertainty', action='store_true', help='Disable uncertainty estimation')
    parser.add_argument('--no_spectral_norm', action='store_true', help='Disable spectral normalization')
    
    args = parser.parse_args()
    
    # Create improved configuration
    config = ImprovedConfig()
    
    # Update config with command line arguments
    config.TRAIN_PATH = args.expanded_train_path if args.use_expanded_dataset else args.train_path
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    config.NUM_WORKERS = args.num_workers
    config.CONVNEXT_BACKBONE = args.backbone
    config.HIDDEN_DIM = args.hidden_dim
    config.DROPOUT_RATE = args.dropout_rate
    config.EPOCHS = args.epochs
    config.ADAMW_LR = args.lr
    config.SGD_LR = args.sgd_lr
    config.WEIGHT_DECAY = args.weight_decay
    config.FINE_TUNE_START_EPOCH = args.fine_tune_start
    config.UNFREEZE_EPOCHS = args.unfreeze_epochs
    config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    config.FOCAL_GAMMA = args.focal_gamma
    config.LABEL_SMOOTHING = args.label_smoothing
    config.CONTRASTIVE_WEIGHT = args.contrastive_weight
    config.CLASS_SEPARATION_WEIGHT = args.class_separation_weight
    config.USE_MIXUP = not args.no_mixup
    config.USE_CUTMIX = not args.no_cutmix
    config.MIXUP_ALPHA = args.mixup_alpha
    config.CUTMIX_ALPHA = args.cutmix_alpha
    config.EXPAND_SYNTHETIC_STYLES = not args.no_style_expansion
    config.SYNTHETIC_STYLE_MULTIPLIER = args.synthetic_multiplier
    config.BRIGHT_STYLE_PROB = args.bright_style_prob
    config.OVERLAY_STYLE_PROB = args.overlay_style_prob
    config.PRESERVE_FORENSIC_ARTIFACTS = not args.no_forensic_preservation
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.CHECKPOINT_EVERY_N_EPOCHS = args.checkpoint_every_n_epochs
    config.DISTRIBUTED = args.distributed
    config.MASTER_ADDR = args.master_addr
    config.MASTER_PORT = args.master_port
    config.USE_FORENSICS_MODULE = not args.no_forensics
    config.USE_UNCERTAINTY_ESTIMATION = not args.no_uncertainty
    config.USE_SPECTRAL_NORM = not args.no_spectral_norm
    
    # Handle tensor device assignment properly
    if not config.DISTRIBUTED:
        config.FOCAL_ALPHA = config.FOCAL_ALPHA.to(config.DEVICE)
        config.CLASS_WEIGHTS = config.CLASS_WEIGHTS.to(config.DEVICE)
    # For distributed training, tensors will be moved to device in worker function
    
    config.validate()
    
    # Generate expanded dataset offline if requested
    if args.generate_expanded_dataset:
        logger.info("Generating expanded synthetic dataset offline...", extra={'rank': 0})
        generate_expanded_synthetic_dataset_offline(args.train_path, args.expanded_train_path, config)
        logger.info("Dataset expansion completed. You can now train with --use_expanded_dataset", extra={'rank': 0})
        return
    
    # Log configuration
    logger.info("=== IMPROVED DEEPFAKE DETECTION TRAINING ===", extra={'rank': 0})
    logger.info(f"Training with expanded synthetic styles: {config.EXPAND_SYNTHETIC_STYLES}", extra={'rank': 0})
    logger.info(f"Synthetic style multiplier: {config.SYNTHETIC_STYLE_MULTIPLIER}x", extra={'rank': 0})
    logger.info(f"Focus on real vs semi-synthetic separation: Enhanced", extra={'rank': 0})
    logger.info(f"Forensic artifact preservation: {config.PRESERVE_FORENSIC_ARTIFACTS}", extra={'rank': 0})
    logger.info(f"Dataset path: {config.TRAIN_PATH}", extra={'rank': 0})
    logger.info(f"Checkpoint directory: {config.CHECKPOINT_DIR}", extra={'rank': 0})
    
    if config.DISTRIBUTED and torch.cuda.device_count() > 1:
        master_port = find_free_port(int(config.MASTER_PORT))
        logger.info(f"Using master port: {master_port}", extra={'rank': 0})
        world_size = torch.cuda.device_count()
        logger.info(f"Starting distributed training with {world_size} GPUs", extra={'rank': 0})
        
        # Use spawn method for multiprocessing
        mp.spawn(
            improved_train_worker,
            args=(config, str(master_port)),
            nprocs=world_size,
            join=True
        )
    else:
        logger.info("Starting single GPU training", extra={'rank': 0})
        improved_train_single_gpu(config)
    
    logger.info("=== TRAINING COMPLETED ===", extra={'rank': 0})

if __name__ == '__main__':
    main()