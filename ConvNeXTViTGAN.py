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

warnings.filterwarnings('ignore')

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

class SuperiorConfig:
    def __init__(self):
        self.MODEL_TYPE = "superior_forensics_model"
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
        self.BATCH_SIZE = 28
        self.EPOCHS = 50
        self.USE_AMP = True
        self.TRAIN_PATH = "datasets/train"
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = 4
        self.UNFREEZE_EPOCHS = [5, 10, 15, 20, 50]
        self.FINE_TUNE_START_EPOCH = 20
        self.EARLY_STOPPING_PATIENCE = 8
        self.ADAMW_LR = 3e-4
        self.SGD_LR = 5e-6
        self.SGD_MOMENTUM = 0.95
        self.WEIGHT_DECAY = 2e-2
        self.FOCAL_ALPHA = torch.tensor([1.0, 3.0, 2.5]).to(self.DEVICE)
        self.FOCAL_GAMMA = 3.5
        self.LABEL_SMOOTHING = 0.15
        self.CLASS_WEIGHTS = torch.tensor([1.0, 3.0, 2.0]).to(self.DEVICE)
        self.USE_FORENSICS_MODULE = True
        self.USE_UNCERTAINTY_ESTIMATION = True
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = 0.3
        self.USE_CUTMIX = True
        self.CUTMIX_ALPHA = 1.0
        self.USE_ENSEMBLE = True
        self.ENSEMBLE_SIZE = 3
        self.CONTRASTIVE_WEIGHT = 0.4
        self.EVIDENTIAL_WEIGHT = 0.3
        self.BOUNDARY_WEIGHT = 0.2
        self.TRIPLET_WEIGHT = 0.25
        self.CHECKPOINT_DIR = "superior_checkpoints"
        self.CHECKPOINT_EVERY_N_EPOCHS = 3
        self.USE_MCC_FOR_BEST_MODEL = True
        self.SAVE_TOP_K_MODELS = 3

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

class SuperiorLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.device = config.DEVICE
        self.focal_alpha = config.FOCAL_ALPHA
        self.focal_gamma = config.FOCAL_GAMMA
        self.class_weights = config.CLASS_WEIGHTS
        self.evidential_weight = config.EVIDENTIAL_WEIGHT
        self.contrastive_weight = config.CONTRASTIVE_WEIGHT
        self.boundary_weight = config.BOUNDARY_WEIGHT
        self.triplet_weight = config.TRIPLET_WEIGHT
        self.label_smoothing = config.LABEL_SMOOTHING
        self.triplet_margin = 1.0
    
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
        annealing_coef = min(1.0, epoch / 15.0)
        return likelihood_loss.mean() + annealing_coef * 0.1 * kl_div.mean()
    
    def contrastive_loss(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features, features.t())
        labels_expanded = labels.unsqueeze(1)
        same_class_mask = (labels_expanded == labels_expanded.t()).float()
        diff_class_mask = 1 - same_class_mask
        semi_synthetic_mask = (labels == 1).float().unsqueeze(1)
        semi_synthetic_pairs = semi_synthetic_mask * semi_synthetic_mask.t()
        pos_loss = same_class_mask * (1 - similarity_matrix) ** 2
        neg_loss = diff_class_mask * torch.clamp(similarity_matrix - 0.3, min=0) ** 2
        semi_loss = semi_synthetic_pairs * (1 - similarity_matrix) ** 2
        return (pos_loss.sum() + neg_loss.sum() + 2.0 * semi_loss.sum()) / (labels.size(0) ** 2)
    
    def boundary_loss(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        real_mask = (targets == 0).float()
        synthetic_mask = (targets == 2).float()
        semi_mask = (targets == 1).float()
        real_boundary = real_mask * probs[:, 1]
        synthetic_boundary = synthetic_mask * probs[:, 1]
        semi_boundary = semi_mask * (1 - probs[:, 1])
        return (real_boundary.sum() + synthetic_boundary.sum() + 2.0 * semi_boundary.sum()) / targets.size(0)
    
    def triplet_loss(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        triplet_losses = []
        for i in range(features.size(0)):
            anchor_label = labels[i]
            anchor_feat = features[i]
            pos_mask = (labels == anchor_label) & (torch.arange(features.size(0)).to(self.device) != i)
            if pos_mask.sum() == 0:
                continue
            neg_mask = (labels != anchor_label)
            if neg_mask.sum() == 0:
                continue
            pos_distances = torch.norm(features[pos_mask] - anchor_feat.unsqueeze(0), p=2, dim=1)
            neg_distances = torch.norm(features[neg_mask] - anchor_feat.unsqueeze(0), p=2, dim=1)
            hardest_pos_dist = pos_distances.max()
            hardest_neg_dist = neg_distances.min()
            triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.triplet_margin)
            triplet_losses.append(triplet_loss)
        return torch.stack(triplet_losses).mean() if triplet_losses else torch.tensor(0.0).to(self.device)
    
    def forward(self, logits, targets, features=None, alpha=None, epoch=1):
        focal_loss = self.focal_loss(logits, targets)
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        boundary_loss = self.boundary_loss(logits, targets)
        total_loss = 0.5 * focal_loss + 0.3 * ce_loss + self.boundary_weight * boundary_loss
        if features is not None:
            contrastive_loss = self.contrastive_loss(features, targets)
            triplet_loss = self.triplet_loss(features, targets)
            total_loss += self.contrastive_weight * contrastive_loss
            total_loss += self.triplet_weight * triplet_loss
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

class SuperiorModel(nn.Module):
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
        logger.info("Unfrozen classifier and fusion layers", extra={'rank': 0})
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]
        features_list = [convnext_feats, vit_feats]
        if self.config.USE_FORENSICS_MODULE:
            forensics_feats = self.forensics_module(x)
            features_list.append(forensics_feats)
        fused_features = torch.cat(features_list, dim=1)
        attended_features = self.attention_module(fused_features)
        processed_features = self.fusion(attended_features)
        logits = self.classifier(processed_features)
        if self.config.USE_UNCERTAINTY_ESTIMATION and hasattr(self, 'uncertainty_module'):
            probs, epistemic_unc, aleatoric_unc, alpha = self.uncertainty_module(processed_features)
            return logits, processed_features, (probs, epistemic_unc, aleatoric_unc, alpha)
        return logits, processed_features

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

class SuperiorDataset(Dataset):
    def __init__(self, root_dir, config, transform=None, is_training=True):
        self.root_dir = Path(root_dir)
        self.config = config
        self.is_training = is_training
        self.class_names = config.CLASS_NAMES
        self.file_indices = []
        self.labels = []
        self.metadata_file = self.root_dir / "superior_metadata.json"
        self.tensor_cache = {}
        self.real_transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=3, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=8, val_shift_limit=8, p=0.3),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.semi_synthetic_transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=8, distort_limit=0.15, p=0.4),
                A.RandomGridShuffle(grid=(4, 4), p=0.3),
                A.GaussNoise(var_limit=(15.0, 40.0), p=0.3),
            ], p=0.8),
            A.OneOf([
                A.ImageCompression(quality_lower=70, quality_upper=95, p=0.4),
                A.Blur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.6),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=12, p=0.4),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.synthetic_transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=8, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.4),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3),
            ], p=0.5),
            A.CoarseDropout(max_holes=6, max_height=12, max_width=12, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.val_transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        if transform is not None:
            self.val_transform = transform
        self._validate_dataset_path()
        self._load_file_mapping()
        self._balance_dataset()
    
    def _validate_dataset_path(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.root_dir} does not exist")
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist", extra={'rank': 0})
    
    def _load_file_mapping(self):
        logger.info("Creating/loading superior file mapping...", extra={'rank': 0})
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.file_indices = metadata['file_indices']
                self.labels = metadata['labels']
                logger.info(f"Loaded metadata from {self.metadata_file}", extra={'rank': 0})
        else:
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = self.root_dir / class_name
                if not class_dir.exists():
                    continue
                pt_files = list(class_dir.glob('*.pt'))
                for pt_file in pt_files:
                    try:
                        if str(pt_file) not in self.tensor_cache:
                            tensor_data = torch.load(pt_file, map_location='cpu')
                            if isinstance(tensor_data, dict):
                                tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                            self.tensor_cache[str(pt_file)] = tensor_data
                        num_images = tensor_data.shape[0]
                        for i in range(num_images):
                            self.file_indices.append((str(pt_file), i))
                            self.labels.append(class_idx)
                    except Exception as e:
                        logger.warning(f"Skipping corrupted file {pt_file}: {e}", extra={'rank': 0})
            metadata = {'file_indices': self.file_indices, 'labels': self.labels}
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
            logger.info(f"Saved metadata to {self.metadata_file}", extra={'rank': 0})
        logger.info(f"Found {len(self.file_indices)} images", extra={'rank': 0})
        class_counts = np.bincount(self.labels)
        for i, count in enumerate(class_counts):
            logger.info(f"Class {self.class_names[i]}: {count} samples", extra={'rank': 0})
    
    def _balance_dataset(self):
        if not self.is_training:
            return
        class_counts = np.bincount(self.labels)
        max_count = max(class_counts)
        balanced_indices = []
        balanced_labels = []
        for class_idx in range(len(self.class_names)):
            class_indices = [i for i, label in enumerate(self.labels) if label == class_idx]
            target_count = int(max_count * 1.5) if class_idx == 1 else max_count
            oversampled_indices = np.random.choice(class_indices, target_count, replace=True)
            balanced_indices.extend(oversampled_indices)
            balanced_labels.extend([class_idx] * target_count)
        balanced_file_indices = [self.file_indices[i] for i in balanced_indices]
        self.file_indices = balanced_file_indices
        self.labels = balanced_labels
        logger.info(f"Balanced dataset: {len(self.file_indices)} samples", extra={'rank': 0})
        new_class_counts = np.bincount(self.labels)
        for i, count in enumerate(new_class_counts):
            logger.info(f"Balanced class {self.class_names[i]}: {count} samples", extra={'rank': 0})
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        try:
            file_path, image_idx = self.file_indices[idx]
            label = self.labels[idx]
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
            image_np = image_tensor.permute(1, 2, 0).numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            if self.is_training:
                if label == 0:
                    transformed = self.real_transform(image=image_np)
                elif label == 1:
                    transformed = self.semi_synthetic_transform(image=image_np)
                else:
                    transformed = self.synthetic_transform(image=image_np)
            else:
                transformed = self.val_transform(image=image_np)
            image_tensor = transformed['image']
            return image_tensor, label
        except Exception as e:
            logger.warning(f"Error loading image at index {idx}: {e}", extra={'rank': 0})
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 0

def create_superior_data_loaders(config, local_rank=-1):
    train_dataset = SuperiorDataset(root_dir=config.TRAIN_PATH, config=config, is_training=True)
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.is_training = False
    test_dataset.dataset.is_training = False
    train_sampler = DistributedSampler(train_dataset, rank=local_rank, shuffle=True) if config.DISTRIBUTED and local_rank != -1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    return train_loader, val_loader, test_loader

def calculate_superior_metrics(y_true, y_pred, y_probs=None, class_names=None):
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    mcc = matthews_corrcoef(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    semi_synthetic_precision = precision[1] if len(precision) > 1 else 0
    semi_synthetic_recall = recall[1] if len(recall) > 1 else 0
    semi_synthetic_f1 = f1[1] if len(f1) > 1 else 0
    cm = confusion_matrix(y_true, y_pred)
    total_semi = cm[1].sum() if len(cm) >= 3 else 0
    semi_confusion_rate = (cm[1, 0] + cm[1, 2]) / total_semi if len(cm) >= 3 and total_semi > 0 else 0
    metrics = {
        'accuracy': accuracy, 'mcc': mcc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
        'per_class_precision': precision.tolist(), 'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(), 'semi_synthetic_precision': semi_synthetic_precision,
        'semi_synthetic_recall': semi_synthetic_recall, 'semi_synthetic_f1': semi_synthetic_f1,
        'semi_confusion_rate': semi_confusion_rate, 'confusion_matrix': cm.tolist()
    }
    return metrics

def evaluate_superior_model(model, data_loader, criterion, config, device, epoch=1, local_rank=0):
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
    metrics = calculate_superior_metrics(all_targets, all_predictions, all_probabilities, config.CLASS_NAMES)
    if all_uncertainties:
        metrics['mean_uncertainty'] = np.mean(all_uncertainties)
        metrics['uncertainty_std'] = np.std(all_uncertainties)
    return avg_loss, metrics, all_predictions, all_targets, all_probabilities

def mixup_criterion(criterion, model_output, target_a, target_b, lam, epoch):
    if len(model_output) == 3:
        logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
        loss_a = criterion(logits, target_a, features, alpha, epoch)
        loss_b = criterion(logits, target_b, features, alpha, epoch)
    else:
        logits, features = model_output
        loss_a = criterion(logits, target_a, features, None, epoch)
        loss_b = criterion(logits, target_b, features, None, epoch)
    return lam * loss_a + (1 - lam) * loss_b

def create_superior_optimizer(model, config, epoch):
    backbone_params = []
    forensics_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'convnext' in name or 'vit' in name:
            backbone_params.append(param)
        elif 'forensics' in name or 'attention' in name:
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

def progressive_unfreeze_advanced(model, epoch, config):
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
            model.unfreeze_vit_layers()
            unfroze_this_epoch = True
        elif unfreeze_stage == 4:
            model.unfreeze_convnext_layers()
            unfroze_this_epoch = True
        elif unfreeze_stage == 5:
            model.unfreeze_vit_layers()
            model.unfreeze_convnext_layers()
            unfroze_this_epoch = True
        trainable_params = model.get_trainable_params()
        logger.info(f"Progressive unfreezing stage {unfreeze_stage}: {trainable_params:,} trainable parameters", extra={'rank': 0})
    return unfroze_this_epoch

def save_superior_checkpoint(model, optimizer, scaler, epoch, metrics, config, filename, is_best=False, local_rank=0):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'config': config.__dict__,
        'is_best': is_best
    }
    torch.save(checkpoint, filename)
    if local_rank in [-1, 0]:
        logger.info(f"Checkpoint saved: {filename}", extra={'rank': local_rank})
        if is_best:
            logger.info(f"New best model - Semi-synthetic F1: {metrics.get('semi_synthetic_f1', 0):.4f}", extra={'rank': local_rank})

def plot_superior_metrics(history, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    if 'val_mcc' in history:
        axes[0, 1].plot(history['val_mcc'], label='Val MCC', color='green')
        axes[0, 1].set_title('Matthews Correlation Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MCC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    if 'semi_f1' in history and 'semi_precision' in history and 'semi_recall' in history:
        axes[0, 2].plot(history['semi_f1'], label='Semi-synthetic F1', color='purple')
        axes[0, 2].plot(history['semi_precision'], label='Semi-synthetic Precision', color='orange')
        axes[0, 2].plot(history['semi_recall'], label='Semi-synthetic Recall', color='brown')
        axes[0, 2].set_title('Semi-synthetic Class Metrics')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    if 'val_accuracy' in history:
        axes[1, 0].plot(history['val_accuracy'], label='Val Accuracy', color='cyan')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    if 'macro_f1' in history and 'weighted_f1' in history:
        axes[1, 1].plot(history['macro_f1'], label='Macro F1', color='magenta')
        axes[1, 1].plot(history['weighted_f1'], label='Weighted F1', color='yellow')
        axes[1, 1].set_title('F1 Scores')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    if 'semi_confusion_rate' in history:
        axes[1, 2].plot(history['semi_confusion_rate'], label='Semi-synthetic Confusion Rate', color='red')
        axes[1, 2].set_title('Semi-synthetic Confusion Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Confusion Rate')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Superior Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    plt.figtext(0.15, 0.02, f'Overall Accuracy: {accuracy:.4f}')
    for i, class_name in enumerate(class_names):
        if i < len(cm):
            class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            plt.figtext(0.15, 0.06 - i*0.02, f'{class_name} Accuracy: {class_acc:.4f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def superior_train_worker(local_rank, config, master_port):
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if config.DISTRIBUTED:
            success = setup_distributed(local_rank, torch.cuda.device_count(), config.BACKEND, config.MASTER_ADDR, master_port)
            if not success:
                logger.error(f"Failed to setup distributed training for rank {local_rank}", extra={'rank': local_rank})
                return
            config.DEVICE = torch.device(f'cuda:{local_rank}')
            config.FOCAL_ALPHA = config.FOCAL_ALPHA.to(config.DEVICE)
            config.CLASS_WEIGHTS = config.CLASS_WEIGHTS.to(config.DEVICE)
        
        logger.info(f"Superior training setup complete for rank {local_rank}", extra={'rank': local_rank})
        
        torch.manual_seed(42 + local_rank)
        np.random.seed(42 + local_rank)
        random.seed(42 + local_rank)
        
        if local_rank in [-1, 0]:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            logger.info(f"Created checkpoint directory: {config.CHECKPOINT_DIR}", extra={'rank': local_rank})
        
        train_loader, val_loader, test_loader = create_superior_data_loaders(config, local_rank)
        
        model = SuperiorModel(config).to(config.DEVICE)
        if config.DISTRIBUTED:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
        criterion = SuperiorLoss(config).to(config.DEVICE)
        scaler = GradScaler(enabled=config.USE_AMP)
        mixup_cutmix = MixupCutmixAugmentation(config)
        
        best_metrics = {'semi_synthetic_f1': -1.0, 'mcc': -1.0}
        epochs_no_improve = 0
        training_history = defaultdict(list)
        current_optimizer = None
        
        if local_rank in [-1, 0]:
            initial_trainable = model.module.get_trainable_params() if config.DISTRIBUTED else model.get_trainable_params()
            logger.info(f"Starting superior training with {initial_trainable:,} trainable parameters", extra={'rank': local_rank})
        
        for epoch in range(config.EPOCHS):
            start_time = time.time()
            model_for_unfreeze = model.module if config.DISTRIBUTED else model
            unfroze_this_epoch = progressive_unfreeze_advanced(model_for_unfreeze, epoch + 1, config)
            if current_optimizer is None or unfroze_this_epoch or epoch == config.FINE_TUNE_START_EPOCH:
                current_optimizer = create_superior_optimizer(model, config, epoch + 1)
            
            model.train()
            train_loss = 0
            train_batches = 0
            if config.DISTRIBUTED:
                train_loader.sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS}', disable=local_rank not in [-1, 0])
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
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
                    progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{train_loss/train_batches:.4f}'})
                if batch_idx % 100 == 0:
                    cleanup_memory()
            
            avg_train_loss = train_loss / train_batches
            if local_rank in [-1, 0]:
                training_history['train_loss'].append(avg_train_loss)
                val_loss, val_metrics, val_preds, val_targets, val_probs = evaluate_superior_model(
                    model, val_loader, criterion, config, config.DEVICE, epoch + 1, local_rank
                )
                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_metrics['accuracy'])
                training_history['val_mcc'].append(val_metrics['mcc'])
                training_history['macro_f1'].append(val_metrics['macro_f1'])
                training_history['weighted_f1'].append(val_metrics['weighted_f1'])
                training_history['semi_f1'].append(val_metrics['semi_synthetic_f1'])
                training_history['semi_precision'].append(val_metrics['semi_synthetic_precision'])
                training_history['semi_recall'].append(val_metrics['semi_synthetic_recall'])
                training_history['semi_confusion_rate'].append(val_metrics['semi_confusion_rate'])
                training_history['val_metrics'].append(val_metrics)
                epoch_time = time.time() - start_time
                logger.info(f"Epoch {epoch+1}/{config.EPOCHS} completed in {epoch_time:.2f}s", extra={'rank': local_rank})
                logger.info(f"Train Loss: {avg_train_loss:.4f}", extra={'rank': local_rank})
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val MCC: {val_metrics['mcc']:.4f}", extra={'rank': local_rank})
                logger.info(f"Semi-synthetic - Precision: {val_metrics['semi_synthetic_precision']:.4f}, "
                           f"Recall: {val_metrics['semi_synthetic_recall']:.4f}, F1: {val_metrics['semi_synthetic_f1']:.4f}", extra={'rank': local_rank})
                logger.info(f"Semi-synthetic Confusion Rate: {val_metrics['semi_confusion_rate']:.4f}", extra={'rank': local_rank})
                current_score = val_metrics['semi_synthetic_f1'] * 0.6 + val_metrics['mcc'] * 0.4
                best_score = best_metrics['semi_synthetic_f1'] * 0.6 + best_metrics['mcc'] * 0.4
                if current_score > best_score:
                    best_metrics = val_metrics.copy()
                    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_superior_model.pth")
                    save_superior_checkpoint(model, current_optimizer, scaler, epoch, val_metrics, config, best_model_path, is_best=True, local_rank=local_rank)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if (epoch + 1) % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"superior_checkpoint_epoch_{epoch+1}.pth")
                    save_superior_checkpoint(model, current_optimizer, scaler, epoch, val_metrics, config, checkpoint_path, local_rank=local_rank)
                plot_superior_metrics(training_history, os.path.join(config.CHECKPOINT_DIR, f'superior_metrics_epoch_{epoch+1}.png'))
                cm = confusion_matrix(val_targets, val_preds)
                plot_confusion_matrix(cm, config.CLASS_NAMES, os.path.join(config.CHECKPOINT_DIR, f'superior_cm_epoch_{epoch+1}.png'))
                if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs", extra={'rank': local_rank})
                    break
            
            if config.DISTRIBUTED:
                dist.barrier()
            cleanup_memory()
        
        if local_rank in [-1, 0]:
            logger.info("SUPERIOR TRAINING COMPLETED - FINAL EVALUATION", extra={'rank': local_rank})
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_superior_model.pth")
            if os.path.exists(best_model_path):
                logger.info("Loading best model for final evaluation...", extra={'rank': local_rank})
                checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
                if config.DISTRIBUTED:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Best model loaded - Semi-synthetic F1: {checkpoint['metrics']['semi_synthetic_f1']:.4f}", extra={'rank': local_rank})
            test_loss, test_metrics, test_preds, test_targets, test_probs = evaluate_superior_model(
                model, test_loader, criterion, config, config.DEVICE, epoch + 1, local_rank
            )
            logger.info(f"Test Loss: {test_loss:.4f}", extra={'rank': local_rank})
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}", extra={'rank': local_rank})
            logger.info(f"Test MCC: {test_metrics['mcc']:.4f}", extra={'rank': local_rank})
            logger.info(f"Semi-synthetic - Precision: {test_metrics['semi_synthetic_precision']:.4f}, "
                       f"Recall: {test_metrics['semi_synthetic_recall']:.4f}, F1: {test_metrics['semi_synthetic_f1']:.4f}", extra={'rank': local_rank})
            class_report = classification_report(test_targets, test_preds, target_names=config.CLASS_NAMES, digits=4)
            logger.info(f"\nDETAILED CLASSIFICATION REPORT:\n{class_report}", extra={'rank': local_rank})
            cm = test_metrics['confusion_matrix']
            logger.info(f"CONFUSION MATRIX:\nClasses: {config.CLASS_NAMES}", extra={'rank': local_rank})
            for i, row in enumerate(cm):
                logger.info(f"{config.CLASS_NAMES[i]}: {row}", extra={'rank': local_rank})
            final_results = {
                'training_history': dict(training_history),
                'best_val_metrics': best_metrics,
                'final_test_metrics': test_metrics,
                'config': config.__dict__
            }
            results_path = os.path.join(config.CHECKPOINT_DIR, "superior_final_results.json")
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
            logger.info("Superior training completed successfully!", extra={'rank': local_rank})
        return training_history
        
    except Exception as e:
        logger.error(f"Error in superior_train_worker for rank {local_rank}: {e}", extra={'rank': local_rank})
        if config.DISTRIBUTED:
            cleanup_distributed()
        cleanup_memory()
        raise

def superior_train_single_gpu(config):
    return superior_train_worker(-1, config, config.MASTER_PORT)

def main():
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(torch.cuda.device_count()))
    parser = argparse.ArgumentParser(description='Superior Deepfake Detection Training')
    parser.add_argument('--train_path', type=str, default='datasets/train')
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='convnext_tiny', choices=['convnext_tiny', 'convnext_small'])
    parser.add_argument('--hidden_dim', type=int, default=1536)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--no_forensics', action='store_true')
    parser.add_argument('--no_uncertainty', action='store_true')
    parser.add_argument('--no_spectral_norm', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sgd_lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--fine_tune_start', type=int, default=15)
    parser.add_argument('--unfreeze_epochs', type=int, nargs='+', default=[10, 20, 30, 40, 50])
    parser.add_argument('--early_stopping_patience', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='superior_checkpoints')
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=3)
    parser.add_argument('--no_mixup', action='store_true')
    parser.add_argument('--no_cutmix', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='12355')
    args = parser.parse_args()
    config = SuperiorConfig()
    config.TRAIN_PATH = args.train_path
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    config.NUM_WORKERS = args.num_workers
    config.CONVNEXT_BACKBONE = args.backbone
    config.HIDDEN_DIM = args.hidden_dim
    config.DROPOUT_RATE = args.dropout_rate
    config.USE_FORENSICS_MODULE = not args.no_forensics
    config.USE_UNCERTAINTY_ESTIMATION = not args.no_uncertainty
    config.USE_SPECTRAL_NORM = not args.no_spectral_norm
    config.EPOCHS = args.epochs
    config.ADAMW_LR = args.lr
    config.SGD_LR = args.sgd_lr
    config.WEIGHT_DECAY = args.weight_decay
    config.FINE_TUNE_START_EPOCH = args.fine_tune_start
    config.UNFREEZE_EPOCHS = args.unfreeze_epochs
    config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.CHECKPOINT_EVERY_N_EPOCHS = args.checkpoint_every_n_epochs
    config.USE_MIXUP = not args.no_mixup
    config.USE_CUTMIX = not args.no_cutmix
    config.MIXUP_ALPHA = args.mixup_alpha
    config.CUTMIX_ALPHA = args.cutmix_alpha
    config.DISTRIBUTED = args.distributed
    config.MASTER_ADDR = args.master_addr
    config.MASTER_PORT = args.master_port
    config.FOCAL_ALPHA = torch.tensor([1.0, 3.0, 2.5]).to(config.DEVICE)
    config.CLASS_WEIGHTS = torch.tensor([1.0, 4.0, 2.0]).to(config.DEVICE)
    config.validate()
    if config.DISTRIBUTED and torch.cuda.device_count() > 1:
        master_port = find_free_port(int(config.MASTER_PORT))
        logger.info(f"Using master port: {master_port}", extra={'rank': 0})
        world_size = torch.cuda.device_count()
        mp.spawn(
            superior_train_worker,
            args=(config, str(master_port)),
            nprocs=world_size,
            join=True
        )
    else:
        superior_train_single_gpu(config)

if __name__ == '__main__':
    main()