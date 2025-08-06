import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import convnext_small, convnext_base, efficientnet_b4
import timm
import os
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
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
from torch.utils.data import random_split

warnings.filterwarnings('ignore')

# Logging setup
def set_multiprocessing_start_method():
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

# Utility functions for device consistency
def ensure_model_device(model, device, rank):
    """Ensure all model parameters and buffers are on the correct device."""
    model.to(device)
    
    # Force move all parameters to the correct device
    for name, param in model.named_parameters():
        if param.device != device:
            logger.warning(f"Moving parameter {name} from {param.device} to {device}", extra={'rank': rank})
            param.data = param.data.to(device, non_blocking=True)
    
    # Force move all buffers to the correct device
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            logger.warning(f"Moving buffer {name} from {buffer.device} to {device}", extra={'rank': rank})
            buffer.data = buffer.data.to(device, non_blocking=True)
    
    # Handle spectral norm parameters specifically
    for module in model.modules():
        if hasattr(module, 'weight_u') and module.weight_u is not None:
            if module.weight_u.device != device:
                module.weight_u = module.weight_u.to(device, non_blocking=True)
        if hasattr(module, 'weight_v') and module.weight_v is not None:
            if module.weight_v.device != device:
                module.weight_v = module.weight_v.to(device, non_blocking=True)

def check_model_devices(model, expected_device, rank):
    """Check if all model parameters are on the expected device."""
    device_issues = []
    for name, param in model.named_parameters():
        if param.device != expected_device:
            device_issues.append(f"Parameter {name} is on {param.device}, expected {expected_device}")
    
    if device_issues:
        logger.error(f"Device mismatch issues found: {len(device_issues)}", extra={'rank': rank})
        for issue in device_issues[:10]:  # Show first 10 issues
            logger.error(issue, extra={'rank': rank})
        if len(device_issues) > 10:
            logger.error(f"... and {len(device_issues) - 10} more issues", extra={'rank': rank})
        return False
    return True

# Distributed training utilities
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def setup_distributed(local_rank, world_size, backend, master_addr, master_port):
    try:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}", extra={'rank': local_rank})
        return False

def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, cleaning up...", extra={'rank': 0})
    cleanup_distributed()
    cleanup_memory()
    exit(0)

# Configuration
class UltimateDeepfakeConfig:
    def __init__(self):
        self.MODEL_TYPE = "ultimate_forensics_ensemble"
        self.BACKBONES = ["convnext_small", "efficientnet_b4", "vit_base_patch16_224"]
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 3072
        self.DROPOUT_RATE = 0.4
        self.FREEZE_BACKBONES = True
        self.ATTENTION_DROPOUT = 0.25
        self.USE_SPECTRAL_NORM = True
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DISTRIBUTED = torch.cuda.device_count() > 1
        self.BACKEND = "nccl"
        self.MASTER_ADDR = "localhost"
        self.MASTER_PORT = "12356"
        self.BATCH_SIZE = 8
        self.EPOCHS = 120
        self.USE_AMP = True
        self.TRAIN_PATH = "datasets/train"
        self.IMAGE_SIZE = (224, 224)
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = 0
        self.UNFREEZE_EPOCHS = [3, 8, 15, 25, 40, 60, 80]
        self.FINE_TUNE_START_EPOCH = 25
        self.EARLY_STOPPING_PATIENCE = 100
        self.ADAMW_LR = 8e-5
        self.SGD_LR = 1.5e-5
        self.SGD_MOMENTUM = 0.95
        self.WEIGHT_DECAY = 3e-2
        self.FOCAL_ALPHA = torch.tensor([1.0, 3.2, 1.0])
        self.FOCAL_GAMMA = 3.2
        self.LABEL_SMOOTHING = 0.05
        self.CLASS_WEIGHTS = torch.tensor([1.0, 2.8, 1.0])
        self.USE_FORENSICS_MODULE = True
        self.USE_UNCERTAINTY_ESTIMATION = False
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = 0.15
        self.USE_CUTMIX = True
        self.CUTMIX_ALPHA = 0.6
        self.USE_ENSEMBLE = False
        self.ENSEMBLE_SIZE = 7
        self.CONTRASTIVE_WEIGHT = 0.4
        self.EVIDENTIAL_WEIGHT = 0.3
        self.BOUNDARY_WEIGHT = 0.25
        self.TRIPLET_WEIGHT = 0.3
        self.SEMI_BOUNDARY_WEIGHT = 0.8
        self.INTER_CLASS_WEIGHT = 0.4
        self.CHECKPOINT_DIR = "ultimate_deepfake_checkpoints"
        self.CHECKPOINT_EVERY_N_EPOCHS = 3
        self.USE_MCC_FOR_BEST_MODEL = True
        self.SAVE_TOP_K_MODELS = 7
        self.USE_STOCHASTIC_DEPTH = True
        self.STOCHASTIC_DEPTH_PROB = 0.08
        self.USE_GRADIENT_CLIPPING = True
        self.GRADIENT_CLIP_VALUE = 0.8
        self.USE_ADVANCED_AUGMENTATION = True
        self.AUGMENTATION_STRENGTH = 0.8
        self.USE_COSINE_ANNEALING = True
        self.USE_WARMUP = True
        self.WARMUP_EPOCHS = 15
        self.USE_TTA = True
        self.TTA_CROPS = 8
        self.USE_CROSS_VALIDATION = False
        self.CV_FOLDS = 5
        self.USE_FREQUENCY_ANALYSIS = True
        self.USE_WAVELET_ANALYSIS = False
        self.USE_GRADIENT_ANALYSIS = True
        self.USE_ADAPTIVE_WEIGHTS = True
        self.WEIGHT_ADAPTATION_FACTOR = 0.05

    def validate(self):
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES
        assert self.FINE_TUNE_START_EPOCH < self.EPOCHS
        assert all(epoch <= self.EPOCHS for epoch in self.UNFREEZE_EPOCHS)
        assert isinstance(self.IMAGE_SIZE, tuple) and len(self.IMAGE_SIZE) == 2

# Dataset
class UltimateDataset(Dataset):
    def __init__(self, root_dir, transform=None, config=None):
        self.root_dir = root_dir
        self.transform = transform
        self.config = config
        self.image_tensors = []
        self.labels = []
        self.class_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
        self.tensor_indices = []
        
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for pt_file in os.listdir(class_dir):
                    if pt_file.lower().endswith('.pt'):
                        pt_path = os.path.join(class_dir, pt_file)
                        try:
                            # Load the .pt file containing 5000 image tensors
                            tensor_batch = torch.load(pt_path, map_location='cpu')
                            if tensor_batch.shape[0] == 5000:  # Verify expected batch size
                                self.image_tensors.append(tensor_batch)
                                # Create labels for all 5000 images in this batch
                                self.labels.extend([self.class_to_idx[class_name]] * 5000)
                                # Store indices to map dataset index to (batch_idx, tensor_idx)
                                batch_idx = len(self.image_tensors) - 1
                                self.tensor_indices.extend([(batch_idx, i) for i in range(5000)])
                            else:
                                logger.warning(f"Skipping {pt_path}: Expected 5000 images, got {tensor_batch.shape[0]}", extra={'rank': 0})
                        except Exception as e:
                            logger.warning(f"Failed to load {pt_path}: {e}", extra={'rank': 0})
        
        self.image_tensors = [t for t in self.image_tensors if t is not None]
        self.labels = np.array(self.labels)
        self.tensor_indices = np.array(self.tensor_indices)
        logger.info(f"Loaded {len(self.labels)} images from {len(self.image_tensors)} .pt files in {root_dir}", extra={'rank': 0})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx >= len(self.labels):
            logger.warning(f"Index {idx} out of range for dataset size {len(self.labels)}", extra={'rank': 0})
            return None, None
        
        batch_idx, tensor_idx = self.tensor_indices[idx]
        image = self.image_tensors[batch_idx][tensor_idx]
        label = self.labels[idx]
        
        # Ensure image is in correct format (C, H, W) and float
        if image.dtype != torch.float32:
            image = image.float()
        if image.shape[0] not in [1, 3]:  # Handle potential grayscale or incorrect channel count
            logger.warning(f"Unexpected channel count in image at index {idx}: {image.shape[0]}", extra={'rank': 0})
            return None, None
        
        # Convert to numpy for albumentations
        image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

# Data augmentation
def create_ultimate_transforms(config):
    train_transforms = A.Compose([
        A.Resize(*config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.3),
        A.MotionBlur(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        A.Resize(*config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transforms, val_transforms

# Data loaders
def create_ultimate_data_loaders(config, local_rank):
    train_transforms, val_transforms = create_ultimate_transforms(config)
    
    full_dataset = UltimateDataset(config.TRAIN_PATH, transform=train_transforms, config=config)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms
    
    if config.DISTRIBUTED:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Loss function
class UltimateLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(
            weight=config.CLASS_WEIGHTS.to(config.DEVICE),
            label_smoothing=config.LABEL_SMOOTHING
        )
        self.focal_loss = FocalLoss(
            alpha=config.FOCAL_ALPHA.to(config.DEVICE),
            gamma=config.FOCAL_GAMMA
        )

    def forward(self, logits, target, features=None, alpha=None, epoch=None, class_f1_scores=None):
        ce_loss = self.ce_loss(logits, target)
        focal_loss = self.focal_loss(logits, target)
        
        total_loss = ce_loss + self.config.FOCAL_GAMMA * focal_loss
        
        if self.config.USE_ADAPTIVE_WEIGHTS and class_f1_scores is not None:
            adaptive_weights = torch.tensor([1.0 / (f1 + 1e-8) for f1 in class_f1_scores]).to(logits.device)
            adaptive_weights = adaptive_weights / adaptive_weights.sum()
            adaptive_loss = F.cross_entropy(logits, target, weight=adaptive_weights)
            total_loss += self.config.WEIGHT_ADAPTATION_FACTOR * adaptive_loss
        
        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Mixup and Cutmix
class UltimateMixupCutmix:
    def __init__(self, config):
        self.config = config

    def __call__(self, images, targets):
        if random.random() < 0.5 and self.config.USE_MIXUP:
            lam = np.random.beta(self.config.MIXUP_ALPHA, self.config.MIXUP_ALPHA)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(images.device)
            mixed_images = lam * images + (1 - lam) * images[index]
            return mixed_images, targets, targets[index], lam
        elif random.random() < 0.5 and self.config.USE_CUTMIX:
            lam = np.random.beta(self.config.CUTMIX_ALPHA, self.config.CUTMIX_ALPHA)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(images.device)
            mixed_images = images.clone()
            h, w = images.size()[2:]
            cut_h, cut_w = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - cut_h // 2, 0, h)
            y2 = np.clip(y + cut_h // 2, 0, h)
            x1 = np.clip(x - cut_w // 2, 0, w)
            x2 = np.clip(x + cut_w // 2, 0, w)
            mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
            return mixed_images, targets, targets[index], lam
        else:
            return images, targets, targets, 1.0

def enhanced_mixup_criterion(criterion, model_output, target_a, target_b, lam, epoch, class_f1_scores=None):
    if len(model_output) == 3:
        logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
        loss_a = criterion(logits, target_a, features, alpha, epoch, class_f1_scores)
        loss_b = criterion(logits, target_b, features, alpha, epoch, class_f1_scores)
    else:
        logits, features = model_output
        loss_a = criterion(logits, target_a, features, None, epoch, class_f1_scores)
        loss_b = criterion(logits, target_b, features, None, epoch, class_f1_scores)
    return lam * loss_a + (1 - lam) * loss_b

# Stochastic depth
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x * random_tensor / keep_prob

# Uncertainty module
class UltimateUncertaintyModule(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.evidential_network = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features // 4, num_classes),
            nn.Softplus()
        )

    def forward(self, x):
        alpha = self.evidential_network(x) + 1
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        S = torch.sum(alpha, dim=1, keepdim=True)
        epistemic_unc = alpha / (S * (S + 1))
        aleatoric_unc = torch.diag(alpha) / (S * (S + 1))
        return probs, epistemic_unc.mean(), aleatoric_unc.mean(), alpha

# Forensics module
class UltimateForensicsModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=8, stride=8),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((12, 12)),
            nn.Flatten(),
            nn.Linear(192 * 12 * 12, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384)
        ).to(config.DEVICE)
        
        self.noise_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        ).to(config.DEVICE)
        
        self.edge_analyzer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        ).to(config.DEVICE)
        
        self.freq_analyzer = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Flatten(),
            nn.Linear(192 * 10 * 10, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192)
        ).to(config.DEVICE)
        
        if config.USE_WAVELET_ANALYSIS:
            self.wavelet_analyzer = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=7, padding=3),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(24, 48, kernel_size=5, padding=2),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(48, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(96 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128)
            ).to(config.DEVICE)
        
        if config.USE_GRADIENT_ANALYSIS:
            self.gradient_analyzer = nn.Sequential(
                nn.Conv2d(2, 24, kernel_size=3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(48, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten(),
                nn.Linear(96 * 6 * 6, 192),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(192, 96)
            ).to(config.DEVICE)
        
        total_forensics_dim = 384 + 256 + 128 + 192
        if config.USE_WAVELET_ANALYSIS:
            total_forensics_dim += 128
        if config.USE_GRADIENT_ANALYSIS:
            total_forensics_dim += 96
        
        self.forensics_fusion = nn.Sequential(
            nn.Linear(total_forensics_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        ).to(config.DEVICE)
        
        if config.USE_SPECTRAL_NORM:
            self._apply_spectral_norm()
        
        self.to(config.DEVICE)
    
    def _apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if not hasattr(module, 'weight_u'):
                    module = nn.utils.spectral_norm(module)
                    # Ensure the spectral norm module is on the correct device
                    module.to(self.config.DEVICE)
                    # Force move spectral norm parameters
                    if hasattr(module, 'weight_u') and module.weight_u is not None:
                        module.weight_u = module.weight_u.to(self.config.DEVICE, non_blocking=True)
                    if hasattr(module, 'weight_v') and module.weight_v is not None:
                        module.weight_v = module.weight_v.to(self.config.DEVICE, non_blocking=True)
    
    def extract_edge_inconsistencies(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return self.edge_analyzer(gray)
    
    def extract_gradients(self, x):
        if not self.config.USE_GRADIENT_ANALYSIS:
            return None
        
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)
        
        grad_x = F.conv2d(gray.unsqueeze(1), sobel_x, padding=1)
        grad_y = F.conv2d(gray.unsqueeze(1), sobel_y, padding=1)
        
        gradients = torch.cat([grad_x, grad_y], dim=1)
        return self.gradient_analyzer(gradients)
    
    def forward(self, x):
        x = x.to(self.config.DEVICE)
        dct_feats = self.dct_analyzer(x)
        noise_feats = self.noise_analyzer(x)
        edge_feats = self.extract_edge_inconsistencies(x)
        freq_feats = self.freq_analyzer(x)
        
        features_list = [dct_feats, noise_feats, edge_feats, freq_feats]
        
        if self.config.USE_WAVELET_ANALYSIS:
            wavelet_feats = self.wavelet_analyzer(x)
            features_list.append(wavelet_feats)
        
        if self.config.USE_GRADIENT_ANALYSIS:
            grad_feats = self.extract_gradients(x)
            if grad_feats is not None:
                features_list.append(grad_feats)
        
        combined_feats = torch.cat(features_list, dim=1)
        forensics_output = self.forensics_fusion(combined_feats)
        return forensics_output

# Attention module
class UltimateAttentionModule(nn.Module):
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        
        self.forensics_attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=16,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True
        ).to(config.DEVICE)
        
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.BatchNorm1d(in_features // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 4, in_features // 8),
            nn.BatchNorm1d(in_features // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        ).to(config.DEVICE)
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        ).to(config.DEVICE)
        
        self.self_attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        ).to(config.DEVICE)
        
        if config.USE_SPECTRAL_NORM:
            self._apply_spectral_norm()
        
        self.to(config.DEVICE)
    
    def _apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if not hasattr(module, 'weight_u'):
                    module = nn.utils.spectral_norm(module)
                    # Ensure the spectral norm module is on the correct device
                    module.to(self.config.DEVICE)
                    # Force move spectral norm parameters
                    if hasattr(module, 'weight_u') and module.weight_u is not None:
                        module.weight_u = module.weight_u.to(self.config.DEVICE, non_blocking=True)
                    if hasattr(module, 'weight_v') and module.weight_v is not None:
                        module.weight_v = module.weight_v.to(self.config.DEVICE, non_blocking=True)
    
    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() != 2:
            x = x.view(batch_size, -1)
        
        x = x.to(self.config.DEVICE)
        
        logger.debug(f"Input x device: {x.device}", extra={'rank': 0})
        logger.debug(f"forensics_attention weights device: {self.forensics_attention.in_proj_weight.device}", extra={'rank': 0})
        
        x_reshaped = x.unsqueeze(1)
        attn_output, _ = self.forensics_attention(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.squeeze(1)
        
        channel_weights = self.channel_attention(x)
        channel_attended = x * channel_weights
        
        spatial_weights = self.spatial_attention(x)
        spatial_attended = x * spatial_weights
        
        self_weights = self.self_attention(x)
        self_attended = x * self_weights
        
        attended_features = (x + 
                           0.25 * attn_output + 
                           0.3 * channel_attended + 
                           0.3 * spatial_attended + 
                           0.15 * self_attended)
        
        return attended_features

# Main model
class UltimateModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        convnext_features = 768
        
        self.efficientnet = efficientnet_b4(weights=config.PRETRAINED_WEIGHTS)
        efficientnet_features = 1792
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        vit_features = self.vit.num_features
        
        if config.FREEZE_BACKBONES:
            self.freeze_backbones()
        
        forensics_features = 128 if config.USE_FORENSICS_MODULE else 0
        total_features = convnext_features + efficientnet_features + vit_features + forensics_features
        
        if config.USE_FORENSICS_MODULE:
            self.forensics_module = UltimateForensicsModule(config).to(config.DEVICE)
        
        self.attention_module = UltimateAttentionModule(total_features, config).to(config.DEVICE)
        
        if config.USE_STOCHASTIC_DEPTH:
            self.stochastic_depth = StochasticDepth(config.STOCHASTIC_DEPTH_PROB).to(config.DEVICE)
        
        if config.USE_UNCERTAINTY_ESTIMATION:
            self.uncertainty_module = UltimateUncertaintyModule(config.HIDDEN_DIM // 4, config.NUM_CLASSES).to(config.DEVICE)
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
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
            nn.Dropout(config.DROPOUT_RATE // 2)
        ).to(config.DEVICE)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM // 4, config.HIDDEN_DIM // 6),
            nn.BatchNorm1d(config.HIDDEN_DIM // 6),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(config.HIDDEN_DIM // 6, config.HIDDEN_DIM // 8),
            nn.BatchNorm1d(config.HIDDEN_DIM // 8),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 0.3),
            nn.Linear(config.HIDDEN_DIM // 8, config.NUM_CLASSES)
        ).to(config.DEVICE)
        
        if config.USE_SPECTRAL_NORM:
            self._apply_spectral_norm()
        
        self.to(config.DEVICE)
    
    def _apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if not hasattr(module, 'weight_u'):
                    module = nn.utils.spectral_norm(module)
                    # Ensure the spectral norm module is on the correct device
                    module.to(self.config.DEVICE)
                    # Force move spectral norm parameters
                    if hasattr(module, 'weight_u') and module.weight_u is not None:
                        module.weight_u = module.weight_u.to(self.config.DEVICE, non_blocking=True)
                    if hasattr(module, 'weight_v') and module.weight_v is not None:
                        module.weight_v = module.weight_v.to(self.config.DEVICE, non_blocking=True)
    
    def freeze_backbones(self):
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        logger.info("Frozen all backbone parameters", extra={'rank': 0})
    
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
    
    def unfreeze_efficientnet_layers(self, num_layers=None):
        if num_layers is None:
            for param in self.efficientnet.parameters():
                param.requires_grad = True
            logger.info("Unfrozen all EfficientNet layers", extra={'rank': 0})
        else:
            layers = list(self.efficientnet.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True
            logger.info(f"Unfrozen last {num_layers} EfficientNet layers", extra={'rank': 0})
    
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
        logger.info("Unfrozen forensics, attention, and fusion modules", extra={'rank': 0})
    
    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True
        logger.info("Unfrozen classifier", extra={'rank': 0})
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        x = x.to(self.config.DEVICE)
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        
        efficientnet_feats = self.efficientnet.features(x)
        efficientnet_feats = self.efficientnet.avgpool(efficientnet_feats)
        efficientnet_feats = torch.flatten(efficientnet_feats, 1)
        
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]
        
        features_list = [convnext_feats, efficientnet_feats, vit_feats]
        
        if self.config.USE_FORENSICS_MODULE:
            forensics_feats = self.forensics_module(x)
            features_list.append(forensics_feats)
        
        fused_features = torch.cat(features_list, dim=1)
        
        attended_features = self.attention_module(fused_features)
        
        if self.config.USE_STOCHASTIC_DEPTH and hasattr(self, 'stochastic_depth'):
            attended_features = self.stochastic_depth(attended_features)
        
        processed_features = self.fusion(attended_features)
        
        logits = self.classifier(processed_features)
        
        if self.config.USE_UNCERTAINTY_ESTIMATION and hasattr(self, 'uncertainty_module'):
            probs, epistemic_unc, aleatoric_unc, alpha = self.uncertainty_module(processed_features)
            return logits, processed_features, (probs, epistemic_unc, aleatoric_unc, alpha)
        
        return logits, processed_features

# Optimizer and scheduler
def create_ultimate_optimizer_scheduler(model, config, epoch):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if epoch < config.FINE_TUNE_START_EPOCH:
        optimizer = optim.AdamW(
            trainable_params,
            lr=config.ADAMW_LR,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            trainable_params,
            lr=config.SGD_LR,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    
    if config.USE_COSINE_ANNEALING:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS - epoch,
            eta_min=config.ADAMW_LR / 100
        )
    elif config.USE_WARMUP and epoch < config.WARMUP_EPOCHS:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.WARMUP_EPOCHS
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            threshold=0.0001
        )
    
    return optimizer, scheduler

# Progressive unfreezing
def ultimate_progressive_unfreeze(model, epoch, config):
    unfroze = False
    if epoch in config.UNFREEZE_EPOCHS:
        stage = config.UNFREEZE_EPOCHS.index(epoch)
        if stage == 0:
            model.unfreeze_classifier()
            unfroze = True
        elif stage == 1:
            model.unfreeze_forensics_and_attention()
            unfroze = True
        elif stage == 2:
            model.unfreeze_vit_layers(num_layers=10)
            unfroze = True
        elif stage == 3:
            model.unfreeze_efficientnet_layers(num_layers=20)
            unfroze = True
        elif stage == 4:
            model.unfreeze_convnext_layers(num_layers=20)
            unfroze = True
        elif stage == 5:
            model.unfreeze_efficientnet_layers()
            unfroze = True
        elif stage == 6:
            model.unfreeze_vit_layers()
            model.unfreeze_convnext_layers()
            unfroze = True
        logger.info(f"Unfroze layers at epoch {epoch}, stage {stage}, trainable params: {model.get_trainable_params():,}", extra={'rank': 0})
    return unfroze

# Evaluation
def evaluate_ultimate_model(model, loader, criterion, config, device, epoch, local_rank):
    model.eval()
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc=f'Evaluating epoch {epoch}', disable=local_rank not in [-1, 0]):
            data, target = data.to(device), target.to(device)
            
            with autocast(enabled=config.USE_AMP):
                model_output = model(data)
                if len(model_output) == 3:
                    logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
                    loss = criterion(logits, target, features, alpha, epoch)
                else:
                    logits, features = model_output
                    loss = criterion(logits, target, features, None, epoch)
            
            total_loss += loss.item()
            total_batches += 1
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    avg_loss = total_loss / total_batches
    accuracy = (all_preds == all_targets).mean()
    mcc = matthews_corrcoef(all_targets, all_preds)
    
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    macro_f1 = precision_recall_fscore_support(all_targets, all_preds, average='macro')[2]
    weighted_f1 = precision_recall_fscore_support(all_targets, all_preds, average='weighted')[2]
    
    semi_synthetic_idx = config.CLASS_NAMES.index('semi-synthetic')
    semi_synthetic_precision = precision[semi_synthetic_idx]
    semi_synthetic_recall = recall[semi_synthetic_idx]
    semi_synthetic_f1 = f1[semi_synthetic_idx]
    
    cm = confusion_matrix(all_targets, all_preds)
    semi_confusion_rate = (cm[semi_synthetic_idx, :].sum() - cm[semi_synthetic_idx, semi_synthetic_idx]) / cm[semi_synthetic_idx, :].sum()
    semi_to_real_confusion = cm[semi_synthetic_idx, config.CLASS_NAMES.index('real')] / cm[semi_synthetic_idx, :].sum()
    semi_to_synthetic_confusion = cm[semi_synthetic_idx, config.CLASS_NAMES.index('synthetic')] / cm[semi_synthetic_idx, :].sum()
    real_to_semi_confusion = cm[config.CLASS_NAMES.index('real'), semi_synthetic_idx] / cm[config.CLASS_NAMES.index('real'), :].sum()
    synthetic_to_semi_confusion = cm[config.CLASS_NAMES.index('synthetic'), semi_synthetic_idx] / cm[config.CLASS_NAMES.index('synthetic'), :].sum()
    
    class_balance_score = np.std(support / support.sum())
    
    ultimate_score = (0.4 * mcc + 0.4 * semi_synthetic_f1 + 0.2 * accuracy)
    
    metrics = {
        'accuracy': accuracy,
        'mcc': mcc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'semi_synthetic_precision': semi_synthetic_precision,
        'semi_synthetic_recall': semi_synthetic_recall,
        'semi_synthetic_f1': semi_synthetic_f1,
        'semi_confusion_rate': semi_confusion_rate,
        'semi_to_real_confusion': semi_to_real_confusion,
        'semi_to_synthetic_confusion': semi_to_synthetic_confusion,
        'real_to_semi_confusion': real_to_semi_confusion,
        'synthetic_to_semi_confusion': synthetic_to_semi_confusion,
        'class_balance_score': class_balance_score,
        'ultimate_score': ultimate_score,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist()
    }
    
    return avg_loss, metrics, all_preds, all_targets, all_probs

# Checkpointing
def save_ultimate_checkpoint(model, optimizer, scaler, scheduler, epoch, metrics, config, path, is_best=False, local_rank=0):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config.__dict__
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved {'best' if is_best else 'checkpoint'} to {path}", extra={'rank': local_rank})

# Plotting
def plot_ultimate_confusion_matrix(cm, class_names, save_path, metrics):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nUltimate Score: {metrics["ultimate_score"]:.4f}, MCC: {metrics["mcc"]:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_ultimate_metrics(history, save_path):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.plot(history['val_mcc'], label='MCC')
    plt.title('Accuracy and MCC')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['macro_f1'], label='Macro F1')
    plt.plot(history['weighted_f1'], label='Weighted F1')
    plt.plot(history['semi_f1'], label='Semi-synthetic F1')
    plt.title('F1 Scores')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Training worker
def ultimate_train_worker(local_rank, config, master_port):
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if config.DISTRIBUTED:
            success = setup_distributed(local_rank, torch.cuda.device_count(), config.BACKEND, config.MASTER_ADDR, master_port)
            if not success:
                logger.error(f"Failed to setup distributed training for rank {local_rank}", extra={'rank': local_rank})
                return
            config.DEVICE = torch.device(f'cuda:{local_rank}')
            # Ensure tensors are on the correct device
            config.FOCAL_ALPHA = config.FOCAL_ALPHA.to(config.DEVICE, non_blocking=True)
            config.CLASS_WEIGHTS = config.CLASS_WEIGHTS.to(config.DEVICE, non_blocking=True)
        else:
            # For single GPU, ensure we use a specific device
            if torch.cuda.is_available():
                config.DEVICE = torch.device('cuda:0')
            else:
                config.DEVICE = torch.device('cpu')
            # Ensure tensors are on the correct device
            config.FOCAL_ALPHA = config.FOCAL_ALPHA.to(config.DEVICE, non_blocking=True)
            config.CLASS_WEIGHTS = config.CLASS_WEIGHTS.to(config.DEVICE, non_blocking=True)
        
        logger.info(f"ULTIMATE TRAINING SETUP COMPLETE for rank {local_rank}", extra={'rank': local_rank})
        
        torch.manual_seed(42 + local_rank)
        np.random.seed(42 + local_rank)
        random.seed(42 + local_rank)
        
        if local_rank in [-1, 0]:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            logger.info(f"Created checkpoint directory: {config.CHECKPOINT_DIR}", extra={'rank': local_rank})
        
        train_loader, val_loader, test_loader = create_ultimate_data_loaders(config, local_rank)
        
        model = UltimateModel(config)
        
        # Ensure model is on the correct device before DDP wrapping
        ensure_model_device(model, config.DEVICE, local_rank)
        
        if config.DISTRIBUTED:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            # Check device consistency after DDP wrapping
            if not check_model_devices(model.module, config.DEVICE, local_rank):
                logger.warning("Device mismatch detected after DDP wrapping, attempting to fix...", extra={'rank': local_rank})
                ensure_model_device(model.module, config.DEVICE, local_rank)
                check_model_devices(model.module, config.DEVICE, local_rank)
        else:
            if not check_model_devices(model, config.DEVICE, local_rank):
                logger.warning("Device mismatch detected, attempting to fix...", extra={'rank': local_rank})
                ensure_model_device(model, config.DEVICE, local_rank)
                check_model_devices(model, config.DEVICE, local_rank)
        
        torch.cuda.empty_cache()
        cleanup_memory()
        
        criterion = UltimateLoss(config).to(config.DEVICE)
        scaler = GradScaler(enabled=config.USE_AMP)
        mixup_cutmix = UltimateMixupCutmix(config)
        
        best_metrics = {'ultimate_score': -1.0, 'semi_synthetic_f1': -1.0, 'mcc': -1.0}
        epochs_no_improve = 0
        training_history = defaultdict(list)
        current_optimizer = None
        current_scheduler = None
        
        start_epoch = 0
        resume_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_ultimate_checkpoint.pth")
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
                logger.info(f"Resumed ultimate training from epoch {start_epoch}", extra={'rank': local_rank})
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.", extra={'rank': local_rank})
                start_epoch = 0
        
        if local_rank in [-1, 0]:
            initial_trainable = model.module.get_trainable_params() if config.DISTRIBUTED else model.get_trainable_params()
            logger.info(f"Starting ULTIMATE training with {initial_trainable:,} trainable parameters", extra={'rank': local_rank})
            logger.info("🎯 TARGET: >92% MCC, >90% Semi-synthetic F1, >96% Accuracy", extra={'rank': local_rank})
        
        for epoch in range(start_epoch, config.EPOCHS):
            epoch_start_time = time.time()
            model_for_unfreeze = model.module if config.DISTRIBUTED else model
            unfroze_this_epoch = ultimate_progressive_unfreeze(model_for_unfreeze, epoch + 1, config)
            
            if current_optimizer is None or unfroze_this_epoch or (epoch + 1) == config.FINE_TUNE_START_EPOCH:
                current_optimizer, current_scheduler = create_ultimate_optimizer_scheduler(model, config, epoch + 1)
            
            model.train()
            train_loss = 0
            train_batches = 0
            
            if config.DISTRIBUTED:
                train_loader.sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_loader, desc=f'🚀 Ultimate Epoch {epoch+1}/{config.EPOCHS}', disable=local_rank not in [-1, 0])
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                # Skip invalid samples
                if data is None or target is None:
                    continue
                    
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                
                if config.USE_MIXUP or config.USE_CUTMIX:
                    data, target_a, target_b, lam = mixup_cutmix(data, target)
                    use_mixup = True
                else:
                    use_mixup = False
                
                current_optimizer.zero_grad()
                
                with autocast(enabled=config.USE_AMP):
                    model_output = model(data)
                    
                    class_f1_scores = None
                    if config.USE_ADAPTIVE_WEIGHTS and len(model_output) >= 2:
                        logits = model_output[0]
                        pred_class = logits.argmax(dim=1)
                        actual_target = target_a if use_mixup else target
                        
                        class_f1_scores = []
                        for c in range(config.NUM_CLASSES):
                            mask = (actual_target == c)
                            if mask.sum() > 0:
                                tp = ((pred_class == c) & (actual_target == c)).sum().float()
                                fp = ((pred_class == c) & (actual_target != c)).sum().float()
                                fn = ((pred_class != c) & (actual_target == c)).sum().float()
                                precision = tp / (tp + fp + 1e-8)
                                recall = tp / (tp + fn + 1e-8)
                                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                                class_f1_scores.append(f1.item())
                            else:
                                class_f1_scores.append(0.0)
                    
                    if use_mixup:
                        loss = enhanced_mixup_criterion(criterion, model_output, target_a, target_b, lam, epoch + 1, class_f1_scores)
                    else:
                        if len(model_output) == 3:
                            logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
                            loss = criterion(logits, target, features, alpha, epoch + 1, class_f1_scores)
                        else:
                            logits, features = model_output
                            loss = criterion(logits, target, features, None, epoch + 1, class_f1_scores)
                
                scaler.scale(loss).backward()
                
                if config.USE_GRADIENT_CLIPPING:
                    scaler.unscale_(current_optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
                
                scaler.step(current_optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_batches += 1
                
                if local_rank in [-1, 0]:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}', 
                        'Avg': f'{train_loss/train_batches:.4f}',
                        'LR': f'{current_optimizer.param_groups[0]["lr"]:.2e}',
                        'Stage': f'{len([e for e in config.UNFREEZE_EPOCHS if e <= epoch + 1])}'
                    })
                
                if batch_idx % 10 == 0:
                    cleanup_memory()
                    torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
            
            if current_scheduler is not None:
                if isinstance(current_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    pass
                else:
                    current_scheduler.step()
            
            if local_rank in [-1, 0]:
                training_history['train_loss'].append(avg_train_loss)
                training_history['learning_rate'].append(current_optimizer.param_groups[0]['lr'])
                
                val_loss, val_metrics, val_preds, val_targets, val_probs = evaluate_ultimate_model(
                    model, val_loader, criterion, config, config.DEVICE, epoch + 1, local_rank
                )
                
                if current_scheduler is not None and isinstance(current_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    current_scheduler.step(val_metrics['ultimate_score'])
                
                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_metrics['accuracy'])
                training_history['val_mcc'].append(val_metrics['mcc'])
                training_history['macro_f1'].append(val_metrics['macro_f1'])
                training_history['weighted_f1'].append(val_metrics['weighted_f1'])
                training_history['semi_f1'].append(val_metrics['semi_synthetic_f1'])
                training_history['semi_precision'].append(val_metrics['semi_synthetic_precision'])
                training_history['semi_recall'].append(val_metrics['semi_synthetic_recall'])
                training_history['semi_confusion_rate'].append(val_metrics['semi_confusion_rate'])
                training_history['class_balance_score'].append(val_metrics['class_balance_score'])
                training_history['ultimate_score'].append(val_metrics['ultimate_score'])
                training_history['per_class_f1'].append(val_metrics['per_class_f1'])
                
                if 'mean_uncertainty' in val_metrics:
                    training_history['mean_uncertainty'].append(val_metrics['mean_uncertainty'])
                    training_history['uncertainty_std'].append(val_metrics['uncertainty_std'])
                
                epoch_time = time.time() - epoch_start_time
                
                logger.info(f"🚀 ULTIMATE EPOCH {epoch+1}/{config.EPOCHS} completed in {epoch_time:.2f}s", extra={'rank': local_rank})
                logger.info(f"📊 Train Loss: {avg_train_loss:.6f}, LR: {current_optimizer.param_groups[0]['lr']:.2e}", extra={'rank': local_rank})
                logger.info(f"📈 Val Loss: {val_loss:.6f}, Accuracy: {val_metrics['accuracy']:.6f}, MCC: {val_metrics['mcc']:.6f}", extra={'rank': local_rank})
                logger.info(f"🎯 ULTIMATE SCORE: {val_metrics['ultimate_score']:.6f}", extra={'rank': local_rank})
                logger.info(f"🔍 Semi-synthetic - P: {val_metrics['semi_synthetic_precision']:.6f}, "
                           f"R: {val_metrics['semi_synthetic_recall']:.6f}, F1: {val_metrics['semi_synthetic_f1']:.6f}", extra={'rank': local_rank})
                
                if val_metrics['mcc'] > 0.92:
                    logger.info("🏆 EXCELLENCE: MCC > 92%!", extra={'rank': local_rank})
                if val_metrics['semi_synthetic_f1'] > 0.90:
                    logger.info("🏆 EXCELLENCE: Semi-synthetic F1 > 90%!", extra={'rank': local_rank})
                if val_metrics['accuracy'] > 0.96:
                    logger.info("🏆 EXCELLENCE: Accuracy > 96%!", extra={'rank': local_rank})
                
                current_ultimate_score = val_metrics['ultimate_score']
                best_ultimate_score = best_metrics['ultimate_score']
                
                if current_ultimate_score > best_ultimate_score:
                    best_metrics = val_metrics.copy()
                    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_ultimate_model.pth")
                    save_ultimate_checkpoint(model, current_optimizer, scaler, current_scheduler, 
                                           epoch, val_metrics, config, best_model_path, is_best=True, local_rank=local_rank)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                latest_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': current_optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': current_scheduler.state_dict() if current_scheduler else None,
                    'best_metrics': best_metrics,
                    'training_history': dict(training_history),
                    'epochs_no_improve': epochs_no_improve,
                    'config': config.__dict__
                }
                torch.save(latest_checkpoint, os.path.join(config.CHECKPOINT_DIR, "latest_ultimate_checkpoint.pth"))
                
                if (epoch + 1) % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"ultimate_checkpoint_epoch_{epoch+1}.pth")
                    save_ultimate_checkpoint(model, current_optimizer, scaler, current_scheduler,
                                           epoch, val_metrics, config, checkpoint_path, local_rank=local_rank)
                
                plot_ultimate_metrics(training_history, 
                                    os.path.join(config.CHECKPOINT_DIR, f'ultimate_metrics_epoch_{epoch+1}.png'))
                
                cm = confusion_matrix(val_targets, val_preds)
                plot_ultimate_confusion_matrix(cm, config.CLASS_NAMES, 
                                             os.path.join(config.CHECKPOINT_DIR, f'ultimate_cm_epoch_{epoch+1}.png'),
                                             val_metrics)
                
                if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"🛑 Early stopping triggered after {epoch+1} epochs (no improvement for {epochs_no_improve} epochs)", 
                               extra={'rank': local_rank})
                    break
                
                if (val_metrics['mcc'] > 0.92 and 
                    val_metrics['semi_synthetic_f1'] > 0.90 and 
                    val_metrics['accuracy'] > 0.96):
                    logger.info("🏆🏆🏆 ULTIMATE EXCELLENCE ACHIEVED! Continuing for stability...", extra={'rank': local_rank})
            
            if config.DISTRIBUTED:
                dist.barrier()
            cleanup_memory()
        
        if local_rank in [-1, 0]:
            logger.info("🏁 ULTIMATE TRAINING COMPLETED - FINAL EVALUATION", extra={'rank': local_rank})
            
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_ultimate_model.pth")
            if os.path.exists(best_model_path):
                logger.info("Loading ultimate best model for final evaluation...", extra={'rank': local_rank})
                checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
                if config.DISTRIBUTED:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                test_loss, test_metrics, test_preds, test_targets, test_probs = evaluate_ultimate_model(
                    model, test_loader, criterion, config, config.DEVICE, 999, local_rank
                )
                
                logger.info("=" * 100, extra={'rank': local_rank})
                logger.info("🏆 ULTIMATE FINAL TEST RESULTS - DEEPFAKE DETECTION EXCELLENCE 🏆", extra={'rank': local_rank})
                logger.info("=" * 100, extra={'rank': local_rank})
                logger.info(f"🎯 ULTIMATE SCORE: {test_metrics['ultimate_score']:.8f}", extra={'rank': local_rank})
                logger.info(f"📊 Test Loss: {test_loss:.8f}", extra={'rank': local_rank})
                logger.info(f"📈 Test Accuracy: {test_metrics['accuracy']:.8f}", extra={'rank': local_rank})
                logger.info(f"🔢 Matthews Correlation Coefficient: {test_metrics['mcc']:.8f}", extra={'rank': local_rank})
                logger.info(f"📊 Macro F1-Score: {test_metrics['macro_f1']:.8f}", extra={'rank': local_rank})
                logger.info(f"⚖️ Weighted F1-Score: {test_metrics['weighted_f1']:.8f}", extra={'rank': local_rank})
                logger.info(f"🎯 Class Balance Score: {test_metrics['class_balance_score']:.8f}", extra={'rank': local_rank})
                logger.info("-" * 80, extra={'rank': local_rank})
                logger.info("📋 PER-CLASS DETAILED ANALYSIS:", extra={'rank': local_rank})
                for i, class_name in enumerate(config.CLASS_NAMES):
                    if i < len(test_metrics['per_class_precision']):
                        logger.info(f"{class_name:>15} - P: {test_metrics['per_class_precision'][i]:.8f}, "
                                  f"R: {test_metrics['per_class_recall'][i]:.8f}, "
                                  f"F1: {test_metrics['per_class_f1'][i]:.8f}, "
                                  f"Support: {test_metrics['per_class_support'][i]}", extra={'rank': local_rank})
                
                logger.info("-" * 80, extra={'rank': local_rank})
                logger.info("🔍 SEMI-SYNTHETIC CLASS ULTIMATE ANALYSIS:", extra={'rank': local_rank})
                logger.info(f"🎯 Semi-synthetic Precision: {test_metrics['semi_synthetic_precision']:.8f}", extra={'rank': local_rank})
                logger.info(f"🎯 Semi-synthetic Recall: {test_metrics['semi_synthetic_recall']:.8f}", extra={'rank': local_rank})
                logger.info(f"🏆 Semi-synthetic F1-Score: {test_metrics['semi_synthetic_f1']:.8f}", extra={'rank': local_rank})
                logger.info(f"⚠️ Semi-synthetic Confusion Rate: {test_metrics['semi_confusion_rate']:.8f}", extra={'rank': local_rank})
                logger.info(f"📊 Semi -> Real Confusion: {test_metrics['semi_to_real_confusion']:.8f}", extra={'rank': local_rank})
                logger.info(f"📊 Semi -> Synthetic Confusion: {test_metrics['semi_to_synthetic_confusion']:.8f}", extra={'rank': local_rank})
                logger.info(f"📊 Real -> Semi Confusion: {test_metrics['real_to_semi_confusion']:.8f}", extra={'rank': local_rank})
                logger.info(f"📊 Synthetic -> Semi Confusion: {test_metrics['synthetic_to_semi_confusion']:.8f}", extra={'rank': local_rank})
                
                if 'mean_uncertainty' in test_metrics:
                    logger.info("-" * 80, extra={'rank': local_rank})
                    logger.info("🤔 UNCERTAINTY ANALYSIS:", extra={'rank': local_rank})
                    logger.info(f"📊 Mean Uncertainty: {test_metrics['mean_uncertainty']:.8f}", extra={'rank': local_rank})
                    logger.info(f"📊 Uncertainty Std: {test_metrics['uncertainty_std']:.8f}", extra={'rank': local_rank})
                
                logger.info("-" * 80, extra={'rank': local_rank})
                logger.info("🏆 ULTIMATE PERFORMANCE ASSESSMENT:", extra={'rank': local_rank})
                
                excellence_count = 0
                if test_metrics['mcc'] > 0.92:
                    logger.info("🏅 EXCELLENCE BADGE: MCC > 92% ✅", extra={'rank': local_rank})
                    excellence_count += 1
                else:
                    logger.info(f"⚠️  MCC Target: {test_metrics['mcc']:.4f} (Target: >0.92)", extra={'rank': local_rank})
                
                if test_metrics['semi_synthetic_f1'] > 0.90:
                    logger.info("🏅 EXCELLENCE BADGE: Semi-synthetic F1 > 90% ✅", extra={'rank': local_rank})
                    excellence_count += 1
                else:
                    logger.info(f"⚠️  Semi-synthetic F1: {test_metrics['semi_synthetic_f1']:.4f} (Target: >0.90)", extra={'rank': local_rank})
                
                if test_metrics['accuracy'] > 0.96:
                    logger.info("🏅 EXCELLENCE BADGE: Accuracy > 96% ✅", extra={'rank': local_rank})
                    excellence_count += 1
                else:
                    logger.info(f"⚠️  Accuracy: {test_metrics['accuracy']:.4f} (Target: >0.96)", extra={'rank': local_rank})
                
                if test_metrics['ultimate_score'] > 0.92:
                    logger.info("🏅 EXCELLENCE BADGE: Ultimate Score > 92% ✅", extra={'rank': local_rank})
                    excellence_count += 1
                else:
                    logger.info(f"⚠️  Ultimate Score: {test_metrics['ultimate_score']:.4f} (Target: >0.92)", extra={'rank': local_rank})
                
                logger.info(f"🏆 TOTAL EXCELLENCE BADGES: {excellence_count}/4", extra={'rank': local_rank})
                
                if excellence_count == 4:
                    logger.info("🎉🎉🎉 ULTIMATE DEEPFAKE DETECTION EXCELLENCE ACHIEVED! 🎉🎉🎉", extra={'rank': local_rank})
                elif excellence_count >= 3:
                    logger.info("🎯 SUPERIOR PERFORMANCE ACHIEVED! Nearly perfect!", extra={'rank': local_rank})
                elif excellence_count >= 2:
                    logger.info("✅ EXCELLENT PERFORMANCE! Good progress toward ultimate goals!", extra={'rank': local_rank})
                else:
                    logger.info("📈 SOLID PERFORMANCE! Room for improvement in key metrics.", extra={'rank': local_rank})
                
                logger.info("=" * 100, extra={'rank': local_rank})
                
                final_results = {
                    'test_metrics': test_metrics,
                    'test_loss': test_loss,
                    'ultimate_score': test_metrics['ultimate_score'],
                    'excellence_badges': excellence_count,
                    'config': config.__dict__,
                    'model_architecture': 'UltimateModel',
                    'training_completed': True
                }
                
                results_path = os.path.join(config.CHECKPOINT_DIR, "ultimate_final_results.json")
                with open(results_path, 'w') as f:
                    json.dump(final_results, f, indent=2, default=str)
                
                cm = confusion_matrix(test_targets, test_preds)
                plot_ultimate_confusion_matrix(
                    cm, config.CLASS_NAMES, 
                    os.path.join(config.CHECKPOINT_DIR, 'ultimate_final_confusion_matrix.png'),
                    test_metrics
                )
                
                plot_ultimate_metrics(training_history, 
                                    os.path.join(config.CHECKPOINT_DIR, 'ultimate_final_training_metrics.png'))
                
                logger.info(f"🎯 Ultimate results saved to {results_path}", extra={'rank': local_rank})
                logger.info("🚀🚀🚀 ULTIMATE DEEPFAKE DETECTION TRAINING PIPELINE COMPLETED! 🚀🚀🚀", extra={'rank': local_rank})
    
    except Exception as e:
        logger.error(f"Ultimate training error on rank {local_rank}: {e}", extra={'rank': local_rank})
        raise e
    finally:
        cleanup_distributed()
        cleanup_memory()

# Main function
def main():
    set_multiprocessing_start_method()
    
    parser = argparse.ArgumentParser(description='Ultimate Deepfake Detection Training')
    parser.add_argument('--train_path', type=str, default='datasets/train', 
                       help='Path to training dataset')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=120, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=8e-5, 
                       help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='ultimate_deepfake_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--distributed', action='store_true', 
                       help='Enable distributed training')
    parser.add_argument('--use_cross_validation', action='store_true',
                       help='Enable cross-validation for ultimate performance')
    args = parser.parse_args()
    
    config = UltimateDeepfakeConfig()
    config.TRAIN_PATH = args.train_path
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.ADAMW_LR = args.lr
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.DISTRIBUTED = args.distributed and torch.cuda.device_count() > 1
    config.USE_CROSS_VALIDATION = args.use_cross_validation
    
    config.validate()
    
    logger.info("🚀 Starting ULTIMATE Deepfake Detection Training Pipeline", extra={'rank': 0})
    logger.info("🎯 ULTIMATE TARGETS:", extra={'rank': 0})
    logger.info("   • MCC > 92%", extra={'rank': 0})
    logger.info("   • Semi-synthetic F1 > 90%", extra={'rank': 0})
    logger.info("   • Overall Accuracy > 96%", extra={'rank': 0})
    logger.info("   • Ultimate Score > 92%", extra={'rank': 0})
    
    if config.DISTRIBUTED:
        master_port = str(find_free_port())
        config.MASTER_PORT = master_port
        logger.info(f"Using master port {master_port} for distributed training", extra={'rank': 0})
        
        world_size = torch.cuda.device_count()
        logger.info(f"Launching ultimate training on {world_size} GPUs", extra={'rank': 0})
        mp.spawn(
            ultimate_train_worker,
            args=(config, master_port),
            nprocs=world_size,
            join=True
        )
    else:
        logger.info("Launching ultimate single-device training", extra={'rank': 0})
        ultimate_train_worker(-1, config, config.MASTER_PORT)
    
    logger.info("🏁 Ultimate training pipeline completed", extra={'rank': 0})

if __name__ == "__main__":
    main()