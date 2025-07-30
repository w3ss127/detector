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
from sklearn.metrics import confusion_matrix, classification_report
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
import math

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedConfig:
    """Configuration for deepfake detection training."""
    def __init__(self):
        self.MODEL_TYPE = "enhanced_convnext_vit"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 1024
        self.DROPOUT_RATE = 0.3
        self.FREEZE_BACKBONES = True
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
        self.ADAMW_LR = 1e-3
        self.SGD_LR = 1e-4
        self.SGD_MOMENTUM = 0.9
        self.SGD_WEIGHT_DECAY = 1e-4
        self.WEIGHT_DECAY = 1e-2
        self.FOCAL_ALPHA = [1.0, 3.0, 2.0]
        self.FOCAL_GAMMA = 2.0
        self.LABEL_SMOOTHING = 0.1
        self.CHECKPOINT_DIR = "frozenimprove_checkpoints"
        self.CHECKPOINT_EVERY_N_EPOCHS = 5
        self.USE_MIXUP = True
        self.USE_CUTMIX = True
        self.MIXUP_ALPHA = 0.2
        self.CUTMIX_ALPHA = 1.0
        self.MIXUP_PROB = 0.5
        self.CUTMIX_PROB = 0.5
        self.SWITCH_PROB = 0.5
        self.TRAINING_STAGES = {
            1: {'epochs': (1, 5), 'freeze_backbone': 'full', 'optimizer': 'adamw'},
            2: {'epochs': (6, 15), 'freeze_backbone': 'classifiers+vit', 'optimizer': 'adamw'},
            3: {'epochs': (16, 25), 'freeze_backbone': 'convnext', 'optimizer': 'adamw'},
            4: {'epochs': (26, 40), 'freeze_backbone': 'none', 'optimizer': 'adamw'},
            5: {'epochs': (41, 50), 'freeze_backbone': 'none', 'optimizer': 'sgd'}
        }

    def get_current_stage(self, epoch):
        """Get current training stage based on epoch number."""
        for stage, config in self.TRAINING_STAGES.items():
            start_epoch, end_epoch = config['epochs']
            if start_epoch <= epoch <= end_epoch:
                return stage, config
        return None, None

    def validate(self):
        """Validate configuration parameters."""
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0, "Batch size must be positive"
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0, "Epochs must be positive"
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES, "Class names must match NUM_CLASSES"
        assert self.CONVNEXT_BACKBONE in ["convnext_tiny", "convnext_small"], "Unsupported backbone"
        assert isinstance(self.CHECKPOINT_EVERY_N_EPOCHS, int) and self.CHECKPOINT_EVERY_N_EPOCHS > 0, "Checkpoint frequency must be positive"
        assert 0 <= self.MIXUP_PROB <= 1, "MIXUP_PROB must be between 0 and 1"
        assert 0 <= self.CUTMIX_PROB <= 1, "CUTMIX_PROB must be between 0 and 1"
        assert 0 <= self.SWITCH_PROB <= 1, "SWITCH_PROB must be between 0 and 1"
        assert self.MIXUP_ALPHA > 0, "MIXUP_ALPHA must be positive"
        assert self.CUTMIX_ALPHA > 0, "CUTMIX_ALPHA must be positive"
        assert self.ADAMW_LR > 0, "ADAMW_LR must be positive"
        assert self.SGD_LR > 0, "SGD_LR must be positive"
        assert 0 <= self.SGD_MOMENTUM < 1, "SGD_MOMENTUM must be between 0 and 1"

class MixUpCutMixCollator:
    """Collator for applying MixUp and CutMix augmentations."""
    def __init__(self, config):
        self.config = config
        self.mixup_alpha = config.MIXUP_ALPHA
        self.cutmix_alpha = config.CUTMIX_ALPHA
        self.mixup_prob = config.MIXUP_PROB
        self.cutmix_prob = config.CUTMIX_PROB
        self.switch_prob = config.SWITCH_PROB
        self.num_classes = config.NUM_CLASSES

    def __call__(self, batch):
        """Apply MixUp or CutMix to a batch."""
        if not batch:
            logger.warning("Empty batch received in collator")
            return None, None
        try:
            images, targets = zip(*batch)
            images = torch.stack(images)
            targets = torch.tensor(targets, dtype=torch.long)
            targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
            
            use_mixup = self.config.USE_MIXUP and np.random.rand() < self.mixup_prob
            use_cutmix = self.config.USE_CUTMIX and np.random.rand() < self.cutmix_prob
            
            if use_mixup and use_cutmix:
                if np.random.rand() < self.switch_prob:
                    return self._cutmix(images, targets_onehot)
                return self._mixup(images, targets_onehot)
            elif use_mixup:
                return self._mixup(images, targets_onehot)
            elif use_cutmix:
                return self._cutmix(images, targets_onehot)
            return images, targets_onehot
        except Exception as e:
            logger.error(f"Error in MixUpCutMixCollator: {e}")
            return images, targets_onehot

    def _mixup(self, images, targets):
        """Apply MixUp augmentation."""
        batch_size = images.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        return mixed_images, mixed_targets

    def _cutmix(self, images, targets):
        """Apply CutMix augmentation."""
        batch_size = images.size(0)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if self.cutmix_alpha > 0 else 1
        index = torch.randperm(batch_size)
        _, _, h, w = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        return mixed_images, mixed_targets

class AdvancedLoss(nn.Module):
    """Loss function with focal loss and support for soft targets."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.focal_alpha = torch.tensor(config.FOCAL_ALPHA, device=config.DEVICE)
        self.focal_gamma = config.FOCAL_GAMMA
        self.label_smoothing = config.LABEL_SMOOTHING

    def focal_loss(self, inputs, targets):
        """Focal loss with support for soft targets."""
        if targets.dim() == 1:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
            pt = torch.exp(-ce_loss)
            alpha_t = self.focal_alpha[targets]
            return (alpha_t * (1 - pt) ** self.focal_gamma * ce_loss).mean()
        else:
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(targets * log_probs).sum(dim=1)
            probs = F.softmax(inputs, dim=1)
            pt = (targets * probs).sum(dim=1)
            alpha_t = (targets * self.focal_alpha.unsqueeze(0)).sum(dim=1)
            return (alpha_t * (1 - pt) ** self.focal_gamma * ce_loss).mean()

    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)

class SpectralNorm(nn.Module):
    """Spectral normalization for regularization."""
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        assert isinstance(module, (nn.Conv2d, nn.Linear)), "SpectralNorm only supports Conv2d and Linear layers"
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        w_shape = w.shape
        height = w_shape[0]
        width = w_shape[1] * w_shape[2] * w_shape[3] if len(w_shape) == 4 else w_shape[1]
        w_reshaped = w.view(height, -1)

        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.matmul(w_reshaped.t(), u.data), dim=0)
            u.data = F.normalize(torch.matmul(w_reshaped, v.data), dim=0)

        sigma = torch.dot(u.data, torch.matmul(w_reshaped, v.data)).clamp(min=1e-10)
        w_normalized = w / sigma
        if len(w_shape) == 4:
            w_normalized = w_normalized.view(w_shape)
        setattr(self.module, self.name, w_normalized)

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        w_shape = w.shape
        height = w_shape[0]
        width = w_shape[1] * w_shape[2] * w_shape[3] if len(w_shape) == 4 else w_shape[1]
        u = nn.Parameter(torch.randn(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(torch.randn(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)
        w_bar = nn.Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class EnhancedAttentionModule(nn.Module):
    """Attention module with channel and spatial attention."""
    def __init__(self, channels, reduction=16, config=None):
        super().__init__()
        self.config = config or EnhancedConfig()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.utils.spectral_norm(nn.Conv2d(channels, channels // reduction, 1, bias=False)) if self.config.USE_SPECTRAL_NORM else nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.ATTENTION_DROPOUT),
            nn.utils.spectral_norm(nn.Conv2d(channels // reduction, channels, 1, bias=False)) if self.config.USE_SPECTRAL_NORM else nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(2, 16, kernel_size=7, padding=3, bias=False)) if self.config.USE_SPECTRAL_NORM else nn.Conv2d(2, 16, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(16, 1, kernel_size=7, padding=3, bias=False)) if self.config.USE_SPECTRAL_NORM else nn.Conv2d(16, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), x.size(1), 1, 1)
        ca_weight = self.channel_attention(x)
        x_ca = x * ca_weight
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_weight = self.spatial_attention(spatial_input)
        x_final = x_ca * sa_weight
        return x_final.view(x_final.size(0), x_final.size(1))

class EnhancedConvNextViTModel(nn.Module):
    """Hybrid ConvNeXt and ViT model with attention."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        else:
            raise ValueError(f"Unsupported ConvNeXt backbone: {config.CONVNEXT_BACKBONE}")
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=config.PRETRAINED_WEIGHTS is not None, num_classes=0)
        
        for module in self.convnext.classifier:
            if isinstance(module, nn.Linear):
                convnext_features = module.in_features
                break
        else:
            raise AttributeError("No Linear layer found in ConvNeXt classifier")
        
        vit_features = self.vit.num_features
        self.backbone_params = {'convnext': list(self.convnext.parameters()), 'vit': list(self.vit.parameters())}
        
        if config.FREEZE_BACKBONES:
            self.set_freeze_strategy('full')
        
        self.attention_module = EnhancedAttentionModule(channels=convnext_features + vit_features, config=config)
        self.fusion = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(convnext_features + vit_features, config.HIDDEN_DIM)) if config.USE_SPECTRAL_NORM else nn.Linear(convnext_features + vit_features, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.utils.spectral_norm(nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)) if config.USE_SPECTRAL_NORM else nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        )

    def set_freeze_strategy(self, strategy):
        """Set backbone freezing strategy based on stage configuration."""
        for param in self.backbone_params['convnext'] + self.backbone_params['vit']:
            param.requires_grad = False
        
        if strategy == 'none':
            for param in self.backbone_params['convnext'] + self.backbone_params['vit']:
                param.requires_grad = True
        elif strategy == 'classifiers+vit':
            for param in self.convnext.classifier.parameters():
                param.requires_grad = True
        elif strategy == 'convnext':
            for param in self.backbone_params['convnext']:
                param.requires_grad = True
        logger.info(f"Set backbone freeze strategy: {strategy}")

    def forward(self, x):
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]
        fused_features = torch.cat([convnext_feats, vit_feats], dim=1)
        fused_features = self.attention_module(fused_features)
        logits = self.fusion(fused_features)
        return logits

class EnhancedCustomDatasetPT(Dataset):
    """Dataset for loading .pt files."""
    def __init__(self, root_dir, config, transform=None):
        self.root_dir = Path(root_dir)
        self.config = config
        self.transform = transform
        self.class_names = config.CLASS_NAMES
        self.images = []
        self.labels = []
        self.file_mapping = []
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from .pt files and validate tensor shapes."""
        logger.info("Loading dataset from .pt files...")
        class_counts = {class_name: 0 for class_name in self.class_names}
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist")
                continue
            pt_files = list(class_dir.glob('*.pt'))
            for pt_file in pt_files:
                try:
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    if isinstance(tensor_data, dict):
                        tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                    if tensor_data.dim() < 4 or tensor_data.shape[1] != 3:
                        logger.warning(f"Invalid tensor shape in {pt_file}: {tensor_data.shape}")
                        continue
                    for i in range(tensor_data.shape[0]):
                        self.labels.append(class_idx)
                        self.file_mapping.append((str(pt_file), i))
                        self.images.append(tensor_data[i])
                        class_counts[class_name] += 1
                except Exception as e:
                    logger.error(f"Error loading {pt_file}: {e}")
        logger.info(f"Loaded {len(self.images)} images")
        for class_name, count in class_counts.items():
            if count == 0:
                logger.warning(f"No images loaded for class {class_name}")
        if not self.images:
            raise ValueError("No valid images loaded from dataset")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_tensor = self.images[idx]
            label = self.labels[idx]
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            image_tensor = torch.clamp(image_tensor, 0, 1)
            if self.transform:
                image_np = image_tensor.permute(1, 2, 0).numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            return image_tensor, label
        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 0

class EnhancedDataAugmentation:
    """Data augmentation preserving forensic artifacts."""
    def __init__(self, config, is_training=True):
        self.config = config
        self.is_training = is_training

    def get_train_transforms(self):
        """Get augmentation transforms for training or validation."""
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

def create_enhanced_data_loaders(config, local_rank=-1):
    """Create data loaders with MixUp/CutMix support."""
    dataset = EnhancedCustomDatasetPT(
        root_dir=config.TRAIN_PATH,
        transform=EnhancedDataAugmentation(config, is_training=True).get_train_transforms(),
        config=config
    )
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = EnhancedDataAugmentation(config, is_training=False).get_train_transforms()
    test_dataset.dataset.transform = EnhancedDataAugmentation(config, is_training=False).get_train_transforms()
    
    mixup_cutmix_collator = MixUpCutMixCollator(config)
    
    train_sampler = DistributedSampler(train_dataset, rank=local_rank, shuffle=True) if config.DISTRIBUTED else None
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=config.NUM_WORKERS, pin_memory=True,
        collate_fn=mixup_cutmix_collator
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

def find_free_port(start_port=12355, max_attempts=100):
    """Find an available port."""
    port = start_port
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError as e:
            if e.errno == 98:
                port += 1
                continue
            raise
    raise RuntimeError(f"No free port found after {max_attempts} attempts")

def setup_distributed(local_rank, world_size, backend='nccl', master_addr='localhost', master_port='12355'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    logger.info(f"Distributed process group initialized for rank {local_rank} on port {master_port}")

def cleanup_distributed():
    """Clean up distributed process group."""
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")
    except Exception as e:
        logger.warning(f"Error during distributed cleanup: {e}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def create_optimizer(model, config, optimizer_type='adamw'):
    """Create optimizer based on type."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No trainable parameters found for optimizer")
    if optimizer_type == 'adamw':
        return optim.AdamW(trainable_params, lr=config.ADAMW_LR, weight_decay=config.WEIGHT_DECAY)
    elif optimizer_type == 'sgd':
        return optim.SGD(trainable_params, lr=config.SGD_LR, momentum=config.SGD_MOMENTUM, weight_decay=config.SGD_WEIGHT_DECAY)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def train_model(config, resume_from=None):
    """Main training function with staged training."""
    local_rank = -1
    world_size = 1
    if config.DISTRIBUTED:
        try:
            local_rank = int(os.environ.get('LOCAL_RANK', -1))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            if local_rank == -1 or world_size <= 1:
                logger.warning("Invalid distributed training configuration. Falling back to single-GPU.")
                config.DISTRIBUTED = False
            else:
                setup_distributed(local_rank, world_size, config.BACKEND, config.MASTER_ADDR, find_free_port(int(config.MASTER_PORT)))
                config.DEVICE = torch.device(f'cuda:{local_rank}')
        except Exception as e:
            logger.error(f"Distributed training failed: {e}. Falling back to single-GPU.")
            config.DISTRIBUTED = False

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model_staged.pth")
    
    train_loader, val_loader, test_loader = create_enhanced_data_loaders(config, local_rank)
    model = EnhancedConvNextViTModel(config).to(config.DEVICE)
    if config.DISTRIBUTED:
        model = DDP(model, device_ids=[local_rank])
    
    criterion = AdvancedLoss(config).to(config.DEVICE)
    scaler = GradScaler(enabled=config.USE_AMP)
    
    best_val_acc = 0.0
    current_stage = 1
    current_optimizer_type = 'adamw'
    optimizer = create_optimizer(model, config, current_optimizer_type)
    start_epoch = 1
    
    if resume_from:
        try:
            start_epoch, current_stage, best_val_acc, freeze_strategy, current_optimizer_type = load_staged_checkpoint(model, optimizer, resume_from, config)
            model.set_freeze_strategy(freeze_strategy)
            config.FREEZE_BACKBONES = (freeze_strategy != 'none')
            optimizer = create_optimizer(model, config, current_optimizer_type)
            scaler = GradScaler(enabled=config.USE_AMP)
            logger.info(f"Resumed training from epoch {start_epoch}, stage {current_stage}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {resume_from}: {e}. Starting from scratch.")
            start_epoch = 1
            current_stage = 1
            best_val_acc = 0.0
    
    logger.info(f"Starting training with {config.EPOCHS} epochs in 5 stages")
    for stage, stage_config in config.TRAINING_STAGES.items():
        logger.info(f"Stage {stage} (epochs {stage_config['epochs'][0]}-{stage_config['epochs'][1]}): "
                    f"Freeze={stage_config['freeze_backbone']}, Optimizer={stage_config['optimizer'].upper()}")
    
    for epoch in range(start_epoch, config.EPOCHS + 1):
        stage_num, stage_config = config.get_current_stage(epoch)
        if stage_num is None:
            logger.warning(f"No stage defined for epoch {epoch}. Skipping.")
            continue
        
        if stage_num != current_stage:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRANSITIONING TO STAGE {stage_num} (Epochs {stage_config['epochs'][0]}-{stage_config['epochs'][1]})")
            logger.info(f"{'='*60}")
            current_stage = stage_num
            model_to_modify = model.module if isinstance(model, DDP) else model
            model_to_modify.set_freeze_strategy(stage_config['freeze_backbone'])
            config.FREEZE_BACKBONES = (stage_config['freeze_backbone'] != 'none')
            
            if stage_config['optimizer'] != current_optimizer_type:
                current_optimizer_type = stage_config['optimizer']
                logger.info(f"Switching to {current_optimizer_type.upper()} optimizer")
                optimizer = create_optimizer(model, config, current_optimizer_type)
                scaler = GradScaler(enabled=config.USE_AMP)
            
            logger.info(f"Stage {stage_num} configuration:")
            logger.info(f"  - Backbone frozen: {stage_config['freeze_backbone']}")
            logger.info(f"  - Optimizer: {current_optimizer_type.upper()}")
            logger.info(f"  - Learning rate: {config.ADAMW_LR if current_optimizer_type == 'adamw' else config.SGD_LR}")
            if current_optimizer_type == 'sgd':
                logger.info(f"  - Momentum: {config.SGD_MOMENTUM}")
        
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Stage {stage_num}, Epoch {epoch}/{config.EPOCHS}')
        for batch_idx, (data, target) in enumerate(pbar):
            if data is None or target is None:
                logger.warning(f"Skipping invalid batch {batch_idx}")
                continue
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()
            
            with autocast(enabled=config.USE_AMP):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            target_hard = target.argmax(dim=1) if target.dim() > 1 else target
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target_hard).sum().item()
            train_total += target.size(0)
            
            train_acc = train_correct / train_total
            augmentation_info = "None"
            if config.USE_MIXUP and config.USE_CUTMIX:
                augmentation_info = "Mix+Cut"
            elif config.USE_MIXUP:
                augmentation_info = "MixUp"
            elif config.USE_CUTMIX:
                augmentation_info = "CutMix"
            
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{train_acc:.4f}',
                'Opt': current_optimizer_type.upper(),
                'Aug': augmentation_info,
                'Freeze': stage_config['freeze_backbone']
            })
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for data, target in val_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            logger.info(f"Stage {stage_num}, Epoch {epoch}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if config.DISTRIBUTED and local_rank != 0:
                continue
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'stage': stage_num,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'backbone_frozen': stage_config['freeze_backbone'],
                    'optimizer_type': current_optimizer_type
                }, best_model_path)
                logger.info(f"Saved best model with Val Acc: {best_val_acc:.4f} at {best_model_path}")
            
            if epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                periodic_path = os.path.join(config.CHECKPOINT_DIR, f"model_stage{stage_num}_epoch{epoch}_{timestamp}.pth")
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'stage': stage_num,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'backbone_frozen': stage_config['freeze_backbone'],
                    'optimizer_type': current_optimizer_type
                }, periodic_path)
                logger.info(f"Saved periodic checkpoint at {periodic_path}")
    
    final_model_path = os.path.join(config.CHECKPOINT_DIR, f"final_model_staged_{time.strftime('%Y%m%d_%H%M%S')}.pth")
    if not config.DISTRIBUTED or local_rank == 0:
        model_to_save = model.module if isinstance(model, DDP) else model
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': config.EPOCHS,
            'stage': 5,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'backbone_frozen': stage_config['freeze_backbone'],
            'optimizer_type': current_optimizer_type
        }, final_model_path)
        logger.info(f"Saved final model at {final_model_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Final validation accuracy: {val_acc:.4f}")
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"Final model saved at: {final_model_path}")
    logger.info(f"{'='*60}")
    
    # Clean up
    del model
    del optimizer
    cleanup_distributed()
    return {}, None

def load_staged_checkpoint(model, optimizer, checkpoint_path, config):
    """Load a staged training checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'stage', 'val_acc']
        if not all(key in checkpoint for key in required_keys):
            raise KeyError("Checkpoint missing required keys")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        stage = checkpoint['stage']
        val_acc = checkpoint['val_acc']
        train_acc = checkpoint.get('train_acc', 0.0)
        backbone_frozen = checkpoint.get('backbone_frozen', 'full')
        optimizer_type = checkpoint.get('optimizer_type', 'adamw')
        logger.info(f"Loaded checkpoint from epoch {epoch}, stage {stage}")
        logger.info(f"  - Validation accuracy: {val_acc:.4f}")
        logger.info(f"  - Training accuracy: {train_acc:.4f}")
        logger.info(f"  - Backbone frozen: {backbone_frozen}")
        logger.info(f"  - Optimizer type: {optimizer_type}")
        return epoch, stage, val_acc, backbone_frozen, optimizer_type
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        raise

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Staged Deepfake Detection Training with MixUp/CutMix')
    parser.add_argument('--data-path', type=str, default='datasets/train')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--adamw-lr', type=float, default=1e-3)
    parser.add_argument('--sgd-lr', type=float, default=1e-4)
    parser.add_argument('--sgd-momentum', type=float, default=0.9)
    parser.add_argument('--no-distributed', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default='frozenimprove_checkpoints')
    parser.add_argument('--checkpoint-every-n-epochs', type=int, default=5)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--no-mixup', action='store_true')
    parser.add_argument('--no-cutmix', action='store_true')
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--cutmix-alpha', type=float, default=1.0)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    parser.add_argument('--cutmix-prob', type=float, default=0.5)
    parser.add_argument('--switch-prob', type=float, default=0.5)
    
    args = parser.parse_args()
    
    config = EnhancedConfig()
    config.TRAIN_PATH = args.data_path
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.ADAMW_LR = args.adamw_lr
    config.SGD_LR = args.sgd_lr
    config.SGD_MOMENTUM = args.sgd_momentum
    config.DISTRIBUTED = not args.no_distributed and torch.cuda.device_count() > 1
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.CHECKPOINT_EVERY_N_EPOCHS = args.checkpoint_every_n_epochs
    config.USE_MIXUP = not args.no_mixup
    config.USE_CUTMIX = not args.no_cutmix
    config.MIXUP_ALPHA = args.mixup_alpha
    config.CUTMIX_ALPHA = args.cutmix_alpha
    config.MIXUP_PROB = args.mixup_prob
    config.CUTMIX_PROB = args.cutmix_prob
    config.SWITCH_PROB = args.switch_prob
    
    try:
        config.validate()
    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    logger.info(f"Model: {config.MODEL_TYPE}, Backbone: {config.CONVNEXT_BACKBONE}, Device: {config.DEVICE}")
    logger.info("Staged Training Configuration:")
    for stage, stage_config in config.TRAINING_STAGES.items():
        logger.info(f"  - Stage {stage} (epochs {stage_config['epochs'][0]}-{stage_config['epochs'][1]}): "
                    f"Freeze={stage_config['freeze_backbone']}, Optimizer={stage_config['optimizer'].upper()}")
    
    logger.info("Data Augmentation Configuration:")
    logger.info(f"  - MixUp: {'Enabled' if config.USE_MIXUP else 'Disabled'}")
    if config.USE_MIXUP:
        logger.info(f"    - Alpha: {config.MIXUP_ALPHA}, Probability: {config.MIXUP_PROB}")
    logger.info(f"  - CutMix: {'Enabled' if config.USE_CUTMIX else 'Disabled'}")
    if config.USE_CUTMIX:
        logger.info(f"    - Alpha: {config.CUTMIX_ALPHA}, Probability: {config.CUTMIX_PROB}")
    if config.USE_MIXUP and config.USE_CUTMIX:
        logger.info(f"    - Switch Probability (CutMix over MixUp): {config.SWITCH_PROB}")
    
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
    
    try:
        history, _ = train_model(config, resume_from=args.resume_from)
        logger.info("Staged training with MixUp/CutMix completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        cleanup_distributed()
    
    return history

if __name__ == '__main__':
    main()