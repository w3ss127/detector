import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny
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
    """Configuration for deepfake detection training"""
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
        self.MASTER_PORT = "13355"
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
        self.FOCAL_ALPHA = [1.0, 4.0, 2.0]
        self.FOCAL_GAMMA = 2.0
        self.LABEL_SMOOTHING = 0.1
        self.CHECKPOINT_DIR = "improved_checkpoints"
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
            2: {'epochs': (6, 10), 'freeze_backbone': 'classifiers_only', 'optimizer': 'adamw'},
            3: {'epochs': (11, 15), 'freeze_backbone': 'vit', 'optimizer': 'adamw'},
            4: {'epochs': (16, 20), 'freeze_backbone': 'convnext', 'optimizer': 'sgd'},
            5: {'epochs': (21, 50), 'freeze_backbone': 'none', 'optimizer': 'sgd'}
        }

    def get_current_stage(self, epoch):
        for stage, config in self.TRAINING_STAGES.items():
            start_epoch, end_epoch = config['epochs']
            if start_epoch <= epoch <= end_epoch:
                return stage, config
        return None, None

    def validate(self):
        assert isinstance(self.BATCH_SIZE, int) and self.BATCH_SIZE > 0, "Batch size must be positive"
        assert isinstance(self.EPOCHS, int) and self.EPOCHS > 0, "Epochs must be positive"
        assert len(self.CLASS_NAMES) == self.NUM_CLASSES, "Class names must match NUM_CLASSES"
        assert self.CONVNEXT_BACKBONE in ["convnext_tiny", "convnext_small"], "Unsupported backbone"
        assert isinstance(self.CHECKPOINT_EVERY_N_EPOCHS, int) and self.CHECKPOINT_EVERY_N_EPOCHS > 0
        assert 0 <= self.MIXUP_PROB <= 1, "MIXUP_PROB must be between 0 and 1"
        assert 0 <= self.CUTMIX_PROB <= 1, "CUTMIX_PROB must be between 0 and 1"
        assert 0 <= self.SWITCH_PROB <= 1, "SWITCH_PROB must be between 0 and 1"
        assert self.MIXUP_ALPHA > 0, "MIXUP_ALPHA must be positive"
        assert self.CUTMIX_ALPHA > 0, "CUTMIX_ALPHA must be positive"

class MixUpCutMixCollator:
    def __init__(self, config):
        self.config = config
        self.mixup_alpha = config.MIXUP_ALPHA
        self.cutmix_alpha = config.CUTMIX_ALPHA
        self.mixup_prob = config.MIXUP_PROB
        self.cutmix_prob = config.CUTMIX_PROB
        self.switch_prob = config.SWITCH_PROB
        self.num_classes = config.NUM_CLASSES

    def __call__(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        use_mixup = self.config.USE_MIXUP and np.random.rand() < self.mixup_prob
        use_cutmix = self.config.USE_CUTMIX and np.random.rand() < self.cutmix_prob
        
        if use_mixup and use_cutmix:
            if np.random.rand() < self.switch_prob:
                return self._cutmix(images, targets_onehot)
            else:
                return self._mixup(images, targets_onehot)
        elif use_mixup:
            return self._mixup(images, targets_onehot)
        elif use_cutmix:
            return self._cutmix(images, targets_onehot)
        else:
            return images, targets_onehot

    def _mixup(self, images, targets):
        batch_size = images.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        return mixed_images, mixed_targets

    def _cutmix(self, images, targets):
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.focal_alpha = torch.tensor(config.FOCAL_ALPHA, device=config.DEVICE)
        self.focal_gamma = config.FOCAL_GAMMA
        self.label_smoothing = config.LABEL_SMOOTHING

    def focal_loss(self, inputs, targets):
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
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
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
    def __init__(self, channels, reduction=16, config=None):
        super().__init__()
        self.config = config or EnhancedConfig()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.ATTENTION_DROPOUT),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        if self.config.USE_SPECTRAL_NORM:
            self.spatial_attention[0] = SpectralNorm(self.spatial_attention[0])
            self.spatial_attention[2] = SpectralNorm(self.spatial_attention[2])

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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=config.PRETRAINED_WEIGHTS is not None, num_classes=0)
        
        self.convnext_feature_params = [p for n, p in self.convnext.named_parameters() if 'classifier' not in n]
        self.convnext_classifier_params = list(self.convnext.classifier.parameters())
        self.vit_feature_params = [p for n, p in self.vit.named_parameters() if 'head' not in n]
        self.vit_classifier_params = list(self.vit.head.parameters()) if hasattr(self.vit, 'head') else []
        self.backbone_params = self.convnext_feature_params + self.convnext_classifier_params + \
                              self.vit_feature_params + self.vit_classifier_params
        
        if config.FREEZE_BACKBONES:
            self.freeze_backbones()
        
        # Feature dimensions
        self.convnext_dim = 768  # convnext_tiny output dimension
        self.vit_dim = 768       # vit_base_patch16_224 output dimension
        
        # Attention modules
        self.convnext_attention = EnhancedAttentionModule(self.convnext_dim, config=config)
        self.vit_attention = EnhancedAttentionModule(self.vit_dim, config=config)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.convnext_dim + self.vit_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        )
        
        if config.USE_SPECTRAL_NORM:
            self.fusion[0] = SpectralNorm(self.fusion[0])
            self.fusion[3] = SpectralNorm(self.fusion[3])

    def freeze_backbones(self):
        for param in self.backbone_params:
            param.requires_grad = False
        logger.info("All backbones frozen")

    def unfreeze_classifiers(self):
        self.freeze_backbones()
        for param in self.convnext_classifier_params + self.vit_classifier_params:
            param.requires_grad = True
        logger.info("ConvNeXt and ViT classifiers unfrozen")

    def unfreeze_vit(self):
        self.freeze_backbones()
        for param in self.vit_feature_params + self.vit_classifier_params:
            param.requires_grad = True
        for param in self.convnext_classifier_params:
            param.requires_grad = True
        logger.info("ViT backbone and classifiers unfrozen, ConvNeXt features remain frozen")

    def unfreeze_convnext(self):
        for param in self.convnext_feature_params + self.convnext_classifier_params + \
                    self.vit_feature_params + self.vit_classifier_params:
            param.requires_grad = True
        logger.info("ConvNeXt and ViT backbones fully unfrozen")

    def unfreeze_backbones(self):
        for param in self.backbone_params:
            param.requires_grad = True
        logger.info("All backbones unfrozen")

    def forward(self, x):
        # Extract features
        convnext_features = self.convnext(x)
        vit_features = self.vit(x)
        
        # Apply attention
        convnext_att = self.convnext_attention(convnext_features)
        vit_att = self.vit_attention(vit_features)
        
        # Fusion
        combined_features = torch.cat([convnext_att, vit_att], dim=1)
        output = self.fusion(combined_features)
        return output

class EnhancedCustomDatasetPT(Dataset):
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
        logger.info("Loading dataset from .pt files...")
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
                    for i in range(tensor_data.shape[0]):
                        self.labels.append(class_idx)
                        self.file_mapping.append((str(pt_file), i))
                        self.images.append(tensor_data[i])
                except Exception as e:
                    logger.error(f"Error loading {pt_file}: {e}")
        logger.info(f"Loaded {len(self.images)} images")

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

def create_enhanced_data_loaders(config, local_rank=-1):
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

def find_free_port(start_port=13355, max_attempts=100):
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

def setup_distributed(local_rank, world_size, backend='nccl', master_addr='localhost', master_port='13355'):
    port = find_free_port(int(master_port))
    if port != int(master_port):
        logger.info(f"Port {master_port} in use, using {port}")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    logger.info(f"Distributed process group initialized for rank {local_rank} on port {port}")

def cleanup_distributed():
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
    if optimizer_type == 'adamw':
        return optim.AdamW(
            model.parameters(), 
            lr=config.ADAMW_LR, 
            weight_decay=config.WEIGHT_DECAY
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.SGD_LR,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.SGD_WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def evaluate_model(model, data_loader, criterion, config, phase='val'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"{phase} Progress"):
            images = images.to(config.DEVICE)
            if targets.dim() > 1:  # Convert soft targets to hard targets for evaluation
                targets = torch.argmax(targets, dim=1)
            targets = targets.to(config.DEVICE)
            
            with autocast(enabled=config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES, output_dict=True)
    
    return epoch_loss, epoch_acc, cm, report

def train_model(local_rank, config, resume_from=None):
    if config.DISTRIBUTED:
        world_size = torch.cuda.device_count()
        setup_distributed(local_rank, world_size, config.BACKEND, config.MASTER_ADDR, config.MASTER_PORT)
    
    train_loader, val_loader, test_loader = create_enhanced_data_loaders(config, local_rank)
    
    model = EnhancedConvNextViTModel(config).to(config.DEVICE)
    if config.DISTRIBUTED:
        model = DDP(model, device_ids=[local_rank])
    
    criterion = AdvancedLoss(config)
    scaler = GradScaler(enabled=config.USE_AMP)
    
    best_val_acc = 0.0
    current_stage = 1
    current_optimizer_type = 'adamw'
    optimizer = create_optimizer(model, config, current_optimizer_type)
    
    if resume_from:
        epoch, current_stage, best_val_acc, config.FREEZE_BACKBONES, current_optimizer_type = load_staged_checkpoint(
            model, optimizer, resume_from, config
        )
        logger.info(f"Resumed training from epoch {epoch}")
    else:
        epoch = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logger.info(f"Starting training with {config.EPOCHS} epochs in 5 stages")
    for epoch in range(epoch, config.EPOCHS):
        epoch_num = epoch + 1
        stage_num, stage_config = config.get_current_stage(epoch_num)
        
        if stage_num != current_stage:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRANSITIONING TO STAGE {stage_num} (Epochs {stage_config['epochs'][0]}-{stage_config['epochs'][1]})")
            logger.info(f"{'='*60}")
            
            current_stage = stage_num
            model_to_modify = model.module if isinstance(model, DDP) else model
            
            if stage_config['freeze_backbone'] == 'full':
                model_to_modify.freeze_backbones()
                config.FREEZE_BACKBONES = True
            elif stage_config['freeze_backbone'] == 'classifiers_only':
                model_to_modify.unfreeze_classifiers()
                config.FREEZE_BACKBONES = True
            elif stage_config['freeze_backbone'] == 'vit':
                model_to_modify.unfreeze_vit()
                config.FREEZE_BACKBONES = True
            elif stage_config['freeze_backbone'] == 'convnext':
                model_to_modify.unfreeze_convnext()
                config.FREEZE_BACKBONES = False
            elif stage_config['freeze_backbone'] == 'none':
                model_to_modify.unfreeze_backbones()
                config.FREEZE_BACKBONES = False
            
            if stage_config['optimizer'] != current_optimizer_type:
                current_optimizer_type = stage_config['optimizer']
                optimizer = create_optimizer(model, config, current_optimizer_type)
                scaler = GradScaler(enabled=config.USE_AMP)
            
            logger.info(f"Stage {stage_num} configuration:")
            logger.info(f"  - Backbone freeze state: {stage_config['freeze_backbone']}")
            logger.info(f"  - Optimizer: {current_optimizer_type.upper()}")
        
        model.train()
        train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch_num}/{config.EPOCHS}"):
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            with autocast(enabled=config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            if targets.dim() > 1:  # Handle soft targets
                targets = torch.argmax(targets, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        
        val_loss, val_acc, val_cm, val_report = evaluate_model(model, val_loader, criterion, config, phase='val')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch_num}/{config.EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        model_to_save = model.module if isinstance(model, DDP) else model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'stage': stage_num,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'backbone_frozen': config.FREEZE_BACKBONES,
            'optimizer_type': current_optimizer_type,
            'freeze_state': stage_config['freeze_backbone']
        }
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            logger.info("Saved best model checkpoint")
        
        if epoch_num % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
            torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch_num}.pth'))
            logger.info(f"Saved checkpoint for epoch {epoch_num}")
    
    test_loss, test_acc, test_cm, test_report = evaluate_model(model, test_loader, criterion, config, phase='test')
    logger.info(f"Final Test Results:")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    logger.info(f"Test Classification Report:\n{test_report}")
    
    torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, 'final_model.pth'))
    logger.info("Saved final model checkpoint")
    
    if config.DISTRIBUTED:
        cleanup_distributed()
    
    return history, model
def load_staged_checkpoint(model, optimizer, checkpoint_path, config):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    stage = checkpoint['stage']
    val_acc = checkpoint['val_acc']
    train_acc = checkpoint.get('train_acc', 0.0)
    backbone_frozen = checkpoint.get('backbone_frozen', True)
    freeze_state = checkpoint.get('freeze_state', 'full')
    optimizer_type = checkpoint.get('optimizer_type', 'adamw')
    
    model_to_modify = model.module if isinstance(model, DDP) else model
    if freeze_state == 'full':
        model_to_modify.freeze_backbones()
        config.FREEZE_BACKBONES = True
    elif freeze_state == 'classifiers_only':
        model_to_modify.unfreeze_classifiers()
        config.FREEZE_BACKBONES = True
    elif freeze_state == 'vit':
        model_to_modify.unfreeze_vit()
        config.FREEZE_BACKBONES = True
    elif freeze_state in ['convnext', 'none']:
        model_to_modify.unfreeze_convnext()
        config.FREEZE_BACKBONES = False
    
    logger.info(f"Loaded checkpoint from epoch {epoch}, stage {stage}")
    return epoch, stage, val_acc, backbone_frozen, optimizer_type

def main():
    parser = argparse.ArgumentParser(description='Staged Deepfake Detection Training with MixUp/CutMix')
    parser.add_argument('--data-path', type=str, default='datasets/train')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--adamw-lr', type=float, default=1e-3)
    parser.add_argument('--sgd-lr', type=float, default=1e-4)
    parser.add_argument('--sgd-momentum', type=float, default=0.9)
    parser.add_argument('--no-distributed', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default='improved_checkpoints')
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
    
    config.validate()
    
    logger.info(f"Model: {config.MODEL_TYPE}, Backbone: {config.CONVNEXT_BACKBONE}, Device: {config.DEVICE}")
    logger.info("Staged Training Configuration:")
    logger.info(f"  - Stage 1 (epochs 1-10): Frozen backbone + AdamW (lr={config.ADAMW_LR})")
    logger.info(f"  - Stages 2-5 (epochs 11-50): Unfrozen backbone + SGD (lr={config.SGD_LR}, momentum={config.SGD_MOMENTUM})")
    
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
        logger.info(f"Will attempt to resume from checkpoint: {args.resume_from}")
    
    if config.DISTRIBUTED:
        mp.spawn(
            train_model,
            args=(config, args.resume_from),  # Pass config and resume_from as args
            nprocs=torch.cuda.device_count(),
            join=True
        )
    else:
        history, model = train_model(0, config, resume_from=args.resume_from)  # Pass local_rank=0 for single-GPU
    
    logger.info("Staged training with MixUp/CutMix completed successfully!")
    return history, model

if __name__ == '__main__':
    main()