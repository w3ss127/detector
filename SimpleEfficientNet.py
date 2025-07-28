import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import random
import warnings
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import argparse
import gc
import traceback
import shutil
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CheckpointManager:
    """Enhanced checkpoint manager for saving and loading model states"""
    def __init__(self, checkpoint_dir: str = "checkpoints", save_every_n_epochs: int = 1,
                 keep_last_n: int = 5, save_best: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric = float('-inf')
        self.checkpoint_files = []
        
        logger.info(f"Checkpoint manager initialized: {checkpoint_dir}")
        logger.info(f"Save every {save_every_n_epochs} epochs, keep last {keep_last_n} checkpoints")
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: optim.lr_scheduler._LRScheduler, epoch: int, 
                       train_loss: float, val_loss: float, val_acc: float,
                       history: Dict, is_best: bool = False, force_save: bool = False):
        """Save a comprehensive checkpoint"""
        
        # Determine if we should save this epoch
        should_save = (
            force_save or 
            is_best or 
            (epoch + 1) % self.save_every_n_epochs == 0 or
            epoch == 0  # Always save first epoch
        )
        
        if not should_save:
            return
        
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': history,
            'best_metric': self.best_metric,
            'timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__,
        }
        
        # Save regular checkpoint
        if not is_best:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
            torch.save(checkpoint, checkpoint_path)
            self.checkpoint_files.append(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.best_metric = val_acc
            logger.info(f"Best model saved: {best_path} (Val Acc: {val_acc:.2f}%)")
        
        # Always save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the last N"""
        if len(self.checkpoint_files) > self.keep_last_n:
            # Sort by epoch number and remove oldest
            self.checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            files_to_remove = self.checkpoint_files[:-self.keep_last_n]
            
            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed old checkpoint: {file_path}")
            
            self.checkpoint_files = self.checkpoint_files[-self.keep_last_n:]
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer: optim.Optimizer = None, 
                       scheduler: optim.lr_scheduler._LRScheduler = None,
                       load_optimizer: bool = True) -> Dict:
        """Load checkpoint and restore model state"""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if requested
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded")
        
        # Load scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state loaded")
        
        # Restore best metric
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        
        logger.info(f"Checkpoint loaded: Epoch {checkpoint['epoch']+1}, Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        return latest_path if latest_path.exists() else None
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best model checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pth"
        return best_path if best_path.exists() else None
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoints

class SimpleEfficientNet(nn.Module):
    """Simplified EfficientNet without complex attention modules - MAIN FIX"""
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.2):
        super(SimpleEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        # Replace the classifier with a simpler one
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Initialize the new classifier
        nn.init.kaiming_normal_(self.backbone.classifier[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.backbone.classifier[1].bias, 0)
        
        logger.info(f"Simplified model created with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class ImprovedTensorDataset(Dataset):
    """Improved dataset with robust tensor preprocessing - MAJOR FIX"""
    def __init__(self, file_list: List[Tuple[str, int]], transform=None):
        self.file_list = file_list
        self.transform = transform
        self.sample_indices = self._build_sample_index()
    
    def _build_sample_index(self):
        """Build sample index with better error handling"""
        logger.info("Building sample index...")
        sample_indices = []
        
        for file_idx, (file_path, class_label) in enumerate(self.file_list):
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                # Quick check without loading full tensor
                tensor_info = torch.load(file_path, map_location='cpu', weights_only=False)
                
                if isinstance(tensor_info, torch.Tensor):
                    if tensor_info.dim() == 4:  # Batch of images
                        batch_size = tensor_info.shape[0]
                        for local_idx in range(batch_size):
                            sample_indices.append((file_idx, local_idx, class_label))
                    else:  # Single image
                        sample_indices.append((file_idx, 0, class_label))
                else:
                    sample_indices.append((file_idx, 0, class_label))
                
                del tensor_info
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Error checking {file_path}: {e}")
                continue
        
        logger.info(f"Total samples: {len(sample_indices)}")
        return sample_indices
    
    def _preprocess_tensor_robust(self, tensor: torch.Tensor) -> Image.Image:
        """Robust tensor preprocessing with minimal conversions - CRITICAL FIX"""
        try:
            # Ensure tensor is float32 and on CPU
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            if tensor.device != torch.device('cpu'):
                tensor = tensor.cpu()
            
            # Handle batch dimension
            if tensor.dim() == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            
            # Handle different formats
            if tensor.dim() == 3:
                if tensor.size(0) == 3:  # Already in CHW format
                    pass
                elif tensor.size(2) == 3:  # HWC format
                    tensor = tensor.permute(2, 0, 1)
                elif tensor.size(0) == 1:  # Grayscale
                    tensor = tensor.repeat(3, 1, 1)
                else:
                    # Fallback: take first 3 channels or repeat if needed
                    if tensor.size(0) >= 3:
                        tensor = tensor[:3]
                    else:
                        tensor = tensor[0:1].repeat(3, 1, 1)
            elif tensor.dim() == 2:  # 2D grayscale
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            else:
                raise ValueError(f"Unexpected tensor dimensions: {tensor.shape}")
            
            # Normalize to [0, 1] if needed
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            # Clamp and ensure proper size
            tensor = torch.clamp(tensor, 0, 1)
            
            # Resize if needed (avoid if possible to preserve quality)
            if tensor.shape[1] != 224 or tensor.shape[2] != 224:
                tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), 
                                     mode='bilinear', align_corners=False).squeeze(0)
            
            # Convert to PIL Image
            numpy_img = (tensor * 255).byte().permute(1, 2, 0).numpy()
            return Image.fromarray(numpy_img.astype(np.uint8))
            
        except Exception as e:
            logger.warning(f"Error in tensor preprocessing: {e}")
            # Return black image as fallback
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int):
        try:
            file_idx, local_idx, class_label = self.sample_indices[idx]
            file_path = self.file_list[file_idx][0]
            
            # Load tensor
            tensor_batch = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Extract specific sample
            if isinstance(tensor_batch, torch.Tensor) and tensor_batch.dim() == 4:
                if local_idx < tensor_batch.size(0):
                    tensor = tensor_batch[local_idx]
                else:
                    tensor = tensor_batch[0]  # Fallback
            else:
                tensor = tensor_batch
            
            del tensor_batch  # Clean up immediately
            
            # Preprocess to PIL Image
            image = self._preprocess_tensor_robust(tensor)
            del tensor  # Clean up
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, class_label
            
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            # Return dummy data
            dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

def create_conservative_transforms(image_size: int = 224, use_light_augmentation: bool = True):
    """Create conservative data transforms - CRITICAL FIX"""
    
    if use_light_augmentation:
        # Much gentler augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Reduced from 20
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Much gentler
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # No RandomErasing initially to avoid destroying important features
        ])
    else:
        # Minimal augmentation for debugging
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_optimal_learning_rate(batch_size: int, base_lr: float = 0.01) -> float:
    """Calculate optimal learning rate based on batch size - CRITICAL FIX"""
    # Use linear scaling rule but cap it for stability
    scaled_lr = base_lr * batch_size / 256
    # Cap at reasonable values to prevent instability
    return min(scaled_lr, 0.01)

def load_tensor_data(data_dir: str, train_split: float = 0.8, val_split: float = 0.1) -> Tuple[List, List, List, List[str]]:
    """Load and split tensor data efficiently"""
    logger.info("Loading tensor dataset...")
    
    data_dir = Path(data_dir)
    
    # Define class mapping
    class_mapping = {
        'real': 0,
        'semi-synthetic': 1,
        'synthetic': 2
    }
    
    file_list = []
    
    # Look for data in multiple possible structures
    possible_dirs = [data_dir / 'basic', data_dir / 'train', data_dir]
    
    found_data = False
    for base_dir in possible_dirs:
        if base_dir.exists():
            for class_name, class_label in class_mapping.items():
                class_dir = base_dir / class_name
                if class_dir.exists():
                    pt_files = list(class_dir.glob('*.pt'))
                    if pt_files:
                        found_data = True
                        logger.info(f"Found {len(pt_files)} .pt files in {class_name} directory")
                        file_list.extend([(str(pt_file), class_label) for pt_file in pt_files])
            
            if found_data:
                break
    
    if not found_data:
        raise FileNotFoundError(f"No valid data structure found in {data_dir}")
    
    # Shuffle and split data
    random.shuffle(file_list)
    n_total = len(file_list)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train+n_val]
    test_files = file_list[n_train+n_val:]
    
    logger.info(f"Data split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")
    
    class_names = list(class_mapping.keys())
    return train_files, val_files, test_files, class_names

def create_dataloaders(train_files: List, val_files: List, test_files: List, 
                      batch_size: int = 64, num_workers: int = 4,  # Reduced for stability
                      use_light_augmentation: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create optimized data loaders"""
    
    train_transform, val_transform = create_conservative_transforms(use_light_augmentation=use_light_augmentation)
    
    train_dataset = ImprovedTensorDataset(train_files, transform=train_transform)
    val_dataset = ImprovedTensorDataset(val_files, transform=val_transform)
    test_dataset = ImprovedTensorDataset(test_files, transform=val_transform)
    
    # More conservative settings for stability
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=False  # Disabled for stability
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader

def debug_data_loading(train_loader, num_batches=2):
    """Debug function to check data loading - NEW ADDITION"""
    logger.info("=== DEBUGGING DATA LOADING ===")
    
    try:
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
                
            logger.info(f"Batch {batch_idx}:")
            logger.info(f"  Data shape: {data.shape}")
            logger.info(f"  Data dtype: {data.dtype}")
            logger.info(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
            logger.info(f"  Target shape: {target.shape}")
            logger.info(f"  Target values: {target.unique().tolist()}")
            logger.info(f"  Class distribution: {torch.bincount(target).tolist()}")
            
            # Check for NaN or inf values
            if torch.isnan(data).any():
                logger.error("Found NaN values in data!")
            if torch.isinf(data).any():
                logger.error("Found inf values in data!")
                
            # Sample some actual values
            logger.info(f"  Sample pixel values: {data[0, 0, :3, :3].tolist()}")
    except Exception as e:
        logger.error(f"Error during data debugging: {e}")
        
    logger.info("=== END DEBUG ===")

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
            self.best_score = float('inf')
        else:
            self.is_better = lambda score, best: score > best + min_delta
            self.best_score = float('-inf')
        
    def __call__(self, val_metric: float, model: nn.Module) -> bool:
        if self.is_better(val_metric, self.best_score):
            self.best_score = val_metric
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                num_epochs: int = 5, learning_rate: float = 0.01,  # FIXED: Much lower LR
                weight_decay: float = 1e-4, patience: int = 10, 
                optimizer_type: str = 'sgd',  # FIXED: Changed to AdamW
                checkpoint_dir: str = "checkpoints",
                save_every_n_epochs: int = 1, keep_last_n: int = 5,
                resume_from: str = None) -> Dict:
    """Enhanced training function with fixes"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=save_every_n_epochs,
        keep_last_n=keep_last_n,
        save_best=True
    )
    
    # Loss function with reduced label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced from 0.1
    
    # Optimizer selection - default to AdamW for stability
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        # Cosine annealing with warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate*0.01)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=0.9, 
            weight_decay=weight_decay,
            nesterov=True
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logger.info(f"Using {optimizer_type.upper()} optimizer with LR={learning_rate}, WD={weight_decay}")
    
    # Mixed precision and early stopping
    scaler = GradScaler() if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    # Reduced gradient clipping for stability
    max_grad_norm = 1.0
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    # Variables for resuming training
    start_epoch = 0
    best_val_acc = 0.0
    
    # Resume from checkpoint if specified
    if resume_from:
        try:
            checkpoint = checkpoint_manager.load_checkpoint(
                resume_from, model, optimizer, scheduler, load_optimizer=True
            )
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', history)
            best_val_acc = checkpoint.get('val_acc', 0.0)
            logger.info(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    logger.info(f"Training on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Starting from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (data, target) in enumerate(train_bar):
            try:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Memory cleanup every 100 batches
                if batch_idx % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validating'):
                try:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    
                    if scaler is not None:
                        with autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            history=history,
            is_best=is_best
        )
        
        # Logging
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logger.info(f'Learning Rate: {current_lr:.6f}')
        if is_best:
            logger.info(f'New best validation accuracy: {val_acc:.2f}%')
        logger.info('-' * 60)
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping
        if early_stopping(val_loss, model):
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            # Save final checkpoint when early stopping
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                history=history,
                is_best=False,
                force_save=True
            )
            break
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return history

def evaluate_model(model: nn.Module, test_loader: DataLoader, class_names: List[str]) -> Dict:
    """Comprehensive model evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            try:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                if torch.cuda.is_available():
                    with autocast():
                        output = model(data)
                        probabilities = F.softmax(output, dim=1)
                        _, predicted = output.max(1)
                else:
                    output = model(data)
                    probabilities = F.softmax(output, dim=1)
                    _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error in evaluation batch: {e}")
                continue
    
    # Calculate metrics
    if len(all_predictions) == 0:
        logger.error("No valid predictions made during evaluation")
        return {'accuracy': 0.0, 'predictions': [], 'targets': [], 'probabilities': [], 'confusion_matrix': np.array([])}
    
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    
    logger.info(f'Test Accuracy: {accuracy:.2f}%')
    logger.info('\nClassification Report:')
    try:
        print(classification_report(all_targets, all_predictions, 
                                  target_names=class_names, digits=4))
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
    
    # Confusion Matrix
    try:
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")
        cm = np.array([])
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'confusion_matrix': cm
    }

def plot_training_history(history: Dict):
    """Enhanced training history visualization"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(epochs, history['lr'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['val_loss'])]
        ax4.plot(epochs, loss_diff, 'm-', linewidth=2)
        ax4.set_title('Train-Validation Loss Gap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")

def save_results(history: Dict, evaluation_results: Dict, config: Dict, filepath: str = 'training_results.json'):
    """Save training and evaluation results"""
    try:
        results = {
            'config': config,
            'training_history': history,
            'evaluation': {
                'test_accuracy': evaluation_results['accuracy'],
                'confusion_matrix': evaluation_results['confusion_matrix'].tolist() if len(evaluation_results['confusion_matrix']) > 0 else []
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """Main training pipeline with enhanced fixes"""
    parser = argparse.ArgumentParser(description='Train EfficientNet for Image Classification - FIXED VERSION')
    parser.add_argument('--data_dir', type=str, default='datasets', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (reduced for stability)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate (FIXED: much lower)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'], help='Optimizer type (FIXED: AdamW default)')
    parser.add_argument('--light_augmentation', action='store_true', default=True, help='Use light data augmentation (FIXED)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers (reduced for stability)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (reduced)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Checkpoint-related arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_every_n_epochs', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--keep_last_n', type=int, default=5, help='Keep last N checkpoints')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint path')
    parser.add_argument('--load_best_for_eval', action='store_true', default=True, help='Load best model for evaluation')
    
    # Debug options
    parser.add_argument('--debug_data', action='store_true', default=True, help='Debug data loading (NEW)')
    parser.add_argument('--auto_lr', action='store_true', default=True, help='Auto-calculate learning rate based on batch size')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Auto-calculate learning rate if requested
    if args.auto_lr:
        auto_lr = get_optimal_learning_rate(args.batch_size)
        logger.info(f"Auto-calculated learning rate: {auto_lr} (batch_size={args.batch_size})")
        args.learning_rate = auto_lr
    
    # Configuration dictionary for saving
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'light_augmentation': args.light_augmentation,
        'num_workers': args.num_workers,
        'patience': args.patience,
        'seed': args.seed,
        'checkpoint_dir': args.checkpoint_dir,
        'auto_lr': args.auto_lr,
        'model_type': 'SimpleEfficientNet'  # Updated model type
    }
    
    logger.info("Starting FIXED training pipeline...")
    logger.info("=== KEY FIXES APPLIED ===")
    logger.info(f"1. Learning rate reduced to: {args.learning_rate} (was 0.1)")
    logger.info(f"2. Using SimpleEfficientNet (removed complex attention)")
    logger.info(f"3. Conservative data augmentation: {args.light_augmentation}")
    logger.info(f"4. Improved tensor preprocessing")
    logger.info(f"5. Optimizer: {args.optimizer} (AdamW default)")
    logger.info(f"6. Batch size: {args.batch_size} (reduced from 128)")
    logger.info("========================")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"CUDA available: {device_name}")
        logger.info(f"CUDA memory: {total_memory:.1f} GB")
        
        # Enable optimizations for supported GPUs
        if any(gpu_type in device_name for gpu_type in ["A100", "V100", "RTX"]):
            logger.info("GPU optimizations enabled")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        torch.cuda.empty_cache()
    else:
        logger.info("CUDA not available, using CPU")
    
    try:
        # Load and split data
        train_files, val_files, test_files, class_names = load_tensor_data(
            args.data_dir, train_split=0.8, val_split=0.1
        )
        logger.info(f"Classes: {class_names}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files, val_files, test_files, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            use_light_augmentation=args.light_augmentation
        )
        
        # Debug data loading if requested
        if args.debug_data:
            debug_data_loading(train_loader, num_batches=2)
        
        # Create simplified model
        model = SimpleEfficientNet(
            num_classes=len(class_names), 
            pretrained=True, 
            dropout_rate=0.2  # Conservative dropout
        )
        
        # Train model
        logger.info("Starting training with FIXED parameters...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            optimizer_type=args.optimizer,
            checkpoint_dir=args.checkpoint_dir,
            save_every_n_epochs=args.save_every_n_epochs,
            keep_last_n=args.keep_last_n,
            resume_from=args.resume_from
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Load best model for evaluation
        if args.load_best_for_eval:
            logger.info("Loading best model for evaluation...")
            checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
            best_checkpoint_path = checkpoint_manager.get_best_checkpoint()
            
            if best_checkpoint_path and best_checkpoint_path.exists():
                try:
                    checkpoint = checkpoint_manager.load_checkpoint(
                        str(best_checkpoint_path), model, load_optimizer=False
                    )
                    logger.info(f"Best model loaded from epoch {checkpoint['epoch'] + 1} with validation accuracy: {checkpoint['val_acc']:.2f}%")
                except Exception as e:
                    logger.warning(f"Could not load best model: {e}. Using current model state.")
            else:
                logger.warning("Best model checkpoint not found. Using current model state.")
        
        # Evaluate model
        logger.info("Evaluating model on test set...")
        evaluation_results = evaluate_model(model, test_loader, class_names)
        
        # Save results
        save_results(history, evaluation_results, config)
        
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Final Test Accuracy: {evaluation_results['accuracy']:.2f}%")
        
        # Show improvement expectations
        logger.info("=== EXPECTED IMPROVEMENTS ===")
        logger.info("With the fixes applied, you should see:")
        logger.info("1. Training accuracy starting around 50-60% (not 36%)")
        logger.info("2. Steady improvement each epoch")
        logger.info("3. More stable loss values")
        logger.info("4. Better convergence overall")
        
        # Print checkpoint information
        checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
        available_checkpoints = checkpoint_manager.list_checkpoints()
        logger.info(f"Available checkpoints: {len(available_checkpoints)}")
        
        if checkpoint_manager.get_best_checkpoint():
            logger.info(f"Best model saved at: {checkpoint_manager.get_best_checkpoint()}")
        
        # Print per-class accuracy if available
        if len(evaluation_results['confusion_matrix']) > 0:
            cm = evaluation_results['confusion_matrix']
            per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
            logger.info("\nPer-class accuracies:")
            for i, class_name in enumerate(class_names):
                logger.info(f"{class_name}: {per_class_acc[i]:.2f}%")
        
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.info("Please ensure your dataset follows this structure:")
        logger.info("datasets/")
        logger.info("├── basic/ (or train/)")
        logger.info("│   ├── real/")
        logger.info("│   │   ├── batch1.pt")
        logger.info("│   │   └── ...")
        logger.info("│   ├── semi-synthetic/")
        logger.info("│   │   └── ...")
        logger.info("│   └── synthetic/")
        logger.info("│       └── ...")
        
        logger.info("\nTo run with fixes:")
        logger.info("python script.py --data_dir datasets --batch_size 64 --learning_rate 0.005 --optimizer adamw")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()