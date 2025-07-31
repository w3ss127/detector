# Key issues identified and solutions:

"""
PROBLEMS WITH ORIGINAL CODE:
1. Aggressive augmentation destroying forensic artifacts
2. Complex loss function causing training instability
3. Over-engineering leading to overfitting
4. Augmentations removing subtle differences between real/semi-synthetic

SOLUTIONS:
1. Forensics-aware augmentation strategy
2. Simplified, focused loss function
3. Balanced training approach
4. Preservation of critical image artifacts
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import timm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForensicsAwareConfig:
    """Simplified, forensics-focused configuration."""
    def __init__(self, args=None):
        # Model architecture
        self.CONVNEXT_BACKBONE = getattr(args, 'convnext_backbone', "convnext_tiny")
        self.NUM_CLASSES = getattr(args, 'num_classes', 3)
        self.HIDDEN_DIM = getattr(args, 'hidden_dim', 1024)  # Reduced complexity
        self.DROPOUT_RATE = getattr(args, 'dropout_rate', 0.2)  # Less aggressive dropout
        
        # Training configuration
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = getattr(args, 'batch_size', 16)  # Smaller batch for stability
        self.EPOCHS = getattr(args, 'epochs', 30)  # Fewer epochs to prevent overfitting
        self.LEARNING_RATE = getattr(args, 'learning_rate', 1e-4)  # Conservative LR
        self.WEIGHT_DECAY = getattr(args, 'weight_decay', 1e-3)  # Lighter regularization
        
        # Data paths
        self.TRAIN_PATH = getattr(args, 'train_path', "datasets/train")
        self.VAL_PATH = getattr(args, 'val_path', "datasets/val")
        self.IMAGE_SIZE = getattr(args, 'image_size', 224)
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = getattr(args, 'num_workers', 4)
        
        # Forensics-aware settings
        self.PRESERVE_ARTIFACTS = getattr(args, 'preserve_artifacts', True)
        self.GENTLE_AUGMENTATION = getattr(args, 'gentle_augmentation', True)
        self.FOCUS_ON_BOUNDARIES = getattr(args, 'focus_on_boundaries', True)
        
        # Loss weights (simplified)
        self.CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 1.5]).to(self.DEVICE)  # Focus on semi-synthetic
        self.FOCAL_GAMMA = 2.0
        self.BOUNDARY_WEIGHT = 0.3  # Reduced from original
        
        # Early stopping
        self.EARLY_STOPPING_PATIENCE = 7
        self.CHECKPOINT_DIR = getattr(args, 'checkpoint_dir', "forensics_checkpoints")

class ForensicsAwareAugmentation:
    """Careful augmentation that preserves forensic artifacts."""
    
    def __init__(self, config, is_training=True):
        self.config = config
        self.is_training = is_training
        
    def get_gentle_transforms(self):
        """Gentle augmentations that preserve forensic details."""
        if not self.is_training:
            return A.Compose([
                A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        if self.config.GENTLE_AUGMENTATION:
            # Conservative augmentation preserving artifacts
            return A.Compose([
                A.Resize(self.config.IMAGE_SIZE + 16, self.config.IMAGE_SIZE + 16),
                A.RandomCrop(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.3),  # Reduced probability
                
                # Very gentle geometric transforms
                A.ShiftScaleRotate(
                    shift_limit=0.05,   # Minimal shift
                    scale_limit=0.05,   # Minimal scale
                    rotate_limit=5,     # Small rotation only
                    p=0.3
                ),
                
                # Minimal color changes to preserve compression artifacts
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,  # Very gentle
                    contrast_limit=0.1,
                    p=0.3
                ),
                
                # Light noise that doesn't destroy artifacts
                A.GaussNoise(var_limit=(5, 15), p=0.2),  # Light noise only
                
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Minimal augmentation for maximum artifact preservation
            return A.Compose([
                A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),  # Only horizontal flip
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

class SimplifiedModel(nn.Module):
    """Simplified model focused on forensics detection."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Single backbone approach for simplicity
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.backbone = convnext_tiny(weights='IMAGENET1K_V1')
            backbone_features = 768
        else:
            self.backbone = convnext_small(weights='IMAGENET1K_V1')
            backbone_features = 768
        
        # Replace classifier with identity to get features
        self.backbone.classifier = nn.Identity()
        
        # Forensics-aware feature extraction
        self.forensics_head = nn.Sequential(
            nn.Linear(backbone_features, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.HIDDEN_DIM // 2),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )
        
        # Feature extractor for boundary loss
        self.feature_extractor = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # Extract backbone features
        features = self.backbone(x)
        
        # Get logits
        logits = self.forensics_head(features)
        
        # Get features for boundary loss
        boundary_features = self.feature_extractor(features)
        
        return {
            'logits': logits,
            'features': boundary_features,
            'backbone_features': features
        }

class FocusedLoss(nn.Module):
    """Simplified loss focusing on key objectives."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.class_weights = config.CLASS_WEIGHTS
        self.focal_gamma = config.FOCAL_GAMMA
        self.boundary_weight = config.BOUNDARY_WEIGHT
        
    def focal_loss(self, inputs, targets):
        """Focal loss to handle class imbalance."""
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def boundary_loss(self, logits, targets):
        """Simplified boundary loss focusing on semi-synthetic separation."""
        probs = F.softmax(logits, dim=1)
        
        # Encourage clear separation for semi-synthetic class
        real_mask = (targets == 0).float()
        semi_mask = (targets == 1).float() 
        synthetic_mask = (targets == 2).float()
        
        # Real images should have low semi-synthetic probability
        real_boundary = real_mask * probs[:, 1]
        
        # Synthetic images should have low semi-synthetic probability  
        synthetic_boundary = synthetic_mask * probs[:, 1]
        
        # Semi-synthetic should have high semi-synthetic probability
        semi_confidence = semi_mask * (1 - probs[:, 1])
        
        boundary_loss = (real_boundary.sum() + synthetic_boundary.sum() + semi_confidence.sum()) / targets.size(0)
        return boundary_loss
    
    def forward(self, logits, targets):
        """Combined loss computation."""
        focal_loss = self.focal_loss(logits, targets)
        boundary_loss = self.boundary_loss(logits, targets)
        
        total_loss = focal_loss + self.boundary_weight * boundary_loss
        return total_loss

class ForensicsDataset(Dataset):
    """Dataset with forensics-aware preprocessing."""
    
    def __init__(self, image_paths, labels, config, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.config = config
        self.is_training = is_training
        
        # Get appropriate transforms
        augmentation = ForensicsAwareAugmentation(config, is_training)
        self.transform = augmentation.get_gentle_transforms()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image carefully preserving quality
            image = PIL.Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # Apply forensics-aware transforms
            transformed = self.transform(image=image)
            image = transformed['image']
            
            return image, label
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), label

def create_balanced_sampler(labels, config):
    """Create balanced sampler with gentle weighting."""
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Gentle class weighting (less aggressive than original)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
    
    # Cap maximum weight to prevent extreme imbalance correction
    max_weight = class_weights.max()
    min_weight = class_weights.min()
    
    if max_weight / min_weight > 2.0:  # More conservative capping
        class_weights = np.clip(class_weights, min_weight, min_weight * 2.0)
    
    sample_weights = [class_weights[label] for label in labels]
    
    logger.info(f"Balanced class weights: {class_weights}")
    logger.info(f"Class distribution: {class_counts}")
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def load_dataset_carefully(data_path, config):
    """Load dataset with careful handling."""
    image_paths = []
    labels = []
    
    data_path = Path(data_path)
    
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = data_path / class_name
        if class_dir.exists():
            # Support multiple image formats
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            class_images = []
            for ext in extensions:
                class_images.extend(list(class_dir.glob(ext)))
                class_images.extend(list(class_dir.glob(ext.upper())))
            
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            
            logger.info(f"Loaded {len(class_images)} {class_name} images")
    
    logger.info(f"Total images loaded: {len(image_paths)}")
    logger.info(f"Class distribution: {np.bincount(labels)}")
    
    return image_paths, labels

def compute_detailed_metrics(y_true, y_pred):
    """Compute comprehensive metrics."""
    from sklearn.metrics import precision_recall_fscore_support
    
    accuracy = (y_true == y_pred).mean()
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Overall metrics
    macro_f1 = f1.mean()
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[2]
    
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'mcc': mcc,
        'per_class_f1': f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'confusion_matrix': cm,
        'support': support
    }

def train_epoch_focused(model, dataloader, optimizer, criterion, config, epoch):
    """Focused training epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(data)
        logits = outputs['logits']
        
        loss = criterion(logits, target)
        loss.backward()
        
        # Gentle gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 50 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate_epoch_focused(model, dataloader, criterion, config):
    """Focused validation epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            outputs = model(data)
            logits = outputs['logits']
            
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_detailed_metrics(np.array(all_targets), np.array(all_preds))
    metrics['val_loss'] = avg_loss
    
    return metrics

def main_focused_training():
    """Main training function with forensics focus."""
    # Configuration
    config = ForensicsAwareConfig()
    
    # Load datasets
    train_paths, train_labels = load_dataset_carefully(config.TRAIN_PATH, config)
    
    if Path(config.VAL_PATH).exists():
        val_paths, val_labels = load_dataset_carefully(config.VAL_PATH, config)
    else:
        # Split training data
        from sklearn.model_selection import train_test_split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )
    
    # Create datasets
    train_dataset = ForensicsDataset(train_paths, train_labels, config, is_training=True)
    val_dataset = ForensicsDataset(val_paths, val_labels, config, is_training=False)
    
    # Create samplers
    train_sampler = create_balanced_sampler(train_labels, config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
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
    
    # Create model
    model = SimplifiedModel(config)
    model = model.to(config.DEVICE)
    
    # Create loss and optimizer
    criterion = FocusedLoss(config)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training setup
    best_mcc = 0
    patience_counter = 0
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    
    logger.info("Starting focused forensics training...")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        # Training
        train_loss, train_acc = train_epoch_focused(
            model, train_loader, optimizer, criterion, config, epoch
        )
        
        # Validation  
        val_metrics = validate_epoch_focused(model, val_loader, criterion, config)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['mcc'])
        
        # Logging
        logger.info(f"Epoch {epoch:2d} | "
                   f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_metrics['val_loss']:.4f} | "
                   f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                   f"MCC: {val_metrics['mcc']:.4f}")
        
        # Per-class logging
        for i, class_name in enumerate(config.CLASS_NAMES):
            logger.info(f"{class_name}: "
                       f"P={val_metrics['per_class_precision'][i]:.3f} "
                       f"R={val_metrics['per_class_recall'][i]:.3f} "
                       f"F1={val_metrics['per_class_f1'][i]:.3f}")
        
        # Save best model
        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config.__dict__
            }, checkpoint_dir / 'best_model.pth')
            
            logger.info(f"New best model saved with MCC: {best_mcc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best MCC achieved: {best_mcc:.4f}")
    
    return model, best_mcc

# Additional utility functions for analysis

def analyze_model_predictions(model, dataloader, config, save_path=None):
    """Analyze model predictions for forensics insights."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            outputs = model(data)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute detailed analysis
    metrics = compute_detailed_metrics(np.array(all_targets), np.array(all_preds))
    
    # Print detailed results
    print("\n" + "="*60)
    print("FORENSICS MODEL ANALYSIS")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    
    print("\nDetailed Per-Class Analysis:")
    print("-" * 50)
    for i, class_name in enumerate(config.CLASS_NAMES):
        print(f"\n{class_name.upper()}:")
        print(f"  Precision: {metrics['per_class_precision'][i]:.4f}")  
        print(f"  Recall:    {metrics['per_class_recall'][i]:.4f}")
        print(f"  F1-Score:  {metrics['per_class_f1'][i]:.4f}")
        print(f"  Support:   {metrics['support'][i]}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    if save_path:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', xticklabels=config.CLASS_NAMES, 
                   yticklabels=config.CLASS_NAMES)
        plt.title('Forensics Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved to: {save_path}")
    
    return metrics

if __name__ == '__main__':
    # Run focused training
    model, best_mcc = main_focused_training()
    print(f"\nTraining completed with best MCC: {best_mcc:.4f}")