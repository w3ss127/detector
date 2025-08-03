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
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForensicsAwareConfig:
    """Simplified, forensics-focused configuration."""
    def __init__(self, args=None):
        # Model architecture
        self.CONVNEXT_BACKBONE = getattr(args, 'convnext_backbone', "convnext_tiny")
        self.NUM_CLASSES = getattr(args, 'num_classes', 3)
        self.HIDDEN_DIM = getattr(args, 'hidden_dim', 1024)
        self.DROPOUT_RATE = getattr(args, 'dropout_rate', 0.2)
        
        # Training configuration
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = getattr(args, 'batch_size', 64)
        self.EPOCHS = getattr(args, 'epochs', 30)
        self.LEARNING_RATE = getattr(args, 'learning_rate', 1e-4)
        self.WEIGHT_DECAY = getattr(args, 'weight_decay', 1e-3)
        
        # Data paths
        self.TRAIN_PATH = getattr(args, 'train_path', "datasets/train")
        self.IMAGE_SIZE = getattr(args, 'image_size', 224)
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = getattr(args, 'num_workers', 4)
        
        # Forensics-aware settings
        self.PRESERVE_ARTIFACTS = getattr(args, 'preserve_artifacts', True)
        self.GENTLE_AUGMENTATION = getattr(args, 'gentle_augmentation', True)
        self.FOCUS_ON_BOUNDARIES = getattr(args, 'focus_on_boundaries', True)
        
        # Loss weights
        self.CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 1.5]).to(self.DEVICE)
        self.FOCAL_GAMMA = 2.0
        self.BOUNDARY_WEIGHT = 0.3
        
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
            return A.Compose([
                A.Resize(self.config.IMAGE_SIZE + 16, self.config.IMAGE_SIZE + 16),
                A.RandomCrop(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.3),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    rotate=(-5, 5),
                    p=0.3
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

class SimplifiedModel(nn.Module):
    """Simplified model focused on forensics detection."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.backbone = convnext_tiny(weights='IMAGENET1K_V1')
            backbone_features = 768
        else:
            self.backbone = convnext_small(weights='IMAGENET1K_V1')
            backbone_features = 768
        
        self.backbone.classifier = nn.Identity()  # Remove the default classifier
        self.pool = nn.AdaptiveAvgPool2d(1)  # Reduce spatial dimensions
        
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
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        features = self.backbone(x)  # Shape: [batch_size, 768, H, W]
        features = self.pool(features)  # Shape: [batch_size, 768, 1, 1]
        features = features.view(features.size(0), -1)  # Shape: [batch_size, 768]
        
        logits = self.forensics_head(features)
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
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def boundary_loss(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        real_mask = (targets == 0).float()
        semi_mask = (targets == 1).float()
        synthetic_mask = (targets == 2).float()
        real_boundary = real_mask * probs[:, 1]
        synthetic_boundary = synthetic_mask * probs[:, 1]
        semi_confidence = semi_mask * (1 - probs[:, 1])
        boundary_loss = (real_boundary.sum() + synthetic_boundary.sum() + semi_confidence.sum()) / targets.size(0)
        return boundary_loss
    
    def forward(self, logits, targets):
        focal_loss = self.focal_loss(logits, targets)
        boundary_loss = self.boundary_loss(logits, targets)
        total_loss = focal_loss + self.boundary_weight * boundary_loss
        return total_loss

class ForensicsDataset(Dataset):
    """Dataset with forensics-aware preprocessing for .pt files."""
    
    def __init__(self, pt_file_info, config, is_training=True):
        """
        pt_file_info: List of tuples (pt_file_path, class_idx, num_images)
        """
        self.pt_file_info = pt_file_info
        self.config = config
        self.is_training = is_training
        augmentation = ForensicsAwareAugmentation(config, is_training)
        self.transform = augmentation.get_gentle_transforms()
        
        # Create index mapping for efficient access
        self.image_indices = []
        for pt_idx, (_, _, num_images) in enumerate(pt_file_info):
            for img_idx in range(num_images):
                self.image_indices.append((pt_idx, img_idx))
        
    def __len__(self):
        return len(self.image_indices)
    
    def __getitem__(self, idx):
        pt_idx, img_idx = self.image_indices[idx]
        pt_path, class_idx, _ = self.pt_file_info[pt_idx]
        
        try:
            # Load only the required .pt file
            pt_data = torch.load(pt_path, map_location='cpu')
            image = pt_data[img_idx]  # Shape: [C, H, W] or [H, W, C]
            
            # Ensure image is in correct format [H, W, C]
            if image.shape[0] in [1, 3]:  # [C, H, W]
                image = image.permute(1, 2, 0)
            
            # Convert to numpy for albumentations
            image = image.numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Apply transformations
            transformed = self.transform(image=image)
            image = transformed['image']
            
            return image, class_idx
        except Exception as e:
            logger.warning(f"Error loading image {img_idx} from {pt_path}: {e}")
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), class_idx

def create_balanced_sampler(labels, config):
    """Create balanced sampler with gentle weighting."""
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
    max_weight = class_weights.max()
    min_weight = class_weights.min()
    if max_weight / min_weight > 2.0:
        class_weights = np.clip(class_weights, min_weight, min_weight * 2.0)
    sample_weights = [class_weights[label] for label in labels]
    logger.info(f"Balanced class weights: {class_weights}")
    logger.info(f"Class distribution: {class_counts}")
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def load_dataset_carefully(data_path, config):
    """Load dataset from .pt files with careful handling."""
    pt_file_info = []  # List of (pt_file_path, class_idx, num_images)
    labels = []
    
    data_path = Path(data_path)
    
    # Check if data_path exists
    if not data_path.exists():
        logger.error(f"Dataset directory {data_path} does not exist.")
        raise FileNotFoundError(f"Dataset directory {data_path} does not exist.")
    
    # Process each class directory
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = data_path / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory {class_dir} does not exist.")
            continue
        
        # Find all .pt files
        pt_files = list(class_dir.glob('*.pt'))
        if not pt_files:
            logger.warning(f"No .pt files found in {class_dir} for class {class_name}.")
            continue
        
        # Load each .pt file and verify contents
        for pt_file in pt_files:
            try:
                pt_data = torch.load(pt_file, map_location='cpu')
                if not isinstance(pt_data, torch.Tensor) and not isinstance(pt_data, list):
                    logger.warning(f"Invalid .pt file format in {pt_file}")
                    continue
                
                num_images = len(pt_data) if isinstance(pt_data, list) else pt_data.shape[0]
                if num_images != 5000:
                    logger.warning(f"Unexpected number of images ({num_images}) in {pt_file}")
                
                pt_file_info.append((pt_file, class_idx, num_images))
                labels.extend([class_idx] * num_images)
                logger.info(f"Loaded {num_images} images from {pt_file} for class {class_name}")
            except Exception as e:
                logger.warning(f"Error loading .pt file {pt_file}: {e}")
                continue
    
    if not pt_file_info:
        logger.error("No valid .pt files loaded. Check dataset path and structure.")
        raise ValueError("No valid .pt files loaded. Ensure dataset directory contains .pt files in subfolders named 'real', 'semi-synthetic', and 'synthetic'.")
    
    logger.info(f"Total images loaded: {sum(info[2] for info in pt_file_info)}")
    logger.info(f"Class distribution: {np.bincount(labels) if labels else []}")
    
    return pt_file_info, labels

def compute_detailed_metrics(y_true, y_pred):
    """Compute comprehensive metrics."""
    from sklearn.metrics import precision_recall_fscore_support
    
    accuracy = (y_true == y_pred).mean()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = f1.mean()
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[2]
    mcc = matthews_corrcoef(y_true, y_pred)
    single_mcc = mcc  # Single MCC is the same as MCC in this context
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'mcc': mcc,
        'single_mcc': single_mcc,
        'per_class_f1': f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'confusion_matrix': cm,
        'support': support
    }

def train_epoch_focused(model, dataloader, optimizer, criterion, config, epoch):
    """Focused training epoch with batch-level accuracy and MCC."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        logits = outputs['logits']
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        if batch_idx % 50 == 0:
            batch_accuracy = correct / total if total > 0 else 0
            batch_mcc = matthews_corrcoef(np.array(all_targets), np.array(all_preds)) if len(np.unique(all_targets)) > 1 else 0
            logger.info(f"Epoch {epoch}, Batch {batch_idx}, Accuracy: {batch_accuracy:.4f}, Loss: {loss.item():.4f}, MCC: {batch_mcc:.4f}")
    
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
    
    metrics = compute_detailed_metrics(np.array(all_targets), np.array(all_preds))
    
    print("\n" + "="*60)
    print("FORENSICS MODEL ANALYSIS")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}")
    print(f"Single MCC: {metrics['single_mcc']:.4f}")
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

def main_focused_training():
    """Main training function with forensics focus."""
    # Configuration
    config = ForensicsAwareConfig()
    
    # Load dataset
    try:
        pt_file_info, labels = load_dataset_carefully(config.TRAIN_PATH, config)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    if not pt_file_info:
        logger.error("No .pt files found in dataset. Exiting.")
        raise ValueError("No .pt files found in dataset.")
    
    # Split dataset into train/val/test (70/15/15) with stratification
    indices = np.arange(len(labels))
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )
    
    # Create pt_file_info subsets
    train_pt_info = pt_file_info
    val_pt_info = pt_file_info
    test_pt_info = pt_file_info
    
    # Log dataset sizes
    total_images = len(labels)
    logger.info(f"Dataset split: Train={len(train_indices)} ({len(train_indices)/total_images*100:.1f}%), "
                f"Validation={len(val_indices)} ({len(val_indices)/total_images*100:.1f}%), "
                f"Test={len(test_indices)} ({len(test_indices)/total_images*100:.1f}%)")
    
    # Create datasets
    train_dataset = ForensicsDataset(train_pt_info, config, is_training=True)
    val_dataset = ForensicsDataset(val_pt_info, config, is_training=False)
    test_dataset = ForensicsDataset(test_pt_info, config, is_training=False)
    
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
    
    test_loader = DataLoader(
        test_dataset,
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
        optimizer, mode='max', factor=0.5, patience=3
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
        train_loss, train_acc = train_epoch_focused(
            model, train_loader, optimizer, criterion, config, epoch
        )
        
        val_metrics = validate_epoch_focused(model, val_loader, criterion, config)
        
        # Step the scheduler and log learning rate
        scheduler.step(val_metrics['mcc'])
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch:2d} | Current Learning Rate: {current_lr:.6f}")
        
        logger.info(f"Epoch {epoch:2d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                    f"MCC: {val_metrics['mcc']:.4f} | "
                    f"Single MCC: {val_metrics['single_mcc']:.4f}")
        
        for i, class_name in enumerate(config.CLASS_NAMES):
            logger.info(f"{class_name}: "
                        f"P={val_metrics['per_class_precision'][i]:.3f} "
                        f"R={val_metrics['per_class_recall'][i]:.3f} "
                        f"F1={val_metrics['per_class_f1'][i]:.3f}")
        
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
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best MCC achieved: {best_mcc:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = validate_epoch_focused(model, test_loader, criterion, config)
    logger.info(f"Test Results: "
                f"Test Loss: {test_metrics['val_loss']:.4f} | "
                f"Test Acc: {test_metrics['accuracy']*100:.2f}% | "
                f"MCC: {test_metrics['mcc']:.4f} | "
                f"Single MCC: {test_metrics['single_mcc']:.4f}")
    
    # Analyze test predictions
    test_analysis = analyze_model_predictions(model, test_loader, config, save_path=checkpoint_dir / 'confusion_matrix.png')
    
    return model, best_mcc, test_metrics

if __name__ == '__main__':
    # Run focused training
    model, best_mcc, test_metrics = main_focused_training()
    print(f"\nTraining completed with best MCC: {best_mcc:.4f}")