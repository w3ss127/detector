import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import timm
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
import time
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestConfig:
    """Configuration for deepfake detection testing."""
    def __init__(self):
        self.MODEL_TYPE = "enhanced_convnext_vit"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 1024
        self.DROPOUT_RATE = 0.3
        self.ATTENTION_DROPOUT = 0.1
        self.USE_SPECTRAL_NORM = True
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 64  # Larger batch size for testing
        self.TEST_PATH = "datasets/test"
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = 4
        self.MODEL_PATH = None
        self.RESULTS_DIR = "test_results"

class EnhancedAttentionModule(nn.Module):
    """Attention module with channel and spatial attention."""
    def __init__(self, channels, reduction=16, config=None):
        super().__init__()
        self.config = config or TestConfig()
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
        
        # Initialize ConvNeXt backbone
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=None)  # No pretrained weights for testing
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=None)
        else:
            raise ValueError(f"Unsupported ConvNeXt backbone: {config.CONVNEXT_BACKBONE}")
        
        # Initialize ViT backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        
        # Get feature dimensions
        for module in self.convnext.classifier:
            if isinstance(module, nn.Linear):
                convnext_features = module.in_features
                break
        else:
            raise AttributeError("No Linear layer found in ConvNeXt classifier")
        
        vit_features = self.vit.num_features
        
        # Initialize attention and fusion layers
        self.attention_module = EnhancedAttentionModule(channels=convnext_features + vit_features, config=config)
        self.fusion = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(convnext_features + vit_features, config.HIDDEN_DIM)) if config.USE_SPECTRAL_NORM else nn.Linear(convnext_features + vit_features, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.utils.spectral_norm(nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)) if config.USE_SPECTRAL_NORM else nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        )

    def forward(self, x):
        # ConvNeXt feature extraction
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        
        # ViT feature extraction
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]  # Use CLS token
        
        # Feature fusion and attention
        fused_features = torch.cat([convnext_feats, vit_feats], dim=1)
        fused_features = self.attention_module(fused_features)
        logits = self.fusion(fused_features)
        return logits

class TestDataset(Dataset):
    """Dataset for loading test .pt files."""
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
        logger.info(f"Loading test dataset from {self.root_dir}...")
        class_counts = {class_name: 0 for class_name in self.class_names}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist")
                continue
                
            pt_files = list(class_dir.glob('*.pt'))
            logger.info(f"Found {len(pt_files)} .pt files in {class_name} directory")
            
            for pt_file in tqdm(pt_files, desc=f"Loading {class_name}"):
                try:
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    
                    # Handle different tensor formats
                    if isinstance(tensor_data, dict):
                        tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                    
                    # Validate tensor shape
                    if tensor_data.dim() != 4 or tensor_data.shape[1] != 3:
                        logger.warning(f"Invalid tensor shape in {pt_file}: {tensor_data.shape}")
                        continue
                    
                    # Add all images from this tensor
                    for i in range(tensor_data.shape[0]):
                        self.labels.append(class_idx)
                        self.file_mapping.append((str(pt_file), i))
                        self.images.append(tensor_data[i])
                        class_counts[class_name] += 1
                        
                except Exception as e:
                    logger.error(f"Error loading {pt_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.images)} total images")
        for class_name, count in class_counts.items():
            logger.info(f"  - {class_name}: {count} images")
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
            
            # Ensure tensor is float32
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            
            # Normalize to [0, 1] if needed
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # Apply transforms if specified
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

def get_test_transforms(config):
    """Get test transforms (no augmentation, just resize and normalize)."""
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def load_model(model_path, config):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    model = EnhancedConvNextViTModel(config).to(config.DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_acc' in checkpoint:
                logger.info(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    return model

def calculate_multiclass_mcc(y_true, y_pred, num_classes):
    """Calculate Matthews Correlation Coefficient for multiclass classification."""
    # Calculate confusion matrix
    C = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Calculate MCC using the multiclass formula
    t_k = np.sum(C, axis=1)  # True samples for each class
    p_k = np.sum(C, axis=0)  # Predicted samples for each class
    c = np.trace(C)  # Correctly predicted samples
    s = np.sum(C)    # Total samples
    
    # Calculate numerator and denominator
    numerator = c * s - np.sum(t_k * p_k)
    denominator = np.sqrt((s**2 - np.sum(p_k**2)) * (s**2 - np.sum(t_k**2)))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def calculate_per_class_mcc(y_true, y_pred, num_classes):
    """Calculate MCC for each class using one-vs-rest approach."""
    per_class_mcc = []
    
    for class_id in range(num_classes):
        # Convert to binary classification problem
        y_true_binary = (np.array(y_true) == class_id).astype(int)
        y_pred_binary = (np.array(y_pred) == class_id).astype(int)
        
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        per_class_mcc.append(mcc)
    
    return per_class_mcc

def evaluate_model(model, test_loader, config):
    """Evaluate model on test dataset."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_samples = 0
    correct_predictions = 0
    
    logger.info("Starting model evaluation...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            # Forward pass
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update counters
            correct_predictions += predictions.eq(target).sum().item()
            total_samples += target.size(0)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    return all_predictions, all_labels, all_probabilities

def calculate_metrics(predictions, labels, probabilities, config):
    """Calculate comprehensive evaluation metrics."""
    num_classes = config.NUM_CLASSES
    class_names = config.CLASS_NAMES
    
    # Basic accuracy
    overall_accuracy = accuracy_score(labels, predictions)
    
    # Per-class accuracy
    per_class_accuracy = []
    for class_id in range(num_classes):
        class_mask = labels == class_id
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(labels[class_mask], predictions[class_mask])
            per_class_accuracy.append(class_acc)
        else:
            per_class_accuracy.append(0.0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    
    # Classification report
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    
    # MCC calculations
    multiclass_mcc = calculate_multiclass_mcc(labels, predictions, num_classes)
    per_class_mcc = calculate_per_class_mcc(labels, predictions, num_classes)
    
    # ROC AUC (one-vs-rest for multiclass)
    try:
        roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = 0.0
        logger.warning("Could not calculate ROC AUC score")
    
    # Precision, Recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), average=None
    )
    
    return {
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'multiclass_mcc': multiclass_mcc,
        'per_class_mcc': per_class_mcc,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }

def print_results(metrics, config):
    """Print evaluation results in a formatted way."""
    class_names = config.CLASS_NAMES
    
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  Multiclass MCC:   {metrics['multiclass_mcc']:.4f}")
    print(f"  ROC AUC (macro):  {metrics['roc_auc']:.4f}")
    
    # Per-class metrics
    print(f"\nPER-CLASS PERFORMANCE:")
    print(f"{'Class':<15} {'Accuracy':<10} {'MCC':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} "
              f"{metrics['per_class_accuracy'][i]:<10.4f} "
              f"{metrics['per_class_mcc'][i]:<10.4f} "
              f"{metrics['precision'][i]:<10.4f} "
              f"{metrics['recall'][i]:<10.4f} "
              f"{metrics['f1_score'][i]:<10.4f} "
              f"{metrics['support'][i]:<10}")
    
    # Confusion matrix
    print(f"\nCONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    print("Actual \\ Predicted".ljust(20), end="")

    for class_name in class_names:
        print(f"{class_name:<15}", end="")
    print()
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:<15}", end="")
        print()
    
    print("\n" + "="*80)

def save_results(metrics, config, output_path=None):
    """Save results to files."""
    if output_path is None:
        output_path = Path(config.RESULTS_DIR)
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save summary metrics
    summary_file = output_path / f"test_results_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("DEEPFAKE DETECTION MODEL EVALUATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
        f.write(f"Multiclass MCC: {metrics['multiclass_mcc']:.4f}\n")
        f.write(f"ROC AUC (macro): {metrics['roc_auc']:.4f}\n\n")
        
        f.write("Per-Class Performance:\n")
        for i, class_name in enumerate(config.CLASS_NAMES):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Accuracy:  {metrics['per_class_accuracy'][i]:.4f}\n")
            f.write(f"  MCC:       {metrics['per_class_mcc'][i]:.4f}\n")
            f.write(f"  Precision: {metrics['precision'][i]:.4f}\n")
            f.write(f"  Recall:    {metrics['recall'][i]:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score'][i]:.4f}\n")
            f.write(f"  Support:   {metrics['support'][i]}\n")
    
    # Save confusion matrix as CSV
    cm_file = output_path / f"confusion_matrix_{timestamp}.csv"
    cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                        index=config.CLASS_NAMES, 
                        columns=config.CLASS_NAMES)
    cm_df.to_csv(cm_file)
    
    # Save detailed metrics as CSV
    metrics_file = output_path / f"detailed_metrics_{timestamp}.csv"
    metrics_df = pd.DataFrame({
        'Class': config.CLASS_NAMES,
        'Accuracy': metrics['per_class_accuracy'],
        'MCC': metrics['per_class_mcc'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1_Score': metrics['f1_score'],
        'Support': metrics['support']
    })
    metrics_df.to_csv(metrics_file, index=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"  - Summary: {summary_file}")
    logger.info(f"  - Confusion Matrix: {cm_file}")
    logger.info(f"  - Detailed Metrics: {metrics_file}")

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test Deepfake Detection Model')
    parser.add_argument('--model-path', type=str, default="progressive_unfreeze_checkpoints/model_stage2_epoch10_20250730_152546.pth",
                        help='Path to trained model checkpoint')
    parser.add_argument('--test-path', type=str, default='datasets/test',
                        help='Path to test dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--results-dir', type=str, default='test_results',
                        help='Directory to save results')
    parser.add_argument('--backbone', type=str, default='convnext_tiny',
                        choices=['convnext_tiny', 'convnext_small'],
                        help='ConvNeXt backbone architecture')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = TestConfig()
    config.MODEL_PATH = args.model_path
    config.TEST_PATH = args.test_path
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.RESULTS_DIR = args.results_dir
    config.CONVNEXT_BACKBONE = args.backbone
    
    logger.info(f"Testing configuration:")
    logger.info(f"  Model path: {config.MODEL_PATH}")
    logger.info(f"  Test data path: {config.TEST_PATH}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Device: {config.DEVICE}")
    logger.info(f"  Backbone: {config.CONVNEXT_BACKBONE}")
    
    try:
        # Load test dataset
        test_transforms = get_test_transforms(config)
        test_dataset = TestDataset(config.TEST_PATH, config, transform=test_transforms)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=config.NUM_WORKERS, 
            pin_memory=True
        )
        
        logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        
        # Load model
        model = load_model(config.MODEL_PATH, config)
        
        # Evaluate model
        predictions, labels, probabilities = evaluate_model(model, test_loader, config)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, labels, probabilities, config)
        
        # Print results
        print_results(metrics, config)
        
        # Save results
        save_results(metrics, config)
        
        logger.info("Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise

if __name__ == '__main__':
    main()