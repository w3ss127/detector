import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import timm
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from collections import defaultdict
import json
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedConfig:
    """Configuration for deepfake detection testing"""
    def __init__(self):
        self.MODEL_TYPE = "enhanced_convnext_vit"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 1024
        self.DROPOUT_RATE = 0.3
        self.ATTENTION_DROPOUT = 0.1
        self.USE_SPECTRAL_NORM = True
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]
        self.NUM_WORKERS = 4
        self.TEST_PATH = "datasets/test"  # Will be overridden by args
        self.CHECKPOINT_PATH = "checkpoints/best_model_staged.pth"
        self.RESULTS_DIR = "test_results"

class SpectralNorm(nn.Module):
    """Spectral normalization for regularization"""
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
    """Attention module with channel and spatial attention"""
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
    """Hybrid ConvNeXt and ViT model with attention"""
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
        
        self.attention_module = EnhancedAttentionModule(channels=convnext_features + vit_features, config=config)
        self.fusion = nn.Sequential(
            nn.Linear(convnext_features + vit_features, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        )
        if config.USE_SPECTRAL_NORM:
            self.fusion[0] = nn.utils.spectral_norm(self.fusion[0])
            self.fusion[3] = nn.utils.spectral_norm(self.fusion[3])

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

class TestDatasetPT(Dataset):
    """Dataset for loading .pt files for testing"""
    def __init__(self, root_dir, config, transform=None):
        self.root_dir = Path(root_dir)
        self.config = config
        self.transform = transform
        self.class_names = config.CLASS_NAMES
        self.images = []
        self.labels = []
        self.file_mapping = []
        self.class_counts = defaultdict(int)
        self._load_dataset()

    def _load_dataset(self):
        logger.info("Loading test dataset from .pt files...")
        total_images = 0
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist")
                continue
            
            pt_files = list(class_dir.glob('*.pt'))
            logger.info(f"Found {len(pt_files)} .pt files in {class_name} directory")
            
            class_image_count = 0
            for pt_file in tqdm(pt_files, desc=f"Loading {class_name} files"):
                try:
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    if isinstance(tensor_data, dict):
                        tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                    
                    # Handle different tensor shapes
                    if tensor_data.dim() == 4:  # [N, C, H, W]
                        batch_size = tensor_data.shape[0]
                        for i in range(batch_size):
                            self.labels.append(class_idx)
                            self.file_mapping.append((str(pt_file), i))
                            self.images.append(tensor_data[i])
                            class_image_count += 1
                    elif tensor_data.dim() == 3:  # [C, H, W] - single image
                        self.labels.append(class_idx)
                        self.file_mapping.append((str(pt_file), 0))
                        self.images.append(tensor_data)
                        class_image_count += 1
                    else:
                        logger.warning(f"Unexpected tensor shape in {pt_file}: {tensor_data.shape}")
                        
                except Exception as e:
                    logger.error(f"Error loading {pt_file}: {e}")
            
            self.class_counts[class_name] = class_image_count
            total_images += class_image_count
            logger.info(f"Loaded {class_image_count} images from {class_name} class")
        
        logger.info(f"Total images loaded: {total_images}")
        for class_name, count in self.class_counts.items():
            logger.info(f"  {class_name}: {count} images")

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
            
            # Apply transforms if provided
            if self.transform:
                # Convert to numpy for albumentations
                image_np = image_tensor.permute(1, 2, 0).numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            
            return image_tensor, label
            
        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 0

class TestDataAugmentation:
    """Test-time data augmentation (minimal, preserving forensic artifacts)"""
    def __init__(self, config):
        self.config = config

    def get_test_transforms(self):
        return A.Compose([
            A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def load_model_checkpoint(model, checkpoint_path, config):
    """Load model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Log checkpoint information
    epoch = checkpoint.get('epoch', 'Unknown')
    stage = checkpoint.get('stage', 'Unknown')
    val_acc = checkpoint.get('val_acc', 'Unknown')
    train_acc = checkpoint.get('train_acc', 'Unknown')
    backbone_frozen = checkpoint.get('backbone_frozen', 'Unknown')
    optimizer_type = checkpoint.get('optimizer_type', 'Unknown')
    
    logger.info(f"Loaded checkpoint from:")
    logger.info(f"  - Epoch: {epoch}")
    logger.info(f"  - Stage: {stage}")
    logger.info(f"  - Validation accuracy: {val_acc}")
    logger.info(f"  - Training accuracy: {train_acc}")
    logger.info(f"  - Backbone frozen: {backbone_frozen}")
    logger.info(f"  - Optimizer type: {optimizer_type}")
    
    return model

def calculate_metrics(y_true, y_pred, y_prob, class_names):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Overall accuracy
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        metrics[f'accuracy_{class_name}'] = per_class_acc[i]
    
    # Matthews Correlation Coefficient (overall)
    metrics['overall_mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class MCC (one-vs-rest)
    for i, class_name in enumerate(class_names):
        y_true_binary = (np.array(y_true) == i).astype(int)
        y_pred_binary = (np.array(y_pred) == i).astype(int)
        metrics[f'mcc_{class_name}'] = matthews_corrcoef(y_true_binary, y_pred_binary)
    
    # Precision, Recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision[i]
        metrics[f'recall_{class_name}'] = recall[i]
        metrics[f'f1_{class_name}'] = f1[i]
        metrics[f'support_{class_name}'] = support[i]
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # AUC (if multiclass)
    if len(class_names) > 2:
        try:
            auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            auc_ovo = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
            metrics['auc_ovr_macro'] = auc_ovr
            metrics['auc_ovo_macro'] = auc_ovo
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
    
    return metrics

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)')
        annotations.append(row)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")

def plot_per_class_metrics(metrics, class_names, save_path):
    """Plot per-class metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    accuracies = [metrics[f'accuracy_{cls}'] for cls in class_names]
    axes[0, 0].bar(class_names, accuracies, color='skyblue')
    axes[0, 0].set_title('Per-Class Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # MCC
    mccs = [metrics[f'mcc_{cls}'] for cls in class_names]
    axes[0, 1].bar(class_names, mccs, color='lightcoral')
    axes[0, 1].set_title('Per-Class Matthews Correlation Coefficient')
    axes[0, 1].set_ylabel('MCC')
    axes[0, 1].set_ylim(-1, 1)
    for i, v in enumerate(mccs):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Precision, Recall, F1
    precisions = [metrics[f'precision_{cls}'] for cls in class_names]
    recalls = [metrics[f'recall_{cls}'] for cls in class_names]
    f1s = [metrics[f'f1_{cls}'] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    axes[1, 0].bar(x - width, precisions, width, label='Precision', color='lightgreen')
    axes[1, 0].bar(x, recalls, width, label='Recall', color='orange')
    axes[1, 0].bar(x + width, f1s, width, label='F1-Score', color='purple')
    axes[1, 0].set_title('Precision, Recall, and F1-Score')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    # Support (sample count)
    supports = [metrics[f'support_{cls}'] for cls in class_names]
    axes[1, 1].bar(class_names, supports, color='gold')
    axes[1, 1].set_title('Support (Number of Samples)')
    axes[1, 1].set_ylabel('Count')
    for i, v in enumerate(supports):
        axes[1, 1].text(i, v + max(supports) * 0.01, f'{int(v)}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Per-class metrics plot saved to {save_path}")

def save_detailed_results(metrics, class_names, save_path):
    """Save detailed results to JSON and CSV"""
    # Save as JSON
    json_path = save_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Create DataFrame for CSV
    results_data = []
    
    # Overall metrics
    results_data.append({
        'Metric': 'Overall Accuracy',
        'Value': metrics['overall_accuracy'],
        'Class': 'Overall'
    })
    results_data.append({
        'Metric': 'Overall MCC',
        'Value': metrics['overall_mcc'],
        'Class': 'Overall'
    })
    
    # Macro averages
    results_data.append({
        'Metric': 'Precision (Macro)',
        'Value': metrics['precision_macro'],
        'Class': 'Overall'
    })
    results_data.append({
        'Metric': 'Recall (Macro)',
        'Value': metrics['recall_macro'],
        'Class': 'Overall'
    })
    results_data.append({
        'Metric': 'F1-Score (Macro)',
        'Value': metrics['f1_macro'],
        'Class': 'Overall'
    })
    
    # Per-class metrics
    for class_name in class_names:
        for metric_type in ['accuracy', 'mcc', 'precision', 'recall', 'f1', 'support']:
            results_data.append({
                'Metric': metric_type.capitalize(),
                'Value': metrics[f'{metric_type}_{class_name}'],
                'Class': class_name
            })
    
    # Save as CSV
    csv_path = save_path.replace('.txt', '.csv')
    df = pd.DataFrame(results_data)
    df.to_csv(csv_path, index=False)
    
    # Save as formatted text
    with open(save_path, 'w') as f:
        f.write("DEEPFAKE DETECTION MODEL TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
        f.write(f"Overall MCC: {metrics['overall_mcc']:.4f}\n")
        f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
        f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (Weighted): {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n")
        
        if 'auc_ovr_macro' in metrics:
            f.write(f"AUC (OvR Macro): {metrics['auc_ovr_macro']:.4f}\n")
            f.write(f"AUC (OvO Macro): {metrics['auc_ovo_macro']:.4f}\n")
        
        f.write("\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 20 + "\n")
        for class_name in class_names:
            f.write(f"\n{class_name.upper()} CLASS:\n")
            f.write(f"  Accuracy: {metrics[f'accuracy_{class_name}']:.4f}\n")
            f.write(f"  MCC: {metrics[f'mcc_{class_name}']:.4f}\n")
            f.write(f"  Precision: {metrics[f'precision_{class_name}']:.4f}\n")
            f.write(f"  Recall: {metrics[f'recall_{class_name}']:.4f}\n")
            f.write(f"  F1-Score: {metrics[f'f1_{class_name}']:.4f}\n")
            f.write(f"  Support: {int(metrics[f'support_{class_name}'])}\n")
    
    logger.info(f"Detailed results saved to:")
    logger.info(f"  - Text: {save_path}")
    logger.info(f"  - JSON: {json_path}")
    logger.info(f"  - CSV: {csv_path}")

def test_model(config):
    """Main testing function"""
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Create test dataset and dataloader
    test_dataset = TestDatasetPT(
        root_dir=config.TEST_PATH,
        config=config,
        transform=TestDataAugmentation(config).get_test_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    model = EnhancedConvNextViTModel(config).to(config.DEVICE)
    model = load_model_checkpoint(model, config.CHECKPOINT_PATH, config)
    model.eval()
    
    # Test the model
    logger.info("Starting model evaluation...")
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            # Forward pass
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    logger.info("Model evaluation completed. Calculating metrics...")
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities, config.CLASS_NAMES)
    
    # Print key results
    logger.info("\nKEY RESULTS:")
    logger.info("=" * 40)
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"Overall MCC: {metrics['overall_mcc']:.4f}")
    logger.info(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    
    logger.info("\nPer-Class Results:")
    for class_name in config.CLASS_NAMES:
        logger.info(f"{class_name.upper()}:")
        logger.info(f"  Accuracy: {metrics[f'accuracy_{class_name}']:.4f}")
        logger.info(f"  MCC: {metrics[f'mcc_{class_name}']:.4f}")
        logger.info(f"  F1-Score: {metrics[f'f1_{class_name}']:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"{cm}")
    
    # Save results and plots
    results_base_path = os.path.join(config.RESULTS_DIR, f"test_results_{timestamp}")
    
    # Save detailed results
    save_detailed_results(metrics, config.CLASS_NAMES, f"{results_base_path}.txt")
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, config.CLASS_NAMES, f"{results_base_path}_confusion_matrix.png")
    
    # Plot and save per-class metrics
    plot_per_class_metrics(metrics, config.CLASS_NAMES, f"{results_base_path}_per_class_metrics.png")
    
    # Save raw predictions for further analysis
    predictions_data = {
        'true_labels': all_labels.tolist(),
        'predicted_labels': all_predictions.tolist(),
        'probabilities': all_probabilities.tolist(),
        'class_names': config.CLASS_NAMES,
        'file_mapping': test_dataset.file_mapping
    }
    
    predictions_path = f"{results_base_path}_predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    logger.info(f"Raw predictions saved to {predictions_path}")
    
    # Create summary report
    create_summary_report(metrics, cm, config.CLASS_NAMES, f"{results_base_path}_summary.txt")
    
    logger.info(f"\nTest completed! Results saved to {config.RESULTS_DIR}")
    return metrics, all_predictions, all_labels, all_probabilities

def create_summary_report(metrics, cm, class_names, save_path):
    """Create a concise summary report"""
    with open(save_path, 'w') as f:
        f.write("DEEPFAKE DETECTION MODEL - TEST SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Class Names: {', '.join(class_names)}\n")
        f.write(f"Total Test Samples: {cm.sum()}\n\n")
        
        # Key metrics
        f.write("KEY PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)\n")
        f.write(f"Overall MCC: {metrics['overall_mcc']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}\n\n")
        
        # Per-class summary table
        f.write("PER-CLASS PERFORMANCE SUMMARY:\n")
        f.write("-" * 35 + "\n")
        f.write(f"{'Class':<15} {'Accuracy':<10} {'MCC':<8} {'F1':<8} {'Support':<8}\n")
        f.write("-" * 50 + "\n")
        
        for class_name in class_names:
            f.write(f"{class_name:<15} "
                   f"{metrics[f'accuracy_{class_name}']:<10.4f} "
                   f"{metrics[f'mcc_{class_name}']:<8.4f} "
                   f"{metrics[f'f1_{class_name}']:<8.4f} "
                   f"{int(metrics[f'support_{class_name}']):<8d}\n")
        
        f.write("\n")
        
        # Confusion matrix
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 20 + "\n")
        f.write("True\\Predicted  ")
        for class_name in class_names:
            f.write(f"{class_name[:8]:<10}")
        f.write("\n")
        
        for i, true_class in enumerate(class_names):
            f.write(f"{true_class[:12]:<15}")
            for j in range(len(class_names)):
                f.write(f"{cm[i, j]:<10}")
            f.write("\n")
        
        f.write("\n")
        
        # Model interpretation
        f.write("PERFORMANCE INTERPRETATION:\n")
        f.write("-" * 30 + "\n")
        
        if metrics['overall_accuracy'] >= 0.9:
            performance_level = "Excellent"
        elif metrics['overall_accuracy'] >= 0.8:
            performance_level = "Good"
        elif metrics['overall_accuracy'] >= 0.7:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        f.write(f"Overall Performance: {performance_level}\n")
        
        # Find best and worst performing classes
        class_f1s = [(class_name, metrics[f'f1_{class_name}']) for class_name in class_names]
        class_f1s.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"Best Performing Class: {class_f1s[0][0]} (F1: {class_f1s[0][1]:.4f})\n")
        f.write(f"Worst Performing Class: {class_f1s[-1][0]} (F1: {class_f1s[-1][1]:.4f})\n")
        
        # Check for class imbalance issues
        support_values = [metrics[f'support_{class_name}'] for class_name in class_names]
        max_support = max(support_values)
        min_support = min(support_values)
        imbalance_ratio = max_support / min_support
        
        f.write(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1\n")
        if imbalance_ratio > 3:
            f.write("WARNING: Significant class imbalance detected. Consider balancing techniques.\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 17 + "\n")
        
        if metrics['overall_accuracy'] < 0.8:
            f.write("- Consider additional training or model architecture improvements\n")
        
        if metrics['overall_mcc'] < 0.6:
            f.write("- MCC suggests room for improvement in prediction quality\n")
        
        worst_class_f1 = class_f1s[-1][1]
        if worst_class_f1 < 0.7:
            f.write(f"- Focus on improving {class_f1s[-1][0]} class performance\n")
        
        if imbalance_ratio > 3:
            f.write("- Address class imbalance through data augmentation or sampling techniques\n")
    
    logger.info(f"Summary report saved to {save_path}")

def analyze_misclassifications(true_labels, pred_labels, probabilities, class_names, file_mapping, save_path):
    """Analyze and save misclassification details"""
    misclassified = []
    
    for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
        if true_label != pred_label:
            file_path, file_index = file_mapping[i]
            confidence = probabilities[i][pred_label]
            true_confidence = probabilities[i][true_label]
            
            misclassified.append({
                'file_path': file_path,
                'file_index': file_index,
                'true_class': class_names[true_label],
                'predicted_class': class_names[pred_label],
                'prediction_confidence': float(confidence),
                'true_class_confidence': float(true_confidence),
                'confidence_difference': float(confidence - true_confidence)
            })
    
    # Sort by confidence difference (most confident misclassifications first)
    misclassified.sort(key=lambda x: x['confidence_difference'], reverse=True)
    
    # Save detailed misclassification analysis
    with open(save_path, 'w') as f:
        json.dump(misclassified, f, indent=2)
    
    # Create summary
    summary_path = save_path.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MISCLASSIFICATION ANALYSIS\n")
        f.write("=" * 30 + "\n\n")
        
        f.write(f"Total Misclassifications: {len(misclassified)}\n")
        f.write(f"Total Samples: {len(true_labels)}\n")
        f.write(f"Misclassification Rate: {len(misclassified)/len(true_labels)*100:.2f}%\n\n")
        
        # Misclassification matrix
        f.write("MISCLASSIFICATION PATTERNS:\n")
        f.write("-" * 35 + "\n")
        
        # Count misclassifications by type
        misclass_counts = defaultdict(int)
        for item in misclassified:
            key = f"{item['true_class']} -> {item['predicted_class']}"
            misclass_counts[key] += 1
        
        # Sort by frequency
        sorted_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
        
        for pattern, count in sorted_misclass:
            percentage = count / len(misclassified) * 100
            f.write(f"{pattern}: {count} ({percentage:.1f}% of misclassifications)\n")
        
        f.write("\n")
        
        # Top 10 most confident misclassifications
        f.write("TOP 10 MOST CONFIDENT MISCLASSIFICATIONS:\n")
        f.write("-" * 45 + "\n")
        
        for i, item in enumerate(misclassified[:10]):
            f.write(f"{i+1}. File: {item['file_path']} (index {item['file_index']})\n")
            f.write(f"   True: {item['true_class']} -> Predicted: {item['predicted_class']}\n")
            f.write(f"   Confidence: {item['prediction_confidence']:.4f}\n")
            f.write(f"   Confidence Difference: {item['confidence_difference']:.4f}\n\n")
    
    logger.info(f"Misclassification analysis saved to {save_path}")
    logger.info(f"Misclassification summary saved to {summary_path}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Test Deepfake Detection Model')
    parser.add_argument('--test-path', type=str, default='datasets/test',
                       help='Path to test data directory containing real/semi-synthetic/synthetic subdirectories')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/model_stage1_epoch5_20250729_205930.pth',
                       help='Path to model checkpoint file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for testing (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--results-dir', type=str, default='test_results',
                       help='Directory to save test results (default: test_results)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--backbone', type=str, default='convnext_tiny',
                       choices=['convnext_tiny', 'convnext_small'],
                       help='ConvNeXt backbone architecture (default: convnext_tiny)')
    parser.add_argument('--analyze-misclassifications', action='store_true',
                       help='Perform detailed misclassification analysis')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for testing (default: auto)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = EnhancedConfig()
    config.TEST_PATH = args.test_path
    config.CHECKPOINT_PATH = args.checkpoint_path
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.RESULTS_DIR = args.results_dir
    config.IMAGE_SIZE = args.image_size
    config.CONVNEXT_BACKBONE = args.backbone
    
    # Set device
    if args.device == 'auto':
        config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        config.DEVICE = torch.device(args.device)
    
    # Validate paths
    if not os.path.exists(args.test_path):
        logger.error(f"Test data path does not exist: {args.test_path}")
        return
    
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file does not exist: {args.checkpoint_path}")
        return
    
    # Check for required subdirectories
    required_dirs = ['real', 'semi-synthetic', 'synthetic']
    for class_dir in required_dirs:
        class_path = os.path.join(args.test_path, class_dir)
        if not os.path.exists(class_path):
            logger.error(f"Required class directory does not exist: {class_path}")
            return
    
    logger.info("=" * 60)
    logger.info("DEEPFAKE DETECTION MODEL TESTING")
    logger.info("=" * 60)
    logger.info(f"Test data path: {config.TEST_PATH}")
    logger.info(f"Checkpoint path: {config.CHECKPOINT_PATH}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Backbone: {config.CONVNEXT_BACKBONE}")
    logger.info(f"Results directory: {config.RESULTS_DIR}")
    logger.info("=" * 60)
    
    try:
        # Run testing
        metrics, predictions, labels, probabilities = test_model(config)
        
        # Perform misclassification analysis if requested
        if args.analyze_misclassifications:
            logger.info("Performing misclassification analysis...")
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            misclass_path = os.path.join(config.RESULTS_DIR, f"misclassifications_{timestamp}.json")
            
            # Load dataset to get file mapping
            test_dataset = TestDatasetPT(
                root_dir=config.TEST_PATH,
                config=config,
                transform=TestDataAugmentation(config).get_test_transforms()
            )
            
            analyze_misclassifications(
                labels, predictions, probabilities, 
                config.CLASS_NAMES, test_dataset.file_mapping, 
                misclass_path
            )
        
        logger.info("\n" + "=" * 60)
        logger.info("TESTING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Final Results:")
        logger.info(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        logger.info(f"  Overall MCC: {metrics['overall_mcc']:.4f}")
        logger.info(f"  Macro F1-Score: {metrics['f1_macro']:.4f}")
        logger.info(f"Results saved to: {config.RESULTS_DIR}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())