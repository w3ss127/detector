import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
import warnings
import gc
from tqdm import tqdm
import json
import pandas as pd
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import timm
import os

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestConfig:
    """Configuration for testing"""
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
        self.TEST_PATH = "datasets/test"  # Default test path
        
        # Forensic augmentation settings
        self.USE_FORENSIC_AUGMENTATION = True
        self.CLASS_WEIGHTS = [1.0, 2.5, 1.5]
        self.USE_COMBINED_LOSS = True
        self.CONTRASTIVE_WEIGHT = 0.3

# Copy necessary classes from training script
class FixedAttentionModule(nn.Module):
    """Fixed attention module that properly handles feature dimensions"""
    def __init__(self, in_features, config=None):
        super().__init__()
        self.config = config or TestConfig()
        self.in_features = in_features
        
        # Channel attention using global average pooling
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )
        
        # Apply spectral normalization if enabled
        if self.config.USE_SPECTRAL_NORM:
            self.channel_attention[0] = nn.utils.spectral_norm(self.channel_attention[0])
            self.channel_attention[3] = nn.utils.spectral_norm(self.channel_attention[3])

    def forward(self, x):
        # x should be [batch_size, features]
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        
        # Channel attention
        attention_weights = self.channel_attention(x)
        attended_features = x * attention_weights
        
        return attended_features

class ImprovedConvNextViTModel(nn.Module):
    """Enhanced model with better feature extraction for semi-synthetic detection"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize backbones
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        else:
            raise ValueError(f"Unsupported ConvNeXt backbone: {config.CONVNEXT_BACKBONE}")
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Add additional forensic-aware layers
        self.forensic_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Update fusion layer to include forensic features
        convnext_features = 768  # ConvNeXt tiny features
        vit_features = self.vit.num_features
        forensic_features = 256
        
        total_features = convnext_features + vit_features + forensic_features
        
        self.attention_module = FixedAttentionModule(total_features, config)
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )
        
        if config.USE_SPECTRAL_NORM:
            self.fusion[0] = nn.utils.spectral_norm(self.fusion[0])
            self.fusion[3] = nn.utils.spectral_norm(self.fusion[3])
            self.fusion[6] = nn.utils.spectral_norm(self.fusion[6])
    
    def forward(self, x):
        # Original ConvNeXt and ViT features
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]
        
        # Additional forensic features from raw input
        forensic_feats = self.forensic_features(x)
        
        # Combine all features
        fused_features = torch.cat([convnext_feats, vit_feats, forensic_feats], dim=1)
        attended_features = self.attention_module(fused_features)
        
        logits = self.fusion(attended_features)
        
        # Return both logits and features for contrastive loss
        return logits, fused_features

class TestDataset(torch.utils.data.Dataset):
    """Dataset for loading test .pt files"""
    
    def __init__(self, test_path, config, transform=None):
        self.test_path = Path(test_path)
        self.config = config
        self.transform = transform
        self.class_names = config.CLASS_NAMES
        self.samples = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """Load all .pt files and create sample list"""
        logger.info(f"Loading test data from {self.test_path}")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.test_path / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist")
                continue
            
            pt_files = list(class_dir.glob('*.pt'))
            logger.info(f"Found {len(pt_files)} .pt files in {class_name}")
            
            for pt_file in pt_files:
                try:
                    # Load and check tensor data
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    if isinstance(tensor_data, dict):
                        tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                    
                    num_images = tensor_data.shape[0]
                    for i in range(num_images):
                        self.samples.append((str(pt_file), i))
                        self.labels.append(class_idx)
                    
                    del tensor_data  # Free memory
                    
                except Exception as e:
                    logger.error(f"Error loading {pt_file}: {e}")
        
        logger.info(f"Total test samples: {len(self.samples)}")
        
        # Print class distribution
        class_counts = [self.labels.count(i) for i in range(len(self.class_names))]
        for i, (class_name, count) in enumerate(zip(self.class_names, class_counts)):
            logger.info(f"{class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            file_path, image_idx = self.samples[idx]
            label = self.labels[idx]
            
            # Load tensor data
            tensor_data = torch.load(file_path, map_location='cpu')
            if isinstance(tensor_data, dict):
                tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
            
            image_tensor = tensor_data[image_idx]
            
            # Normalize to [0, 1] if needed
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # Apply transforms
            if self.transform:
                # Convert to numpy for albumentations
                image_np = image_tensor.permute(1, 2, 0).numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            
            return image_tensor, label, file_path, image_idx
            
        except Exception as e:
            logger.error(f"Error loading sample at index {idx}: {e}")
            # Return dummy data in case of error
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 0, "error", 0

def get_test_transform(config):
    """Get test time transforms"""
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def calculate_multiclass_mcc(y_true, y_pred, num_classes):
    """Calculate Matthews Correlation Coefficient for multiclass"""
    if num_classes == 2:
        return matthews_corrcoef(y_true, y_pred)
    else:
        # For multiclass, calculate MCC using confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate MCC for multiclass
        cov_ytyp = np.trace(cm)
        cov_ypyp = np.sum(cm)
        cov_ytyt = np.sum(cm)
        
        sum_yt_yp = np.sum(np.sum(cm, axis=1) * np.sum(cm, axis=0))
        sum_yt2 = np.sum(np.sum(cm, axis=1) ** 2)
        sum_yp2 = np.sum(np.sum(cm, axis=0) ** 2)
        
        numerator = cov_ytyp * cov_ypyp - sum_yt_yp
        denominator = np.sqrt((cov_ypyp**2 - sum_yp2) * (cov_ytyt**2 - sum_yt2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator

def calculate_per_class_mcc(y_true, y_pred, num_classes):
    """Calculate MCC for each class using one-vs-rest approach"""
    per_class_mcc = []
    
    for class_idx in range(num_classes):
        # Create binary labels for current class
        y_true_binary = (np.array(y_true) == class_idx).astype(int)
        y_pred_binary = (np.array(y_pred) == class_idx).astype(int)
        
        # Calculate MCC for this class
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        per_class_mcc.append(mcc)
    
    return per_class_mcc

def load_model(model_path, config):
    """Load trained model from checkpoint"""
    logger.info(f"Loading model from {model_path}")
    
    # Initialize model
    model = ImprovedConvNextViTModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        if 'val_mcc' in checkpoint:
            logger.info(f"Checkpoint validation MCC: {checkpoint['val_mcc']:.4f}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        logger.info("Model loaded from direct state dict")
    
    model.to(config.DEVICE)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, config):
    """Comprehensive model evaluation"""
    logger.info("Starting model evaluation...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_file_paths = []
    all_image_indices = []
    
    # Per-file predictions for debugging
    file_predictions = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, (data, targets, file_paths, image_indices) in enumerate(tqdm(test_loader, desc="Evaluating")):
            data = data.to(config.DEVICE, non_blocking=True)
            targets = targets.to(config.DEVICE, non_blocking=True)
            
            # Forward pass
            outputs = model(data)
            if isinstance(outputs, tuple):
                logits, features = outputs
            else:
                logits = outputs
            
            # Get predictions and probabilities
            probabilities = F.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_file_paths.extend(file_paths)
            all_image_indices.extend(image_indices)
            
            # Store per-file predictions for analysis
            for i, (file_path, img_idx, pred, target, prob) in enumerate(zip(
                file_paths, image_indices, predictions.cpu().numpy(), targets.cpu().numpy(), probabilities.cpu().numpy()
            )):
                file_predictions[file_path].append({
                    'image_idx': img_idx,
                    'prediction': pred,
                    'target': target,
                    'probabilities': prob,
                    'correct': pred == target
                })
            
            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets) 
    all_probabilities = np.array(all_probabilities)
    
    return all_predictions, all_targets, all_probabilities, file_predictions

def calculate_detailed_metrics(y_true, y_pred, y_prob, class_names):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Overall accuracy
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Multiclass MCC
    metrics['multiclass_mcc'] = calculate_multiclass_mcc(y_true, y_pred, len(class_names))
    
    # Per-class MCC
    per_class_mcc = calculate_per_class_mcc(y_true, y_pred, len(class_names))
    metrics['per_class_mcc'] = dict(zip(class_names, per_class_mcc))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics['per_class_precision'] = dict(zip(class_names, precision))
    metrics['per_class_recall'] = dict(zip(class_names, recall))
    metrics['per_class_f1'] = dict(zip(class_names, f1))
    metrics['per_class_support'] = dict(zip(class_names, support))
    
    # Macro averages
    metrics['macro_precision'] = precision.mean()
    metrics['macro_recall'] = recall.mean()
    metrics['macro_f1'] = f1.mean()
    
    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics['weighted_precision'] = precision_w
    metrics['weighted_recall'] = recall_w
    metrics['weighted_f1'] = f1_w
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations combining counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
        annotations.append(row)
    
    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {save_path}")
    
    plt.show()

def plot_per_class_metrics(metrics, class_names, save_path=None):
    """Plot per-class metrics comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract per-class metrics
    classes = class_names
    precision = [metrics['per_class_precision'][cls] for cls in classes]
    recall = [metrics['per_class_recall'][cls] for cls in classes]
    f1 = [metrics['per_class_f1'][cls] for cls in classes]
    mcc = [metrics['per_class_mcc'][cls] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.6
    
    # Precision
    bars1 = ax1.bar(x, precision, width, color='skyblue', alpha=0.8)
    ax1.set_ylabel('Precision')
    ax1.set_title('Per-Class Precision')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{precision[i]:.3f}', ha='center', va='bottom')
    
    # Recall
    bars2 = ax2.bar(x, recall, width, color='lightcoral', alpha=0.8)
    ax2.set_ylabel('Recall')
    ax2.set_title('Per-Class Recall')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{recall[i]:.3f}', ha='center', va='bottom')
    
    # F1-Score
    bars3 = ax3.bar(x, f1, width, color='lightgreen', alpha=0.8)
    ax3.set_ylabel('F1-Score')
    ax3.set_title('Per-Class F1-Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=45)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1[i]:.3f}', ha='center', va='bottom')
    
    # MCC
    bars4 = ax4.bar(x, mcc, width, color='gold', alpha=0.8)
    ax4.set_ylabel('MCC')
    ax4.set_title('Per-Class Matthews Correlation Coefficient')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes, rotation=45)
    ax4.set_ylim(-1, 1)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., 
                height + (0.02 if height >= 0 else -0.05),
                f'{mcc[i]:.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved: {save_path}")
    
    plt.show()

def analyze_file_level_performance(file_predictions, class_names):
    """Analyze performance at file level"""
    logger.info("Analyzing file-level performance...")
    
    file_stats = []
    
    for file_path, predictions in file_predictions.items():
        if not predictions:
            continue
            
        # Get class from file path
        file_class = None
        for i, class_name in enumerate(class_names):
            if class_name in file_path:
                file_class = i
                break
        
        if file_class is None:
            continue
        
        # Calculate file-level statistics
        total_images = len(predictions)
        correct_predictions = sum(p['correct'] for p in predictions)
        accuracy = correct_predictions / total_images
        
        # Most common prediction
        pred_counts = defaultdict(int)
        for p in predictions:
            pred_counts[p['prediction']] += 1
        
        most_common_pred = max(pred_counts.keys(), key=lambda x: pred_counts[x])
        majority_vote_correct = (most_common_pred == file_class)
        
        # Average confidence for correct class
        avg_confidence = np.mean([p['probabilities'][file_class] for p in predictions])
        
        file_stats.append({
            'file_path': file_path,
            'true_class': file_class,
            'true_class_name': class_names[file_class],
            'total_images': total_images,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'most_common_pred': most_common_pred,
            'most_common_pred_name': class_names[most_common_pred],
            'majority_vote_correct': majority_vote_correct,
            'avg_confidence': avg_confidence
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(file_stats)
    
    # Overall file-level statistics
    logger.info("\n" + "="*50)
    logger.info("FILE-LEVEL ANALYSIS")
    logger.info("="*50)
    
    for class_idx, class_name in enumerate(class_names):
        class_files = df[df['true_class'] == class_idx]
        if len(class_files) == 0:
            continue
            
        avg_accuracy = class_files['accuracy'].mean()
        majority_vote_accuracy = class_files['majority_vote_correct'].mean()
        avg_confidence = class_files['avg_confidence'].mean()
        
        logger.info(f"\n{class_name.upper()} FILES:")
        logger.info(f"  Number of files: {len(class_files)}")
        logger.info(f"  Average image-level accuracy: {avg_accuracy:.4f}")
        logger.info(f"  Majority vote accuracy: {majority_vote_accuracy:.4f}")
        logger.info(f"  Average confidence: {avg_confidence:.4f}")
        
        # Find worst performing files
        worst_files = class_files.nsmallest(3, 'accuracy')
        logger.info(f"  Worst performing files:")
        for _, row in worst_files.iterrows():
            logger.info(f"    {Path(row['file_path']).name}: {row['accuracy']:.4f}")
    
    return df

def save_detailed_results(metrics, file_stats, class_names, output_dir):
    """Save detailed results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics as JSON
    metrics_to_save = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_to_save[key] = value.tolist()
        elif isinstance(value, dict):
            metrics_to_save[key] = {k: float(v) if isinstance(v, np.floating) else v 
                                  for k, v in value.items()}
        else:
            metrics_to_save[key] = float(value) if isinstance(value, np.floating) else value
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Save file-level statistics
    if file_stats is not None:
        file_stats.to_csv(output_dir / 'file_level_results.csv', index=False)
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                        index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / 'confusion_matrix.csv')
    
    logger.info(f"Detailed results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Test Deepfake Detection Model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--test_path', type=str, default='datasets/test',
                      help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='test_results',
                      help='Directory to save test results')
    parser.add_argument('--backbone', type=str, default='convnext_tiny',
                      choices=['convnext_tiny', 'convnext_small'],
                      help='ConvNeXt backbone architecture')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size')
    parser.add_argument('--save_plots', action='store_true',
                      help='Save plots to output directory')
    parser.add_argument('--no_file_analysis', action='store_true',
                      help='Skip file-level analysis')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TestConfig()
    config.TEST_PATH = args.test_path
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.CONVNEXT_BACKBONE = args.backbone
    config.IMAGE_SIZE = args.image_size
    
    logger.info("="*60)
    logger.info("DEEPFAKE DETECTION MODEL TESTING")
    logger.info("="*60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test path: {args.test_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Image size: {config.IMAGE_SIZE}")
    logger.info("="*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load model
        model = load_model(args.model_path, config)
        logger.info("Model loaded successfully")
        
        # Create test dataset and dataloader
        transform = get_test_transform(config)
        test_dataset = TestDataset(args.test_path, config, transform)
        
        if len(test_dataset) == 0:
            logger.error("No test data found!")
            return
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        
        logger.info(f"Test dataset created with {len(test_dataset)} samples")
        
        # Evaluate model
        predictions, targets, probabilities, file_predictions = evaluate_model(
            model, test_loader, config
        )
        
        # Calculate detailed metrics
        metrics = calculate_detailed_metrics(
            targets, predictions, probabilities, config.CLASS_NAMES
        )
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        logger.info(f"Multiclass MCC: {metrics['multiclass_mcc']:.4f}")
        logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
        logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")
        logger.info(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        logger.info(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
        logger.info(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
        logger.info(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        
        # Per-class results
        logger.info("\n" + "-"*50)
        logger.info("PER-CLASS RESULTS")
        logger.info("-"*50)
        
        for class_name in config.CLASS_NAMES:
            logger.info(f"\n{class_name.upper()}:")
            logger.info(f"  Precision: {metrics['per_class_precision'][class_name]:.4f}")
            logger.info(f"  Recall: {metrics['per_class_recall'][class_name]:.4f}")
            logger.info(f"  F1-Score: {metrics['per_class_f1'][class_name]:.4f}")
            logger.info(f"  MCC: {metrics['per_class_mcc'][class_name]:.4f}")
            logger.info(f"  Support: {metrics['per_class_support'][class_name]}")
        
        # Confusion Matrix
        logger.info("\n" + "-"*50)
        logger.info("CONFUSION MATRIX")
        logger.info("-"*50)
        logger.info("Rows: True labels, Columns: Predicted labels")
        logger.info(f"Classes: {config.CLASS_NAMES}")
        
        cm = metrics['confusion_matrix']
        # Print confusion matrix with class names
        print("\n" + " "*12 + "".join(f"{name:>12}" for name in config.CLASS_NAMES))
        for i, true_class in enumerate(config.CLASS_NAMES):
            print(f"{true_class:>10}  " + "".join(f"{cm[i,j]:>12}" for j in range(len(config.CLASS_NAMES))))
        
        # Calculate and display percentage matrix
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        logger.info("\nConfusion Matrix (Percentages):")
        print("\n" + " "*12 + "".join(f"{name:>12}" for name in config.CLASS_NAMES))
        for i, true_class in enumerate(config.CLASS_NAMES):
            print(f"{true_class:>10}  " + "".join(f"{cm_percent[i,j]:>10.1f}%" for j in range(len(config.CLASS_NAMES))))
        
        # Classification report
        logger.info("\n" + "-"*50)
        logger.info("DETAILED CLASSIFICATION REPORT")
        logger.info("-"*50)
        from sklearn.metrics import classification_report
        report = classification_report(
            targets, predictions, 
            target_names=config.CLASS_NAMES,
            digits=4
        )
        logger.info(f"\n{report}")
        
        # File-level analysis
        file_stats = None
        if not args.no_file_analysis:
            file_stats = analyze_file_level_performance(file_predictions, config.CLASS_NAMES)
        
        # Save detailed results
        save_detailed_results(metrics, file_stats, config.CLASS_NAMES, args.output_dir)
        
        # Generate and save plots
        if args.save_plots:
            logger.info("Generating plots...")
            
            # Confusion matrix plot
            cm_plot_path = output_dir / 'confusion_matrix.png'
            plot_confusion_matrix(metrics['confusion_matrix'], config.CLASS_NAMES, cm_plot_path)
            
            # Per-class metrics plot
            metrics_plot_path = output_dir / 'per_class_metrics.png'
            plot_per_class_metrics(metrics, config.CLASS_NAMES, metrics_plot_path)
            
            logger.info("Plots saved successfully")
        
        # Summary of problematic cases
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        # Find most confused classes
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        max_confusion = 0
        most_confused_pair = None
        
        for i in range(len(config.CLASS_NAMES)):
            for j in range(len(config.CLASS_NAMES)):
                if i != j and cm_normalized[i, j] > max_confusion:
                    max_confusion = cm_normalized[i, j]
                    most_confused_pair = (i, j)
        
        if most_confused_pair:
            true_class, pred_class = most_confused_pair
            logger.info(f"Most confused classes: {config.CLASS_NAMES[true_class]} -> {config.CLASS_NAMES[pred_class]}")
            logger.info(f"Confusion rate: {max_confusion:.1%}")
        
        # Identify best and worst performing classes
        per_class_f1_values = [metrics['per_class_f1'][cls] for cls in config.CLASS_NAMES]
        best_class_idx = np.argmax(per_class_f1_values)
        worst_class_idx = np.argmin(per_class_f1_values)
        
        logger.info(f"Best performing class: {config.CLASS_NAMES[best_class_idx]} (F1: {per_class_f1_values[best_class_idx]:.4f})")
        logger.info(f"Worst performing class: {config.CLASS_NAMES[worst_class_idx]} (F1: {per_class_f1_values[worst_class_idx]:.4f})")
        
        # Semi-synthetic specific analysis
        if 'semi-synthetic' in config.CLASS_NAMES:
            semi_idx = config.CLASS_NAMES.index('semi-synthetic')
            logger.info(f"\nSEMI-SYNTHETIC CLASS ANALYSIS:")
            logger.info(f"  Precision: {metrics['per_class_precision']['semi-synthetic']:.4f}")
            logger.info(f"  Recall: {metrics['per_class_recall']['semi-synthetic']:.4f}")
            logger.info(f"  F1-Score: {metrics['per_class_f1']['semi-synthetic']:.4f}")
            logger.info(f"  MCC: {metrics['per_class_mcc']['semi-synthetic']:.4f}")
            
            # Check what semi-synthetic is most confused with
            semi_row = cm[semi_idx]
            semi_row_normalized = semi_row.astype('float') / semi_row.sum()
            
            most_confused_with_idx = np.argmax(semi_row_normalized[np.arange(len(semi_row)) != semi_idx])
            if most_confused_with_idx >= semi_idx:
                most_confused_with_idx += 1
            
            logger.info(f"  Most confused with: {config.CLASS_NAMES[most_confused_with_idx]} ({semi_row_normalized[most_confused_with_idx]:.1%})")
        
        logger.info("\n" + "="*60)
        logger.info("TESTING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Final summary for easy copying
        logger.info(f"\nQUICK SUMMARY:")
        logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        logger.info(f"Multiclass MCC: {metrics['multiclass_mcc']:.4f}")
        logger.info(f"Per-class MCC: {', '.join([f'{cls}: {metrics['per_class_mcc'][cls]:.4f}' for cls in config.CLASS_NAMES])}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()