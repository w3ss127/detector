import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import timm
import numpy as np
from pathlib import Path
import json
import argparse
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

warnings.filterwarnings('ignore')

class SuperiorConfig:
    """Configuration class matching the training script"""
    def __init__(self):
        self.MODEL_TYPE = "superior_forensics_model"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        
        # Binary configuration settings
        self.USE_BINARY_MODE = True
        self.ORDINAL_REGRESSION = True
        self.NUM_CLASSES = 3
        self.REALNESS_SCORE_MODE = True
        
        self.HIDDEN_DIM = 1024
        self.DROPOUT_RATE = 0.3
        self.FREEZE_BACKBONES = True
        self.ATTENTION_DROPOUT = 0.2
        self.USE_SPECTRAL_NORM = False
        
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["synthetic", "semi-synthetic", "real"]  # Fixed order to match training
        self.USE_FORENSICS_MODULE = True
        self.USE_UNCERTAINTY_ESTIMATION = False
        self.TEMPERATURE = 1.0
        
        # Additional parameters from training script
        self.ORDINAL_WEIGHT = 0.7
        self.BINARY_WEIGHT = 0.3
        self.CUTPOINT_REGULARIZATION = 0.01

class ForensicsAwareModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Simplified and memory-efficient forensics module
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=8),  # Reduced channels
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Reduced channels
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # Smaller output
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),  # Reduced size
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.noise_analyzer = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),   # Reduced channels
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Reduced channels
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Smaller output
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),   # Reduced size
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.forensics_fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),  # Simplified fusion
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64)  # Final output size
        )
    
    def forward(self, x):
        dct_feats = self.dct_analyzer(x)
        noise_feats = self.noise_analyzer(x)
        combined_feats = torch.cat([dct_feats, noise_feats], dim=1)
        forensics_output = self.forensics_fusion(combined_feats)
        return forensics_output

class SuperiorAttentionModule(nn.Module):
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        
        # Simplified attention module
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 8),  # Reduced bottleneck
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() != 2:
            x = x.view(batch_size, -1)
        
        channel_weights = self.channel_attention(x)
        attended_features = x * channel_weights
        return attended_features

class SuperiorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize backbones with memory optimization
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        
        # Use smaller ViT model for memory efficiency
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        
        # Calculate feature dimensions
        convnext_features = 768
        vit_features = self.vit.num_features
        forensics_features = 64 if config.USE_FORENSICS_MODULE else 0
        total_features = convnext_features + vit_features + forensics_features
        
        # Initialize modules
        if config.USE_FORENSICS_MODULE:
            self.forensics_module = ForensicsAwareModule(config)
        
        self.attention_module = SuperiorAttentionModule(total_features, config)
        
        # Simplified fusion network
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
        
        # Traditional classifier
        self.classifier = nn.Linear(config.HIDDEN_DIM // 4, config.NUM_CLASSES)
        
        # Ordinal regression components
        if config.ORDINAL_REGRESSION:
            self.realness_predictor = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM // 4, config.HIDDEN_DIM // 8),
                nn.GELU(),
                nn.Dropout(config.DROPOUT_RATE * 0.5),
                nn.Linear(config.HIDDEN_DIM // 8, 1)
            )
            
            # Initialize cutpoints
            initial_cutpoints = torch.linspace(-1.0, 1.0, config.NUM_CLASSES - 1)
            self.cutpoints = nn.Parameter(initial_cutpoints.clone())
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features from backbones
        convnext_features = self.convnext.features(x)
        convnext_features = self.convnext.avgpool(convnext_features).flatten(1)
        
        vit_features = self.vit(x)
        
        # Combine features
        features_list = [convnext_features, vit_features]
        
        # Add forensics features if enabled
        if self.config.USE_FORENSICS_MODULE:
            forensics_features = self.forensics_module(x)
            features_list.append(forensics_features)
        
        # Concatenate all features
        combined_features = torch.cat(features_list, dim=1)
        
        # Apply attention
        attended_features = self.attention_module(combined_features)
        
        # Feature fusion
        fused_features = self.fusion(attended_features)
        
        # Traditional classification
        logits = self.classifier(fused_features)
        
        outputs = [logits]
        
        # Ordinal regression if enabled
        if self.config.ORDINAL_REGRESSION:
            realness_scores = self.realness_predictor(fused_features)
            outputs.extend([realness_scores, self.cutpoints])
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

def get_test_transforms(config):
    """Get test transforms - matches training validation transforms exactly"""
    transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return transform

def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint - Fixed for PyTorch 2.6+ and better error handling"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model = SuperiorModel(config).to(device)
    
    try:
        # First try with weights_only=True (safer)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print("Loaded checkpoint with weights_only=True")
    except Exception as e:
        print(f"Failed to load with weights_only=True: {e}")
        print("Trying to load with weights_only=False")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            print("Loaded checkpoint with weights_only=False")
        except Exception as e2:
            print(f"Failed to load checkpoint: {e2}")
            raise e2
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Found model_state_dict in checkpoint")
        
        # Extract config from checkpoint if available
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            print("Found config in checkpoint, updating current config")
            for key, value in checkpoint_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found state_dict in checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint as state_dict directly")
    
    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    module_prefix_found = False
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            module_prefix_found = True
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    if module_prefix_found:
        print("Removed 'module.' prefix from state dict keys (DDP model)")
    
    # Load state dict with better error reporting
    try:
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            # Check if missing keys are critical
            critical_missing = [k for k in missing_keys if not k.startswith('cutpoints')]
            if critical_missing:
                print(f"Critical missing keys found: {critical_missing}")
                
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Model architecture might not match the checkpoint")
        raise e
    
    model.eval()
    print("Model loaded successfully!")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    return model

def process_pt_file(pt_file_path, class_name, class_idx, model, config, device, transform):
    """Process a single .pt file with improved error handling and memory management"""
    try:
        print(f"Processing {Path(pt_file_path).name}")
        
        # Load tensor file with better error handling
        try:
            tensor_data = torch.load(pt_file_path, map_location='cpu', weights_only=True)
        except:
            try:
                tensor_data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Error loading tensor file {pt_file_path}: {e}")
                return []
        
        # Extract images from tensor data with better validation
        images = None
        if isinstance(tensor_data, torch.Tensor):
            if tensor_data.dim() == 4 and tensor_data.size(1) in [1, 3]:  # [N, C, H, W]
                images = tensor_data
            elif tensor_data.dim() == 4 and tensor_data.size(3) in [1, 3]:  # [N, H, W, C]
                images = tensor_data.permute(0, 3, 1, 2)
            else:
                print(f"Unexpected tensor dimensions: {tensor_data.shape}")
                return []
                
        elif isinstance(tensor_data, (list, tuple)):
            try:
                # Validate that all items are tensors
                valid_tensors = []
                for item in tensor_data:
                    if isinstance(item, torch.Tensor):
                        if item.dim() == 3:  # [C, H, W] or [H, W, C]
                            if item.shape[0] in [1, 3]:
                                valid_tensors.append(item)
                            elif item.shape[2] in [1, 3]:
                                valid_tensors.append(item.permute(2, 0, 1))
                        elif item.dim() == 2:  # [H, W] grayscale
                            valid_tensors.append(item.unsqueeze(0).repeat(3, 1, 1))
                
                if valid_tensors:
                    images = torch.stack(valid_tensors)
                else:
                    print(f"No valid tensors found in list/tuple")
                    return []
                    
            except Exception as e:
                print(f"Error stacking images: {e}")
                return []
                
        elif isinstance(tensor_data, dict):
            # Try common dictionary keys
            for key in ['images', 'data', 'tensors', 'samples']:
                if key in tensor_data:
                    data = tensor_data[key]
                    if isinstance(data, torch.Tensor):
                        if data.dim() == 4:
                            images = data
                        elif data.dim() == 3:
                            images = data.unsqueeze(0)
                    elif isinstance(data, (list, tuple)):
                        try:
                            images = torch.stack([img for img in data if isinstance(img, torch.Tensor)])
                        except:
                            continue
                    break
            
            if images is None:
                print(f"Could not find image data in dict with keys: {list(tensor_data.keys())}")
                return []
        else:
            print(f"Unknown tensor format: {type(tensor_data)}")
            return []
        
        if images is None:
            print(f"Failed to extract images from {pt_file_path}")
            return []
            
        print(f"Found {images.size(0)} images in {Path(pt_file_path).name}")
        
        results = []
        batch_size = 8  # Conservative batch size for testing
        
        # Process images in batches
        for i in range(0, images.size(0), batch_size):
            batch_end = min(i + batch_size, images.size(0))
            batch_images = images[i:batch_end]
            
            # Process batch
            processed_batch = []
            for img_idx, img in enumerate(batch_images):
                try:
                    # Ensure correct format
                    if img.dtype != torch.float32:
                        img = img.float()
                    
                    # Handle different value ranges
                    if img.max() > 2.0:  # Likely 0-255 range
                        img = img / 255.0
                    elif img.max() > 1.0 and img.max() <= 2.0:  # Possibly normalized differently
                        img = torch.clamp(img, 0, 1)
                    
                    # Ensure correct shape [C, H, W]
                    if img.dim() == 3:
                        if img.shape[0] not in [1, 3]:
                            if img.shape[-1] in [1, 3]:
                                img = img.permute(2, 0, 1)
                    elif img.dim() == 2:
                        img = img.unsqueeze(0)
                    
                    # Convert grayscale to RGB if needed
                    if img.shape[0] == 1:
                        img = img.repeat(3, 1, 1)
                    
                    # Ensure we have exactly 3 channels
                    if img.shape[0] != 3:
                        print(f"Warning: Image has {img.shape[0]} channels, expected 3")
                        continue
                    
                    # Resize if needed
                    if img.shape[-2:] != (config.IMAGE_SIZE, config.IMAGE_SIZE):
                        img = F.interpolate(img.unsqueeze(0), 
                                          size=(config.IMAGE_SIZE, config.IMAGE_SIZE), 
                                          mode='bilinear', align_corners=False).squeeze(0)
                    
                    # Apply transforms
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                    
                    try:
                        transformed = transform(image=img_np)
                        processed_batch.append(transformed['image'])
                    except Exception as transform_error:
                        print(f"Transform error for image {i + img_idx}: {transform_error}")
                        continue
                    
                except Exception as e:
                    print(f"Error processing individual image {i + img_idx}: {e}")
                    continue
            
            # Skip empty batches
            if not processed_batch:
                continue
            
            # Stack and predict
            try:
                batch_tensor = torch.stack(processed_batch).to(device)
                
                with torch.no_grad():
                    output = model(batch_tensor)
                    
                    if isinstance(output, tuple) and len(output) >= 3:
                        logits, realness_scores, cutpoints = output[:3]
                        
                        # Ordinal predictions using same logic as training
                        cutpoints_sorted = torch.sort(cutpoints)[0]
                        scores_expanded = realness_scores.expand(-1, len(cutpoints_sorted))
                        cutpoints_expanded = cutpoints_sorted.unsqueeze(0).expand(len(realness_scores), -1)
                        ordinal_probs = torch.sigmoid((scores_expanded - cutpoints_expanded) / config.TEMPERATURE)
                        
                        class_probs = torch.zeros(len(realness_scores), config.NUM_CLASSES, device=device)
                        class_probs[:, 0] = 1 - ordinal_probs[:, 0]
                        for k in range(1, config.NUM_CLASSES - 1):
                            class_probs[:, k] = ordinal_probs[:, k-1] - ordinal_probs[:, k]
                        class_probs[:, -1] = ordinal_probs[:, -1]
                        
                        class_probs = torch.clamp(class_probs, min=1e-7, max=1-1e-7)
                        
                        predictions = torch.argmax(class_probs, dim=1)
                        confidences = torch.max(class_probs, dim=1)[0]
                        realness_batch = torch.sigmoid(realness_scores.squeeze())
                        
                        # Also get traditional classifier predictions for comparison
                        traditional_probs = F.softmax(logits, dim=1)
                        traditional_preds = torch.argmax(traditional_probs, dim=1)
                        
                    else:
                        logits = output if not isinstance(output, tuple) else output[0]
                        probs = F.softmax(logits, dim=1)
                        predictions = torch.argmax(probs, dim=1)
                        confidences = torch.max(probs, dim=1)[0]
                        realness_batch = None
                        traditional_preds = predictions
                
                # Store results (move to CPU to free GPU memory)
                for j, (pred, conf) in enumerate(zip(predictions.cpu().numpy(), confidences.cpu().numpy())):
                    realness = None
                    if realness_batch is not None:
                        if realness_batch.dim() == 0:
                            realness = realness_batch.item()
                        else:
                            realness = realness_batch[j].item()
                    
                    trad_pred = traditional_preds[j].item() if traditional_preds is not None else pred
                    
                    results.append({
                        'file': str(pt_file_path),
                        'image_idx': i + j,
                        'true_class': class_name,
                        'true_label': class_idx,
                        'predicted_label': int(pred),
                        'predicted_class': config.CLASS_NAMES[pred],
                        'traditional_prediction': int(trad_pred),
                        'confidence': float(conf),
                        'realness_score': realness,
                        'correct': int(pred) == class_idx
                    })
                
                # Clean up GPU memory
                del batch_tensor, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error during batch prediction: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up memory
        del tensor_data, images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Completed {Path(pt_file_path).name} - {len(results)} predictions")
        return results
        
    except Exception as e:
        print(f"Error processing {pt_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_model(test_path, config, checkpoint_path, device):
    """Test model on single GPU/CPU with improved error handling"""
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, config, device)
    transform = get_test_transforms(config)
    
    # Collect all .pt files
    test_path = Path(test_path)
    pt_files = []
    
    print(f"Scanning test directory: {test_path}")
    
    for class_name in config.CLASS_NAMES:
        class_path = test_path / class_name
        if class_path.exists():
            class_idx = config.CLASS_NAMES.index(class_name)
            pt_files_in_class = list(class_path.glob("*.pt"))
            print(f"Found {len(pt_files_in_class)} .pt files in {class_path}")
            
            for pt_file in pt_files_in_class:
                pt_files.append((str(pt_file), class_name, class_idx))
        else:
            print(f"Warning: Class directory not found: {class_path}")
    
    print(f"Total .pt files to process: {len(pt_files)}")
    
    if len(pt_files) == 0:
        print("No .pt files found!")
        print("Expected directory structure:")
        print("test_path/")
        for class_name in config.CLASS_NAMES:
            print(f"  {class_name}/")
            print(f"    *.pt")
        return None
    
    all_results = []
    
    for pt_file_path, class_name, class_idx in tqdm(pt_files, desc="Processing files"):
        results = process_pt_file(
            pt_file_path, class_name, class_idx, 
            model, config, device, transform
        )
        all_results.extend(results)
    
    print(f"Total predictions made: {len(all_results)}")
    
    if len(all_results) == 0:
        print("No predictions were made!")
        return None
    
    # Organize results
    predictions = [r['predicted_label'] for r in all_results]
    true_labels = [r['true_label'] for r in all_results]
    confidences = [r['confidence'] for r in all_results]
    realness_scores = [r['realness_score'] for r in all_results if r['realness_score'] is not None]
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'confidences': confidences,
        'realness_scores': realness_scores if realness_scores else None,
        'detailed_results': all_results
    }

def calculate_metrics(results, config):
    """Calculate comprehensive test metrics with better error handling"""
    predictions = np.array(results['predictions'])
    true_labels = np.array(results['true_labels'])
    confidences = np.array(results['confidences'])
    
    # Check if we have any predictions
    if len(predictions) == 0:
        print("No predictions to calculate metrics!")
        return None
    
    # Basic metrics
    accuracy = np.mean(predictions == true_labels)
    
    # Get unique labels present in the data
    unique_labels = np.unique(np.concatenate([true_labels, predictions]))
    labels_present = sorted(unique_labels.tolist())
    
    # Create label names for present classes only
    target_names_present = [config.CLASS_NAMES[i] for i in labels_present if i < len(config.CLASS_NAMES)]
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=labels_present)
    
    # Classification report with only present labels
    try:
        report = classification_report(true_labels, predictions, 
                                     labels=labels_present,
                                     target_names=target_names_present, 
                                     output_dict=True, zero_division=0)
    except Exception as e:
        print(f"Error generating classification report: {e}")
        report = {}
    
    # Matthews Correlation Coefficient
    try:
        mcc = matthews_corrcoef(true_labels, predictions)
    except Exception as e:
        print(f"Error calculating MCC: {e}")
        mcc = 0.0
    
    # Per-class metrics
    per_class_acc = []
    for i in range(config.NUM_CLASSES):
        class_mask = true_labels == i
        if class_mask.sum() > 0:
            class_acc = np.mean(predictions[class_mask] == i)
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Confidence statistics
    avg_confidence = np.mean(confidences)
    correct_mask = predictions == true_labels
    avg_confidence_correct = np.mean(confidences[correct_mask]) if correct_mask.sum() > 0 else 0
    avg_confidence_incorrect = np.mean(confidences[~correct_mask]) if (~correct_mask).sum() > 0 else 0
    
    # Ordinal-specific metrics
    ordinal_metrics = {}
    if results['realness_scores'] is not None and len(results['realness_scores']) > 0:
        realness_scores = np.array(results['realness_scores'])
        
        # MAE for ordinal predictions
        mae = np.mean(np.abs(true_labels - predictions))
        
        # Ordinal accuracy (within 1 class)
        ordinal_acc = np.mean(np.abs(true_labels - predictions) <= 1)
        
        # Spearman correlation between realness scores and true labels
        try:
            from scipy.stats import spearmanr
            spearman_corr, spearman_p = spearmanr(realness_scores, true_labels)
            ordinal_metrics['spearman_correlation'] = spearman_corr
            ordinal_metrics['spearman_p_value'] = spearman_p
        except ImportError:
            pass
        
        ordinal_metrics.update({
            'mae': mae,
            'ordinal_accuracy': ordinal_acc,
            'avg_realness_score': np.mean(realness_scores),
            'realness_score_std': np.std(realness_scores),
            'realness_score_range': [np.min(realness_scores), np.max(realness_scores)]
        })
    
    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_accuracy': per_class_acc,
        'avg_confidence': avg_confidence,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_incorrect': avg_confidence_incorrect,
        'macro_f1': report.get('macro avg', {}).get('f1-score', 0),
        'weighted_f1': report.get('weighted avg', {}).get('f1-score', 0),
        'labels_present': labels_present,
        'target_names_present': target_names_present,
        **ordinal_metrics
    }

def print_results(metrics, config, results):
    """Print comprehensive test results with better formatting"""
    if metrics is None:
        print("No metrics to display!")
        return
        
    print("\n" + "="*70)
    print("                       TEST RESULTS")
    print("="*70)
    
    # Dataset information
    print(f"\nDATASET INFORMATION:")
    print(f"  Total samples tested: {len(results['predictions'])}")
    
    # Count samples per class
    unique_labels, counts = np.unique(results['true_labels'], return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label < len(config.CLASS_NAMES):
            print(f"  {config.CLASS_NAMES[label]}: {count} samples")
    
    # Overall metrics
    print(f"\nOVERALL METRICS:")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Matthews Correlation Coefficient: {metrics['mcc']:.4f}")
    print(f"  Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    
    # Confidence metrics
    print(f"\nCONFIDENCE METRICS:")
    print(f"  Average Confidence: {metrics['avg_confidence']:.4f}")
    print(f"  Avg Confidence (Correct): {metrics['avg_confidence_correct']:.4f}")
    print(f"  Avg Confidence (Incorrect): {metrics['avg_confidence_incorrect']:.4f}")
    confidence_gap = metrics['avg_confidence_correct'] - metrics['avg_confidence_incorrect']
    print(f"  Confidence Gap: {confidence_gap:.4f}")
    
    # Ordinal-specific metrics
    if 'mae' in metrics:
        print(f"\nORDINAL REGRESSION METRICS:")
        print(f"  Mean Absolute Error: {metrics['mae']:.4f}")
        print(f"  Ordinal Accuracy (±1 class): {metrics['ordinal_accuracy']:.4f}")
        print(f"  Average Realness Score: {metrics['avg_realness_score']:.4f}")
        print(f"  Realness Score Std: {metrics['realness_score_std']:.4f}")
        if 'realness_score_range' in metrics:
            score_range = metrics['realness_score_range']
            print(f"  Realness Score Range: [{score_range[0]:.4f}, {score_range[1]:.4f}]")
        if 'spearman_correlation' in metrics:
            print(f"  Spearman Correlation: {metrics['spearman_correlation']:.4f}")
    
    # Per-class metrics
    print(f"\nPER-CLASS ACCURACY:")
    for i, (class_name, acc) in enumerate(zip(config.CLASS_NAMES, metrics['per_class_accuracy'])):
        sample_count = sum(1 for label in results['true_labels'] if label == i)
        print(f"  {class_name}: {acc:.4f} ({acc*100:.1f}%) - {sample_count} samples")
    
    # Detailed classification report
    if 'classification_report' in metrics and metrics['classification_report']:
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        report = metrics['classification_report']
        
        # Header
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        # Per-class metrics
        for i, class_name in enumerate(metrics['target_names_present']):
            if str(i) in report:
                class_metrics = report[str(i)]
                print(f"{class_name:<15} {class_metrics['precision']:<10.4f} "
                      f"{class_metrics['recall']:<10.4f} {class_metrics['f1-score']:<10.4f} "
                      f"{class_metrics['support']:<10}")
        
        # Average metrics
        print("-" * 60)
        if 'macro avg' in report:
            macro = report['macro avg']
            print(f"{'Macro Avg':<15} {macro['precision']:<10.4f} "
                  f"{macro['recall']:<10.4f} {macro['f1-score']:<10.4f} "
                  f"{macro['support']:<10}")
        
        if 'weighted avg' in report:
            weighted = report['weighted avg']
            print(f"{'Weighted Avg':<15} {weighted['precision']:<10.4f} "
                  f"{weighted['recall']:<10.4f} {weighted['f1-score']:<10.4f} "
                  f"{weighted['support']:<10}")
    
    # Confusion matrix
    print(f"\nCONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    labels_present = metrics['labels_present']
    target_names_present = metrics['target_names_present']
    
    if len(labels_present) > 0:
        # Print header
        print("    Predicted:")
        header = "Actual".ljust(12)
        for name in target_names_present:
            header += f"{name[:10]:>12}"
        print(header)
        
        # Print matrix rows
        for i, (true_class, row) in enumerate(zip(target_names_present, cm)):
            row_str = f"{true_class[:10]:>10}  "
            for val in row:
                row_str += f"{val:>12}"
            print(row_str)
    
    # Error analysis
    print(f"\nERROR ANALYSIS:")
    error_count = sum(1 for r in results['detailed_results'] if not r['correct'])
    print(f"  Total errors: {error_count}/{len(results['detailed_results'])} ({error_count/len(results['detailed_results'])*100:.1f}%)")
    
    # Most common errors
    error_types = defaultdict(int)
    for r in results['detailed_results']:
        if not r['correct']:
            error_key = f"{r['true_class']} → {r['predicted_class']}"
            error_types[error_key] += 1
    
    if error_types:
        print(f"  Most common error types:")
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:5]:  # Top 5 errors
            percentage = count / error_count * 100
            print(f"    {error_type}: {count} ({percentage:.1f}% of errors)")
    
    print("="*70)

def save_results(results, metrics, config, output_dir):
    """Save detailed results to files with improved organization"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    detailed_results_path = output_path / "detailed_results.json"
    with open(detailed_results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'predictions': [int(x) for x in results['predictions']],
            'true_labels': [int(x) for x in results['true_labels']],
            'confidences': [float(x) for x in results['confidences']],
            'realness_scores': [float(x) for x in results['realness_scores']] if results['realness_scores'] else None,
            'detailed_results': results['detailed_results'],
            'config': {
                'class_names': config.CLASS_NAMES,
                'num_classes': config.NUM_CLASSES,
                'ordinal_regression': config.ORDINAL_REGRESSION,
                'use_forensics_module': config.USE_FORENSICS_MODULE
            }
        }
        json.dump(json_results, f, indent=2)
    
    # Save metrics
    if metrics is not None:
        metrics_path = output_path / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                elif isinstance(value, (np.float64, np.float32)):
                    json_metrics[key] = float(value)
                elif isinstance(value, (np.int64, np.int32)):
                    json_metrics[key] = int(value)
                else:
                    json_metrics[key] = value
            json.dump(json_metrics, f, indent=2, default=str)
        
        # Save confusion matrix plot
        if 'confusion_matrix' in metrics and len(metrics['labels_present']) > 0:
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics['confusion_matrix'], 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=metrics['target_names_present'],
                        yticklabels=metrics['target_names_present'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save confidence distribution plot
        if len(results['confidences']) > 0:
            plt.figure(figsize=(12, 4))
            
            # Overall confidence distribution
            plt.subplot(1, 2, 1)
            plt.hist(results['confidences'], bins=50, alpha=0.7, edgecolor='black')
            plt.title('Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Confidence by correctness
            plt.subplot(1, 2, 2)
            correct_confidences = [r['confidence'] for r in results['detailed_results'] if r['correct']]
            incorrect_confidences = [r['confidence'] for r in results['detailed_results'] if not r['correct']]
            
            plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
            plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
            plt.title('Confidence Distribution by Correctness')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "confidence_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save realness score analysis if available
        if results['realness_scores'] is not None:
            plt.figure(figsize=(12, 4))
            
            # Realness score distribution
            plt.subplot(1, 2, 1)
            plt.hist(results['realness_scores'], bins=50, alpha=0.7, edgecolor='black')
            plt.title('Realness Score Distribution')
            plt.xlabel('Realness Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Realness score by true class
            plt.subplot(1, 2, 2)
            realness_by_class = defaultdict(list)
            for r in results['detailed_results']:
                if r['realness_score'] is not None:
                    realness_by_class[r['true_class']].append(r['realness_score'])
            
            positions = []
            labels = []
            data = []
            for i, class_name in enumerate(config.CLASS_NAMES):
                if class_name in realness_by_class and realness_by_class[class_name]:
                    positions.append(i)
                    labels.append(class_name)
                    data.append(realness_by_class[class_name])
            
            if data:
                plt.boxplot(data, positions=positions, labels=labels)
                plt.title('Realness Score by True Class')
                plt.ylabel('Realness Score')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "realness_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Save summary report
    summary_path = output_path / "test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("FORENSICS MODEL TEST SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(results['predictions'])}\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  - Classes: {config.CLASS_NAMES}\n")
        f.write(f"  - Ordinal Regression: {config.ORDINAL_REGRESSION}\n")
        f.write(f"  - Forensics Module: {config.USE_FORENSICS_MODULE}\n\n")
        
        if metrics:
            f.write(f"Key Metrics:\n")
            f.write(f"  - Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  - MCC: {metrics['mcc']:.4f}\n")
            f.write(f"  - Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"  - Weighted F1: {metrics['weighted_f1']:.4f}\n")
            
            if 'mae' in metrics:
                f.write(f"  - MAE: {metrics['mae']:.4f}\n")
                f.write(f"  - Ordinal Accuracy: {metrics['ordinal_accuracy']:.4f}\n")
    
    print(f"\nResults saved to: {output_path}")
    print(f"  - detailed_results.json: Individual predictions with metadata")
    if metrics is not None:
        print(f"  - test_metrics.json: Comprehensive metrics")
        print(f"  - confusion_matrix.png: Confusion matrix visualization")
        print(f"  - confidence_analysis.png: Confidence score analysis")
        if results['realness_scores'] is not None:
            print(f"  - realness_analysis.png: Realness score analysis")
        print(f"  - test_summary.txt: Summary report")

def load_config_from_checkpoint(checkpoint_path):
    """Try to load config from checkpoint with better error handling"""
    try:
        # Try loading with weights_only=True first
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = SuperiorConfig()
            
            # Update config with checkpoint values
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
            print("Successfully loaded config from checkpoint")
            return config
            
    except Exception as e:
        print(f"Could not load config from checkpoint: {e}")
    
    print("Using default config")
    return SuperiorConfig()

def validate_test_environment(test_path, config):
    """Validate the test environment and data structure"""
    test_path = Path(test_path)
    
    if not test_path.exists():
        print(f"Error: Test path does not exist: {test_path}")
        return False
    
    print(f"Validating test environment at: {test_path}")
    
    # Check class directories
    valid_classes = 0
    total_files = 0
    
    for class_name in config.CLASS_NAMES:
        class_path = test_path / class_name
        if class_path.exists() and class_path.is_dir():
            pt_files = list(class_path.glob("*.pt"))
            if pt_files:
                valid_classes += 1
                total_files += len(pt_files)
                print(f"  ✓ {class_name}: {len(pt_files)} .pt files")
            else:
                print(f"  ✗ {class_name}: No .pt files found")
        else:
            print(f"  ✗ {class_name}: Directory not found")
    
    if valid_classes == 0:
        print("\nError: No valid class directories with .pt files found!")
        print("\nExpected directory structure:")
        print(f"{test_path}/")
        for class_name in config.CLASS_NAMES:
            print(f"  {class_name}/")
            print(f"    *.pt")
        return False
    
    print(f"\nValidation successful: {valid_classes}/{len(config.CLASS_NAMES)} classes, {total_files} total files")
    return True

def main():
    parser = argparse.ArgumentParser(description='Test Superior Forensics Model')
    parser.add_argument('--test_path', type=str, default="datasets/test",
                       help='Path to test data directory containing class subdirectories with .pt files')
    parser.add_argument('--checkpoint', type=str, default="ordinal_checkpoints/best_model.pt",
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save test results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Optional config file path (JSON format)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference (default: 8 for memory safety)')
    parser.add_argument('--save_errors', action='store_true',
                       help='Save detailed error analysis')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Load configuration
    if args.config_file and Path(args.config_file).exists():
        print(f"Loading config from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = SuperiorConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        # Try to load config from checkpoint, otherwise use default
        config = load_config_from_checkpoint(args.checkpoint)
    
    print(f"\nModel Configuration:")
    print(f"  Model Type: {config.MODEL_TYPE}")
    print(f"  ConvNeXt Backbone: {config.CONVNEXT_BACKBONE}")
    print(f"  Number of Classes: {config.NUM_CLASSES}")
    print(f"  Class Names: {config.CLASS_NAMES}")
    print(f"  Use Ordinal Regression: {config.ORDINAL_REGRESSION}")
    print(f"  Use Forensics Module: {config.USE_FORENSICS_MODULE}")
    print(f"  Image Size: {config.IMAGE_SIZE}")
    print(f"  Temperature: {config.TEMPERATURE}")
    
    # Validate inputs
    if not validate_test_environment(args.test_path, config):
        return
    
    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint does not exist: {checkpoint_path}")
        return
    
    print(f"\nStarting model testing...")
    print(f"Test data: {args.test_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    
    import time
    start_time = time.time()
    
    try:
        # Run testing
        results = test_model(args.test_path, config, args.checkpoint, device)
        
        if results is None:
            print("Testing failed - no results obtained")
            return
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = calculate_metrics(results, config)
        
        # Print results
        print_results(metrics, config, results)
        
        # Save results
        print(f"\nSaving results...")
        save_results(results, metrics, config, args.output_dir)
        
        # Additional summary
        test_time = time.time() - start_time
        print(f"\n" + "="*70)
        print("TESTING COMPLETED SUCCESSFULLY")
        print(f"Total time: {test_time:.2f} seconds")
        print(f"Total images processed: {len(results['predictions'])}")
        print(f"Processing speed: {len(results['predictions'])/test_time:.1f} images/second")
        print(f"Overall accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        if 'mae' in metrics:
            print(f"Mean Absolute Error: {metrics['mae']:.4f}")
            print(f"Ordinal Accuracy (±1): {metrics['ordinal_accuracy']:.4f}")
        print(f"Results saved to: {Path(args.output_dir).absolute()}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        return
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return

def create_sample_config():
    """Create a sample configuration file"""
    config = SuperiorConfig()
    config_dict = {
        'MODEL_TYPE': config.MODEL_TYPE,
        'CONVNEXT_BACKBONE': config.CONVNEXT_BACKBONE,
        'PRETRAINED_WEIGHTS': config.PRETRAINED_WEIGHTS,
        'USE_BINARY_MODE': config.USE_BINARY_MODE,
        'ORDINAL_REGRESSION': config.ORDINAL_REGRESSION,
        'NUM_CLASSES': config.NUM_CLASSES,
        'REALNESS_SCORE_MODE': config.REALNESS_SCORE_MODE,
        'HIDDEN_DIM': config.HIDDEN_DIM,
        'DROPOUT_RATE': config.DROPOUT_RATE,
        'FREEZE_BACKBONES': config.FREEZE_BACKBONES,
        'ATTENTION_DROPOUT': config.ATTENTION_DROPOUT,
        'USE_SPECTRAL_NORM': config.USE_SPECTRAL_NORM,
        'IMAGE_SIZE': config.IMAGE_SIZE,
        'CLASS_NAMES': config.CLASS_NAMES,
        'USE_FORENSICS_MODULE': config.USE_FORENSICS_MODULE,
        'USE_UNCERTAINTY_ESTIMATION': config.USE_UNCERTAINTY_ESTIMATION,
        'TEMPERATURE': config.TEMPERATURE,
        'ORDINAL_WEIGHT': config.ORDINAL_WEIGHT,
        'BINARY_WEIGHT': config.BINARY_WEIGHT,
        'CUTPOINT_REGULARIZATION': config.CUTPOINT_REGULARIZATION
    }
    
    with open('sample_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("Sample configuration saved to 'sample_config.json'")

if __name__ == "__main__":
    # Uncomment the line below to create a sample config file
    # create_sample_config()
    
    main()