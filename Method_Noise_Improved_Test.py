#!/usr/bin/env python3
"""
Neural Network Deepfake Detection - Test Program
Loads PT files from test dataset and evaluates using trained models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model classes (copy from training script)
class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature recalibration"""
    
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention"""
    
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EnhancedDenosingAutoencoder(nn.Module):
    """Enhanced autoencoder with attention mechanisms and skip connections"""
    
    def __init__(self, channels=3, use_attention=True):
        super(EnhancedDenosingAutoencoder, self).__init__()
        self.use_attention = use_attention
        
        # Encoder with attention
        self.enc_conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        
        self.enc_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(128)
        
        self.enc_conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc_bn5 = nn.BatchNorm2d(256)
        self.enc_conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.enc_bn6 = nn.BatchNorm2d(256)
        
        # Attention modules
        if self.use_attention:
            self.attention1 = CBAM(64)
            self.attention2 = CBAM(128)
            self.attention3 = CBAM(256)
        
        # Decoder with skip connections
        self.dec_conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        
        self.dec_conv3 = nn.Conv2d(256, 128, 3, padding=1)  # 256 due to skip connection
        self.dec_bn3 = nn.BatchNorm2d(128)
        self.dec_conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(64)
        
        self.dec_conv5 = nn.Conv2d(128, 32, 3, padding=1)  # 128 due to skip connection
        self.dec_bn5 = nn.BatchNorm2d(32)
        self.dec_conv6 = nn.Conv2d(32, channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.maxpool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc_bn1(self.enc_conv1(x)))
        e2 = self.relu(self.enc_bn2(self.enc_conv2(e1)))
        if self.use_attention:
            e2 = self.attention1(e2)
        e2_pool = self.maxpool(e2)
        
        e3 = self.relu(self.enc_bn3(self.enc_conv3(e2_pool)))
        e4 = self.relu(self.enc_bn4(self.enc_conv4(e3)))
        if self.use_attention:
            e4 = self.attention2(e4)
        e4_pool = self.maxpool(e4)
        
        e5 = self.relu(self.enc_bn5(self.enc_conv5(e4_pool)))
        e6 = self.relu(self.enc_bn6(self.enc_conv6(e5)))
        if self.use_attention:
            e6 = self.attention3(e6)
        
        # Decoder with skip connections
        d1 = self.relu(self.dec_bn1(self.dec_conv1(e6)))
        d2 = self.relu(self.dec_bn2(self.dec_conv2(d1)))
        d2_up = self.upsample(d2)
        
        # Skip connection from e4
        d3 = torch.cat([d2_up, e4], dim=1)
        d3 = self.relu(self.dec_bn3(self.dec_conv3(d3)))
        d4 = self.relu(self.dec_bn4(self.dec_conv4(d3)))
        d4_up = self.upsample(d4)
        
        # Skip connection from e2
        d5 = torch.cat([d4_up, e2], dim=1)
        d5 = self.relu(self.dec_bn5(self.dec_conv5(d5)))
        d6 = self.sigmoid(self.dec_conv6(d5))
        
        return d6

class AdvancedNoiseFeatureExtractor(nn.Module):
    """Neural network for extracting noise features from reconstructed images"""
    
    def __init__(self, input_channels=3, feature_dim=512):
        super(AdvancedNoiseFeatureExtractor, self).__init__()
        
        # Convolutional feature extractor
        self.conv_features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Statistical feature extractor
        self.stat_features = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism for feature selection
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, original, reconstructed):
        # Calculate noise
        noise = original - reconstructed
        
        # Extract convolutional features
        conv_feat = self.conv_features(noise)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)
        
        # Extract statistical features
        stat_feat = self.stat_features(conv_feat)
        
        # Apply attention
        attention_weights = self.attention(stat_feat)
        attended_features = stat_feat * attention_weights
        
        return attended_features

class NeuralNoiseClassifier(nn.Module):
    """Neural network classifier for three-class noise classification"""
    
    def __init__(self, input_dim=512, num_classes=3, hidden_dims=[256, 128, 64]):
        super(NeuralNoiseClassifier, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)

class TensorDataset(Dataset):
    """Custom dataset for tensor data"""
    def __init__(self, tensors: torch.Tensor):
        self.tensors = tensors
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx]

class DeepfakeTestEvaluator:
    """Comprehensive test evaluator for the deepfake detection system"""
    
    def __init__(self, model_dir='models', device=None):
        self.model_dir = model_dir
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['real', 'semi-synthetic', 'synthetic']
        
        # Initialize models
        self.autoencoder = EnhancedDenosingAutoencoder().to(self.device)
        self.feature_extractor = AdvancedNoiseFeatureExtractor().to(self.device)
        self.classifier = NeuralNoiseClassifier().to(self.device)
        self.scaler = None
        
        # Load trained models
        self.load_models()
        
        # Set models to evaluation mode
        self.autoencoder.eval()
        self.feature_extractor.eval()
        self.classifier.eval()
        
        logger.info(f"Test evaluator initialized on device: {self.device}")
    
    def load_models(self):
        """Load all trained models"""
        # Define model paths
        autoencoder_path = os.path.join(self.model_dir, 'autoencoder.pth')
        feature_extractor_path = os.path.join(self.model_dir, 'feature_extractor.pth')
        classifier_path = os.path.join(self.model_dir, 'classifier.pth')
        scaler_path = os.path.join(self.model_dir, 'scaler.pth')
        checkpoint_path = os.path.join(self.model_dir, 'checkpoint.pth')
        
        # Try to load complete pipeline first
        if os.path.exists(checkpoint_path):
            logger.info("Loading complete pipeline...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.autoencoder.load_state_dict(checkpoint['autoencoder_state'])
                self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state'])
                self.classifier.load_state_dict(checkpoint['classifier_state'])
                
                if 'scaler' in checkpoint:
                    self.scaler = checkpoint['scaler']
                
                logger.info("✅ Complete pipeline loaded successfully!")
                return True
            except Exception as e:
                logger.warning(f"Failed to load complete pipeline: {str(e)}")
        
        # Load individual models
        models_loaded = 0
        
        # Load autoencoder
        if os.path.exists(autoencoder_path):
            try:
                checkpoint = torch.load(autoencoder_path, map_location=self.device)
                self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Autoencoder loaded")
                models_loaded += 1
            except Exception as e:
                logger.error(f"Failed to load autoencoder: {str(e)}")
        
        # Load feature extractor
        if os.path.exists(feature_extractor_path):
            try:
                checkpoint = torch.load(feature_extractor_path, map_location=self.device)
                self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Feature extractor loaded")
                models_loaded += 1
            except Exception as e:
                logger.error(f"Failed to load feature extractor: {str(e)}")
        
        # Load classifier
        if os.path.exists(classifier_path):
            try:
                checkpoint = torch.load(classifier_path, map_location=self.device)
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Classifier loaded")
                models_loaded += 1
            except Exception as e:
                logger.error(f"Failed to load classifier: {str(e)}")
        
        # Load scaler
        if os.path.exists(scaler_path):
            try:
                self.scaler = torch.load(scaler_path, map_location='cpu')
                logger.info("✅ Scaler loaded")
                models_loaded += 1
            except Exception as e:
                logger.error(f"Failed to load scaler: {str(e)}")
        
        if models_loaded == 0:
            raise FileNotFoundError(f"No trained models found in {self.model_dir}!")
        elif models_loaded < 4:
            logger.warning(f"Only {models_loaded}/4 model components loaded. Results may be unreliable.")
        
        return models_loaded == 4
    
    def load_test_data(self, test_dir='datasets/test'):
        """Load test data from PT files"""
        logger.info(f"Loading test data from {test_dir}...")
        
        all_tensors = []
        all_labels = []
        total_images = 0
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(test_dir, class_name)
            
            if not os.path.exists(class_path):
                logger.error(f"Directory {class_path} not found!")
                continue
            
            pt_files = [f for f in os.listdir(class_path) if f.endswith('.pt')]
            if not pt_files:
                logger.warning(f"No .pt files found in {class_path}")
                continue
            
            class_images = 0
            class_tensors = []
            
            for pt_file in tqdm(pt_files, desc=f'Loading {class_name}'):
                pt_path = os.path.join(class_path, pt_file)
                try:
                    tensor_batch = torch.load(pt_path, map_location='cpu')
                    
                    if len(tensor_batch.shape) != 4:
                        logger.warning(f"Invalid tensor shape in {pt_file}: {tensor_batch.shape}")
                        continue
                    
                    class_tensors.append(tensor_batch)
                    class_images += len(tensor_batch)
                    
                except Exception as e:
                    logger.error(f"Error loading {pt_file}: {str(e)}")
                    continue
            
            if class_tensors:
                # Concatenate all tensors for this class
                class_data = torch.cat(class_tensors, dim=0)
                class_labels = torch.full((len(class_data),), class_idx, dtype=torch.long)
                
                all_tensors.append(class_data)
                all_labels.append(class_labels)
                
                logger.info(f"✅ {class_name}: {class_images} images loaded")
                total_images += class_images
            else:
                logger.warning(f"❌ No valid data loaded for {class_name}")
        
        if not all_tensors:
            raise RuntimeError("No test data loaded!")
        
        # Combine all classes
        test_tensors = torch.cat(all_tensors, dim=0)
        test_labels = torch.cat(all_labels, dim=0)
        
        logger.info(f"📊 Total test images loaded: {total_images}")
        logger.info(f"📋 Class distribution:")
        for i, class_name in enumerate(self.class_names):
            count = torch.sum(test_labels == i).item()
            logger.info(f"   {class_name}: {count} images ({count/len(test_labels)*100:.1f}%)")
        
        return test_tensors, test_labels
    
    def extract_features(self, tensors, batch_size=32):
        """Extract features using autoencoder and feature extractor"""
        logger.info("Extracting features...")
        
        dataset = TensorDataset(tensors)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=4, pin_memory=(self.device.type == 'cuda'))
        
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Feature extraction'):
                batch = batch.to(self.device, non_blocking=True)
                
                # Normalize to [0, 1] if needed
                if batch.max() > 1.0:
                    batch = batch / 255.0
                
                # Get reconstructed images from autoencoder
                reconstructed = self.autoencoder(batch)
                
                # Extract features using noise analysis
                features = self.feature_extractor(batch, reconstructed)
                all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def predict_with_loss(self, test_tensors, test_labels, batch_size=32):
        """Make predictions and calculate loss"""
        logger.info("Making predictions...")
        
        # Extract features
        features = self.extract_features(test_tensors, batch_size)
        
        # Normalize features
        if self.scaler is None:
            logger.error("Scaler not loaded! Cannot normalize features.")
            return None, None, float('inf')
        
        features_norm = torch.tensor(self.scaler.transform(features.numpy()), dtype=torch.float32)
        
        # Create dataset for prediction
        dataset = torch.utils.data.TensorDataset(features_norm, test_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size*2, shuffle=False,
                              num_workers=4, pin_memory=(self.device.type == 'cuda'))
        
        all_predictions = []
        all_probabilities = []
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(dataloader, desc="Prediction"):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions and probabilities
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)
        avg_loss = total_loss / num_batches
        
        return predictions, probabilities, avg_loss
    
    def evaluate_comprehensive(self, test_dir='datasets/test', batch_size=32):
        """Comprehensive evaluation of the model"""
        logger.info("🧪 Starting comprehensive evaluation...")
        
        # Load test data
        test_tensors, test_labels = self.load_test_data(test_dir)
        
        # Make predictions
        predictions, probabilities, avg_loss = self.predict_with_loss(
            test_tensors, test_labels, batch_size
        )
        
        if predictions is None:
            logger.error("❌ Prediction failed!")
            return None
        
        # Convert labels to numpy
        test_labels_np = test_labels.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels_np, predictions)
        mcc = matthews_corrcoef(test_labels_np, predictions)
        
        # Generate classification report
        report = classification_report(test_labels_np, predictions, 
                                     target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        # Print results
        self.print_results(accuracy, mcc, avg_loss, report, test_labels_np, predictions)
        
        # Create confusion matrix
        self.plot_confusion_matrix(test_labels_np, predictions)
        
        # Additional analysis
        self.analyze_per_class_performance(test_labels_np, predictions, probabilities)
        
        return {
            'accuracy': accuracy,
            'mcc': mcc,
            'loss': avg_loss,
            'classification_report': report,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': confusion_matrix(test_labels_np, predictions)
        }
    
    def print_results(self, accuracy, mcc, loss, report, y_true, y_pred):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("🎯 NEURAL NETWORK DEEPFAKE DETECTION - TEST RESULTS")
        print("="*80)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📈 MCC:       {mcc:.4f}")
        print(f"   💸 Loss:      {loss:.4f}")
        print(f"   📋 F1-Score:  {report['macro avg']['f1-score']:.4f}")
        
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 65)
        
        for class_name in self.class_names:
            if class_name in report:
                p = report[class_name]['precision']
                r = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                s = int(report[class_name]['support'])
                print(f"{class_name:<15} {p:<10.3f} {r:<10.3f} {f1:<10.3f} {s:<10}")
        
        print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
        
        # Calculate additional metrics
        total_samples = len(y_true)
        correct_predictions = np.sum(y_true == y_pred)
        error_rate = 1 - accuracy
        
        print(f"\n📈 ADDITIONAL STATISTICS:")
        print(f"   Total Samples:      {total_samples:,}")
        print(f"   Correct Predictions: {correct_predictions:,}")
        print(f"   Error Rate:         {error_rate:.4f} ({error_rate*100:.2f}%)")
        
        # Per-class accuracy
        print(f"\n🎯 PER-CLASS ACCURACY:")
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if np.any(class_mask):
                class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
                class_count = np.sum(class_mask)
                print(f"   {class_name:<15}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) [{class_count} samples]")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='test_confusion_matrix.png'):
        """Plot and save confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       cbar_kws={'label': 'Count'})
            
            plt.title('Confusion Matrix - Neural Network Deepfake Detection (Test Results)', 
                     fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            
            # Add accuracy text
            accuracy = np.trace(cm) / np.sum(cm)
            plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                       ha='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"📊 Confusion matrix saved to {save_path}")
            
        except Exception as e:
            logger.warning(f"Could not create confusion matrix plot: {str(e)}")
    
    def analyze_per_class_performance(self, y_true, y_pred, probabilities):
        """Analyze performance for each class in detail"""
        print(f"\n🔍 DETAILED PER-CLASS ANALYSIS:")
        print("="*60)
        
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if not np.any(class_mask):
                continue
            
            class_predictions = y_pred[class_mask]
            class_probabilities = probabilities[class_mask, i]  # Confidence for correct class
            
            # Calculate metrics
            correct_predictions = np.sum(class_predictions == i)
            total_samples = len(class_predictions)
            class_accuracy = correct_predictions / total_samples
            
            # Confidence statistics
            avg_confidence = np.mean(class_probabilities)
            min_confidence = np.min(class_probabilities)
            max_confidence = np.max(class_probabilities)
            
            # Misclassification analysis
            misclassified_mask = (class_predictions != i)
            misclassification_rate = np.mean(misclassified_mask)
            
            print(f"\n📊 {class_name.upper()} CLASS:")
            print(f"   Samples:           {total_samples}")
            print(f"   Correct:           {correct_predictions}")
            print(f"   Accuracy:          {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
            print(f"   Avg Confidence:    {avg_confidence:.4f}")
            print(f"   Confidence Range:  {min_confidence:.4f} - {max_confidence:.4f}")
            print(f"   Misclass Rate:     {misclassification_rate:.4f}")
            
            # Show most common misclassifications
            if np.any(misclassified_mask):
                misclassified_preds = class_predictions[misclassified_mask]
                unique_errors, error_counts = np.unique(misclassified_preds, return_counts=True)
                print(f"   Common Errors:")
                for error_class, count in zip(unique_errors, error_counts):
                    error_rate = count / total_samples
                    print(f"     → {self.class_names[error_class]}: {count} ({error_rate*100:.1f}%)")


def main():
    """Main function to run the test evaluation"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Neural Network Deepfake Detection - Test Program')
    parser.add_argument('--test_dir', type=str, default='datasets/test',
                       help='Directory containing test data (default: datasets/test)')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing trained models (default: models)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing (default: 32)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--save_results', type=str, default='test_results.txt',
                       help='File to save detailed results (default: test_results.txt)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 Neural Network Deepfake Detection - Test Program")
    print("="*60)
    print(f"📁 Test Directory:    {args.test_dir}")
    print(f"🤖 Model Directory:   {args.model_dir}")
    print(f"💻 Device:            {device}")
    print(f"🎯 Batch Size:        {args.batch_size}")
    print("="*60)
    
    try:
        # Initialize evaluator
        evaluator = DeepfakeTestEvaluator(model_dir=args.model_dir, device=device)
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_comprehensive(
            test_dir=args.test_dir, 
            batch_size=args.batch_size
        )
        
        if results is None:
            logger.error("❌ Evaluation failed!")
            return
        
        # Save detailed results to file
        if args.save_results:
            save_results_to_file(results, args.save_results)
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"🎯 Final Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"📈 Final MCC:      {results['mcc']:.4f}")
        print(f"💸 Final Loss:     {results['loss']:.4f}")
        print("="*80)
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {str(e)}")
        print("Make sure you have:")
        print("1. Trained models in the models/ directory")
        print("2. Test data in datasets/test/{real,semi-synthetic,synthetic}/ directories")
        print("3. PT files in each class directory")
        
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


def save_results_to_file(results, filename):
    """Save detailed results to a text file"""
    try:
        with open(filename, 'w') as f:
            f.write("Neural Network Deepfake Detection - Test Results\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Accuracy: {results['accuracy']:.6f}\n")
            f.write(f"MCC: {results['mcc']:.6f}\n")
            f.write(f"Loss: {results['loss']:.6f}\n\n")
            
            f.write("Classification Report:\n")
            f.write("-"*30 + "\n")
            
            report = results['classification_report']
            class_names = ['real', 'semi-synthetic', 'synthetic']
            
            for class_name in class_names:
                if class_name in report:
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
                    f.write(f"  Recall: {report[class_name]['recall']:.4f}\n")
                    f.write(f"  F1-Score: {report[class_name]['f1-score']:.4f}\n")
                    f.write(f"  Support: {report[class_name]['support']}\n\n")
            
            f.write(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write("-"*20 + "\n")
            cm = results['confusion_matrix']
            for i, row in enumerate(cm):
                f.write(f"{class_names[i]:>15}: {' '.join(f'{val:>6d}' for val in row)}\n")
        
        logger.info(f"📁 Detailed results saved to {filename}")
        
    except Exception as e:
        logger.warning(f"Could not save results to file: {str(e)}")


def quick_test():
    """Quick test function for debugging"""
    print("🧪 Running Quick Test...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluator = DeepfakeTestEvaluator(device=device)
        
        # Check if models are loaded
        print(f"✅ Models loaded successfully on {device}")
        
        # Try to load a small amount of test data
        try:
            test_tensors, test_labels = evaluator.load_test_data()
            print(f"✅ Test data loaded: {len(test_tensors)} samples")
            
            # Test feature extraction on a small batch
            small_batch = test_tensors[:8]  # Just 8 samples
            features = evaluator.extract_features(small_batch, batch_size=8)
            print(f"✅ Feature extraction working: {features.shape}")
            
            print("🎉 Quick test passed! Ready for full evaluation.")
            
        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")


if __name__ == "__main__":
    print("🔬 Neural Network Deepfake Detection - Test Program")
    print("="*60)
    print("Available modes:")
    print("  python test_program.py                    # Full evaluation")
    print("  python test_program.py --help             # Show all options")
    print("  python test_program.py --quick            # Quick test")
    print("="*60)
    
    import sys
    if '--quick' in sys.argv:
        quick_test()
    else:
        main()