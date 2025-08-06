#!/usr/bin/env python3
"""
Complete Test Evaluation Program for Deepfake Detection
Tests the trained model on PT files from datasets/test/ folders
Shows comprehensive results with accuracy, MCC, loss, and enhanced progress displays
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, confusion_matrix
from typing import List, Tuple, Dict
from tqdm import tqdm
import joblib
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import cv2
from scipy import stats
from scipy.signal import welch
import warnings
import time
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import autoencoder architectures from the main module
try:
    from Method_Noise_fixed import DenosingAutoencoder, EnhancedDenosingAutoencoder, EnsembleAutoencoder, NoiseDistributionAnalyzer
    logger.info("Successfully imported autoencoder architectures from Method_Noise_fixed")
except ImportError as e:
    logger.error(f"Failed to import autoencoder architectures: {e}")
    raise

class TestDataset(Dataset):
    """Custom dataset for test tensor data"""
    def __init__(self, tensors: torch.Tensor, labels: torch.Tensor):
        self.tensors = tensors
        self.labels = labels
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]

class ComprehensiveModelTester:
    """Comprehensive model tester with enhanced metrics and progress display"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.autoencoder = None
        self.classifier = None
        self.noise_analyzer = None
        self.class_names = ['real', 'semi-synthetic', 'synthetic']
        
        logger.info(f"Tester initialized on device: {self.device}")
    
    def load_models(self, autoencoder_checkpoint_path='noise_autoencoder_checkpoints/noise_autoencoder_epoch_1_rank_0.pth', classifier_path='noise_classifier_model.joblib'):
        """Load trained autoencoder and classifier models with intelligent architecture detection"""
        
        # Load autoencoder
        if autoencoder_checkpoint_path is None:
            # Find latest checkpoint
            autoencoder_checkpoint_path = self.find_latest_checkpoint()
        
        if autoencoder_checkpoint_path and os.path.exists(autoencoder_checkpoint_path):
            logger.info(f"Loading autoencoder from: {autoencoder_checkpoint_path}")
            
            # Load checkpoint to inspect its structure
            checkpoint = torch.load(autoencoder_checkpoint_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']
            
            # Detect architecture based on state_dict keys
            has_enhanced_keys = any(key.startswith('enc_conv') or key.startswith('attention') for key in state_dict.keys())
            has_ensemble_keys = any(key.startswith('models.') for key in state_dict.keys())
            has_old_keys = any(key.startswith('encoder.') or key.startswith('decoder.') for key in state_dict.keys())
            
            # Initialize the correct autoencoder architecture
            if has_ensemble_keys:
                logger.info("🔍 Detected EnsembleAutoencoder checkpoint")
                self.autoencoder = EnsembleAutoencoder().to(self.device)
            elif has_enhanced_keys:
                logger.info("🔍 Detected EnhancedDenosingAutoencoder checkpoint")
                self.autoencoder = EnhancedDenosingAutoencoder().to(self.device)
            elif has_old_keys:
                logger.info("🔍 Detected DenosingAutoencoder (legacy) checkpoint")
                self.autoencoder = DenosingAutoencoder().to(self.device)
            else:
                logger.warning("⚠️  Could not detect autoencoder architecture, defaulting to EnhancedDenosingAutoencoder")
                self.autoencoder = EnhancedDenosingAutoencoder().to(self.device)
            
            # Try to load the state dict
            try:
                self.autoencoder.load_state_dict(state_dict, strict=True)
                logger.info("✅ Autoencoder loaded successfully (strict mode)")
            except RuntimeError as e:
                logger.warning(f"⚠️  Strict loading failed: {str(e)}")
                logger.info("🔄 Attempting to load with strict=False...")
                try:
                    missing_keys, unexpected_keys = self.autoencoder.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"❌ Missing keys: {len(missing_keys)} keys (showing first 5): {missing_keys[:5]}")
                    if unexpected_keys:
                        logger.warning(f"➕ Unexpected keys: {len(unexpected_keys)} keys (showing first 5): {unexpected_keys[:5]}")
                    logger.info("✅ Autoencoder loaded successfully (non-strict mode)")
                except Exception as e2:
                    logger.error(f"❌ Failed to load autoencoder: {str(e2)}")
                    raise
            
            self.autoencoder.eval()
            logger.info("🎯 Autoencoder set to evaluation mode")
        else:
            raise FileNotFoundError(f"❌ Autoencoder checkpoint not found: {autoencoder_checkpoint_path}")
        
        # Load classifier and scaler
        if os.path.exists(classifier_path):
            logger.info(f"📊 Loading classifier from: {classifier_path}")
            try:
                # Try joblib first (for .joblib files)
                if classifier_path.endswith('.joblib'):
                    logger.info("🔧 Using joblib.load() for .joblib file")
                    model_data = joblib.load(classifier_path)
                    self.classifier = model_data['classifier']
                    
                    # Initialize noise analyzer and load scaler
                    self.noise_analyzer = NoiseDistributionAnalyzer(device=self.device)
                    if model_data['scaler'] is not None:
                        self.noise_analyzer.scaler = model_data['scaler']
                        self.noise_analyzer.is_fitted = True
                        logger.info("✅ Classifier and scaler loaded successfully from joblib")
                    else:
                        logger.warning("⚠️  No scaler found in classifier file")
                
                # Try torch.load for .pth files
                elif classifier_path.endswith('.pth'):
                    logger.info("🔧 Using torch.load() for .pth file")
                    checkpoint = torch.load(classifier_path, map_location=self.device)
                    
                    # Check if this checkpoint contains classifier data
                    if 'classifier' in checkpoint:
                        self.classifier = checkpoint['classifier']
                        logger.info("✅ Classifier loaded from PyTorch checkpoint")
                    else:
                        logger.warning("⚠️  No classifier found in PyTorch checkpoint")
                        logger.info("💡 This appears to be an autoencoder checkpoint, not a classifier")
                        raise ValueError("No classifier found in the checkpoint file")
                    
                    # Initialize noise analyzer and load scaler if available
                    self.noise_analyzer = NoiseDistributionAnalyzer(device=self.device)
                    if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                        self.noise_analyzer.scaler = checkpoint['scaler']
                        self.noise_analyzer.is_fitted = True
                        logger.info("✅ Scaler loaded from PyTorch checkpoint")
                    else:
                        logger.warning("⚠️  No scaler found in PyTorch checkpoint")
                
                else:
                    # Try both methods as fallback
                    logger.info("🔧 Trying joblib.load() first, then torch.load() as fallback")
                    try:
            model_data = joblib.load(classifier_path)
            self.classifier = model_data['classifier']
            
            # Initialize noise analyzer and load scaler
            self.noise_analyzer = NoiseDistributionAnalyzer(device=self.device)
            if model_data['scaler'] is not None:
                self.noise_analyzer.scaler = model_data['scaler']
                            self.noise_analyzer.is_fitted = True
                            logger.info("✅ Classifier and scaler loaded successfully with joblib fallback")
                        else:
                            logger.warning("⚠️  No scaler found in classifier file")
                    
                    except Exception as joblib_error:
                        logger.warning(f"⚠️  joblib.load() failed: {str(joblib_error)}")
                        logger.info("🔄 Trying torch.load() as fallback...")
                        
                        try:
                            checkpoint = torch.load(classifier_path, map_location=self.device)
                            
                            if 'classifier' in checkpoint:
                                self.classifier = checkpoint['classifier']
                                logger.info("✅ Classifier loaded from PyTorch checkpoint (fallback)")
                            else:
                                logger.warning("⚠️  No classifier found in PyTorch checkpoint")
                                raise ValueError("No classifier found in the checkpoint file")
                            
                            # Initialize noise analyzer and load scaler if available
                            self.noise_analyzer = NoiseDistributionAnalyzer(device=self.device)
                            if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                                self.noise_analyzer.scaler = checkpoint['scaler']
                                self.noise_analyzer.is_fitted = True
                                logger.info("✅ Scaler loaded from PyTorch checkpoint (fallback)")
            else:
                                logger.warning("⚠️  No scaler found in PyTorch checkpoint")
                        
                        except Exception as torch_error:
                            logger.error(f"❌ torch.load() also failed: {str(torch_error)}")
                            logger.error(f"❌ Both joblib and torch loading methods failed")
                            raise
                            
            except Exception as e:
                logger.error(f"❌ Error loading classifier: {str(e)}")
                raise
        else:
            logger.error(f"❌ Classifier file not found: {classifier_path}")
            logger.info("💡 To create the classifier, you need to run the training first:")
            logger.info("   python Method_Noise_fixed.py")
            logger.info("   This will train the autoencoder and create the noise classifier.")
            raise FileNotFoundError(f"❌ Classifier file not found: {classifier_path}")
    
    def find_latest_checkpoint(self, checkpoint_dir='noise_autoencoder_checkpoints'):
        """Find the latest checkpoint file"""
        if not os.path.exists(checkpoint_dir):
            return None
        
        pattern = f"{checkpoint_dir}/noise_autoencoder_epoch_*_rank_*.pth"
        checkpoint_files = glob.glob(pattern)
        
        if not checkpoint_files:
            return None
        
        # Sort by epoch number (extract from filename)
        def extract_epoch(filename):
            try:
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    if part == 'epoch' and i + 1 < len(parts):
                        return int(parts[i + 1])
                return 0
            except:
                return 0
        
        latest_checkpoint = max(checkpoint_files, key=extract_epoch)
        logger.info(f"🔍 Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    def load_test_data(self, test_dir='datasets/test'):
        """Load test data from PT files with enhanced progress display"""
        logger.info(f"📂 Loading test data from: {test_dir}")
        
        all_tensors = []
        all_labels = []
        class_dirs = ['real', 'semi-synthetic', 'synthetic']
        
        # Progress tracking
        total_files = 0
        for class_name in class_dirs:
            class_path = os.path.join(test_dir, class_name)
            if os.path.exists(class_path):
                pt_files = [f for f in os.listdir(class_path) if f.endswith('.pt')]
                total_files += len(pt_files)
        
        logger.info(f"📊 Found {total_files} PT files across {len(class_dirs)} classes")
        
        # Create overall progress bar
        overall_pbar = tqdm(total=total_files, desc="Loading test files", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
        
        class_stats = {}
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(test_dir, class_name)
            if not os.path.exists(class_path):
                logger.warning(f"⚠️  Directory {class_path} not found, skipping...")
                continue
            
            pt_files = [f for f in os.listdir(class_path) if f.endswith('.pt')]
            pt_files.sort()
            
            if not pt_files:
                logger.warning(f"⚠️  No .pt files found in {class_path}")
                continue
            
            class_tensors = []
            class_labels = []
            
            logger.info(f"📁 Loading {len(pt_files)} files for class: {class_name}")
            
            for pt_file in pt_files:
                pt_path = os.path.join(class_path, pt_file)
                overall_pbar.set_description(f"Loading {class_name}/{pt_file}")
                
                try:
                    # Load tensor data
                    tensor_batch = torch.load(pt_path, map_location='cpu')
                    
                    if len(tensor_batch.shape) != 4:
                        logger.warning(f"⚠️  Invalid tensor shape in {pt_file}: {tensor_batch.shape}, skipping...")
                        overall_pbar.update(1)
                        continue
                    
                    # Normalize to [0, 1] if needed
                    if tensor_batch.max() > 1.0:
                        tensor_batch = tensor_batch / 255.0
                    
                    class_tensors.append(tensor_batch)
                    class_labels.append(torch.full((tensor_batch.shape[0],), class_idx, dtype=torch.long))
                    
                    overall_pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {pt_file}: {str(e)}")
                    overall_pbar.update(1)
                    continue
            
            if class_tensors:
                # Concatenate all tensors for this class
                class_data = torch.cat(class_tensors, dim=0)
                class_label_data = torch.cat(class_labels, dim=0)
                
                all_tensors.append(class_data)
                all_labels.append(class_label_data)
                
                class_stats[class_name] = len(class_data)
                logger.info(f"✅ Loaded {len(class_data)} images for class {class_name}")
            else:
                logger.warning(f"⚠️  No valid data loaded for class {class_name}")
        
        overall_pbar.close()
        
        if not all_tensors:
            raise RuntimeError("❌ No test data loaded from any class")
        
        # Combine all classes
        final_tensors = torch.cat(all_tensors, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        
        # Display loading summary
        logger.info(f"\n📊 Test Data Loading Summary:")
        logger.info(f"   Total images loaded: {len(final_tensors)}")
        for class_name, count in class_stats.items():
            percentage = (count / len(final_tensors)) * 100
            logger.info(f"   {class_name:>15}: {count:>6} images ({percentage:>5.1f}%)")
        
        logger.info(f"   Tensor shape: {final_tensors.shape}")
        logger.info(f"   Data range: [{final_tensors.min():.3f}, {final_tensors.max():.3f}]")
        
        return final_tensors, final_labels
    
    def extract_noise_features_batch(self, tensors: torch.Tensor, batch_size: int = 16):
        """Extract noise features with enhanced progress display"""
        logger.info(f"🔬 Extracting noise features from {len(tensors)} images...")
        
        # Optimize batch size based on GPU memory
        if self.device.type == 'cuda':
            memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            batch_size = min(batch_size, 32 if memory_gb < 8 else 64 if memory_gb < 16 else 128)
            logger.info(f"🚀 Using batch size: {batch_size} (GPU memory: {memory_gb:.1f}GB)")
        
        dataset = TestDataset(tensors, torch.zeros(len(tensors)))  # Dummy labels
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(self.device.type == 'cuda'))
        
        features_list = []
        self.autoencoder.eval()
        
        # Enhanced progress bar for feature extraction
        feature_pbar = tqdm(dataloader, desc='Extracting noise features', 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}')
        
        with torch.no_grad():
            for batch_idx, (batch_tensors, _) in enumerate(feature_pbar):
                try:
                    batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                    
                    # Generate reconstructions
                    if hasattr(self.autoencoder, 'forward'):
                    reconstructed_batch = self.autoencoder(batch_tensors)
                        # Handle ensemble autoencoder output
                        if isinstance(reconstructed_batch, tuple):
                            reconstructed_batch = reconstructed_batch[0]  # Use fused output
                    else:
                        raise AttributeError("Autoencoder doesn't have forward method")
                    
                    # Extract noise features
                    features = self.noise_analyzer.extract_noise_distribution_features(
                        batch_tensors, reconstructed_batch)
                    features_list.append(features.cpu().numpy())
                    
                    # Update progress bar with speed info
                    processed = (batch_idx + 1) * batch_size
                    total = len(tensors)
                    feature_pbar.set_postfix({
                        'Processed': f'{min(processed, total)}/{total}',
                        'GPU_Mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if self.device.type == 'cuda' else 'N/A'
                    })
                    
                    # Memory cleanup
                    if self.device.type == 'cuda' and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    logger.error(f"❌ Error in batch {batch_idx + 1}: {str(e)}")
                    if "out of memory" in str(e):
                        logger.warning("🔄 GPU OOM - reducing batch size and retrying")
                        torch.cuda.empty_cache()
                        return self.extract_noise_features_batch(tensors, batch_size=batch_size//2)
                    raise
        
        # Combine all features
        all_features = np.concatenate(features_list, axis=0)
        logger.info(f"✅ Feature extraction completed: {all_features.shape}")
        
        return all_features
    
    def evaluate_comprehensive(self, test_tensors: torch.Tensor, test_labels: torch.Tensor):
        """Comprehensive evaluation with enhanced metrics and progress display"""
        logger.info(f"\n🧪 Starting Comprehensive Model Evaluation")
        logger.info(f"   Test samples: {len(test_tensors)}")
        logger.info(f"   Classes: {self.class_names}")
        
        # Create main evaluation progress bar
        eval_steps = ['Feature Extraction', 'Noise Analysis', 'Classification', 'Metrics Calculation', 'Results Display']
        main_pbar = tqdm(total=len(eval_steps), desc="Evaluation Progress", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
        
        start_time = time.time()
        
        # Step 1: Extract features
        main_pbar.set_description("🔬 Extracting noise features")
        test_features = self.extract_noise_features_batch(test_tensors)
        main_pbar.update(1)
        
        # Step 2: Normalize features
        main_pbar.set_description("📊 Normalizing features")
        normalized_features = []
        norm_pbar = tqdm(test_features, desc="Normalizing", leave=False)
        for features in norm_pbar:
            normalized_features.append(self.noise_analyzer.normalize_features(features))
        normalized_features = np.vstack(normalized_features)
        main_pbar.update(1)
        
        # Step 3: Make predictions
        main_pbar.set_description("🎯 Making predictions")
        predictions = self.classifier.predict(normalized_features)
        probabilities = self.classifier.predict_proba(normalized_features)
        main_pbar.update(1)
        
        # Step 4: Calculate reconstruction loss
        main_pbar.set_description("📈 Calculating reconstruction loss")
        total_reconstruction_loss = 0
        mse_criterion = nn.MSELoss()
        
        loss_dataset = TestDataset(test_tensors, test_labels)
        loss_dataloader = DataLoader(loss_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        self.autoencoder.eval()
        with torch.no_grad():
            loss_pbar = tqdm(loss_dataloader, desc="Computing loss", leave=False)
            for batch_tensors, _ in loss_pbar:
                batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                reconstructed = self.autoencoder(batch_tensors)
                if isinstance(reconstructed, tuple):
                    reconstructed = reconstructed[0]
                loss = mse_criterion(reconstructed, batch_tensors)
                total_reconstruction_loss += loss.item()
        
        avg_reconstruction_loss = total_reconstruction_loss / len(loss_dataloader)
        main_pbar.update(1)
        
        # Step 5: Calculate comprehensive metrics
        main_pbar.set_description("📊 Calculating metrics")
        test_labels_np = test_labels.cpu().numpy()
        
        # Basic metrics
        accuracy = accuracy_score(test_labels_np, predictions)
        mcc = matthews_corrcoef(test_labels_np, predictions)
        
        # Detailed classification report
        report = classification_report(test_labels_np, predictions, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels_np, predictions)
        
        main_pbar.update(1)
        main_pbar.close()
        
        # Calculate evaluation time
        eval_time = time.time() - start_time
        
        # Display comprehensive results
        self.display_results(accuracy, mcc, avg_reconstruction_loss, report, cm, 
                           probabilities, test_labels_np, predictions, eval_time)
        
        return {
            'accuracy': accuracy,
            'mcc': mcc,
            'reconstruction_loss': avg_reconstruction_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'evaluation_time': eval_time
        }
    
    def display_results(self, accuracy, mcc, reconstruction_loss, report, cm, 
                       probabilities, true_labels, predictions, eval_time):
        """Display comprehensive results with enhanced formatting"""
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Overall metrics
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   🎯 Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📈 Matthews Correlation:  {mcc:.4f}")
        print(f"   🔧 Reconstruction Loss:   {reconstruction_loss:.6f}")
        print(f"   ⏱️  Evaluation Time:       {eval_time:.2f} seconds")
        print(f"   🚀 Speed:                 {len(true_labels)/eval_time:.1f} images/sec")
        
        # Per-class performance
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 65)
        
        for class_name in self.class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                print(f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")
        
        # Macro averages
        macro_avg = report['macro avg']
        print("-" * 65)
        print(f"{'Macro Average':<15} {macro_avg['precision']:<10.3f} {macro_avg['recall']:<10.3f} {macro_avg['f1-score']:<10.3f} {macro_avg['support']:<10}")
        
        # Confusion Matrix
        print(f"\n📊 CONFUSION MATRIX:")
        print("     Predicted:")
        print(f"     {'':>12}", end="")
        for class_name in self.class_names:
            print(f"{class_name[:8]:>12}", end="")
        print()
        
        print("Actual:")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name[:8]:>12}", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[i,j]:>12}", end="")
            print()
        
        # Class distribution
        print(f"\n📊 CLASS DISTRIBUTION:")
        unique, counts = np.unique(true_labels, return_counts=True)
        total_samples = len(true_labels)
        
        for class_idx, count in zip(unique, counts):
            class_name = self.class_names[class_idx]
            percentage = (count / total_samples) * 100
            print(f"   {class_name:>15}: {count:>6} samples ({percentage:>5.1f}%)")
        
        # Prediction confidence analysis
        print(f"\n🎯 PREDICTION CONFIDENCE ANALYSIS:")
        max_probs = np.max(probabilities, axis=1)
        print(f"   Mean confidence:     {np.mean(max_probs):.3f}")
        print(f"   Std confidence:      {np.std(max_probs):.3f}")
        print(f"   Min confidence:      {np.min(max_probs):.3f}")
        print(f"   Max confidence:      {np.max(max_probs):.3f}")
        
        # Confidence by correctness
        correct_mask = predictions == true_labels
        correct_conf = max_probs[correct_mask]
        incorrect_conf = max_probs[~correct_mask]
        
        if len(correct_conf) > 0:
            print(f"   Correct predictions: {np.mean(correct_conf):.3f} ± {np.std(correct_conf):.3f}")
        if len(incorrect_conf) > 0:
            print(f"   Wrong predictions:   {np.mean(incorrect_conf):.3f} ± {np.std(incorrect_conf):.3f}")
        
        # Performance interpretation
        print(f"\n🔍 PERFORMANCE INTERPRETATION:")
        if accuracy >= 0.9:
            print("   🟢 Excellent performance (≥90% accuracy)")
        elif accuracy >= 0.8:
            print("   🟡 Good performance (80-90% accuracy)")
        elif accuracy >= 0.7:
            print("   🟠 Fair performance (70-80% accuracy)")
        else:
            print("   🔴 Poor performance (<70% accuracy)")
        
        if mcc >= 0.8:
            print("   🟢 Excellent correlation (MCC ≥ 0.8)")
        elif mcc >= 0.6:
            print("   🟡 Good correlation (MCC 0.6-0.8)")
        elif mcc >= 0.4:
            print("   🟠 Fair correlation (MCC 0.4-0.6)")
        else:
            print("   🔴 Poor correlation (MCC < 0.4)")
        
        print("="*80)
    
    def save_results(self, results, output_dir='test_results'):
        """Save detailed results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        results_text = f"""
Comprehensive Model Evaluation Results
=====================================

Overall Performance:
- Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
- Matthews Correlation Coefficient: {results['mcc']:.4f}
- Reconstruction Loss: {results['reconstruction_loss']:.6f}
- Evaluation Time: {results['evaluation_time']:.2f} seconds

Classification Report:
{classification_report(results['predictions'], results['predictions'], target_names=self.class_names)}

Confusion Matrix:
{results['confusion_matrix']}
"""
        
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(results_text)
        
        logger.info(f"📁 Results saved to: {output_dir}")

def main():
    """Main function to run comprehensive model testing"""
    
    print("🚀 Comprehensive Deepfake Detection Model Tester")
    print("=" * 60)
    print("📊 Features:")
    print("   ✅ Intelligent model architecture detection")
    print("   ✅ Enhanced progress displays with tqdm")
    print("   ✅ Comprehensive metrics (Accuracy, MCC, Loss)")
    print("   ✅ Per-class performance analysis")
    print("   ✅ Confidence analysis and interpretation")
    print("   ✅ Visual confusion matrix")
    print("=" * 60)
    
    try:
        # Initialize tester
        tester = ComprehensiveModelTester(device='cuda')
        
        # Load models
        logger.info("🔧 Loading trained models...")
        tester.load_models()
        
        # Load test data
        logger.info("📂 Loading test data...")
        test_tensors, test_labels = tester.load_test_data('datasets/test')
        
        # Run comprehensive evaluation
        logger.info("🧪 Running comprehensive evaluation...")
        results = tester.evaluate_comprehensive(test_tensors, test_labels)
        
        # Save results
        logger.info("💾 Saving results...")
        tester.save_results(results)
        
        logger.info("✅ Evaluation completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()