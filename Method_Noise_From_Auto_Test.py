import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pywt
import os
from typing import Tuple, List, Dict
from tqdm import tqdm
import pickle
import json
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
warnings.filterwarnings('ignore')

# Optional: use joblib for large sklearn objects
try:
    import joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:
    _JOBLIB_AVAILABLE = False

class ImprovedDenoiseAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(ImprovedDenoiseAutoencoder, self).__init__()
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(2, 2),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(2, 2),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
        ])
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        reconstructed = self.decoder(x)
        return reconstructed

class NoiseFeatureExtractor:
    def __init__(self, use_gpu_features=True):
        self.scaler = None
        self.use_gpu_features = use_gpu_features and torch.cuda.is_available()
    
    def extract_noise_features_single(self, noise_map: np.ndarray) -> np.ndarray:
        """Extract features from a single noise map"""
        if self.use_gpu_features and noise_map.size > 512:
            try:
                noise_tensor = torch.from_numpy(noise_map).float()
                if torch.cuda.is_available():
                    noise_tensor = noise_tensor.cuda()
                return self._extract_gpu_features(noise_tensor).cpu().numpy()
            except:
                pass
        return self._extract_cpu_features_optimized(noise_map)
    
    def _extract_gpu_features(self, noise_tensor: torch.Tensor) -> torch.Tensor:
        features = []
        noise_flat = noise_tensor.flatten()
        
        # Basic statistics
        features.extend([
            torch.mean(noise_flat).item(),
            torch.std(noise_flat).item(),
            torch.var(noise_flat).item()
        ])
        
        # Percentiles
        sorted_noise = torch.sort(noise_flat)[0]
        n = len(sorted_noise)
        indices = [int(n * p) for p in [0.05, 0.25, 0.5, 0.75, 0.95]]
        percentiles = [sorted_noise[min(idx, n-1)].item() for idx in indices]
        features.extend(percentiles)
        features.append(percentiles[-1] - percentiles[0])
        
        # Histogram
        hist = torch.histc(noise_flat, bins=12, min=-1, max=1)
        hist = hist / (torch.sum(hist) + 1e-8)
        features.extend(hist.cpu().tolist())
        
        # Gradient features
        if noise_tensor.dim() == 2:
            noise_tensor = noise_tensor.unsqueeze(0).unsqueeze(0)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if torch.cuda.is_available():
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        grad_x = F.conv2d(noise_tensor, sobel_x, padding=1)
        grad_y = F.conv2d(noise_tensor, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        grad_flat = grad_mag.flatten()
        
        features.extend([
            torch.mean(grad_flat).item(),
            torch.std(grad_flat).item(),
            torch.quantile(grad_flat, 0.9).item()
        ])
        
        # Spatial features
        h, w = noise_tensor.shape[-2:]
        if h > 4 and w > 4:
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                noise_tensor[..., :mid_h, :mid_w],
                noise_tensor[..., :mid_h, mid_w:],
                noise_tensor[..., mid_h:, :mid_w],
                noise_tensor[..., mid_h:, mid_w:]
            ]
            quad_means = [torch.mean(quad).item() for quad in quadrants]
            quad_vars = [torch.var(quad).item() for quad in quadrants]
            features.extend([
                np.std(quad_means),
                np.std(quad_vars),
                max(quad_vars) - min(quad_vars)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Wavelet features
        coeffs = pywt.wavedec2(noise_tensor.cpu().numpy().squeeze(), 'db1', level=2)
        for coeff in coeffs:
            if isinstance(coeff, np.ndarray):
                features.extend([np.mean(coeff), np.std(coeff)])
            else:
                for c in coeff:
                    features.extend([np.mean(c), np.std(c)])
        
        # Pad to target length
        target_length = 40
        while len(features) < target_length:
            features.append(0.0)
        
        return torch.tensor(features[:target_length], dtype=torch.float32)
    
    def _extract_cpu_features_optimized(self, noise_map: np.ndarray) -> np.ndarray:
        features = []
        noise_flat = noise_map.flatten()
        
        # Basic statistics
        features.extend([
            np.mean(noise_flat),
            np.std(noise_flat),
            np.var(noise_flat)
        ])
        
        # Percentiles
        percentiles = np.percentile(noise_flat, [5, 25, 50, 75, 95])
        features.extend(percentiles.tolist())
        features.append(percentiles[4] - percentiles[0])
        
        # Histogram
        hist, _ = np.histogram(noise_flat, bins=12, range=(-1, 1))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist.tolist())
        
        # Gradient features
        h, w = noise_map.shape
        if h > 8 and w > 8:
            grad_x = cv2.Sobel(noise_map, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(noise_map, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_flat = grad_mag.flatten()
            features.extend([
                np.mean(grad_flat),
                np.std(grad_flat),
                np.percentile(grad_flat, 90)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Spatial features
        if h > 4 and w > 4:
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                noise_map[:mid_h, :mid_w],
                noise_map[:mid_h, mid_w:],
                noise_map[mid_h:, :mid_w],
                noise_map[mid_h:, mid_w:]
            ]
            quad_vars = [np.var(quad) for quad in quadrants]
            features.extend([
                np.std(quad_vars),
                np.max(quad_vars) - np.min(quad_vars),
                np.mean(quad_vars)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Wavelet features
        coeffs = pywt.wavedec2(noise_map, 'db1', level=2)
        for coeff in coeffs:
            if isinstance(coeff, np.ndarray):
                features.extend([np.mean(coeff), np.std(coeff)])
            else:
                for c in coeff:
                    features.extend([np.mean(c), np.std(c)])
        
        # Pad to target length
        target_length = 40
        while len(features) < target_length:
            features.append(0.0)
        
        return np.array(features[:target_length], dtype=np.float32)
    
    def load_scaler(self, scaler_path: str):
        """Load the pre-trained scaler"""
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✅ Feature scaler loaded from: {scaler_path}")

class IndividualImageTester:
    def __init__(self, autoencoder_path: str, classifier_path: str, scaler_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Resolve paths relative to this script for robustness
        autoencoder_path = self._resolve_path(autoencoder_path)
        classifier_path = self._resolve_path(classifier_path)
        scaler_path = self._resolve_path(scaler_path)
        
        # Basic file checks
        for pth, label in [
            (autoencoder_path, 'Autoencoder weights'),
            (classifier_path, 'Classifier'),
            (scaler_path, 'Feature scaler')
        ]:
            if not os.path.exists(pth):
                raise FileNotFoundError(f"{label} file not found at: {pth}")
            size_bytes = os.path.getsize(pth)
            if size_bytes == 0:
                raise RuntimeError(f"{label} file is empty at: {pth}")
        
        # Load autoencoder
        self.autoencoder = ImprovedDenoiseAutoencoder().to(self.device)
        self.load_autoencoder(autoencoder_path)
        
        # Load classifier
        try:
            if _JOBLIB_AVAILABLE:
                self.classifier = joblib.load(classifier_path)
            else:
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load classifier from {classifier_path}: {e}. "
                f"If this is a large scikit-learn model, re-save it with joblib.dump(...) and retry."
            )
        print(f"✅ Random Forest classifier loaded from: {classifier_path}")
        
        # Load feature extractor and scaler
        self.feature_extractor = NoiseFeatureExtractor(use_gpu_features=torch.cuda.is_available())
        try:
            self.feature_extractor.load_scaler(scaler_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load feature scaler from {scaler_path}: {e}. "
                f"Recreate the scaler or ensure the file is not corrupted."
            )
        
        self.class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    
    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))
        
    def load_autoencoder(self, checkpoint_path: str):
        """Load the pre-trained autoencoder"""
        print(f"🔄 Loading autoencoder from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'autoencoder_state_dict' in checkpoint:
            state_dict = checkpoint['autoencoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel state dict
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.eval()
        print(f"✅ Autoencoder loaded successfully!")
    
    def extract_noise_map(self, image: torch.Tensor) -> np.ndarray:
        """Extract noise map from a single image"""
        self.autoencoder.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(self.device)
            reconstructed = self.autoencoder(image)
            noise = image - reconstructed
            noise_map = torch.mean(noise.squeeze(), dim=0).cpu().numpy().astype(np.float32)
        return noise_map
    
    def predict_single_image(self, image: torch.Tensor) -> Tuple[int, np.ndarray, np.ndarray]:
        """Predict class for a single image
        Returns: (prediction, probabilities, noise_map)
        """
        # Extract noise map
        noise_map = self.extract_noise_map(image)
        
        # Extract features
        features = self.feature_extractor.extract_noise_features_single(noise_map)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features = self.feature_extractor.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        
        return prediction, probabilities, noise_map

def load_test_data_individual(test_dir: str) -> List[Tuple[torch.Tensor, int, str, str]]:
    """Load test data and return individual images with metadata
    Returns: List of (image_tensor, true_label, class_name, file_path)
    """
    all_images = []
    class_folders = ['real', 'synthetic', 'semi-synthetic']
    class_mapping = {'real': 0, 'synthetic': 1, 'semi-synthetic': 2}
    
    for class_name in class_folders:
        class_idx = class_mapping[class_name]
        class_dir = os.path.join(test_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"⚠️ Directory {class_dir} not found, skipping...")
            continue
            
        print(f"📁 Loading {class_name} images from: {class_dir}")
        pt_files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
        pt_files.sort()
        
        for pt_file in tqdm(pt_files, desc=f"Loading {class_name} files"):
            pt_path = os.path.join(class_dir, pt_file)
            try:
                tensor_data = torch.load(pt_path, map_location='cpu')
                
                # Handle different tensor formats
                if isinstance(tensor_data, dict):
                    if 'images' in tensor_data:
                        images = tensor_data['images']
                    elif 'data' in tensor_data:
                        images = tensor_data['data']
                    else:
                        images = list(tensor_data.values())[0]
                else:
                    images = tensor_data
                
                # Ensure proper format
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                elif images.dim() == 5:
                    images = images.squeeze()
                
                # Normalize
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                else:
                    images = images.float()
                    if images.max() > 1.0:
                        images = images / 255.0
                
                # Add individual images to list
                for i in range(images.shape[0]):
                    image = images[i]
                    image_id = f"{pt_file}_img_{i:03d}"
                    all_images.append((image, class_idx, class_name, image_id))
                    
            except Exception as e:
                print(f"❌ Error loading {pt_file}: {e}")
                continue
    
    print(f"✅ Total individual images loaded: {len(all_images)}")
    return all_images

def visualize_results(results: Dict, save_dir: str):
    """Create comprehensive visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    predictions = results['individual_predictions']
    true_labels = [item['true_label'] for item in predictions]
    pred_labels = [item['prediction'] for item in predictions]
    probabilities = [item['probabilities'] for item in predictions]
    class_names = results['class_names']
    
    # 1. Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nAccuracy: {results["overall_accuracy"]:.4f}, MCC: {results["overall_mcc"]:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class accuracy
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        class_mask = np.array(true_labels) == i
        if np.sum(class_mask) > 0:
            class_pred = np.array(pred_labels)[class_mask]
            class_true = np.array(true_labels)[class_mask]
            class_acc = accuracy_score(class_true, class_pred)
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracies, color=['#FF9F40', '#4BC0C0', '#9966FF'], alpha=0.8)
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1.1)
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, class_name in enumerate(class_names):
        class_confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        for item in predictions:
            if item['prediction'] == i:
                confidence = item['probabilities'][i]
                class_confidences.append(confidence)
                if item['prediction'] == item['true_label']:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)
        
        axes[i].hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green', density=True)
        axes[i].hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red', density=True)
        axes[i].set_xlabel('Confidence')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{class_name} - Prediction Confidence')
        axes[i].legend()
        axes[i].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Sample noise maps for each class
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Get samples for each class
    for class_idx in range(3):
        class_samples = [item for item in predictions if item['true_label'] == class_idx][:3]
        
        for sample_idx, sample in enumerate(class_samples):
            if sample_idx < len(class_samples):
                row = class_idx
                col = sample_idx
                noise_map = sample['noise_map']
                
                im = axes[row, col].imshow(noise_map, cmap='gray')
                pred_class = class_names[sample['prediction']]
                confidence = sample['probabilities'][sample['prediction']]
                correct = "✓" if sample['prediction'] == sample['true_label'] else "✗"
                
                axes[row, col].set_title(f'True: {class_names[class_idx]}\n'
                                       f'Pred: {pred_class} {correct}\n'
                                       f'Conf: {confidence:.3f}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_noise_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {save_dir}")

def main_individual_test():
    print("🔍 INDIVIDUAL IMAGE TESTING PROGRAM")
    print("="*70)
    
    # Configuration (resolve relative to script directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, 'datasets', 'test')
    autoencoder_path = os.path.join(base_dir, 'checkpoints', 'best_autoencoder.pth')
    classifier_path = os.path.join(base_dir, 'checkpoints', 'random_forest_classifier.pkl')
    scaler_path = os.path.join(base_dir, 'checkpoints', 'feature_scaler.pkl')
    results_dir = os.path.join(base_dir, 'test_results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize tester
    print(f"🚀 Initializing individual image tester...")
    try:
        tester = IndividualImageTester(autoencoder_path, classifier_path, scaler_path)
    except Exception as e:
        print(f"❌ Error initializing tester: {e}")
        # Provide quick diagnostics if initialization fails
        checkpoints_dir = os.path.join(base_dir, 'checkpoints')
        if os.path.isdir(checkpoints_dir):
            print("\n📁 Checkpoints directory contents:")
            for name in sorted(os.listdir(checkpoints_dir)):
                p = os.path.join(checkpoints_dir, name)
                try:
                    size = os.path.getsize(p)
                    print(f"  - {name} ({size/1024:.1f} KB)")
                except Exception:
                    print(f"  - {name}")
        else:
            print(f"\n❌ Checkpoints directory not found at {checkpoints_dir}")
        return
    
    # Load test data
    print(f"\n📊 Loading individual test images from: {test_dir}")
    try:
        test_images = load_test_data_individual(test_dir)
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    # Print dataset statistics
    print(f"\n📈 Test Dataset Statistics:")
    class_counts = {0: 0, 1: 0, 2: 0}
    for _, label, _, _ in test_images:
        class_counts[label] += 1
    
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    for class_idx, count in class_counts.items():
        print(f"   {class_names[class_idx]}: {count:,} images")
    
    # Process each image individually
    print(f"\n🔍 Processing {len(test_images)} individual images...")
    start_time = time.time()
    
    all_results = []
    correct_predictions = 0
    
    for i, (image, true_label, class_name, image_id) in enumerate(tqdm(test_images, desc="Testing images")):
        try:
            prediction, probabilities, noise_map = tester.predict_single_image(image)
            
            is_correct = prediction == true_label
            if is_correct:
                correct_predictions += 1
            
            result = {
                'image_id': image_id,
                'true_label': true_label,
                'true_class': class_name,
                'prediction': prediction,
                'predicted_class': class_names[prediction],
                'probabilities': probabilities.tolist(),
                'confidence': float(probabilities[prediction]),
                'is_correct': is_correct,
                'noise_map': noise_map  # For visualization
            }
            all_results.append(result)
            
            # Print progress every 100 images
            if (i + 1) % 100 == 0:
                current_acc = correct_predictions / (i + 1)
                print(f"   Progress: {i+1}/{len(test_images)} - Current Accuracy: {current_acc:.4f}")
        
        except Exception as e:
            print(f"❌ Error processing image {image_id}: {e}")
            # Add failed result
            result = {
                'image_id': image_id,
                'true_label': true_label,
                'true_class': class_name,
                'prediction': -1,
                'predicted_class': 'ERROR',
                'probabilities': [0.0, 0.0, 0.0],
                'confidence': 0.0,
                'is_correct': False,
                'noise_map': np.zeros((64, 64))  # Dummy noise map
            }
            all_results.append(result)
    
    processing_time = time.time() - start_time
    
    # Calculate overall metrics
    valid_results = [r for r in all_results if r['prediction'] != -1]
    true_labels = [r['true_label'] for r in valid_results]
    predictions = [r['prediction'] for r in valid_results]
    
    overall_accuracy = accuracy_score(true_labels, predictions)
    overall_mcc = matthews_corrcoef(true_labels, predictions)
    
    # Per-class metrics
    report = classification_report(true_labels, predictions, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    # Print results
    print(f"\n🎯 INDIVIDUAL IMAGE TEST RESULTS")
    print("="*50)
    print(f"📊 Total Images Processed: {len(test_images):,}")
    print(f"✅ Successfully Processed: {len(valid_results):,}")
    print(f"❌ Failed: {len(test_images) - len(valid_results):,}")
    print(f"🎯 Overall Accuracy: {overall_accuracy:.4f}")
    print(f"📈 Matthews Correlation Coefficient: {overall_mcc:.4f}")
    print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
    print(f"🚀 Speed: {len(test_images)/processing_time:.1f} images/second")
    
    print(f"\n📈 Per-Class Results:")
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-score:  {metrics['f1-score']:.4f}")
            print(f"    Support:   {int(metrics['support'])}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\n📊 Confusion Matrix:")
    print("True\\Pred    Real  Synth  Semi")
    for i, (name, row) in enumerate(zip(['Real     ', 'Synthetic', 'Semi-synth'], cm)):
        print(f"{name} {row[0]:5d} {row[1]:6d} {row[2]:5d}")
    
    # Error Analysis
    print(f"\n🔍 Error Analysis:")
    error_count = {}
    for result in valid_results:
        if not result['is_correct']:
            true_class = result['true_class']
            pred_class = result['predicted_class']
            error_key = f"{true_class} → {pred_class}"
            error_count[error_key] = error_count.get(error_key, 0) + 1
    
    for error_type, count in sorted(error_count.items(), key=lambda x: x[1], reverse=True):
        print(f"   {error_type}: {count} errors")
    
    # Confidence Analysis
    print(f"\n📊 Confidence Analysis:")
    correct_confidences = [r['confidence'] for r in valid_results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in valid_results if not r['is_correct']]
    
    if correct_confidences:
        print(f"   Correct predictions - Mean confidence: {np.mean(correct_confidences):.4f} ± {np.std(correct_confidences):.4f}")
    if incorrect_confidences:
        print(f"   Incorrect predictions - Mean confidence: {np.mean(incorrect_confidences):.4f} ± {np.std(incorrect_confidences):.4f}")
    
    # High confidence errors (potential issues)
    high_conf_errors = [r for r in valid_results if not r['is_correct'] and r['confidence'] > 0.8]
    if high_conf_errors:
        print(f"⚠️  High confidence errors (>0.8): {len(high_conf_errors)}")
        for error in high_conf_errors[:5]:  # Show top 5
            print(f"     {error['image_id']}: {error['true_class']} → {error['predicted_class']} (conf: {error['confidence']:.3f})")
    
    # Low confidence correct predictions
    low_conf_correct = [r for r in valid_results if r['is_correct'] and r['confidence'] < 0.6]
    if low_conf_correct:
        print(f"🤔 Low confidence correct predictions (<0.6): {len(low_conf_correct)}")
    
    # Save detailed results
    detailed_results = {
        'test_summary': {
            'total_images': len(test_images),
            'successfully_processed': len(valid_results),
            'failed': len(test_images) - len(valid_results),
            'overall_accuracy': overall_accuracy,
            'overall_mcc': overall_mcc,
            'processing_time': processing_time,
            'images_per_second': len(test_images)/processing_time
        },
        'class_names': class_names,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'individual_predictions': [
            {k: v for k, v in result.items() if k != 'noise_map'}  # Exclude noise_map for JSON
            for result in all_results
        ],
        'error_analysis': error_count,
        'confidence_stats': {
            'correct_mean': float(np.mean(correct_confidences)) if correct_confidences else 0.0,
            'correct_std': float(np.std(correct_confidences)) if correct_confidences else 0.0,
            'incorrect_mean': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
            'incorrect_std': float(np.std(incorrect_confidences)) if incorrect_confidences else 0.0,
            'high_conf_errors': len(high_conf_errors),
            'low_conf_correct': len(low_conf_correct)
        },
        'overall_accuracy': overall_accuracy,
        'overall_mcc': overall_mcc
    }
    
    # Save results to JSON
    results_file = os.path.join(results_dir, 'individual_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\n✅ Detailed results saved to: {results_file}")
    
    # Save individual predictions to CSV-like format
    predictions_file = os.path.join(results_dir, 'individual_predictions.txt')
    with open(predictions_file, 'w') as f:
        f.write("Image_ID\tTrue_Label\tTrue_Class\tPrediction\tPredicted_Class\tConfidence\tCorrect\tProb_Real\tProb_Synthetic\tProb_Semi\n")
        for result in all_results:
            probs = result['probabilities']
            f.write(f"{result['image_id']}\t{result['true_label']}\t{result['true_class']}\t"
                   f"{result['prediction']}\t{result['predicted_class']}\t{result['confidence']:.4f}\t"
                   f"{result['is_correct']}\t{probs[0]:.4f}\t{probs[1]:.4f}\t{probs[2]:.4f}\n")
    print(f"✅ Individual predictions saved to: {predictions_file}")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    try:
        # Add noise_map back for visualization
        vis_results = detailed_results.copy()
        vis_results['individual_predictions'] = all_results  # Include noise_map
        visualize_results(vis_results, results_dir)
    except Exception as e:
        print(f"⚠️ Warning: Visualization creation failed: {e}")
    
    # Create detailed error report
    print(f"\n📝 Creating detailed error report...")
    error_report_file = os.path.join(results_dir, 'error_analysis_report.txt')
    with open(error_report_file, 'w') as f:
        f.write("DETAILED ERROR ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"- Total Images: {len(test_images):,}\n")
        f.write(f"- Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"- MCC: {overall_mcc:.4f}\n")
        f.write(f"- Processing Speed: {len(test_images)/processing_time:.1f} images/second\n\n")
        
        f.write("Per-Class Performance:\n")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                f.write(f"- {class_name}:\n")
                f.write(f"  * Precision: {metrics['precision']:.4f}\n")
                f.write(f"  * Recall: {metrics['recall']:.4f}\n")
                f.write(f"  * F1-score: {metrics['f1-score']:.4f}\n")
                f.write(f"  * Support: {int(metrics['support'])}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("True\\Predicted  Real  Synthetic  Semi-synthetic\n")
        for i, (name, row) in enumerate(zip(class_names, cm)):
            f.write(f"{name:<12} {row[0]:4d} {row[1]:9d} {row[2]:14d}\n")
        f.write("\n")
        
        f.write("Error Types (True → Predicted):\n")
        for error_type, count in sorted(error_count.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / len([r for r in valid_results if not r['is_correct']])
            f.write(f"- {error_type}: {count} errors ({percentage:.1f}% of all errors)\n")
        f.write("\n")
        
        f.write("High Confidence Errors (Confidence > 0.8):\n")
        if high_conf_errors:
            for i, error in enumerate(high_conf_errors[:10]):  # Top 10
                f.write(f"{i+1}. {error['image_id']}: {error['true_class']} → {error['predicted_class']} "
                       f"(confidence: {error['confidence']:.3f})\n")
                f.write(f"   Probabilities: Real={error['probabilities'][0]:.3f}, "
                       f"Synthetic={error['probabilities'][1]:.3f}, "
                       f"Semi-synthetic={error['probabilities'][2]:.3f}\n\n")
        else:
            f.write("None found.\n\n")
        
        f.write("Low Confidence Correct Predictions (Confidence < 0.6):\n")
        if low_conf_correct:
            for i, correct in enumerate(low_conf_correct[:10]):  # Top 10
                f.write(f"{i+1}. {correct['image_id']}: {correct['true_class']} "
                       f"(confidence: {correct['confidence']:.3f})\n")
        else:
            f.write("None found.\n")
    
    print(f"✅ Error analysis report saved to: {error_report_file}")
    
    # Performance summary by file (if multiple images per file)
    print(f"\n📁 Creating per-file summary...")
    file_performance = {}
    for result in valid_results:
        file_name = result['image_id'].split('_img_')[0]  # Extract base filename
        if file_name not in file_performance:
            file_performance[file_name] = {'total': 0, 'correct': 0, 'true_class': result['true_class']}
        file_performance[file_name]['total'] += 1
        if result['is_correct']:
            file_performance[file_name]['correct'] += 1
    
    file_summary_file = os.path.join(results_dir, 'per_file_performance.txt')
    with open(file_summary_file, 'w') as f:
        f.write("PER-FILE PERFORMANCE SUMMARY\n")
        f.write("="*50 + "\n")
        f.write("Filename\tTrue_Class\tTotal_Images\tCorrect\tAccuracy\n")
        
        for file_name, stats in sorted(file_performance.items()):
            accuracy = stats['correct'] / stats['total']
            f.write(f"{file_name}\t{stats['true_class']}\t{stats['total']}\t"
                   f"{stats['correct']}\t{accuracy:.4f}\n")
    
    print(f"✅ Per-file performance saved to: {file_summary_file}")
    
    print(f"\n🎉 INDIVIDUAL IMAGE TESTING COMPLETED!")
    print("="*70)
    print(f"📊 Final Results Summary:")
    print(f"   • Total Images: {len(test_images):,}")
    print(f"   • Overall Accuracy: {overall_accuracy:.4f}")
    print(f"   • Matthews Correlation: {overall_mcc:.4f}")
    print(f"   • Processing Speed: {len(test_images)/processing_time:.1f} images/sec")
    print(f"   • Results Directory: {results_dir}")
    
    print(f"\n📂 Generated Files:")
    print(f"   • individual_test_results.json - Complete results")
    print(f"   • individual_predictions.txt - Detailed predictions")
    print(f"   • error_analysis_report.txt - Error analysis")
    print(f"   • per_file_performance.txt - Per-file summary")
    print(f"   • confusion_matrix.png - Confusion matrix")
    print(f"   • per_class_accuracy.png - Class accuracies")
    print(f"   • confidence_distribution.png - Confidence analysis")
    print(f"   • sample_noise_maps.png - Sample noise visualizations")
    print("="*70)
    
    return detailed_results

def test_single_image_example():
    """Example function to test a single image"""
    print("\n🔍 SINGLE IMAGE TEST EXAMPLE")
    print("-" * 30)
    
    # Configuration (resolve relative to script directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    autoencoder_path = os.path.join(base_dir, 'checkpoints', 'best_autoencoder.pth')
    classifier_path = os.path.join(base_dir, 'checkpoints', 'random_forest_classifier.pkl')
    scaler_path = os.path.join(base_dir, 'checkpoints', 'feature_scaler.pkl')
    
    try:
        tester = IndividualImageTester(autoencoder_path, classifier_path, scaler_path)
        
        # Create a dummy image for testing
        dummy_image = torch.randn(3, 224, 224)  # Adjust size as needed
        print(f"Testing with dummy image of shape: {dummy_image.shape}")
        
        prediction, probabilities, noise_map = tester.predict_single_image(dummy_image)
        
        print(f"🎯 Prediction Results:")
        print(f"   Predicted Class: {tester.class_names[prediction]}")
        print(f"   Confidence: {probabilities[prediction]:.4f}")
        print(f"   All Probabilities:")
        for i, (class_name, prob) in enumerate(zip(tester.class_names, probabilities)):
            print(f"     {class_name}: {prob:.4f}")
        print(f"   Noise Map Shape: {noise_map.shape}")
        print(f"   Noise Map Range: [{noise_map.min():.4f}, {noise_map.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Single image test failed: {e}")
        return False

if __name__ == "__main__":
    # Test single image first
    print("🧪 Testing single image processing...")
    if test_single_image_example():
        print("✅ Single image test passed!")
        
        # Run full test suite
        results = main_individual_test()
    else:
        print("❌ Single image test failed! Check your model files.")
        print("\nRequired files (resolved relative to this script):")
        print("- ./checkpoints/best_autoencoder.pth (autoencoder weights)")
        print("- ./checkpoints/random_forest_classifier.pkl (classifier)")
        print("- ./checkpoints/feature_scaler.pkl (feature scaler)")
        print("- ./datasets/test/ (test data directory)")
        
        # Print current directory structure for debugging with sizes
        _base = os.path.dirname(os.path.abspath(__file__))
        print(f"\n📁 Current directory contents:")
        ckpt_dir = os.path.join(_base, 'checkpoints')
        if os.path.exists(ckpt_dir):
            print(f"  Checkpoints directory: {ckpt_dir}")
            for file in sorted(os.listdir(ckpt_dir)):
                fp = os.path.join(ckpt_dir, file)
                try:
                    size = os.path.getsize(fp)
                    print(f"    - {file} ({size/1024:.1f} KB)")
                except Exception:
                    print(f"    - {file}")
        else:
            print(f"  ❌ checkpoints directory not found at {ckpt_dir}")
        
        ds_dir = os.path.join(_base, 'datasets')
        if os.path.exists(ds_dir):
            print(f"  Datasets directory: {ds_dir}")
            for item in sorted(os.listdir(ds_dir)):
                print(f"    - {item}")
        else:
            print(f"  ❌ datasets directory not found at {ds_dir}")