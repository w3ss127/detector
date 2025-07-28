import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reuse the ProductionEfficientNet class from the training code
class ProductionEfficientNet(torch.nn.Module):
    """Production EfficientNet with balanced anti-overfitting measures"""
    
    def __init__(self, num_classes=3, model_size='b0', weights='DEFAULT', dropout_rate=0.5):
        super(ProductionEfficientNet, self).__init__()
        
        # Select model architecture
        if model_size == 'b0':
            self.backbone = models.efficientnet_b0(weights=weights)
        elif model_size == 'b1':
            self.backbone = models.efficientnet_b1(weights=weights)
        else:
            raise ValueError("Model size must be 'b0' or 'b1'")
        
        # Moderate layer freezing
        self._freeze_layers(freeze_ratio=0.5)
        
        # Get feature dimensions
        num_features = self.backbone.classifier[1].in_features
        
        # Simplified classifier with moderate regularization
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_features, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate * 0.8),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate * 0.6),
            torch.nn.Linear(128, num_classes)
        )
        
        # Replace backbone classifier
        self.backbone.classifier = self.classifier
        
        # Apply weight initialization
        self._initialize_weights()
    
    def _freeze_layers(self, freeze_ratio=0.5):
        """Freeze specified ratio of backbone layers"""
        total_layers = len(list(self.backbone.features.parameters()))
        freeze_count = int(total_layers * freeze_ratio)
        
        for i, param in enumerate(self.backbone.features.parameters()):
            if i < freeze_count:
                param.requires_grad = False
        
        logger.info(f"Frozen {freeze_count}/{total_layers} backbone parameters")
    
    def _initialize_weights(self):
        """Moderate weight initialization"""
        for m in self.classifier.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)

def create_test_transforms(input_size=224):
    """Create test transforms consistent with training"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return transform

def process_tensor(tensor):
    """Process a single image tensor to ensure proper format [C, H, W]"""
    try:
        # Convert sparse tensor to dense if necessary
        if tensor.is_sparse or tensor.layout == torch.sparse_coo:
            logger.info("Converting sparse tensor to dense")
            tensor = tensor.to_dense()
        
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        if tensor.max() > 10.0:
            tensor = tensor / 255.0
        elif tensor.max() > 2.0:
            tensor = torch.clamp(tensor / tensor.max(), 0, 1)
        
        # Ensure proper dimensions [C, H, W]
        if tensor.dim() == 3:
            if tensor.size(0) in [1, 3]:  # [C, H, W]
                pass  # Already in correct format
            elif tensor.size(-1) in [1, 3]:  # [H, W, C]
                tensor = tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            else:
                raise ValueError(f"Unexpected channel dimension in tensor shape: {tensor.shape}")
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
        
        # Adjust channel dimension
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
        elif tensor.size(0) > 3:
            tensor = tensor[:3]  # [N, H, W] -> [3, H, W]
        elif tensor.size(0) == 2:
            tensor = torch.cat([tensor, tensor[:1]], dim=0)  # [2, H, W] -> [3, H, W]
        
        tensor = torch.clamp(tensor, 0, 1)
        numpy_img = (tensor * 255).byte().permute(1, 2, 0).numpy()
        image = Image.fromarray(numpy_img.astype(np.uint8))
        
        return image
    except Exception as e:
        logger.error(f"Error processing tensor: {e}")
        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

def test_single_image(model_checkpoint_path, pt_file_path, class_names, input_size=224, model_size='b0'):
    """Test all images in a single .pt file with shape [5000, C, H, W]"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = ProductionEfficientNet(
        num_classes=len(class_names),
        model_size=model_size,
        dropout_rate=0.5
    ).to(device)
    
    # Load model checkpoint
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model checkpoint from {model_checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model checkpoint: {e}")
        raise
    
    model.eval()
    
    # Create test transform
    transform = create_test_transforms(input_size)
    
    # Verify .pt file exists
    if not os.path.exists(pt_file_path) or not pt_file_path.endswith('.pt'):
        logger.error(f"Invalid or missing .pt file: {pt_file_path}")
        raise ValueError(f"File {pt_file_path} does not exist or is not a .pt file")
    
    logger.info(f"Loading and testing images from: {pt_file_path}")
    
    # Load tensor
    try:
        tensor = torch.load(pt_file_path, map_location='cpu', weights_only=True)
        
        # Handle different tensor formats
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4 and tensor.size(0) == 5000:
                logger.info(f"Loaded tensor with shape: {tensor.shape}")
            else:
                logger.error(f"Unexpected tensor shape in {pt_file_path}: {tensor.shape}")
                raise ValueError(f"Expected tensor shape [5000, C, H, W], got {tensor.shape}")
        elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
            tensor = tensor[0] if isinstance(tensor[0], torch.Tensor) else None
            if tensor.dim() == 4 and tensor.size(0) == 5000:
                logger.info(f"Loaded tensor with shape: {tensor.shape}")
            else:
                logger.error(f"Unexpected tensor shape in {pt_file_path}: {tensor.shape}")
                raise ValueError(f"Expected tensor shape [5000, C, H, W], got {tensor.shape}")
        else:
            logger.error(f"Invalid tensor in {pt_file_path}")
            raise ValueError(f"Invalid tensor in {pt_file_path}")
        
        # Convert sparse tensor to dense if necessary
        if tensor.is_sparse or tensor.layout == torch.sparse_coo:
            logger.info("Converting batch tensor to dense")
            tensor = tensor.to_dense()
        
        # Process each image in the batch
        results = []
        true_class = os.path.basename(os.path.dirname(pt_file_path))
        true_class_idx = class_names.index(true_class) if true_class in class_names else -1
        
        for i in tqdm(range(tensor.size(0)), desc="Testing images"):
            try:
                # Extract single image tensor [C, H, W]
                single_tensor = tensor[i]
                logger.debug(f"Processing image {i+1}/{tensor.size(0)} with shape {single_tensor.shape}")
                
                # Process tensor to PIL Image
                image = process_tensor(single_tensor)
                
                # Apply test transform
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get model prediction
                with torch.no_grad():
                    output = model(image_tensor)
                    probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
                    predicted_class_idx = probs.argmax()
                    predicted_class = class_names[predicted_class_idx]
                    confidence = probs[predicted_class_idx] * 100
                    
                    correct = predicted_class == true_class if true_class_idx != -1 else False
                    
                    # Store and print result
                    result = {
                        'file': pt_file_path,
                        'image_index': i,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'true_class': true_class,
                        'correct': correct,
                        'probabilities': {class_names[j]: prob * 100 for j, prob in enumerate(probs)}
                    }
                    results.append(result)
                    
                    print(f"\n{'='*50}")
                    print(f"File: {pt_file_path} (Image {i+1}/{tensor.size(0)})")
                    print(f"True Class: {true_class}")
                    print(f"Predicted Class: {predicted_class}")
                    print(f"Confidence: {confidence:.2f}%")
                    print("Probabilities:")
                    for cls, prob in result['probabilities'].items():
                        print(f"  {cls}: {prob:.2f}%")
                    print(f"Correct: {result['correct']}")
                    print(f"{'='*50}")
                    
            except Exception as e:
                logger.error(f"Error processing image {i+1} in {pt_file_path}: {e}")
                continue
        
        # Summary
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = 100. * correct / total if total > 0 else 0
        
        print(f"\n{'='*60}")
        print("TESTING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Images Tested: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {pt_file_path}: {e}")
        raise

def main():
    """Main function for testing images in a single .pt file"""
    CONFIG = {
        'MODEL_CHECKPOINT_PATH': 'results_20250727_154058/best_model.pth',  # Adjust to your checkpoint path
        'PT_FILE_PATH': 'datasets/test/semi-synthetic/seg_inpainting_0.pt',                 # Adjust to your .pt file path
        'INPUT_SIZE': 224,
        'MODEL_SIZE': 'b0',
        'CLASS_NAMES': ['real', 'semi-synthetic', 'synthetic']
    }
    
    print("🚀 SINGLE PT FILE TESTING PIPELINE")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    try:
        results = test_single_image(
            model_checkpoint_path=CONFIG['MODEL_CHECKPOINT_PATH'],
            pt_file_path=CONFIG['PT_FILE_PATH'],
            class_names=CONFIG['CLASS_NAMES'],
            input_size=CONFIG['INPUT_SIZE'],
            model_size=CONFIG['MODEL_SIZE']
        )
        
        print("\n✅ Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        print(f"\n❌ TESTING FAILED")
        print(f"Error: {e}")
        print("Check the logs above for detailed error information.")

if __name__ == "__main__":
    main()