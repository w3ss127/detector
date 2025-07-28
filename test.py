"""
Clean and efficient model testing script for image classification.
Tests a ResNet-ViT hybrid model on real/synthetic/semi-synthetic images.
Processes and shows results for each image individually.
UPDATED TO EXACTLY MATCH TRAINING CODE ARCHITECTURE AND PREPROCESSING.
"""

import os
import sys
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from typing import List, Tuple, Dict


class Config:
    """Configuration settings for the model testing."""
    
    # Paths
    BASE_DIR = Path("datasets")
    TEST_DIR = BASE_DIR / "train"
    MODEL_PATH = Path("checkpoints/best_model_mcc_0.9900.pt")
    
    # Class settings
    CLASS_MAP = {"real": 0, "synthetic": 1, "semi-synthetic": 2}
    CLASS_NAMES = ["real", "synthetic", "semi-synthetic"]
    NUM_CLASSES = 3
    
    # Model settings (MUST match training - FIXED)
    DROPOUT_RATE = 0.3  # FIXED: Was 0.5, now matches training
    IMAGE_SIZE = 224
    
    # System settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42


class ResNetViTHybrid(nn.Module):
    """Hybrid ResNet + ViT model - EXACT copy from training code."""
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        
        # Feature extractors
        self.resnet = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        
        # Combined features
        combined_features = self.resnet.num_features + self.vit.num_features
        
        # FIXED: Use EXACT same classifier architecture as training
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Same as training: 0.5 * dropout_rate
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model."""
        resnet_features = self.resnet(x)
        vit_features = self.vit(x)
        combined = torch.cat([resnet_features, vit_features], dim=1)
        return self.classifier(combined)


def get_test_transforms():
    """Get test transforms that EXACTLY match training base transforms."""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ModelTester:
    """Main class for testing the trained model."""
    
    def __init__(self, config: Config):
        self.config = config
        torch.manual_seed(config.SEED)
        
        self.transforms = get_test_transforms()
        self.model = self._load_model()
        
        print(f"🔧 Using device: {config.DEVICE}")
        print(f"🔧 Model dropout rate: {config.DROPOUT_RATE}")
    
    def _load_model(self) -> ResNetViTHybrid:
        """Load the trained model from checkpoint."""
        if not self.config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {self.config.MODEL_PATH}")
        
        model = ResNetViTHybrid(self.config.NUM_CLASSES, self.config.DROPOUT_RATE)
        model = model.to(self.config.DEVICE)
        
        try:
            checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.config.DEVICE, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded model from {self.config.MODEL_PATH}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image EXACTLY like training code."""
        # FIXED: Use exact same preprocessing as training
        
        # Convert to float [0,1] range - EXACT training logic
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 2.0:  # Likely [0,255] range
            image = image / 255.0
        
        # Ensure [0,1] range and 3 channels - EXACT training logic
        image = torch.clamp(image, 0, 1)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] != 3:
            image = image[:3] if image.shape[0] > 3 else image.repeat(3, 1, 1)

        # Apply transforms
        return self.transforms(image)
    
    @staticmethod
    def is_valid_image(image: torch.Tensor) -> bool:
        """Check if image tensor is valid."""
        return (
            isinstance(image, torch.Tensor) and
            not torch.isnan(image).any() and
            not torch.isinf(image).any() and
            len(image.shape) >= 2
        )
    
    def test_single_image(self, image: torch.Tensor, true_label: int, image_idx: int) -> Dict:
        """Test a single image and return results."""
        try:
            # Debug first few images
            if image_idx <= 3:
                print(f"🔍 Debug image {image_idx}: shape={image.shape}, dtype={image.dtype}, "
                      f"range=[{image.min():.3f}, {image.max():.3f}]")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Debug preprocessed image
            if image_idx <= 3:
                print(f"🔍 After preprocessing: shape={processed_image.shape}, "
                      f"range=[{processed_image.min():.3f}, {processed_image.max():.3f}], "
                      f"mean={processed_image.mean():.3f}")
            
            # Add batch dimension and move to device
            batch_image = processed_image.unsqueeze(0).to(self.config.DEVICE)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(batch_image)
                probabilities = F.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
            
            # Calculate results
            prob_list = [round(p, 4) for p in probabilities.squeeze().cpu().tolist()]
            confidence = max(prob_list)
            is_correct = prediction == true_label
            
            # Print result immediately
            correct_symbol = "✅" if is_correct else "❌"
            print(f"🖼️ Image {image_idx:4d}: {correct_symbol} Pred: {self.config.CLASS_NAMES[prediction]:13s} "
                  f"(conf: {confidence:.4f}) | True: {self.config.CLASS_NAMES[true_label]:13s} | "
                  f"Probs: {prob_list}")
            
            return {
                'prediction': prediction,
                'true_label': true_label,
                'probabilities': prob_list,
                'confidence': confidence,
                'correct': is_correct
            }
            
        except Exception as e:
            print(f"❌ Error processing image {image_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_test_streaming(self, subfolder: str) -> Dict[str, float]:
        """Process and test images one by one, showing results immediately."""
        if subfolder not in self.config.CLASS_MAP:
            raise ValueError(f"Invalid subfolder: {subfolder}. Choose from {list(self.config.CLASS_MAP.keys())}")
        
        subfolder_path = self.config.TEST_DIR / subfolder
        if not subfolder_path.exists():
            raise FileNotFoundError(f"Folder not found: {subfolder_path}")
        
        print(f"\n📂 Testing images from: {subfolder_path}")
        
        pt_files = sorted(list(subfolder_path.glob("*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {subfolder_path}")
        
        class_label = self.config.CLASS_MAP[subfolder]
        
        # Statistics tracking
        total_correct = 0
        total_images = 0
        all_results = []
        
        print(f"\n🧪 Starting inference on '{subfolder}' images...")
        print("=" * 80)
        
        # Process each .pt file
        for file_idx, pt_file in enumerate(pt_files):
            print(f"\n📄 Processing file {file_idx + 1}/{len(pt_files)}: {pt_file.name}")
            
            try:
                # Load images from current file
                images = torch.load(pt_file, map_location="cpu", weights_only=False)
                
                if not isinstance(images, torch.Tensor) or len(images.shape) != 4:
                    print(f"❌ Invalid format in {pt_file.name}")
                    continue
                
                print(f"📊 File contains {images.shape[0]} images (shape: {images.shape})")
                
                # Debug first file's first image
                if file_idx == 0:
                    first_img = images[0]
                    print(f"🔍 First file data check: dtype={first_img.dtype}, "
                          f"range=[{first_img.min():.3f}, {first_img.max():.3f}]")
                
                print("-" * 60)
                
                # Process each image in the file individually
                file_correct = 0
                for img_idx in range(images.shape[0]):
                    image = images[img_idx]
                    
                    # FIXED: Ensure correct format (C, H, W) - same as training
                    if len(image.shape) == 3 and image.shape[2] in [1, 3]:
                        image = image.permute(2, 0, 1)
                    
                    if self.is_valid_image(image):
                        # Global image index across all files
                        global_img_idx = total_images + 1
                        
                        # Test this single image
                        result = self.test_single_image(image, class_label, global_img_idx)
                        
                        if result:
                            all_results.append(result)
                            total_images += 1
                            if result['correct']:
                                total_correct += 1
                                file_correct += 1
                            
                            # Show running accuracy every 100 images
                            if total_images % 100 == 0:
                                running_acc = 100 * total_correct / total_images
                                print(f"  📈 Running accuracy after {total_images} images: {running_acc:.2f}%")
                    else:
                        print(f"⚠️ Skipping invalid image {img_idx + 1} in {pt_file.name}")
                
                # File summary
                file_acc = 100 * file_correct / images.shape[0] if images.shape[0] > 0 else 0
                print(f"\n📋 File summary: {file_correct}/{images.shape[0]} correct ({file_acc:.2f}%)")
                
                # Clear memory
                del images
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ Failed to process {pt_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate final statistics
        final_accuracy = 100 * total_correct / total_images if total_images > 0 else 0
        
        print("\n" + "=" * 80)
        print(f"✅ Testing complete for '{subfolder}'")
        print(f"🎯 Final Results:")
        print(f"   Total images processed: {total_images}")
        print(f"   Correct predictions: {total_correct}")
        print(f"   Final accuracy: {final_accuracy:.2f}%")
        
        # Additional statistics
        if all_results:
            confidences = [r['confidence'] for r in all_results]
            avg_confidence = sum(confidences) / len(confidences)
            correct_confidences = [r['confidence'] for r in all_results if r['correct']]
            incorrect_confidences = [r['confidence'] for r in all_results if not r['correct']]
            
            print(f"📊 Additional Statistics:")
            print(f"   Average confidence: {avg_confidence:.4f}")
            if correct_confidences:
                print(f"   Avg confidence (correct): {sum(correct_confidences)/len(correct_confidences):.4f}")
            if incorrect_confidences:
                print(f"   Avg confidence (incorrect): {sum(incorrect_confidences)/len(incorrect_confidences):.4f}")
        
        # Performance assessment
        if final_accuracy >= 90:
            performance = "EXCELLENT"
        elif final_accuracy >= 80:
            performance = "VERY GOOD"
        elif final_accuracy >= 70:
            performance = "GOOD"
        elif final_accuracy >= 60:
            performance = "FAIR"
        else:
            performance = "POOR"
        
        print(f"🏆 Performance rating: {performance}")
        print("=" * 80)
        
        return {
            'accuracy': final_accuracy,
            'correct': total_correct,
            'total': total_images,
            'all_results': all_results
        }


def main():
    """Main function to run the model testing."""
    config = Config()
    
    # Specify which subfolder to test
    TARGET_SUBFOLDER = "semi-synthetic"  # Change this to test different classes
    
    print(f"🚀 Starting model testing with FIXED configuration:")
    print(f"   Target subfolder: {TARGET_SUBFOLDER}")
    print(f"   Model path: {config.MODEL_PATH}")
    print(f"   Dropout rate: {config.DROPOUT_RATE}")
    print(f"   Device: {config.DEVICE}")
    
    # Initialize tester and run streaming test
    tester = ModelTester(config)
    results = tester.run_test_streaming(TARGET_SUBFOLDER)
    
    return results


if __name__ == "__main__":
    main()