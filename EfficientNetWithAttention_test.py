import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

class AttentionModule(nn.Module):
    """Spatial Attention Module with adaptive input handling"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Channel attention with adaptive sizing
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.channel_att = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        channel_att = self.sigmoid(self.channel_att(avg_pool) + self.channel_att(max_pool))
        channel_att = channel_att.view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.spatial_att(spatial_input)
        
        x = x * spatial_att
        return x

class EfficientNetWithAttention(nn.Module):
    """EfficientNet with integrated attention modules - adaptive version"""
    def __init__(self, num_classes=3, pretrained=True):
        super(EfficientNetWithAttention, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = efficientnet_b0(pretrained=pretrained)
        
        # Remove the classifier
        self.features = self.backbone.features
        
        # Pre-initialize attention modules to avoid dynamic creation issues
        self.attention_modules = nn.ModuleDict()
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Pre-create classifier to avoid dynamic creation issues
        # EfficientNet-B0 final feature size is typically 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),  # EfficientNet-B0 has 1280 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Pass through feature blocks
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Add attention modules dynamically but only if not already created
            attention_key = f"attention_{i}"
            if i in [2, 4, 6] and attention_key not in self.attention_modules:
                # Create attention module for this layer's channels
                self.attention_modules[attention_key] = AttentionModule(x.size(1))
                if x.is_cuda:
                    self.attention_modules[attention_key] = self.attention_modules[attention_key].cuda()
            
            # Apply attention if available
            if attention_key in self.attention_modules:
                x = self.attention_modules[attention_key](x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Use pre-created classifier
        x = self.classifier(x)
        return x

def load_model(model_path, num_classes=3):
    """Load the trained model"""
    model = EfficientNetWithAttention(num_classes=num_classes, pretrained=False)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Initialize model with a dummy forward pass to create dynamic components
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Now load the state dict
    model.load_state_dict(state_dict, strict=False)
    
    return model

def preprocess_tensor(tensor):
    """Preprocess a single tensor for model input"""
    # Ensure tensor is float and in correct range
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    # Normalize tensor values if needed
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    # Ensure tensor is in [C, H, W] format
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    elif tensor.dim() == 2:  # Grayscale [H, W]
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3:
        # Check if it's [H, W, C] and convert to [C, H, W]
        if tensor.size(2) == 3 or tensor.size(2) == 1:
            tensor = tensor.permute(2, 0, 1)
    
    # Ensure we have 3 channels (RGB)
    if tensor.size(0) == 1:  # Grayscale
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.size(0) > 3:  # More than 3 channels
        tensor = tensor[:3]
    
    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image for transforms
    numpy_img = (tensor * 255).byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(numpy_img.astype(np.uint8))
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)

def test_single_file(pt_file_path, model_path, true_class=None):
    """
    Test a single .pt file with the trained model - processes each image individually
    
    Args:
        pt_file_path (str): Path to the .pt file containing image tensors
        model_path (str): Path to the trained model weights
        true_class (int): True class label (0=real, 1=semi-synthetic, 2=synthetic)
    """
    
    # Class names
    class_names = ['real', 'semi-synthetic', 'synthetic']
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load tensor data
    print(f"Loading data from {pt_file_path}...")
    try:
        tensor_data = torch.load(pt_file_path, map_location='cpu')
        print(f"Data loaded successfully")
        
        # Handle different tensor formats
        images = []
        if isinstance(tensor_data, torch.Tensor):
            if tensor_data.dim() == 4:  # [N, C, H, W]
                print(f"Found batch tensor with {tensor_data.size(0)} images")
                images = [tensor_data[i] for i in range(tensor_data.size(0))]
            elif tensor_data.dim() == 3:  # [C, H, W] - single image
                print(f"Found single image tensor")
                images = [tensor_data]
        elif isinstance(tensor_data, (list, tuple)):
            print(f"Found list/tuple with {len(tensor_data)} items")
            images = [tensor for tensor in tensor_data if isinstance(tensor, torch.Tensor)]
        elif isinstance(tensor_data, dict):
            print(f"Found dictionary with keys: {list(tensor_data.keys())}")
            if 'images' in tensor_data:
                tensor_batch = tensor_data['images']
                if tensor_batch.dim() == 4:
                    images = [tensor_batch[i] for i in range(tensor_batch.size(0))]
            elif 'data' in tensor_data:
                tensor_batch = tensor_data['data']
                if tensor_batch.dim() == 4:
                    images = [tensor_batch[i] for i in range(tensor_batch.size(0))]
            else:
                # Try first value
                first_key = list(tensor_data.keys())[0]
                first_value = tensor_data[first_key]
                if isinstance(first_value, torch.Tensor):
                    if first_value.dim() == 4:
                        images = [first_value[i] for i in range(first_value.size(0))]
                    elif first_value.dim() == 3:
                        images = [first_value]
        
        if not images:
            raise ValueError("No valid image tensors found in the file")
            
        print(f"Total images to process: {len(images)}")
        
    except Exception as e:
        print(f"Error loading {pt_file_path}: {e}")
        return
    
    # Process each image individually
    all_predictions = []
    all_confidences = []
    correct_count = 0
    
    print("\nProcessing images individually...")
    print("-" * 80)
    
    with torch.no_grad():
        for i, img_tensor in enumerate(images):
            try:
                # Preprocess single image
                processed_tensor = preprocess_tensor(img_tensor)
                
                # Add batch dimension and move to device
                input_tensor = processed_tensor.unsqueeze(0).to(device)
                
                # Forward pass
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
                
                all_predictions.append(prediction)
                all_confidences.append(confidence)
                
                # Check if correct
                is_correct = ""
                if true_class is not None:
                    if prediction == true_class:
                        correct_count += 1
                        is_correct = " ✓"
                    else:
                        is_correct = " ✗"
                
                # Print result for each image
                print(f"Image {i+1:4d}: {class_names[prediction]:15s} (conf: {confidence:.3f}){is_correct}")
                
            except Exception as e:
                print(f"Image {i+1:4d}: ERROR - {e}")
                continue
    
    # Calculate final results
    total_images = len(all_predictions)
    
    print("-" * 80)
    print(f"FINAL RESULTS")
    print("-" * 80)
    
    if true_class is not None:
        accuracy = (correct_count / total_images) * 100 if total_images > 0 else 0
        print(f"True class: {class_names[true_class]}")
        print(f"Correct predictions: {correct_count}/{total_images} = {accuracy:.2f}%")
    else:
        print(f"Total images processed: {total_images}")
    
    # Show class distribution
    class_counts = {i: sum(1 for pred in all_predictions if pred == i) for i in range(3)}
    print(f"\nPrediction distribution:")
    for class_idx, count in class_counts.items():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  {class_names[class_idx]}: {count} ({percentage:.1f}%)")
    
    print("=" * 80)
    
    return all_predictions, all_confidences

def main():
    # Try to parse command line arguments first
    parser = argparse.ArgumentParser(description='Test individual .pt file with trained model')
    parser.add_argument('pt_file', type=str, nargs='?', help='Path to .pt file containing images')
    parser.add_argument('model_path', type=str, nargs='?', help='Path to trained model weights (.pth file)')
    parser.add_argument('--true_class', type=int, choices=[0, 1, 2], 
                       help='True class: 0=real, 1=semi-synthetic, 2=synthetic')
    
    args = parser.parse_args()
    
    # If no arguments provided, ask for them interactively
    if not args.pt_file or not args.model_path:
        print("Interactive mode - please provide the required information:")
        
        if not args.pt_file:
            args.pt_file = input("Enter path to .pt file: ").strip()
        
        if not args.model_path:
            args.model_path = input("Enter path to model file (.pth): ").strip()
            if not args.model_path:
                args.model_path = 'best_model.pth'  # Default
        
        if args.true_class is None:
            true_class_input = input("Enter true class (0=real, 1=semi-synthetic, 2=synthetic) or press Enter to skip: ").strip()
            if true_class_input and true_class_input.isdigit():
                args.true_class = int(true_class_input)
    
    # Check if files exist
    if not os.path.exists(args.pt_file):
        print(f"Error: .pt file '{args.pt_file}' not found")
        print("Please check the file path and try again.")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        print("Please check the file path and try again.")
        return
    
    print(f"\nTesting file: {args.pt_file}")
    print(f"Using model: {args.model_path}")
    if args.true_class is not None:
        class_names = ['real', 'semi-synthetic', 'synthetic']
        print(f"True class: {class_names[args.true_class]}")
    print("=" * 80)
    
    # Run test
    predictions, confidences = test_single_file(
        args.pt_file, 
        args.model_path, 
        args.true_class
    )

def quick_test(pt_file, model_path='best_model.pth', true_class=None):
    """Quick test function for easy usage"""
    return test_single_file(pt_file, model_path, true_class)

if __name__ == "__main__":
    main()