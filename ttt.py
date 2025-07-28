import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import sys
import time

class CustomImageDataset(torch.utils.data.Dataset):
    """Custom dataset for loading .pt files from class directories"""
    def __init__(self, root_dir, transform=None, test_folder=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.file_info = []  # Track which file each image came from
        self.class_names = ["real", "synthetic", "semi-synthetic"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        folders = [test_folder] if test_folder in self.class_names else self.class_names
        
        for class_name in folders:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"❌ Folder not found: {class_dir}")
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.pt'):
                    file_path = os.path.join(class_dir, file_name)
                    self._load_pt_file(file_path, class_idx, file_name)
        
        print(f"Loaded {len(self.images)} images total")
        self._print_class_distribution()
    
    def _load_pt_file(self, file_path, class_idx, file_name):
        """Load and process a single .pt file"""
        try:
            start_time = time.time()
            data = torch.load(file_path, map_location='cpu')
            print(f"Loaded {file_path} in {time.time() - start_time:.2f} seconds")
            
            if len(data.shape) != 4 or data.shape[1] != 3:
                print(f"❌ Invalid tensor shape in {file_path}: {data.shape}")
                return
            
            # Normalize to [0, 1] if needed
            if data.dtype == torch.uint8:
                data = data.float() / 255.0
            
            # Store raw data - transforms will be applied in __getitem__
            for i in range(data.shape[0]):
                self.images.append(data[i])
                self.labels.append(class_idx)
                self.file_info.append((file_name, i))
                
            print(f"Processed {file_path} with {data.shape[0]} images")
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
    
    def _print_class_distribution(self):
        """Print distribution of classes in dataset"""
        print("Class distribution:")
        for class_name in self.class_names:
            count = self.labels.count(self.class_to_idx[class_name])
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        file_info = self.file_info[idx]
        
        # Apply transforms here, not during loading
        if self.transform:
            image = self.transform(image)
        
        return image, label, file_info

def repeat_grayscale_to_rgb(x):
    """Convert single-channel images to three channels."""
    if x.shape[0] == 1:
        return x.expand(3, -1, -1)
    return x

def create_model(num_classes=3):
    """Create ResNet34 model for classification - MUST match training architecture"""
    model = models.resnet34(pretrained=False)  # Set to False since we're loading trained weights
    num_features = model.fc.in_features
        
    # Replace the final fully connected layer - MUST match training exactly
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def load_model(model_path, device, num_classes=3):
    """Load trained model from checkpoint"""
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found.")
        sys.exit(1)
    
    start_time = time.time()
    model = create_model(num_classes)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Handle DataParallel models
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        print(f"✅ Loaded model weights from {model_path} in {time.time() - start_time:.2f} seconds")
        return model
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def run_inference_individual(model, dataset, device, class_names, max_images_to_print=100, results_file=None):
    """Run inference on each image individually and print results"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    current_file = None
    print_count = {}  # Track printed images per file
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Processing images"):
            image, label, file_info = dataset[i]
            file_name, img_idx = file_info
            
            # Print file header when we encounter a new file
            if current_file != file_name:
                current_file = file_name
                print(f"\n📄 Results for {file_name}:")
                print_count[file_name] = 0
            
            # Move image to device and add batch dimension
            image = image.to(device).unsqueeze(0)
            
            # Run inference
            output = model(image)
            prob = F.softmax(output, dim=1).squeeze()
            pred = torch.argmax(prob).item()
            
            # Store for metrics
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(prob.cpu().numpy())
            
            # Print and save individual result
            actual_class = class_names[label]
            predicted_class = class_names[pred]
            probs_formatted = [round(p, 4) for p in prob.cpu().numpy()]
            is_correct = "✅" if pred == label else "❌"
            
            result_str = f"  Image {img_idx+1}: {is_correct} Predicted → {predicted_class}, Actual → {actual_class}, Probs → {probs_formatted}"
            
            if print_count[file_name] < max_images_to_print:
                print(result_str)
                print_count[file_name] += 1
            
            if results_file:
                with open(results_file, 'a') as f:
                    f.write(result_str + "\n")
        
        # Show truncation message if needed
        for file_name in print_count:
            total_images_in_file = len([fi for fi in dataset.file_info if fi[0] == file_name])
            if print_count[file_name] < total_images_in_file:
                print(f"  ... (showing {print_count[file_name]} of {total_images_in_file} images)")
    
    return all_preds, all_labels

def print_overall_results(predictions, labels, class_names, results_file=None):
    """Print overall accuracy and classification report"""
    accuracy = accuracy_score(labels, predictions)
    correct_count = sum(p == l for p, l in zip(predictions, labels))
    total_count = len(labels)
    report = classification_report(labels, predictions, target_names=class_names, zero_division=0)
    
    print(f"\n✅ Inference complete.")
    print(f"🎯 Overall Test Accuracy: {accuracy:.4f}")
    print(f"Correct: {correct_count} / Total: {total_count}")
    print("\nClassification Report:")
    print(report)
    
    if results_file:
        with open(results_file, 'a') as f:
            f.write(f"\nInference complete.\n")
            f.write(f"Overall Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Correct: {correct_count} / Total: {total_count}\n")
            f.write("\nClassification Report:\n")
            f.write(report)

def main(test_folder=None, save_results_to_file=True, max_images_to_print=100):
    """
    Args:
        test_folder: Optional specific class folder to load (e.g., 'semi-synthetic')
        save_results_to_file: If True, save per-image results to a file
        max_images_to_print: Maximum number of per-image results to print to console
    """
    # Configuration
    BASE_DIR = "datasets"
    TEST_DIR = os.path.join(BASE_DIR, "test")  # Changed from "basic" to "test"
    MODEL_PATH = "best_model_mcc_0.9266_618116ec.pth"
    CLASS_NAMES = ["real", "synthetic", "semi-synthetic"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    print(f"🔧 Using device: {DEVICE}")
    
    # Define transforms - MUST match training transforms exactly
    test_transforms = transforms.Compose([
        transforms.Lambda(repeat_grayscale_to_rgb),  # Handle grayscale images
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization.
    ])
    
    # Load model
    start_time = time.time()
    model = load_model(MODEL_PATH, DEVICE)
    print(f"Model loading completed in {time.time() - start_time:.2f} seconds")
    
    # Load dataset
    print(f"\n📂 Loading test dataset from: {TEST_DIR}")
    if test_folder:
        print(f"📌 Testing only folder: {test_folder}")
    
    start_time = time.time()
    dataset = CustomImageDataset(TEST_DIR, transform=test_transforms, test_folder=test_folder)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    
    if len(dataset) == 0:
        print("❌ No valid test images found.")
        sys.exit(1)
    
    print(f"📊 Total test samples: {len(dataset)}")
    
    # Prepare output file if saving results
    results_file = f"test_results_{test_folder or 'all'}.txt" if save_results_to_file else None
    if results_file:
        print(f"📝 Saving detailed results to {results_file}")
        with open(results_file, 'w') as f:
            f.write(f"Test Results for {test_folder or 'all folders'}\n")
            f.write(f"Model: {MODEL_PATH}\n")
            f.write(f"Device: {DEVICE}\n")
            f.write(f"Total samples: {len(dataset)}\n\n")
    
    # Run inference
    start_time = time.time()
    predictions, labels = run_inference_individual(model, dataset, DEVICE, CLASS_NAMES, max_images_to_print, results_file)
    print(f"Inference completed in {time.time() - start_time:.2f} seconds")
    
    # Print overall results
    if predictions:
        print_overall_results(predictions, labels, CLASS_NAMES, results_file)
    else:
        print("❌ No predictions made. Check input data or model.")
    
    # Clear GPU memory
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # You can modify these parameters as needed
    main(test_folder="None", save_results_to_file=True, max_images_to_print=50)