import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import random
import warnings
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AttentionModule(nn.Module):
    """Spatial Attention Module with adaptive input handling"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.channel_att = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        channel_att = self.sigmoid(self.channel_att(avg_pool) + self.channel_att(max_pool))
        channel_att = channel_att.view(b, c, 1, 1)
        x = x * channel_att
        
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
        
        self.backbone = efficientnet_b0(pretrained=pretrained)
        self.features = self.backbone.features
        self.attention_modules = nn.ModuleDict()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = None
        self._setup_classifier = False
        
    def _create_classifier(self, feature_size, num_classes):
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        if not hasattr(self, '_input_shape_printed'):
            print(f"Input tensor shape: {x.shape}")
            self._input_shape_printed = True
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            if not hasattr(self, '_feature_shapes_printed') and i < 8:
                print(f"After layer {i}: {x.shape}")
            
            attention_key = f"attention_{i}"
            if i in [2, 4, 6] and attention_key not in self.attention_modules:
                self.attention_modules[attention_key] = AttentionModule(x.size(1))
                if x.is_cuda:
                    self.attention_modules[attention_key] = self.attention_modules[attention_key].cuda()
            
            if attention_key in self.attention_modules:
                x = self.attention_modules[attention_key](x)
        
        if not hasattr(self, '_feature_shapes_printed'):
            print(f"Final feature shape before pooling: {x.shape}")
            self._feature_shapes_printed = True
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if not self._setup_classifier:
            self._create_classifier(x.size(1), 3)
            if x.is_cuda:
                self.classifier = self.classifier.cuda()
            self._setup_classifier = True
            print(f"Created classifier with input size: {x.size(1)}")
        
        x = self.classifier(x)
        return x

class LazyTensorDataset(Dataset):
    """Custom dataset for lazy loading .pt files"""
    def __init__(self, file_list, transform=None):
        self.file_list = file_list  # List of (file_path, class_label) tuples
        self.transform = transform
        self.cumulative_sizes = []
        current_idx = 0
        for file_path, _ in self.file_list:
            try:
                tensor_batch = torch.load(file_path, map_location='cpu')
                batch_size = tensor_batch.size(0) if tensor_batch.dim() == 4 else 1
                current_idx += batch_size
                self.cumulative_sizes.append(current_idx)
            except Exception as e:
                print(f"Skipping {file_path}: Error loading - {e}")
                self.cumulative_sizes.append(current_idx)
    
    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx):
        file_idx = next(i for i, size in enumerate(self.cumulative_sizes) if idx < size)
        file_path, class_label = self.file_list[file_idx]
        tensor_batch = torch.load(file_path, map_location='cpu')
        
        local_idx = idx - (self.cumulative_sizes[file_idx-1] if file_idx > 0 else 0)
        tensor = tensor_batch[local_idx] if tensor_batch.dim() == 4 else tensor_batch
        
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3 and (tensor.size(2) == 3 or tensor.size(2) == 1):
            tensor = tensor.permute(2, 0, 1)
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.size(0) > 3:
            tensor = tensor[:3]
        tensor = torch.clamp(tensor, 0, 1)
        numpy_img = (tensor * 255).byte().permute(1, 2, 0).numpy()
        
        if not hasattr(self, '_debug_printed') and idx < 3:
            print(f"Sample {idx}: Original tensor shape: {tensor.shape}")
            print(f"Sample {idx}: Processed tensor shape: {tensor.shape}")
            print(f"Sample {idx}: Numpy image shape: {numpy_img.shape}")
            print(f"Sample {idx}: Value range: {numpy_img.min()} to {numpy_img.max()}")
            if idx == 2:
                self._debug_printed = True
        
        image = Image.fromarray(numpy_img.astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        
        return image, class_label

def create_data_transforms():
    """Create simplified data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_tensor_data(data_dir, train_split=0.8, val_split=0.1):
    """Load tensor data lazily from .pt files"""
    print("Loading tensor dataset lazily...")
    
    class_mapping = {
        'real': 0,
        'semi-synthetic': 1,
        'synthetic': 2
    }
    
    file_list = []
    train_dir = os.path.join(data_dir, 'basic')
    
    for class_name, class_label in class_mapping.items():
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping...")
            continue
        pt_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.pt')]
        print(f"Found {len(pt_files)} .pt files in {class_name} directory")
        file_list.extend([(pt_file, class_label) for pt_file in pt_files])
    
    random.shuffle(file_list)
    n_total = len(file_list)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train+n_val]
    test_files = file_list[n_train+n_val:]
    
    print(f"Data split: {len(train_files)} train files, {len(val_files)} val files, {len(test_files)} test files")
    class_names = list(class_mapping.keys())
    return train_files, val_files, test_files, class_names

def create_tensor_dataloaders(train_files, val_files, test_files, batch_size=32):
    """Create data loaders for tensor data"""
    train_transform, val_transform = create_data_transforms()
    
    train_dataset = LazyTensorDataset(train_files, transform=train_transform)
    val_dataset = LazyTensorDataset(val_files, transform=val_transform)
    test_dataset = LazyTensorDataset(test_files, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=1e-4, accum_steps=4):
    """Train the model with mixed precision and gradient accumulation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            
            with autocast():
                output = model(data)
                loss = criterion(output, target) / accum_steps
            
            scaler.scale(loss).backward()
            train_loss += loss.item() * accum_steps
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item() * accum_steps:.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        if (batch_idx + 1) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                with autocast():
                    output = model(data)
                    val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_240000.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, class_names):
    """Evaluate the model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            with autocast():
                output = model(data)
                _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    print('\nClassification Report:')
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return accuracy

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main training pipeline"""
    DATA_DIR = 'datasets'
    BATCH_SIZE = 32  # Physical batch size, effective batch size of 128 with accum_steps=4
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    ACCUM_STEPS = 4  # For effective batch size of 128
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        train_files, val_files, test_files, class_names = load_tensor_data(DATA_DIR)
        print(f"Classes: {class_names}")
        
        train_loader, val_loader, test_loader = create_tensor_dataloaders(
            train_files, val_files, test_files, BATCH_SIZE
        )
        
        model = EfficientNetWithAttention(num_classes=len(class_names))
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, ACCUM_STEPS
        )
        
        plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
        
        model.load_state_dict(torch.load('model_240000.pth'))
        test_accuracy = evaluate_model(model, test_loader, class_names)
        
        print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
        
    except FileNotFoundError:
        print(f"Dataset directory '{DATA_DIR}' not found.")
        print("Please ensure your dataset follows this structure:")
        print("datasets/")
        print("└── train/")
        print("    ├── real/")
        print("    │   ├── batch1.pt")
        print("    │   ├── batch2.pt")
        print("    │   └── ...")
        print("    ├── semi-synthetic/")
        print("    │   ├── batch1.pt")
        print("    │   ├── batch2.pt")
        print("    │   └── ...")
        print("    └── synthetic/")
        print("        ├── batch1.pt")
        print("        ├── batch2.pt")
        print("        └── ...")
        print("\nEach .pt file should contain image tensors.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your .pt files format and ensure they contain image tensors.")

if __name__ == "__main__":
    main()