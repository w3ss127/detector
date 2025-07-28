import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import numpy as np
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import random
import warnings
from torch.cuda.amp import GradScaler, autocast
from collections import Counter

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ImprovedAttentionModule(nn.Module):
    """Simplified and more efficient attention module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ImprovedAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        reduced_channels = max(in_channels // reduction_ratio, 8)  # Minimum 8 channels
        self.channel_att = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Only use average pooling for efficiency
        avg_pool = self.avg_pool(x).view(b, c)
        channel_att = self.channel_att(avg_pool).view(b, c, 1, 1)
        
        return x * channel_att

class ImprovedEfficientNetWithAttention(nn.Module):
    """Improved EfficientNet with better regularization"""
    def __init__(self, num_classes=3, pretrained=True, dropout_rate=0.5):
        super(ImprovedEfficientNetWithAttention, self).__init__()
        
        self.backbone = efficientnet_b0(pretrained=pretrained)
        self.features = self.backbone.features
        
        # Reduce number of attention modules to prevent overfitting
        self.attention_modules = nn.ModuleDict()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = None
        self._setup_classifier = False
        self.dropout_rate = dropout_rate
        
    def _create_classifier(self, feature_size, num_classes):
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_size, 256),  # Reduced from 512
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.7),  # Progressive dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Only add attention at key bottleneck layers (reduced from 3 to 2)
            attention_key = f"attention_{i}"
            if i in [4, 6] and attention_key not in self.attention_modules:  # Reduced attention points
                self.attention_modules[attention_key] = ImprovedAttentionModule(x.size(1))
                if x.is_cuda:
                    self.attention_modules[attention_key] = self.attention_modules[attention_key].cuda()
            
            if attention_key in self.attention_modules:
                x = self.attention_modules[attention_key](x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if not self._setup_classifier:
            self._create_classifier(x.size(1), 3)
            if x.is_cuda:
                self.classifier = self.classifier.cuda()
            self._setup_classifier = True
        
        x = self.classifier(x)
        return x

class LazyTensorDataset(Dataset):
    """Custom dataset with improved data handling"""
    def __init__(self, file_list, transform=None, return_indices=False):
        self.file_list = file_list
        self.transform = transform
        self.return_indices = return_indices
        self.cumulative_sizes = []
        self.class_counts = Counter()
        
        current_idx = 0
        for file_path, class_label in self.file_list:
            try:
                tensor_batch = torch.load(file_path, map_location='cpu')
                batch_size = tensor_batch.size(0) if tensor_batch.dim() == 4 else 1
                current_idx += batch_size
                self.cumulative_sizes.append(current_idx)
                self.class_counts[class_label] += batch_size
            except Exception as e:
                print(f"Skipping {file_path}: Error loading - {e}")
                self.cumulative_sizes.append(current_idx)
    
    def get_class_distribution(self):
        return dict(self.class_counts)
    
    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx):
        file_idx = next(i for i, size in enumerate(self.cumulative_sizes) if idx < size)
        file_path, class_label = self.file_list[file_idx]
        tensor_batch = torch.load(file_path, map_location='cpu')
        
        local_idx = idx - (self.cumulative_sizes[file_idx-1] if file_idx > 0 else 0)
        tensor = tensor_batch[local_idx] if tensor_batch.dim() == 4 else tensor_batch
        
        # Improved tensor preprocessing
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
        image = Image.fromarray(numpy_img.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        if self.return_indices:
            return image, class_label, idx
        return image, class_label

def create_improved_data_transforms():
    """Enhanced data augmentation to reduce overfitting"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Slightly larger for random crop
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))  # Cutout augmentation
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def create_weighted_sampler(dataset):
    """Create weighted sampler for class imbalance"""
    class_distribution = dataset.get_class_distribution()
    print(f"Class distribution: {class_distribution}")
    
    total_samples = sum(class_distribution.values())
    class_weights = {cls: total_samples / count for cls, count in class_distribution.items()}
    
    # Create sample weights
    sample_weights = []
    for i in range(len(dataset)):
        _, class_label = dataset[i]
        sample_weights.append(class_weights[class_label])
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def load_tensor_data_improved(data_dir, train_split=0.8, val_split=0.1):
    """Improved data loading with better organization"""
    print("Loading tensor dataset with improved handling...")
    
    class_mapping = {
        'real': 0,
        'semi-synthetic': 1,  # Note: you mentioned "semi-synthetic" twice in your question
        'synthetic': 2
    }
    
    file_list = []
    train_dir = os.path.join(data_dir, 'basic')
    
    # Collect files per class for better splitting
    class_files = {class_name: [] for class_name in class_mapping.keys()}
    
    for class_name, class_label in class_mapping.items():
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping...")
            continue
        pt_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.pt')]
        print(f"Found {len(pt_files)} .pt files in {class_name} directory")
        class_files[class_name] = [(pt_file, class_label) for pt_file in pt_files]
    
    # Stratified splitting
    train_files, val_files, test_files = [], [], []
    for class_name, files in class_files.items():
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train+n_val])
        test_files.extend(files[n_train+n_val:])
    
    # Final shuffle
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    
    print(f"Stratified split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")
    return train_files, val_files, test_files, list(class_mapping.keys())

def train_model_improved(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3):
    """Improved training with better optimization and regularization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Improved loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Better optimizer choice
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Improved learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Tracking metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    learning_rates = []
    
    best_val_acc = 0.0
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for data, target in train_bar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Validation phase
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
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        learning_rates.append(current_lr)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.2e}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, 'best_model_improved.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if early_stopping(val_acc, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
        
        print('-' * 60)
    
    return train_losses, val_losses, train_accuracies, val_accuracies, learning_rates

def main_improved():
    """Improved main training pipeline"""
    DATA_DIR = 'datasets'
    BATCH_SIZE = 16  # Reduced batch size for better generalization
    NUM_EPOCHS = 50  # Increased epochs with early stopping
    LEARNING_RATE = 1e-3  # More conservative learning rate
    DROPOUT_RATE = 0.5  # Increased dropout
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        # Load data with improved splitting
        train_files, val_files, test_files, class_names = load_tensor_data_improved(DATA_DIR)
        print(f"Classes: {class_names}")
        
        # Create datasets with improved transforms
        train_transform, val_transform = create_improved_data_transforms()
        
        train_dataset = LazyTensorDataset(train_files, transform=train_transform)
        val_dataset = LazyTensorDataset(val_files, transform=val_transform)
        test_dataset = LazyTensorDataset(test_files, transform=val_transform)
        
        # Create weighted sampler for training
        weighted_sampler = create_weighted_sampler(train_dataset)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                sampler=weighted_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=4, pin_memory=True)
        
        # Create improved model
        model = ImprovedEfficientNetWithAttention(
            num_classes=len(class_names), 
            dropout_rate=DROPOUT_RATE
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train model
        history = train_model_improved(
            model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE
        )
        
        train_losses, val_losses, train_accuracies, val_accuracies, learning_rates = history
        

        
        # Load best model and evaluate
        checkpoint = torch.load('best_model_improved.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        test_accuracy = evaluate_model_improved(model, test_loader, class_names)
        
        print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
        print(f"Best Validation Accuracy during training: {checkpoint['val_acc']:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()



def evaluate_model_improved(model, test_loader, class_names):
    """Enhanced model evaluation with more metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            with autocast():
                output = model(data)
                probabilities = F.softmax(output, dim=1)
                _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    print('\nDetailed Classification Report:')
    print(classification_report(all_targets, all_predictions, target_names=class_names, digits=4))
    
    # Enhanced confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.show()
    
    return accuracy

if __name__ == "__main__":
    main_improved()