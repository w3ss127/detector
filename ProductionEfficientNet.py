import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, efficientnet_b1
import numpy as np
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import random
import warnings
from collections import Counter, defaultdict
import json
import time
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class ProductionEfficientNet(nn.Module):
    """Production EfficientNet with balanced anti-overfitting measures"""
    
    def __init__(self, num_classes=3, model_size='b0', pretrained=True, dropout_rate=0.5):
        super(ProductionEfficientNet, self).__init__()
        
        # Select model architecture
        if model_size == 'b0':
            self.backbone = efficientnet_b0(pretrained=pretrained)
        elif model_size == 'b1':
            self.backbone = efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError("Model size must be 'b0' or 'b1'")
        
        # Moderate layer freezing
        self._freeze_layers(freeze_ratio=0.5)  # Freeze 50% of layers
        
        # Get feature dimensions
        num_features = self.backbone.classifier[1].in_features
        
        # Simplified classifier with moderate regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, num_classes)
        )
        
        # Replace backbone classifier
        self.backbone.classifier = self.classifier
        
        # Apply weight initialization
        self._initialize_weights()
        
        # Remove spectral normalization for faster training
        # self._apply_spectral_norm()
    
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
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # Increased std for better initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)

class MixupDataset(Dataset):
    """Dataset with Mixup augmentation"""
    
    def __init__(self, base_dataset, alpha=0.2, prob=0.3):
        self.base_dataset = base_dataset
        self.alpha = alpha
        self.prob = prob
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        if random.random() < self.prob:
            mix_idx = random.randint(0, len(self.base_dataset) - 1)
            mix_img, mix_label = self.base_dataset[mix_idx]
            
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1
                
            img = lam * img + (1 - lam) * mix_img
            return img, (label, mix_label, lam)
        
        return img, label

class RobustTensorDataset(Dataset):
    """Robust tensor dataset with comprehensive preprocessing"""
    
    def __init__(self, tensor_data, transform=None, validate_data=True):
        self.tensor_data = tensor_data
        self.transform = transform
        
        if validate_data:
            self._validate_and_clean_data()
    
    def _validate_and_clean_data(self):
        """Validate and clean tensor data"""
        valid_data = []
        for i, (tensor, label) in enumerate(self.tensor_data):
            try:
                if self._is_valid_tensor(tensor):
                    valid_data.append((tensor, label))
            except Exception as e:
                logger.warning(f"Skipping invalid tensor at index {i}: {e}")
        
        self.tensor_data = valid_data
        logger.info(f"Validated dataset: {len(self.tensor_data)} valid samples")
    
    def _is_valid_tensor(self, tensor):
        """Check if tensor is valid"""
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.numel() == 0:
            return False
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
        return True
    
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        tensor, label = self.tensor_data[idx]
        
        try:
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            
            if tensor.max() > 10.0:
                tensor = tensor / 255.0
            elif tensor.max() > 2.0:
                tensor = torch.clamp(tensor / tensor.max(), 0, 1)
            
            tensor = self._ensure_proper_dimensions(tensor)
            
            tensor = torch.clamp(tensor, 0, 1)
            numpy_img = (tensor * 255).byte().permute(1, 2, 0).numpy()
            image = Image.fromarray(numpy_img.astype(np.uint8))
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.warning(f"Error processing tensor at index {idx}: {e}")
            blank_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, label
    
    def _ensure_proper_dimensions(self, tensor):
        """Ensure tensor has proper dimensions [C, H, W]"""
        if tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3 and tensor.size(2) in [1, 3]:
            tensor = tensor.permute(2, 0, 1)
        
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.size(0) > 3:
            tensor = tensor[:3]
        elif tensor.size(0) == 2:
            tensor = torch.cat([tensor, tensor[:1]], dim=0)
        
        return tensor

def create_production_transforms(input_size=224, stage='train'):
    """Create balanced transforms with different stages"""
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    if stage == 'train':
        transform = transforms.Compose([
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    
    elif stage == 'val':
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    
    elif stage == 'test_tta':
        transform = transforms.Compose([
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    
    return transform

def load_and_balance_data(data_dir, max_samples_per_class=1500, min_samples_per_class=500):
    """Load and balance dataset with comprehensive validation"""
    
    logger.info("Loading and balancing dataset...")
    
    class_mapping = {
        'real': 0,
        'semi-synthetic': 1,
        'synthetic': 2
    }
    
    train_dir = os.path.join(data_dir, 'basic')
    if not os.path.exists(train_dir):
        raise ValueError(f"Data directory not found: {train_dir}")
    
    class_stats = defaultdict(list)
    
    for class_name in class_mapping.keys():
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        pt_files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
        
        for pt_file in tqdm(pt_files, desc=f'Analyzing {class_name}'):
            pt_path = os.path.join(class_dir, pt_file)
            try:
                tensor_batch = torch.load(pt_path, map_location='cpu')
                
                if isinstance(tensor_batch, torch.Tensor):
                    if tensor_batch.dim() == 4:
                        for i in range(tensor_batch.size(0)):
                            class_stats[class_name].append(tensor_batch[i])
                    else:
                        class_stats[class_name].append(tensor_batch)
                elif isinstance(tensor_batch, (list, tuple)):
                    for tensor in tensor_batch:
                        if isinstance(tensor, torch.Tensor):
                            class_stats[class_name].append(tensor)
                            
            except Exception as e:
                logger.warning(f"Error loading {pt_path}: {e}")
    
    balanced_data = []
    class_counts = {}
    
    for class_name, tensors in class_stats.items():
        class_label = class_mapping[class_name]
        
        valid_tensors = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                if not (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                    valid_tensors.append(tensor)
        
        logger.info(f"Class {class_name}: {len(valid_tensors)} valid tensors")
        
        if len(valid_tensors) < min_samples_per_class:
            logger.warning(f"Skipping {class_name}: only {len(valid_tensors)} samples (min: {min_samples_per_class})")
            continue
        
        target_samples = min(len(valid_tensors), max_samples_per_class)
        
        if len(valid_tensors) > target_samples:
            selected_tensors = random.sample(valid_tensors, target_samples)
        else:
            selected_tensors = valid_tensors
        
        for tensor in selected_tensors:
            balanced_data.append((tensor, class_label))
        
        class_counts[class_name] = len(selected_tensors)
        logger.info(f"Selected {len(selected_tensors)} samples for {class_name}")
    
    if not balanced_data:
        raise ValueError("No valid data loaded after balancing")
    
    random.shuffle(balanced_data)
    
    n_total = len(balanced_data)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    train_data = balanced_data[:n_train]
    val_data = balanced_data[n_train:n_train+n_val]
    test_data = balanced_data[n_train+n_val:]
    
    logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    logger.info(f"Class distribution: {class_counts}")
    
    class_names = [name for name in class_mapping.keys() if name in class_counts]
    
    return train_data, val_data, test_data, class_names, class_counts

def custom_collate_fn(batch):
    """Custom collate function to handle mixup and non-mixup samples"""
    images = []
    labels = []
    
    for item in batch:
        images.append(item[0])
        labels.append(item[1])
    
    images = torch.stack(images)
    return images, labels

def create_production_dataloaders(train_data, val_data, test_data, batch_size=16, input_size=224, use_mixup=True):
    """Create production dataloaders with balanced enhancements"""
    
    train_transform = create_production_transforms(input_size, 'train')
    val_transform = create_production_transforms(input_size, 'val')
    
    train_base = RobustTensorDataset(train_data, train_transform)
    val_dataset = RobustTensorDataset(val_data, val_transform)
    test_dataset = RobustTensorDataset(test_data, val_transform)
    
    if use_mixup:
        train_dataset = MixupDataset(train_base, alpha=0.2, prob=0.3)
        logger.info("Mixup augmentation enabled for training")
        use_custom_collate = True
    else:
        train_dataset = train_base
        use_custom_collate = False
        logger.info("Mixup disabled, using standard collate function")
    
    labels = [item[1] for item in train_data]
    label_counts = Counter(labels)
    
    total_samples = len(labels)
    num_classes = len(label_counts)
    class_weights = {}
    
    for label, count in label_counts.items():
        class_weights[label] = total_samples / (num_classes * count)
    
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn if use_custom_collate else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def safe_tensor_conversion(labels, device):
    """Safely convert labels to tensor, handling nested structures"""
    try:
        if isinstance(labels, torch.Tensor):
            return labels.to(device)
        
        if isinstance(labels, list):
            if all(isinstance(x, (int, np.integer)) for x in labels):
                return torch.tensor(labels, dtype=torch.long, device=device)
            
            if all(isinstance(x, torch.Tensor) for x in labels):
                scalar_values = []
                for x in labels:
                    if x.dim() == 0:
                        scalar_values.append(x.item())
                    else:
                        scalar_values.append(x.squeeze().item() if x.numel() == 1 else x[0].item())
                return torch.tensor(scalar_values, dtype=torch.long, device=device)
            
            if all(isinstance(x, (np.number, np.ndarray)) for x in labels):
                scalar_values = []
                for x in labels:
                    if isinstance(x, np.ndarray):
                        scalar_values.append(int(x.item() if x.size == 1 else x[0]))
                    else:
                        scalar_values.append(int(x))
                return torch.tensor(scalar_values, dtype=torch.long, device=device)
            
            extracted_values = []
            for item in labels:
                if isinstance(item, (int, np.integer)):
                    extracted_values.append(int(item))
                elif isinstance(item, torch.Tensor):
                    if item.dim() == 0:
                        extracted_values.append(item.item())
                    else:
                        extracted_values.append(item.squeeze().item() if item.numel() == 1 else int(item[0]))
                elif isinstance(item, (np.number, np.ndarray)):
                    if isinstance(item, np.ndarray):
                        extracted_values.append(int(item.item() if item.size == 1 else item[0]))
                    else:
                        extracted_values.append(int(item))
                elif isinstance(item, (list, tuple)):
                    if len(item) > 0:
                        first_item = item[0]
                        if isinstance(first_item, (int, np.integer)):
                            extracted_values.append(int(first_item))
                        else:
                            extracted_values.append(0)
                    else:
                        extracted_values.append(0)
                else:
                    logger.warning(f"Unexpected item type in labels: {type(item)}, using fallback value 0")
                    extracted_values.append(0)
            
            if extracted_values:
                return torch.tensor(extracted_values, dtype=torch.long, device=device)
        
        if isinstance(labels, (int, np.integer)):
            return torch.tensor([labels], dtype=torch.long, device=device)
        
        logger.warning(f"Creating fallback for unexpected label format: {type(labels)}")
        if isinstance(labels, list) and len(labels) > 0:
            return torch.zeros(len(labels), dtype=torch.long, device=device)
        else:
            return torch.tensor([0], dtype=torch.long, device=device)
            
    except Exception as e:
        logger.error(f"Error in label conversion: {e}, labels type: {type(labels)}")
        if isinstance(labels, list) and len(labels) > 0:
            return torch.zeros(len(labels), dtype=torch.long, device=device)
        return torch.tensor([0], dtype=torch.long, device=device)

class ProductionTrainer:
    """Production trainer with comprehensive monitoring"""
    
    def __init__(self, model, train_loader, val_loader, num_classes, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config = {
            'max_epochs': 30,
            'learning_rate': 1e-4,  # Increased for better convergence
            'weight_decay': 0.05,   # Reduced for less regularization
            'label_smoothing': 0.1, # Reduced for clearer class boundaries
            'gradient_clip': 1.0,   # Increased for stability
            'patience': 30,         # Increased for more training time
            'overfitting_threshold': 5.0,
            'min_lr': 1e-6,        # Adjusted for better scheduling
            'warmup_epochs': 20,    # Extended warmup
            'save_frequency': 10
        }
        
        if config:
            self.config.update(config)
        
        self.model.to(self.device)
        self._setup_training()
        
        self.history = defaultdict(list)
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        self.results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])
        
        #self.optimizer = optim.AdamW(
           # self.model.parameters(),
            #lr=self.config['learning_rate'],
           # weight_decay=self.config['weight_decay'],
           # betas=(0.9, 0.999),
            #eps=1e-8
        #)
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.01,momentum=0.8,nesterov=True,weight_decay=self.config['weight_decay'])
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=30,  # Increased for longer cycles
            T_mult=1,
            eta_min=self.config['min_lr']
        )
        
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config['warmup_epochs']
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch with robust label handling"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                data, labels = batch_data
                data = data.to(self.device)
                
                is_mixup_batch = False
                if isinstance(labels, list) and len(labels) > 0:
                    first_label = labels[0]
                    if isinstance(first_label, tuple) and len(first_label) == 3:
                        try:
                            label_a, label_b, lam = first_label
                            if (isinstance(label_a, (int, np.integer)) and 
                                isinstance(label_b, (int, np.integer)) and 
                                isinstance(lam, (float, np.floating))):
                                if all(isinstance(x, tuple) and len(x) == 3 for x in labels):
                                    is_mixup_batch = True
                        except (ValueError, TypeError):
                            pass
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                if is_mixup_batch:
                    try:
                        target_a = torch.tensor([label[0] for label in labels], dtype=torch.long).to(self.device)
                        target_b = torch.tensor([label[1] for label in labels], dtype=torch.long).to(self.device)
                        lam_values = torch.tensor([label[2] for label in labels], dtype=torch.float).to(self.device)
                        
                        loss = 0
                        for i in range(len(lam_values)):
                            lam = lam_values[i]
                            loss += lam * self.criterion(output[i:i+1], target_a[i:i+1]) + \
                                   (1 - lam) * self.criterion(output[i:i+1], target_b[i:i+1])
                        loss = loss / len(lam_values)
                        
                        _, predicted = output.max(1)
                        total += target_a.size(0)
                        correct += predicted.eq(target_a).sum().item()
                        
                    except Exception as e:
                        logger.error(f"Error in mixup processing: {e}")
                        target = safe_tensor_conversion(labels, self.device)
                        if target.size(0) != data.size(0):
                            target = target[:data.size(0)]
                        loss = self.criterion(output, target)
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                        
                else:
                    target = safe_tensor_conversion(labels, self.device)
                    
                    if target.size(0) != data.size(0):
                        if target.size(0) > data.size(0):
                            target = target[:data.size(0)]
                        else:
                            logger.warning(f"Target size {target.size(0)} < data size {data.size(0)}")
                            padding = torch.full((data.size(0) - target.size(0),), 
                                               target[-1].item() if target.numel() > 0 else 0, 
                                               dtype=torch.long, device=self.device)
                            target = torch.cat([target, padding])
                    
                    loss = self.criterion(output, target)
                    
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                
                self.optimizer.step()
                
                if epoch < self.config['warmup_epochs']:
                    self.warmup_scheduler.step()
                else:
                    self.scheduler.step(epoch + batch_idx / len(self.train_loader))
                
                running_loss += loss.item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        train_loss = running_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        train_acc = 100. * correct / total if total > 0 else 0
        
        return train_loss, train_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return val_loss, val_acc, f1
    
    def train(self):
        """Main training loop"""
        logger.info("Starting production training...")
        logger.info(f"Configuration: {self.config}")
        
        start_time = time.time()
        
        for epoch in range(self.config['max_epochs']):
            try:
                train_loss, train_acc = self.train_epoch(epoch)
                
                val_loss, val_acc, val_f1 = self.validate_epoch()
                
                overfitting_gap = train_acc - val_acc
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)
                self.history['overfitting_gap'].append(overfitting_gap)
                self.history['learning_rate'].append(current_lr)
                
                print(f'\nEpoch {epoch+1}/{self.config["max_epochs"]}:')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'Val F1: {val_f1:.4f}, Overfitting Gap: {overfitting_gap:.2f}%')
                print(f'Learning Rate: {current_lr:.2e}')
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, 'best')
                    print(f'✅ New best model saved! Val Acc: {val_acc:.2f}%')
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if overfitting_gap > self.config['overfitting_threshold']:
                    print(f'⚠️ Overfitting detected! Gap: {overfitting_gap:.2f}%')
                    self.patience_counter += 2
                
                if self.patience_counter >= self.config['patience']:
                    print(f'Early stopping at epoch {epoch+1}')
                    print(f'Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}')
                    break
                
                if (epoch + 1) % self.config['save_frequency'] == 0:
                    self._save_checkpoint(epoch, f'epoch_{epoch+1}')
                
                print('-' * 80)
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch+1}: {e}")
                continue
        
        total_time = time.time() - start_time
        logger.info(f'Training completed in {total_time:.2f} seconds')
        
        self._save_results()
        
        return self.history
    
    def _save_checkpoint(self, epoch, name):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': dict(self.history),
            'config': self.config
        }
        
        save_path = os.path.join(self.results_dir, f'{name}_model.pth')
        torch.save(checkpoint, save_path)
    
    def _save_results(self):
        """Save training results"""
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        
        config_path = os.path.join(self.results_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

def evaluate_with_tta(model, test_loader, class_names, input_size=224, tta_rounds=5):
    """Evaluate model with Test Time Augmentation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    tta_transform = create_production_transforms(input_size, 'test_tta')
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    logger.info(f"Evaluating with {tta_rounds} TTA rounds...")
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='TTA Evaluation'):
            batch_size = data.size(0)
            batch_predictions = []
            
            for i in range(batch_size):
                single_img = data[i]
                
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = single_img * std + mean
                img_denorm = torch.clamp(img_denorm, 0, 1)
                
                img_pil = transforms.ToPILImage()(img_denorm)
                
                tta_outputs = []
                for _ in range(tta_rounds):
                    tta_img = tta_transform(img_pil).unsqueeze(0).to(device)
                    output = model(tta_img)
                    tta_outputs.append(F.softmax(output, dim=1))
                
                avg_output = torch.mean(torch.stack(tta_outputs), dim=0)
                batch_predictions.append(avg_output)
            
            batch_outputs = torch.cat(batch_predictions, dim=0)
            _, predicted = batch_outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probs.extend(batch_outputs.cpu().numpy())
    
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    print(f'\nTest Results with TTA ({tta_rounds} rounds):')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'F1 Score: {f1:.4f}')
    print('\nDetailed Classification Report:')
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    return accuracy, f1, all_predictions, all_targets, all_probs

def analyze_model_performance(model, test_loader, class_names, results_dir):
    """Comprehensive model performance analysis"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Standard Evaluation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    print(f'\n{"="*60}')
    print('COMPREHENSIVE MODEL PERFORMANCE ANALYSIS')
    print(f'{"="*60}')
    print(f'Overall Accuracy: {accuracy:.2f}%')
    print(f'Weighted F1 Score: {f1:.4f}')
    print(f'Total Test Samples: {len(all_targets)}')
    
    print(f'\n{"Per-Class Performance":^60}')
    print('-' * 60)
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_targets) == i
        if class_mask.sum() > 0:
            class_acc = (np.array(all_predictions)[class_mask] == i).mean() * 100
            class_samples = class_mask.sum()
            print(f'{class_name:15} | Accuracy: {class_acc:6.2f}% | Samples: {class_samples:4d}')
    
    cm = confusion_matrix(all_targets, all_predictions)
    print(f'\nConfusion Matrix:')
    print(f'{"":>15}', end='')
    for name in class_names:
        print(f'{name:>12}', end='')
    print()
    
    for i, name in enumerate(class_names):
        print(f'{name:>15}', end='')
        for j in range(len(class_names)):
            print(f'{cm[i][j]:>12}', end='')
        print()
    
    return accuracy, f1

def main():
    """Main production pipeline"""
    
    CONFIG = {
        'DATA_DIR': 'datasets',
        'BATCH_SIZE': 16,           # Increased for faster training
        'INPUT_SIZE': 224,
        'MAX_SAMPLES_PER_CLASS': 1500,
        'MIN_SAMPLES_PER_CLASS': 500,
        'MODEL_SIZE': 'b0',
        'DROPOUT_RATE': 0.5,        # Reduced for better learning
        'USE_MIXUP': False,         # Disabled to avoid potential issues
        'MAX_EPOCHS': 30,          # Reduced but sufficient with better LR
        'LEARNING_RATE': 1e-4,      # Increased for faster convergence
        'WEIGHT_DECAY': 0.05,       # Reduced for less regularization
        'PATIENCE': 30,             # Increased for more training time
        'OVERFITTING_THRESHOLD': 5.0,
        'ENABLE_TTA': True,
        'TTA_ROUNDS': 5
    }
    
    print("🚀 PRODUCTION ANTI-OVERFITTING TRAINING PIPELINE")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        print("\n📊 STEP 1: Loading and balancing dataset...")
        train_data, val_data, test_data, class_names, class_counts = load_and_balance_data(
            CONFIG['DATA_DIR'],
            max_samples_per_class=CONFIG['MAX_SAMPLES_PER_CLASS'],
            min_samples_per_class=CONFIG['MIN_SAMPLES_PER_CLASS']
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Classes: {class_names}")
        print(f"Class distribution: {class_counts}")
        
        print("\n🔄 STEP 2: Creating production data loaders...")
        train_loader, val_loader, test_loader = create_production_dataloaders(
            train_data, val_data, test_data,
            batch_size=CONFIG['BATCH_SIZE'],
            input_size=CONFIG['INPUT_SIZE'],
            use_mixup=CONFIG['USE_MIXUP']
        )
        
        print(f"✅ Data loaders created successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        print("\n🏗️ STEP 3: Creating production model...")
        model = ProductionEfficientNet(
            num_classes=len(class_names),
            model_size=CONFIG['MODEL_SIZE'],
            dropout_rate=CONFIG['DROPOUT_RATE']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")
        
        print("\n🎯 STEP 4: Training production model...")
        
        training_config = {
            'max_epochs': CONFIG['MAX_EPOCHS'],
            'learning_rate': CONFIG['LEARNING_RATE'],
            'weight_decay': CONFIG['WEIGHT_DECAY'],
            'patience': CONFIG['PATIENCE'],
            'overfitting_threshold': CONFIG['OVERFITTING_THRESHOLD']
        }
        
        trainer = ProductionTrainer(model, train_loader, val_loader, len(class_names), training_config)
        history = trainer.train()
        
        print(f"✅ Training completed!")
        print(f"Results saved in: {trainer.results_dir}")
        
        print("\n📈 STEP 5: Final evaluation...")
        
        best_model_path = os.path.join(trainer.results_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Best model loaded from epoch {checkpoint['epoch'] + 1}")
            print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        else:
            print("No best model checkpoint found, using current model state")
        
        print("\nStandard Evaluation:")
        std_accuracy, std_f1 = analyze_model_performance(model, test_loader, class_names, trainer.results_dir)
        
        if CONFIG['ENABLE_TTA']:
            print("\nTest Time Augmentation Evaluation:")
            tta_accuracy, tta_f1, tta_preds, tta_targets, tta_probs = evaluate_with_tta(
                model, test_loader, class_names, CONFIG['INPUT_SIZE'], CONFIG['TTA_ROUNDS']
            )
        
        print(f"\n🎊 FINAL RESULTS SUMMARY")
        print("=" * 50)
        print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
        print(f"Standard Test Accuracy: {std_accuracy:.2f}%")
        if CONFIG['ENABLE_TTA']:
            print(f"TTA Test Accuracy: {tta_accuracy:.2f}%")
        
        if len(history['overfitting_gap']) > 0:
            print(f"Final Overfitting Gap: {history['overfitting_gap'][-1]:.2f}%")
            
            final_gap = history['overfitting_gap'][-1]
            if final_gap < 5.0:
                print("\n✅ SUCCESS: Overfitting successfully prevented!")
                print(f"   Overfitting gap: {final_gap:.2f}% < 5.0%")
            elif final_gap < CONFIG['OVERFITTING_THRESHOLD']:
                print("\n⚠️ ACCEPTABLE: Mild overfitting within threshold")
                print(f"   Overfitting gap: {final_gap:.2f}% < {CONFIG['OVERFITTING_THRESHOLD']}%")
            else:
                print("\n❌ WARNING: Overfitting detected despite measures")
                print(f"   Overfitting gap: {final_gap:.2f}% > {CONFIG['OVERFITTING_THRESHOLD']}%")
        
        print(f"\n📁 All results saved in: {trainer.results_dir}")
        print("   - Training history data")
        print("   - Model checkpoints")
        print("   - Evaluation metrics")
        print("   - Configuration files")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n❌ TRAINING FAILED")
        print(f"Error: {e}")
        print("Check the logs above for detailed error information.")

if __name__ == "__main__":
    main()