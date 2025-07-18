import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms
import timm
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "datasets"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
RESNET_VARIANT = "resnet50"  # Set to "resnet121" to use ResNet-121
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 5
CHECKPOINT_PATH = "checkpoint.pt"
FINE_TUNE_AFTER = 5
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check CUDA availability and GPU details
if torch.cuda.is_available():
    print(f"CUDA Available: Using {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total VRAM: {total_vram:.2f} GB")
    if total_vram < 12 and BATCH_SIZE > 16:
        print(f"Warning: VRAM ({total_vram:.2f} GB) may be low for batch_size={BATCH_SIZE}. Consider reducing to 16.")
        if RESNET_VARIANT == "resnet121":
            print("Warning: ResNet-121 requires more VRAM (~10-12GB). Strongly recommend batch_size=16 for <16GB VRAM.")
else:
    print("Warning: CUDA not available. Falling back to CPU. Performance will be slower.")

# ==========================
# DATA TRANSFORMS
# ==========================
train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x if x.shape[1:] == (224, 224) else transforms.Resize((224, 224))(x)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x if x.shape[1:] == (224, 224) else transforms.Resize((224, 224))(x)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================
# DATASET LOADER
# ==========================
class LargeTensorDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'real': 0, 'syn': 1, 'semi': 2}
        self.tensors = []
        self.class_counts = [0] * NUM_CLASSES
        self.rgb_counts = [0] * NUM_CLASSES
        self.grayscale_counts = [0] * NUM_CLASSES

        for class_name in self.class_to_idx:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found.")
                continue
            pt_files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))
            if not pt_files:
                print(f"Warning: No .pt files found in {class_dir}.")
                continue
            for pt_file in pt_files:
                try:
                    images = torch.load(pt_file, map_location="cpu")
                    if len(images.shape) != 4 or images.shape[1] not in [1, 3] or images.shape[2:] != (256, 256):
                        print(f"Warning: Invalid tensor shape in {pt_file}: {images.shape}")
                        continue
                    num_images = images.shape[0]
                    label = self.class_to_idx[class_name]
                    self.image_paths.extend([pt_file] * num_images)
                    self.labels.extend([label] * num_images)
                    self.tensors.append(images)
                    self.class_counts[label] += num_images
                    if images.shape[1] == 3:
                        self.rgb_counts[label] += num_images
                    else:
                        self.grayscale_counts[label] += num_images
                except Exception as e:
                    print(f"Error loading {pt_file}: {e}")
                    continue

        self.index_map = []
        for i, images in enumerate(self.tensors):
            length = images.shape[0]
            self.index_map.extend([(i, j) for j in range(length)])

        if not self.image_paths:
            raise ValueError("No valid .pt files found in dataset.")
        
        print(f"\nDataset: {base_dir}")
        total_images = len(self.image_paths)
        print(f"Total images: {total_images}")
        for idx, class_name in enumerate(self.class_to_idx):
            print(f"{class_name}: {self.class_counts[idx]} images "
                  f"({100 * self.class_counts[idx] / total_images:.2f}%), "
                  f"{self.rgb_counts[idx]} RGB, {self.grayscale_counts[idx]} grayscale")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, inner_idx = self.index_map[idx]
        x = self.tensors[file_idx][inner_idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def get_class_weights(self):
        total = sum(self.class_counts)
        weights = [total / (NUM_CLASSES * count) if count > 0 else 0 for count in self.class_counts]
        return torch.tensor(weights, dtype=torch.float).to(DEVICE)

    def get_sampler_weights(self):
        weights = [1.0 / self.class_counts[label] for label in self.labels]
        return weights

# ==========================
# DIAGNOSTIC FUNCTION
# ==========================
def check_dataset(dataset, loader, name="Dataset"):
    print(f"\n=== {name} Check ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Class counts: {dataset.class_counts}")
    print(f"RGB counts: {dataset.rgb_counts}")
    print(f"Grayscale counts: {dataset.grayscale_counts}")
    
    # Sample a few images
    print(f"\nSampling 5 images from {name}:")
    indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    for idx in indices:
        img, label = dataset[idx]
        print(f"Index {idx}: Shape={img.shape}, Label={label}, "
              f"Type={'RGB' if img.shape[0] == 3 else 'Grayscale'}")
    
    # Check first batch from loader
    print(f"\nFirst batch from {name} loader:")
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= 1:
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        print(f"Batch shape: images={x.shape}, labels={y.shape}, Device={x.device}")
        print(f"Labels: {y.tolist()}")
        print(f"Class distribution in batch: real={torch.sum(y == 0)}, "
              f"syn={torch.sum(y == 1)}, semi={torch.sum(y == 2)}")

# ==========================
# MODEL DEFINITION
# ==========================
class ResNetViTHybrid(nn.Module):
    def __init__(self, num_classes, resnet_variant="resnet50"):
        super().__init__()
        if resnet_variant == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
        elif resnet_variant == "resnet121":
            self.resnet = models.resnet121(pretrained=True)
        else:
            raise ValueError("resnet_variant must be 'resnet50' or 'resnet121'")
        resnet_fc_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit_fc_in = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.fc1 = nn.Linear(resnet_fc_in + vit_fc_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x):
        res_feat = self.resnet(x)
        vit_feat = self.vit(x)
        combined = torch.cat([res_feat, vit_feat], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out)

# ==========================
# METRICS
# ==========================
def calculate_metrics(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (preds == labels).mean() * 100
    overall_mcc = matthews_corrcoef(labels, preds)
    
    # Per-class MCC
    per_class_mcc = []
    class_names = ['real', 'syn', 'semi']
    for class_idx in range(NUM_CLASSES):
        binary_labels = (labels == class_idx).astype(int)
        binary_preds = (preds == class_idx).astype(int)
        try:
            mcc = matthews_corrcoef(binary_labels, binary_preds)
        except ValueError:
            mcc = 0.0  # Handle edge cases
        per_class_mcc.append(mcc)
    
    return accuracy, overall_mcc, per_class_mcc, class_names

# ==========================
# TRAINING + VALIDATION
# ==========================
def train(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            outputs = model(x)
            loss = criterion(outputs, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds)
        labels_all.extend(y)

    avg_loss = running_loss / len(loader.dataset)
    accuracy, overall_mcc, per_class_mcc, class_names = calculate_metrics(torch.stack(preds_all), torch.stack(labels_all))
    return avg_loss, accuracy, overall_mcc, per_class_mcc, class_names

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validate", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds)
            labels_all.extend(y)

    avg_loss = running_loss / len(loader.dataset)
    accuracy, overall_mcc, per_class_mcc, class_names = calculate_metrics(torch.stack(preds_all), torch.stack(labels_all))
    return avg_loss, accuracy, overall_mcc, per_class_mcc, class_names

# ==========================
# TEST EVALUATION
# ==========================
def test(model, loader):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Test", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast():
                outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds)
            labels_all.extend(y)

    accuracy, overall_mcc, per_class_mcc, class_names = calculate_metrics(torch.stack(preds_all), torch.stack(labels_all))
    return accuracy, overall_mcc, per_class_mcc, class_names

# ==========================
# EARLY STOPPING
# ==========================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_model = None

    def step(self, val_score, model):
        if self.best_score is None or val_score > self.best_score:
            self.best_score = val_score
            self.best_model = model.state_dict()
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    # Load datasets
    train_dataset = LargeTensorDataset(TRAIN_DIR, transform=train_transforms)
    test_dataset = LargeTensorDataset(TEST_DIR, transform=test_transforms)

    # Check datasets
    print("\n=== Train Dataset Check ===")
    check_dataset(train_dataset, DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True), "Train")
    print("\n=== Test Dataset Check ===")
    check_dataset(test_dataset, DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True), "Test")

    # Compute class weights for loss and sampler
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights.tolist()}")
    sampler_weights = train_dataset.get_sampler_weights()
    sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement=True)

    # Split training data into train and validation (20% for validation)
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=[train_dataset[i][1] for i in indices], random_state=SEED
    )
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    # Check train and validation subsets
    train_labels = [train_dataset.labels[i] for i in train_idx]
    val_labels = [train_dataset.labels[i] for i in val_idx]
    print(f"\nTrain subset size: {len(train_subset)}")
    print(f"Train subset class distribution: real={sum(l == 0 for l in train_labels)}, "
          f"syn={sum(l == 1 for l in train_labels)}, semi={sum(l == 2 for l in train_labels)}")
    print(f"Validation subset size: {len(val_subset)}")
    print(f"Validation subset class distribution: real={sum(l == 0 for l in val_labels)}, "
          f"syn={sum(l == 1 for l in val_labels)}, semi={sum(l == 2 for l in val_labels)}")

    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Check train loader batch distribution
    print("\nChecking train loader batch distribution...")
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= 1:
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        print(f"Train batch shape: images={x.shape}, labels={y.shape}, Device={x.device}")
        print(f"Train batch class distribution: real={torch.sum(y == 0)}, "
              f"syn={torch.sum(y == 1)}, semi={torch.sum(y == 2)}")

    # Initialize model, optimizer, criterion, scaler, scheduler
    model = ResNetViTHybrid(NUM_CLASSES, resnet_variant=RESNET_VARIANT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Resume training if checkpoint exists
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔁 Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch + 1}/{NUM_EPOCHS}")

        # Fine-tuning
        if epoch == FINE_TUNE_AFTER:
            for param in model.resnet.layer4.parameters():
                param.requires_grad = True
            for param in model.vit.blocks[-1].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE / 100, weight_decay=1e-4)
            print("🔓 Unfroze ResNet layer4 and ViT last block")

        train_loss, train_acc, train_mcc, train_per_class_mcc, class_names = train(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_mcc, val_per_class_mcc, _ = validate(model, val_loader, criterion)

        print(f"📊 Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MCC: {train_mcc:.4f}")
        print(f"Train Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, train_per_class_mcc)]))
        print(f"📉 Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MCC: {val_mcc:.4f}")
        print(f"Val Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, val_per_class_mcc)]))
        scheduler.step()

        # Save checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'val_mcc': val_mcc
        }, CHECKPOINT_PATH)

        # Early stopping based on validation MCC
        if early_stopping.step(val_mcc, model):
            print("⏹️ Early stopping triggered.")
            model.load_state_dict(early_stopping.best_model)
            break

    # Final test evaluation
    print("\n📈 Evaluating on test set...")
    test_acc, test_mcc, test_per_class_mcc, class_names = test(model, test_loader)
    print(f"✅ Test Accuracy: {test_acc:.2f}%, Test MCC: {test_mcc:.4f}")
    print(f"Test Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, test_per_class_mcc)]))

    # Final model save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"model_final_{timestamp}.pt")
    print(f"💾 Final model saved as model_final_{timestamp}.pt")

if __name__ == "__main__":
    main()