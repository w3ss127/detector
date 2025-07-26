import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.amp import GradScaler, autocast
import hashlib
from datetime import datetime
from tqdm import tqdm
import sys
import multiprocessing

# Configuration for training settings
class Config:
    NUM_CLASSES = 3
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    FINETUNE_LR = 0.00001
    FREEZE_EPOCHS = 10
    TRAIN_DIR = "datasets/train"
    TEST_DIR = "datasets/test"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    CLASS_NAMES = ["real", "synthetic", "semi-synthetic"]
    ACCUMULATION_STEPS = 2
    PATIENCE = 10
    MIN_DELTA = 0.001  # Minimum MCC improvement for early stopping
    NUM_WORKERS = min(4, multiprocessing.cpu_count() // 2)
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    AUGMENTATION_PARAMS = {
        "rotation_degrees": 15,
        "brightness": 0.2,
        "contrast": 0.2
    }

def setup_logging(log_dir, log_file):
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging to {log_file}")

def cleanup_logging():
    """Clear logging handlers."""
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

def repeat_grayscale_to_rgb(x):
    """Convert single-channel images to three channels."""
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

def get_transforms(is_training=True):
    """Get data transformations for training or validation/testing."""
    transforms_list = [
        transforms.Lambda(repeat_grayscale_to_rgb),
        transforms.Resize(Config.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    if is_training:
        transforms_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(Config.AUGMENTATION_PARAMS["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=Config.AUGMENTATION_PARAMS["brightness"],
                contrast=Config.AUGMENTATION_PARAMS["contrast"]
            ),
        ])
    transforms_list.append(
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    )
    return transforms.Compose(transforms_list)

class ResNetViTHybrid(nn.Module):
    """Hybrid model combining ResNet and Vision Transformer."""
    def __init__(self, num_classes):
        super(ResNetViTHybrid, self).__init__()
        self.resnet = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.fc1 = nn.Linear(self.resnet.num_features + self.vit.num_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        resnet_features = self.resnet(x)
        vit_features = self.vit(x)
        combined = torch.cat((resnet_features, vit_features), dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out)

    def freeze_backbone(self):
        """Freeze ResNet and ViT backbones."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ResNet and ViT backbones."""
        for param in self.resnet.parameters():
            param.requires_grad = True
        for param in self.vit.parameters():
            param.requires_grad = True

class LargeTensorDataset(Dataset):
    """Dataset for loading .pt files with uint8 images."""
    def __init__(self, base_dir, transform=None, is_test=False):
        self.base_dir = base_dir
        self.transform = transform
        self.is_test = is_test
        self.image_paths = []
        self.labels = []
        self.class_counts = [0] * Config.NUM_CLASSES

        # Validate folder structure
        expected_folders = set(Config.CLASS_NAMES)
        actual_folders = set(os.path.basename(f) for f in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(f))
        if actual_folders != expected_folders:
            logging.error(f"Folder mismatch in {base_dir}. Expected: {expected_folders}, Found: {actual_folders}")
            raise ValueError(f"Folder mismatch in {base_dir}")

        # Load .pt files and assign labels
        for class_idx, class_name in enumerate(Config.CLASS_NAMES):
            class_dir = os.path.join(base_dir, class_name)
            pt_files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))
            for pt_file in tqdm(pt_files, desc=f"Loading {class_name} ({'test' if is_test else 'train'})"):
                try:
                    images = torch.load(pt_file, map_location="cpu")
                    if images.shape[0] == 0:
                        continue
                    for sample_idx in range(images.shape[0]):
                        self.image_paths.append((pt_file, sample_idx))
                        self.labels.append(class_idx)
                    self.class_counts[class_idx] += images.shape[0]
                except Exception as e:
                    logging.error(f"Error loading {pt_file}: {e}")
                    continue

        if not self.image_paths:
            logging.error(f"No valid images found in {base_dir}")
            raise ValueError(f"No valid images found in {base_dir}")

        # Shuffle dataset
        indices = list(range(len(self.image_paths)))
        np.random.seed(42)
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

        logging.info(f"Loaded {len(self.image_paths):,} images from {base_dir} ({'test' if is_test else 'train'})")
        logging.info(f"Label counts: " + ", ".join(
            [f"{name}: {count:,}" for name, count in zip(Config.CLASS_NAMES, self.class_counts)]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pt_file, sample_idx = self.image_paths[idx]
        try:
            images = torch.load(pt_file, map_location="cpu")
            image = images[sample_idx]
            if image.shape[0] not in {1, 3}:
                raise ValueError(f"Invalid channel count: {image.shape[0]} in {pt_file}")
            image = image.float() / 255.0  # Assume uint8 input
            if self.transform:
                image = self.transform(image)
            if image.shape != (3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]):
                raise ValueError(f"Invalid shape after transform: {image.shape} in {pt_file}")
            return image, self.labels[idx]
        except Exception as e:
            logging.error(f"Error loading {pt_file}[{sample_idx}]: {e}")
            raise

def train(model, loader, optimizer, criterion, scaler, epoch, run_id):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    dataset_len = len(loader.dataset)
    loader = tqdm(loader, desc=f"Training Epoch {epoch + 1}")
    for i, (x, y) in enumerate(loader):
        x, y = x.to('cuda'), y.to('cuda')
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(x)
            loss = criterion(outputs, y) / Config.ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        if (i + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * x.size(0) * Config.ACCUMULATION_STEPS
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    epoch_loss = running_loss / dataset_len
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = [matthews_corrcoef([1 if l == cls else 0 for l in all_labels],
                                       [1 if p == cls else 0 for p in all_preds])
                     for cls in range(Config.NUM_CLASSES)]
    logging.info(f"Train: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, MCC={epoch_mcc:.4f}")
    logging.info(f"Per-Class MCC: " + ", ".join(
        [f"{name}: {mcc:.4f}" for name, mcc in zip(Config.CLASS_NAMES, per_class_mcc)]))
    with open(os.path.join(Config.LOG_DIR, f"metrics_{run_id}.csv"), "a") as f:
        if epoch == 0:
            f.write("Epoch,Phase,Loss,Accuracy,MCC," + ",".join([f"{name}_MCC" for name in Config.CLASS_NAMES]) + "\n")
        f.write(f"{epoch + 1},Train,{epoch_loss:.4f},{epoch_acc:.2f},{epoch_mcc:.4f}," +
                ",".join([f"{mcc:.4f}" for mcc in per_class_mcc]) + "\n")
    return epoch_loss, epoch_acc, epoch_mcc, per_class_mcc

def validate(model, loader, criterion, epoch, run_id):
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    dataset_len = len(loader.dataset)
    loader = tqdm(loader, desc=f"Validation Epoch {epoch + 1}")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to('cuda'), y.to('cuda')
            with autocast('cuda'):
                outputs = model(x)
                loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    epoch_loss = running_loss / dataset_len
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = [matthews_corrcoef([1 if l == cls else 0 for l in all_labels],
                                       [1 if p == cls else 0 for p in all_preds])
                     for cls in range(Config.NUM_CLASSES)]
    logging.info(f"Validation: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, MCC={epoch_mcc:.4f}")
    logging.info(f"Per-Class MCC: " + ", ".join(
        [f"{name}: {mcc:.4f}" for name, mcc in zip(Config.CLASS_NAMES, per_class_mcc)]))
    with open(os.path.join(Config.LOG_DIR, f"metrics_{run_id}.csv"), "a") as f:
        f.write(f"{epoch + 1},Validation,{epoch_loss:.4f},{epoch_acc:.2f},{epoch_mcc:.4f}," +
                ",".join([f"{mcc:.4f}" for mcc in per_class_mcc]) + "\n")
    return epoch_loss, epoch_acc, epoch_mcc, per_class_mcc

def test(model, loader, run_id, epoch=None):
    """Test the model on the test dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    desc = f"Testing Epoch {epoch + 1}" if epoch is not None else "Testing"
    loader = tqdm(loader, desc=desc)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to('cuda'), y.to('cuda')
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    test_acc = accuracy_score(all_labels, all_preds) * 100
    test_mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = [matthews_corrcoef([1 if l == cls else 0 for l in all_labels],
                                       [1 if p == cls else 0 for p in all_preds])
                     for cls in range(Config.NUM_CLASSES)]
    logging.info(f"Test: Accuracy={test_acc:.2f}%, MCC={test_mcc:.4f}")
    logging.info(f"Per-Class MCC: " + ", ".join(
        [f"{name}: {mcc:.4f}" for name, mcc in zip(Config.CLASS_NAMES, per_class_mcc)]))
    with open(os.path.join(Config.LOG_DIR, f"metrics_{run_id}.csv"), "a") as f:
        f.write(f"{epoch + 1 if epoch is not None else 0},Test,0.0000,{test_acc:.2f},{test_mcc:.4f}," +
                ",".join([f"{mcc:.4f}" for mcc in per_class_mcc]) + "\n")
    return test_acc, test_mcc, per_class_mcc, Config.CLASS_NAMES

def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    """Load model, optimizer, and scaler state from a checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        epoch = checkpoint['epoch']
        best_mcc = checkpoint.get('best_mcc', -1.0)
        no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
        frozen = checkpoint.get('frozen', True)
        run_id = checkpoint.get('run_id', 'unknown')
        logging.info(f"Loaded checkpoint: {checkpoint_path}, Epoch {epoch}")
        return epoch, best_mcc, no_improve_epochs, frozen, run_id
    except Exception:
        logging.error(f"Error loading checkpoint {checkpoint_path}")
        return 0, -1.0, 0, True, 'unknown'

def main():
    """Run training on a single GPU."""
    run_id = hashlib.sha256(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    log_file = f"training_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(Config.LOG_DIR, log_file)

    if not torch.cuda.is_available():
        logging.error("No GPU available")
        sys.exit(1)

    # Load datasets
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    logging.info("Loading datasets...")
    test_dataset = LargeTensorDataset(Config.TEST_DIR, transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    train_dataset = LargeTensorDataset(Config.TRAIN_DIR, transform=train_transform, is_test=False)
    indices = list(range(len(train_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(0.8 * len(train_dataset))
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True,
                             num_workers=Config.NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    logging.info(f"Train size: {len(train_subset):,}, Validation size: {len(val_subset):,}")

    # Initialize model
    model = ResNetViTHybrid(Config.NUM_CLASSES).to('cuda')
    model.freeze_backbone()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_mcc = -1.0
    no_improve_epochs = 0
    start_epoch = 0

    # Load checkpoint
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    latest_checkpoint = max(glob.glob(os.path.join(Config.CHECKPOINT_DIR, f"model_{run_id}.pt")), key=os.path.getctime, default=None)
    if latest_checkpoint:
        start_epoch, best_mcc, no_improve_epochs, frozen, run_id = load_checkpoint(model, optimizer, scaler, latest_checkpoint)
        if frozen and start_epoch >= Config.FREEZE_EPOCHS:
            model.unfreeze_backbone()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.FINETUNE_LR, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            logging.info(f"Unfreezing model and setting learning rate to {Config.FINETUNE_LR}")

    # Training loop
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        if epoch == Config.FREEZE_EPOCHS and not model.resnet.conv1.weight.requires_grad:
            model.unfreeze_backbone()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.FINETUNE_LR, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            logging.info(f"Unfreezing model and setting learning rate to {Config.FINETUNE_LR}")

        train_loss, train_acc, train_mcc, _ = train(model, train_loader, optimizer, criterion, scaler, epoch, run_id)
        val_loss, val_acc, val_mcc, _ = validate(model, val_loader, criterion, epoch, run_id)

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate: {current_lr:.6f}")
        scheduler.step(val_mcc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            logging.info(f"Learning rate reduced to {new_lr:.6f}")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'epoch': epoch + 1,
            'best_mcc': best_mcc,
            'no_improve_epochs': no_improve_epochs,
            'frozen': not model.resnet.conv1.weight.requires_grad,
            'run_id': run_id
        }
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_{run_id}.pt")
        torch.save(checkpoint, checkpoint_path)
        if val_mcc > best_mcc + Config.MIN_DELTA:
            best_mcc = val_mcc
            no_improve_epochs = 0
            best_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_mcc_{best_mcc:.4f}_{run_id}.pt")
            torch.save(checkpoint, best_checkpoint_path)
            logging.info(f"Saved best model (MCC: {best_mcc:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= Config.PATIENCE:
                logging.info(f"Early stopping after {no_improve_epochs} epochs without improvement")
                break

    logging.info("Testing model...")
    test(model, test_loader, run_id)
    cleanup_logging()

if __name__ == "__main__":
    main()