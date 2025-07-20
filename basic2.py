import os
import glob
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import hashlib
from datetime import datetime
from tqdm import tqdm

# Constants
NUM_CLASSES = 3
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
FINETUNE_LR = 1e-5  # Lower LR for fine-tuning
FREEZE_EPOCHS = 2  # Epochs to train with frozen backbone
PATIENCE = 2  # Early stopping patience
MIN_DELTA = 0.001  # Minimum MCC improvement
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET_VARIANT = "resnet50"
TRAIN_DIR = "datasets/train"
TEST_DIR = "datasets/test"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
CLASS_NAMES = ["real", "synthetic", "semi-synthetic"]

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

# Generate a unique run ID
run_id = hashlib.sha256(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
logging.info(f"Run ID: {run_id}")

class ResNetViTHybrid(nn.Module):
    def __init__(self, num_classes, resnet_variant="resnet50"):
        super(ResNetViTHybrid, self).__init__()
        self.resnet = timm.create_model(resnet_variant, pretrained=True, num_classes=0)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.resnet_out_features = self.resnet.num_features
        self.vit_out_features = self.vit.num_features
        self.fc = nn.Linear(self.resnet_out_features + self.vit_out_features, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        resnet_features = self.resnet(x)
        vit_features = self.vit(x)
        combined_features = torch.cat((resnet_features, vit_features), dim=1)
        combined_features = self.dropout(combined_features)
        return self.fc(combined_features)

    def freeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
        for param in self.vit.parameters():
            param.requires_grad = True

class LargeTensorDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_counts = [0] * NUM_CLASSES
        self.file_indices = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(base_dir, class_name)
            pt_files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))
            if not pt_files:
                logging.error(f"No .pt files found in {class_dir}")
                raise FileNotFoundError(f"No .pt files found in {class_dir}")
            for pt_file in pt_files:
                try:
                    images = torch.load(pt_file, map_location="cpu")
                    if images.shape[0] == 0:
                        logging.warning(f"{pt_file}: Empty tensor file")
                        continue
                    if images.shape[1:] != (3, 256, 256):
                        logging.warning(f"{pt_file}: Unexpected shape {images.shape}")
                        continue
                    mean_pixel = images.float().mean().item() / 255.0
                    if mean_pixel < 0.1:
                        logging.warning(f"{pt_file}: Potentially dark (mean pixel value: {mean_pixel:.3f})")
                        img_np = images[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        Image.fromarray(img_np).save(f"debug_image_{class_name}_{os.path.basename(pt_file)}_0.png")
                    for sample_idx in range(images.shape[0]):
                        self.image_paths.append((pt_file, sample_idx))
                        self.labels.append(class_idx)
                        self.file_indices.append((pt_file, sample_idx, class_idx))
                    self.class_counts[class_idx] += images.shape[0]
                except Exception as e:
                    logging.error(f"Error loading {pt_file}: {e}")
                    continue

        if not self.image_paths:
            logging.error("No valid images found in dataset")
            raise ValueError("No valid images found in dataset")
        indices = list(range(len(self.image_paths)))
        np.random.seed(42)
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.file_indices = [self.file_indices[i] for i in indices]
        logging.info(f"Loaded {len(self.image_paths)} images: " + ", ".join([f"{name}: {count}" for name, count in zip(CLASS_NAMES, self.class_counts)]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pt_file, sample_idx = self.image_paths[idx]
        try:
            images = torch.load(pt_file, map_location="cpu")
            image = images[sample_idx]
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image.float() / 255.0)
            return image, label
        except Exception as e:
            logging.error(f"Error loading image from {pt_file}, index {sample_idx}: {e}")
            raise

    def get_sampler_weights(self):
        weights = [1.0 / self.class_counts[label] if self.class_counts[label] > 0 else 0 for label in self.labels]
        assert all(w > 0 for w in weights), "Zero weights detected in sampler"
        return weights

def check_dataset(dataset, loader, name="Dataset"):
    logging.info(f"{name}:")
    print(f"{name}:")
    logging.info(f"Total images: {len(dataset)}")
    logging.info(", ".join([f"{name}: {count} images ({count/len(dataset)*100:.2f}% if non-zero)" 
                            for name, count in zip(CLASS_NAMES, dataset.class_counts)]))
    logging.info(f"=== {name} Check ===")
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Class counts: {dataset.class_counts}")
    logging.info(f"RGB counts: {dataset.class_counts}")
    logging.info(f"Grayscale counts: {[0] * NUM_CLASSES}")
    logging.info(f"Sampling 5 images from {name}:")
    indices = np.random.choice(len(dataset), 5, replace=False)
    for idx in indices:
        image, label = dataset[idx]
        pixel_range = [image.min().item(), image.max().item()]
        logging.info(f"Index {idx}: Shape={image.shape}, Label={label}, Type=RGB, Post-normalization pixel range {pixel_range}")
    logging.info(f"First 3 batches from {name} loader:")
    print(f"First 3 batches from {name} loader:")
    try:
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= 3:
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logging.info(f"Batch {batch_idx + 1} shape: images={x.shape}, labels={y.shape}, Device={x.device}")
            logging.info(f"Labels: {y.tolist()}")
            logging.info(f"Class distribution: " + ", ".join([f"{name}={torch.sum(y == i)}" for i, name in enumerate(CLASS_NAMES)]))
            print(f"Batch {batch_idx + 1} shape: images={x.shape}, labels={y.shape}, Device={x.device}")
            print(f"Labels: {y.tolist()}")
            print(f"Class distribution: " + ", ".join([f"{name}={torch.sum(y == i)}" for i, name in enumerate(CLASS_NAMES)]))
    except Exception as e:
        logging.error(f"Error loading batch from {name} loader: {e}")
        print(f"Error loading batch from {name} loader: {e}")

def train(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    loader = tqdm(loader, desc=f"Training Epoch {epoch + 1}", leave=True)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            outputs = model(x)
            loss = criterion(outputs, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = []
    for cls in range(NUM_CLASSES):
        binary_labels = [1 if l == cls else 0 for l in all_labels]
        binary_preds = [1 if p == cls else 0 for p in all_preds]
        per_class_mcc.append(matthews_corrcoef(binary_labels, binary_preds))
    pred_counts = [sum(1 for p in all_preds if p == i) for i in range(NUM_CLASSES)]
    logging.info(f"Train predicted class distribution: " + ", ".join([f"{name}={count}" for name, count in zip(CLASS_NAMES, pred_counts)]))
    print(f"Train predicted class distribution: " + ", ".join([f"{name}={count}" for name, count in zip(CLASS_NAMES, pred_counts)]))
    return epoch_loss, epoch_acc, epoch_mcc, per_class_mcc

def validate(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    loader = tqdm(loader, desc=f"Validation Epoch {epoch + 1}", leave=True)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast('cuda'):
                outputs = model(x)
                loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = []
    for cls in range(NUM_CLASSES):
        binary_labels = [1 if l == cls else 0 for l in all_labels]
        binary_preds = [1 if p == cls else 0 for p in all_preds]
        per_class_mcc.append(matthews_corrcoef(binary_labels, binary_preds))
    pred_counts = [sum(1 for p in all_preds if p == i) for i in range(NUM_CLASSES)]
    logging.info(f"Validation predicted class distribution: " + ", ".join([f"{name}={count}" for name, count in zip(CLASS_NAMES, pred_counts)]))
    print(f"Validation predicted class distribution: " + ", ".join([f"{name}={count}" for name, count in zip(CLASS_NAMES, pred_counts)]))
    return epoch_loss, epoch_acc, epoch_mcc, per_class_mcc

def test(model, loader, epoch=None):
    model.eval()
    all_preds = []
    all_labels = []
    desc = f"Testing Epoch {epoch + 1}" if epoch is not None else "Testing"
    loader = tqdm(loader, desc=desc, leave=True)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    test_acc = accuracy_score(all_labels, all_preds) * 100
    test_mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = []
    for cls in range(NUM_CLASSES):
        binary_labels = [1 if l == cls else 0 for l in all_labels]
        binary_preds = [1 if p == cls else 0 for p in all_preds]
        per_class_mcc.append(matthews_corrcoef(binary_labels, binary_preds))
    pred_counts = [sum(1 for p in all_preds if p == i) for i in range(NUM_CLASSES)]
    logging.info(f"Test predicted class distribution: " + ", ".join([f"{name}={count}" for name, count in zip(CLASS_NAMES, pred_counts)]))
    print(f"Test predicted class distribution: " + ", ".join([f"{name}={count}" for name, count in zip(CLASS_NAMES, pred_counts)]))
    return test_acc, test_mcc, per_class_mcc, CLASS_NAMES

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    logging.info(f"Loading test dataset (Run ID: {run_id})")
    print(f"Loading test dataset (Run ID: {run_id})")
    test_dataset = LargeTensorDataset(TEST_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    check_dataset(test_dataset, test_loader, "Test")

    # Load pretrained weights
    logging.info(f"Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)")
    model = ResNetViTHybrid(NUM_CLASSES, resnet_variant=RESNET_VARIANT).to(DEVICE)
    model.freeze_backbone()  # Freeze backbone initially
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    best_mcc = -1.0
    start_epoch = 0
    no_improve_epochs = 0

    # Check for latest checkpoint
    latest_checkpoint = max(glob.glob(os.path.join(CHECKPOINT_DIR, "model.pt")), key=os.path.getctime, default=None)
    if latest_checkpoint:
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            start_epoch = checkpoint['epoch']
            best_mcc = checkpoint.get('best_mcc', -1.0)
            no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
            frozen = checkpoint.get('frozen', True)
            if frozen and start_epoch >= FREEZE_EPOCHS:
                model.unfreeze_backbone()
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR, weight_decay=1e-4)
                logging.info(f"Unfreezing backbone and setting LR to {FINETUNE_LR} (resumed from checkpoint)")
                print(f"Unfreezing backbone and setting LR to {FINETUNE_LR} (resumed from checkpoint)")
            logging.info(f"Resuming from checkpoint: {latest_checkpoint}, Epoch {start_epoch}, Frozen: {frozen} (Run ID: {run_id})")
            print(f"Resuming from checkpoint: {latest_checkpoint}, Epoch {start_epoch}, Frozen: {frozen} (Run ID: {run_id})")
        except Exception as e:
            logging.error(f"Error loading checkpoint {latest_checkpoint}: {e}")
            print(f"Error loading checkpoint {latest_checkpoint}: {e}")
            start_epoch = 0
            model.freeze_backbone()

    # Load full training dataset
    logging.info(f"Loading full training dataset (Run ID: {run_id})")
    print(f"Loading full training dataset (Run ID: {run_id})")
    train_dataset = LargeTensorDataset(TRAIN_DIR, transform=train_transform)
    if len(train_dataset) < 500000:
        logging.warning(f"Expected ~600,000 images, got {len(train_dataset)}")
        print(f"Expected ~600,000 images, got {len(train_dataset)}")

    # Split into train and validation
    indices = list(range(len(train_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(0.8 * len(train_dataset))
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    train_weights = [train_dataset.get_sampler_weights()[i] for i in train_indices]
    train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights), replacement=True, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
    check_dataset(train_subset, train_loader, "Train")
    check_dataset(val_subset, val_loader, "Validation")

    logging.info(f"Train subset size: {len(train_subset)}")
    logging.info(f"Train subset distribution: " + ", ".join([f"{name}={sum(1 for i in train_indices if train_dataset.labels[i] == idx)}" 
                                                           for idx, name in enumerate(CLASS_NAMES)]))
    logging.info(f"Validation subset size: {len(val_subset)}")
    logging.info(f"Validation subset distribution: " + ", ".join([f"{name}={sum(1 for i in val_indices if train_dataset.labels[i] == idx)}" 
                                                               for idx, name in enumerate(CLASS_NAMES)]))
    print(f"Train subset size: {len(train_subset)}")
    print(f"Train subset distribution: " + ", ".join([f"{name}={sum(1 for i in train_indices if train_dataset.labels[i] == idx)}" 
                                                     for idx, name in enumerate(CLASS_NAMES)]))
    print(f"Validation subset size: {len(val_subset)}")
    print(f"Validation subset distribution: " + ", ".join([f"{name}={sum(1 for i in val_indices if train_dataset.labels[i] == idx)}" 
                                                         for idx, name in enumerate(CLASS_NAMES)]))

    for epoch in range(start_epoch, NUM_EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} (Run ID: {run_id})")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} (Run ID: {run_id})")
        
        if epoch == FREEZE_EPOCHS and model.resnet.conv1.weight.requires_grad == False:
            model.unfreeze_backbone()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR, weight_decay=1e-4)
            logging.info(f"Unfreezing backbone and setting LR to {FINETUNE_LR}")
            print(f"Unfreezing backbone and setting LR to {FINETUNE_LR}")

        train_loss, train_acc, train_mcc, train_per_class_mcc = train(model, train_loader, optimizer, criterion, scaler, epoch)
        logging.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MCC: {train_mcc:.4f}")
        logging.info(f"Train Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(CLASS_NAMES, train_per_class_mcc)]))
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MCC: {train_mcc:.4f}")
        print(f"Train Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(CLASS_NAMES, train_per_class_mcc)]))
        val_loss, val_acc, val_mcc, val_per_class_mcc = validate(model, val_loader, criterion, epoch)
        logging.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MCC: {val_mcc:.4f}")
        logging.info(f"Val Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(CLASS_NAMES, val_per_class_mcc)]))
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MCC: {val_mcc:.4f}")
        print(f"Val Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(CLASS_NAMES, val_per_class_mcc)]))

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'epoch': epoch + 1,
            'best_mcc': best_mcc,
            'no_improve_epochs': no_improve_epochs,
            'frozen': model.resnet.conv1.weight.requires_grad == False,
            'run_id': run_id
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "model.pt"))
        if val_mcc > best_mcc + MIN_DELTA:
            best_mcc = val_mcc
            no_improve_epochs = 0
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            logging.info(f"Best model saved to checkpoints/best_model.pt (Run ID: {run_id})")
            print(f"Best model saved to checkpoints/best_model.pt (Run ID: {run_id})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                logging.info(f"Early stopping triggered after {no_improve_epochs} epochs with no MCC improvement")
                print(f"Early stopping triggered after {no_improve_epochs} epochs with no MCC improvement")
                break

    # Final test evaluation
    logging.info(f"Evaluating on test set (Run ID: {run_id})")
    print(f"\nEvaluating on test set (Run ID: {run_id})...")
    test_acc, test_mcc, test_per_class_mcc, class_names = test(model, test_loader)
    logging.info(f"Test Accuracy: {test_acc:.2f}%, Test MCC: {test_mcc:.4f}")
    logging.info(f"Test Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, test_per_class_mcc)]))
    print(f"Test Accuracy: {test_acc:.2f}%, Test MCC: {test_mcc:.4f}")
    print(f"Test Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, test_per_class_mcc)]))

if __name__ == "__main__":
    main()
