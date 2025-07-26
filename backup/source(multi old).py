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
from torch.amp import GradScaler, autocast
import hashlib
from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Settings for training
NUM_CLASSES = 3  # Classes: real (0), synthetic (1), semi-synthetic (2)
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
MIN_DELTA = 0.001
EXPECTED_IMAGES = 150000
NUM_WORKERS = 12

LOG_FILE = os.path.join(LOG_DIR, f"training_allranks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_distributed(rank, world_size):
    """Set up multiple GPUs for training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up GPU training setup."""
    dist.destroy_process_group()

class ResNetViTHybrid(nn.Module):
    """Model combining ResNet and Vision Transformer for image classification."""
    def __init__(self, num_classes):
        super(ResNetViTHybrid, self).__init__()
        self.resnet = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # self.fc = nn.Linear(self.resnet.num_features + self.vit.num_features, num_classes)
        # self.dropout = nn.Dropout(0.5)
        

    def forward(self, x):
        resnet_features = self.resnet(x)
        vit_features = self.vit(x)
        combined = torch.cat((resnet_features, vit_features), dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out)

        # combined = self.dropout(combined)
        # return self.fc(combined)

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

def repeat_grayscale_to_rgb(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x
class LargeTensorDataset(Dataset):
    """Loads .pt files containing images for training or testing."""
    def __init__(self, base_dir, transform=None, rank=0):
        self.base_dir = base_dir
        self.transform = transform
        self.rank = rank
        self.image_paths = []
        self.labels = []
        self.class_counts = [0] * NUM_CLASSES
        self.preprocess = transforms.Compose([
            transforms.Lambda(repeat_grayscale_to_rgb),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        ])

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(base_dir, class_name)
            pt_files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))
            if not pt_files:
                if self.rank == 0:
                    logging.error(f"No .pt files in {class_dir}")
                raise FileNotFoundError(f"No .pt files in {class_dir}")
            for pt_file in pt_files:
                try:
                    images = torch.load(pt_file, map_location="cpu")
                    if images.shape[0] == 0:
                        if self.rank == 0:
                            logging.warning(f"{pt_file}: Empty file")
                        continue
                    if images.shape[1] not in {1, 3} or images.shape[2:] != (224, 224):
                        if self.rank == 0:
                            logging.warning(f"{pt_file}: Shape {images.shape}, resizing to (3, 224, 224)")
                        images = self.preprocess(images)
                    if images.shape[1:] != (3, 224, 224):
                        if self.rank == 0:
                            logging.warning(f"{pt_file}: Shape {images.shape} after preprocess")
                        continue
                    for sample_idx in range(images.shape[0]):
                        self.image_paths.append((pt_file, sample_idx))
                        self.labels.append(class_idx)
                    self.class_counts[class_idx] += images.shape[0]
                except Exception as e:
                    if self.rank == 0:
                        logging.error(f"Error loading {pt_file}: {e}")
                    continue

        if not self.image_paths:
            if self.rank == 0:
                logging.error("No valid images found")
            raise ValueError("No valid images found")
        indices = list(range(len(self.image_paths)))
        np.random.seed(42)
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        if self.labels and self.rank == 0:
            label_counts = np.bincount(self.labels, minlength=len(CLASS_NAMES))
            logging.info("Label counts: " + ", ".join([f"{name}: {count}" for name, count in zip(CLASS_NAMES, label_counts)]))
            logging.info(f"Loaded {len(self.image_paths):,} images: " + ", ".join(
                [f"{name}: {count:,}" for name, count in zip(CLASS_NAMES, self.class_counts)]))
            if len(self.image_paths) < EXPECTED_IMAGES * 0.8 and "train" in base_dir:
                logging.warning(f"Expected ~{EXPECTED_IMAGES:,} images, got {len(self.image_paths):,} in {base_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pt_file, sample_idx = self.image_paths[idx]
        try:
            images = torch.load(pt_file, map_location="cpu")
            image = images[sample_idx]
            image = self.preprocess(image)
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            elif image.dtype == torch.float32 or image.dtype == torch.float64:
                image = torch.clamp(image, 0, 255) / 255.0 if image.max() > 1.0 else image
            else:
                raise ValueError(f"Unsupported image dtype: {image.dtype} in {pt_file}")
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading {pt_file}, index {sample_idx}: {e}")
            raise

    def get_sampler_weights(self):
        weights = [1.0 / self.class_counts[label] if self.class_counts[label] > 0 else 0 for label in self.labels]
        assert all(w > 0 for w in weights), "Zero weights in sampler"
        return weights

def check_dataset(dataset, loader, name="Dataset", rank=0):
    """Check dataset details."""
    if rank == 0:
        logging.info(f"{name}: {len(dataset):,} images")
        logging.info(", ".join([f"{name}: {count:,} ({count/len(dataset)*100:.2f}%)"
                                for name, count in zip(CLASS_NAMES, dataset.class_counts)]))
        logging.info(f"Checking {name}...")
        indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
        for idx in indices:
            image, label = dataset[idx]
            pt_file, sample_idx = dataset.image_paths[idx]
            raw_image = torch.load(pt_file, map_location="cpu")[sample_idx]
            raw_image = dataset.preprocess(raw_image)
            raw_min, raw_max = raw_image.min().item(), raw_image.max().item()
            logging.info(f"Image {idx}: Shape={image.shape}, Label={label}, "
                         f"Raw Range=[{raw_min:.2f}, {raw_max:.2f}], "
                         f"Transformed Range=[{image.min():.2f}, {image.max():.2f}]")
        logging.info(f"First 3 batches from {name}:")
    try:
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= 3:
                break
            x = x.to(rank)
            y = y.to(rank)
            if rank == 0:
                logging.info(f"Batch {batch_idx + 1}: Images={x.shape}, Labels={y.shape}, Device={x.device}")
                logging.info(f"Class counts: " + ", ".join([f"{name}={torch.sum(y == i)}" for i, name in enumerate(CLASS_NAMES)]))
                logging.info(f"Batch Range=[{x.min():.2f}, {x.max():.2f}]")
    except Exception as e:
        if rank == 0:
            logging.error(f"Error in {name} batch: {e}")

def train(model, loader, optimizer, criterion, scaler, rank, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    dataset_len = len(loader.dataset)
    if rank == 0:
        loader = tqdm(loader, desc=f"Training Epoch {epoch + 1}")
    for i, (x, y) in enumerate(loader):
        x, y = x.to(rank), y.to(rank)
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(x)
            loss = criterion(outputs, y) / ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * x.size(0) * ACCUMULATION_STEPS
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    epoch_loss = running_loss / dataset_len
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = [matthews_corrcoef([1 if l == cls else 0 for l in all_labels],
                                       [1 if p == cls else 0 for p in all_preds]) for cls in range(NUM_CLASSES)]
    if rank == 0:
        logging.info(f"Train: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, MCC={epoch_mcc:.4f}")
        logging.info(f"Train Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(CLASS_NAMES, per_class_mcc)]))
    return epoch_loss, epoch_acc, epoch_mcc, per_class_mcc

def validate(model, loader, criterion, rank, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    dataset_len = len(loader.dataset)
    if rank == 0:
        loader = tqdm(loader, desc=f"Validation Epoch {epoch + 1}")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(rank), y.to(rank)
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
                                       [1 if p == cls else 0 for p in all_preds]) for cls in range(NUM_CLASSES)]
    if rank == 0:
        logging.info(f"Validation: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, MCC={epoch_mcc:.4f}")
        logging.info(f"Validation Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(CLASS_NAMES, per_class_mcc)]))
    return epoch_loss, epoch_acc, epoch_mcc, per_class_mcc

def test(model, loader, rank, epoch=None):
    model.eval()
    all_preds = []
    all_labels = []
    if rank == 0:
        desc = f"Testing Epoch {epoch + 1}" if epoch is not None else "Testing"
        loader = tqdm(loader, desc=desc)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(rank), y.to(rank)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    test_acc = accuracy_score(all_labels, all_preds) * 100
    test_mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = [matthews_corrcoef([1 if l == cls else 0 for l in all_labels],
                                       [1 if p == cls else 0 for p in all_preds]) for cls in range(NUM_CLASSES)]
    if rank == 0:
        logging.info(f"Test: Accuracy={test_acc:.2f}%, MCC={test_mcc:.4f}")
        logging.info(f"Test Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(CLASS_NAMES, per_class_mcc)]))
    return test_acc, test_mcc, per_class_mcc, CLASS_NAMES

def load_checkpoint(model, optimizer, scaler, checkpoint_path, rank):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        epoch = checkpoint['epoch']
        best_mcc = checkpoint.get('best_mcc', -1.0)
        no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
        frozen = checkpoint.get('frozen', True)
        run_id = checkpoint.get('run_id', 'unknown')
        if rank == 0:
            logging.info(f"Loaded checkpoint: {checkpoint_path}, Epoch {epoch}")
        return epoch, best_mcc, no_improve_epochs, frozen, run_id
    except Exception as e:
        if rank == 0:
            logging.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return 0, -1.0, 0, True, 'unknown'

def main(rank, world_size):
    setup_distributed(rank, world_size)
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    formatter = logging.Formatter(f"[RANK {rank}] %(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(file_handler)
    if rank == 0:
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    if rank == 0:
        run_id = hashlib.sha256(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        logging.info(f"Starting training (Run ID: {run_id})")
    else:
        run_id = "unknown"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if rank == 0:
        logging.info("Loading test dataset...")
    test_dataset = LargeTensorDataset(TEST_DIR, transform=val_transform, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
                             num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    if rank == 0:
        check_dataset(test_dataset, test_loader, "Test", rank)

    model = ResNetViTHybrid(NUM_CLASSES).to(rank)
    model.freeze_backbone()
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_mcc = -1.0
    no_improve_epochs = 0
    start_epoch = 0

    latest_checkpoint = max(glob.glob(os.path.join(CHECKPOINT_DIR, "model.pt")), key=os.path.getctime, default=None)
    if latest_checkpoint:
        start_epoch, best_mcc, no_improve_epochs, frozen, run_id = load_checkpoint(model, optimizer, scaler, latest_checkpoint, rank)
        if frozen and start_epoch >= FREEZE_EPOCHS:
            model.module.unfreeze_backbone()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            if rank == 0:
                logging.info(f"Unfreezing model and setting learning rate to {FINETUNE_LR}")

    if rank == 0:
        logging.info("Loading training dataset...")
    train_dataset = LargeTensorDataset(TRAIN_DIR, transform=train_transform, rank=rank)
    if rank == 0 and len(train_dataset) < EXPECTED_IMAGES * 0.8:
        logging.warning(f"Expected ~{EXPECTED_IMAGES:,} images, got {len(train_dataset):,} (Run ID: {run_id})")

    indices = list(range(len(train_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(0.8 * len(train_dataset))
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler,
                             num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, sampler=val_sampler,
                            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    if rank == 0:
        check_dataset(train_subset, train_loader, "Train", rank)
        check_dataset(val_subset, val_loader, "Validation", rank)
        logging.info(f"Train size: {len(train_subset):,}")
        logging.info(f"Train distribution: " + ", ".join(
            [f"{name}={sum(1 for i in train_indices if train_dataset.labels[i] == idx):,}"
             for idx, name in enumerate(CLASS_NAMES)]))
        logging.info(f"Validation size: {len(val_subset):,}")
        logging.info(f"Validation distribution: " + ", ".join(
            [f"{name}={sum(1 for i in val_indices if train_dataset.labels[i] == idx):,}"
             for idx, name in enumerate(CLASS_NAMES)]))

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        if rank == 0:
            logging.info(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}")

        if epoch == FREEZE_EPOCHS and model.module.resnet.conv1.weight.requires_grad == False:
            model.module.unfreeze_backbone()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            if rank == 0:
                logging.info(f"Unfreezing model and setting learning rate to {FINETUNE_LR}")

        train_loss, train_acc, train_mcc, train_per_class_mcc = train(model, train_loader, optimizer, criterion, scaler, rank, epoch)
        val_loss, val_acc, val_mcc, val_per_class_mcc = validate(model, val_loader, criterion, rank, epoch)
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Learning rate: {current_lr:.6f}")
            scheduler.step(val_mcc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                logging.info(f"Learning rate reduced to {new_lr:.6f}")

        if rank == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'epoch': epoch + 1,
                'best_mcc': best_mcc,
                'no_improve_epochs': no_improve_epochs,
                'frozen': model.module.resnet.conv1.weight.requires_grad == False,
                'run_id': run_id
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "model.pt"))
            if val_mcc > best_mcc + MIN_DELTA:
                best_mcc = val_mcc
                no_improve_epochs = 0
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"best_model_mcc_{best_mcc:.4f}.pt"))
                logging.info(f"Saved best model (MCC: {best_mcc:.4f})")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= PATIENCE:
                    logging.info(f"Stopping early after {no_improve_epochs} rounds with no improvement")
                    break

    if rank == 0:
        logging.info("Testing model...")
    test_acc, test_mcc, test_per_class_mcc, class_names = test(model, test_loader, rank)
    cleanup_distributed()

def run_training(world_size):
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    world_size = torch.cuda.device_count() or 1
    if world_size == 1 and torch.cuda.device_count() == 0:
        logging.warning("No GPUs found, using CPU")
    run_training(world_size)