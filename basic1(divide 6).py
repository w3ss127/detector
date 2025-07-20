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
import logging
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "datasets"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 128  # Adjusted for A100 80GB
ACCUMULATION_STEPS = 1
NUM_CLASSES = 3
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
RESNET_VARIANT = "resnet50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 5
FINE_TUNE_AFTER = 5
SEED = 42
NUM_PARTS = 6
IMAGES_PER_PART = 100000
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

# Setup logging
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting training script")

# Check CUDA availability
if torch.cuda.is_available():
    logging.info(f"CUDA Available: Using {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logging.info(f"Total VRAM: {total_vram:.2f} GB")
    print(f"CUDA Available: Using {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram:.2f} GB")
else:
    logging.warning("CUDA not available. Falling back to CPU.")
    print("Warning: CUDA not available. Falling back to CPU.")

# ==========================
# DATA TRANSFORMS
# ==========================
train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================
# DATASET LOADER
# ==========================
class LargeTensorDataset(Dataset):
    def __init__(self, base_dir, transform=None, part_indices=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'real': 0, 'synthetic': 1, 'semi-synthetic': 2}
        self.class_dirs = {
            'real': os.path.join(base_dir, 'real'),
            'synthetic': os.path.join(base_dir, 'synthetic'),
            'semi-synthetic': os.path.join(base_dir, 'semi-synthetic')
        }
        self.class_counts = [0] * NUM_CLASSES
        self.rgb_counts = [0] * NUM_CLASSES
        self.grayscale_counts = [0] * NUM_CLASSES
        self.active_classes = []
        self.file_indices = []

        for class_name in self.class_to_idx:
            class_dir = self.class_dirs[class_name]
            if not os.path.exists(class_dir):
                logging.warning(f"Directory {class_dir} not found. Skipping {class_name}.")
                print(f"Warning: Directory {class_dir} not found. Skipping {class_name}.")
                continue
            pt_files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))
            if part_indices is not None and class_name in part_indices:
                pt_files = [pt_files[i] for i in part_indices[class_name] if i < len(pt_files)]
            if not pt_files:
                logging.warning(f"No .pt files found in {class_dir}. Skipping {class_name}.")
                print(f"Warning: No .pt files found in {class_dir}. Skipping {class_name}.")
                continue
            self.active_classes.append(class_name)
            label = self.class_to_idx[class_name]
            start_idx = len(self.image_paths)
            for pt_file in tqdm(pt_files, desc=f"Indexing {class_name}", leave=False):
                try:
                    with torch.no_grad():
                        images = torch.load(pt_file, map_location="cpu")
                    if len(images.shape) != 4 or images.shape[1] not in [1, 3] or images.shape[2:] != (256, 256):
                        logging.warning(f"Invalid tensor shape in {pt_file}: {images.shape}")
                        print(f"Warning: Invalid tensor shape in {pt_file}: {images.shape}")
                        continue
                    mean_pixel = (images.float() / 255.0).mean().item()
                    hist = np.histogram(images.cpu().numpy().flatten() / 255.0, bins=10, range=(0, 1))[0].tolist()
                    if mean_pixel < 0.078:
                        logging.warning(f"{pt_file}: Very dark (mean pixel value: {mean_pixel:.3f}), histogram: {hist}")
                        print(f"{pt_file}: Very dark (mean pixel value: {mean_pixel:.3f}), histogram: {hist}")
                        img_np = images[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        Image.fromarray(img_np).save(f"debug_image_{class_name}_{os.path.basename(pt_file)}_0.png")
                    logging.info(f"{pt_file}: Pre-normalization pixel range [min={images.min().item():.2f}, max={images.max().item():.2f}]")
                    print(f"{pt_file}: Pre-normalization pixel range [min={images.min().item():.2f}, max={images.max().item():.2f}]")
                    num_images = images.shape[0]
                    self.image_paths.extend([pt_file] * num_images)
                    self.labels.extend([label] * num_images)
                    self.file_indices.append((pt_file, start_idx, num_images))
                    self.class_counts[label] += num_images
                    if images.shape[1] == 3:
                        self.rgb_counts[label] += num_images
                    else:
                        self.grayscale_counts[label] += num_images
                    start_idx += num_images
                    images = None
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    logging.error(f"Error indexing {pt_file}: {e}")
                    print(f"Error indexing {pt_file}: {e}")
                    continue

        if not self.image_paths:
            logging.error("No valid .pt files found in dataset.")
            raise ValueError("No valid .pt files found in dataset.")

        logging.info(f"Dataset: {base_dir}")
        total_images = len(self.image_paths)
        logging.info(f"Total images: {total_images}")
        print(f"\nDataset: {base_dir}")
        print(f"Total images: {total_images}")
        for idx, class_name in enumerate(self.class_to_idx):
            percentage = 100 * self.class_counts[idx] / total_images if total_images > 0 else 0
            logging.info(f"{class_name}: {self.class_counts[idx]} images "
                        f"({percentage:.2f}% if non-zero), "
                        f"{self.rgb_counts[idx]} RGB, {self.grayscale_counts[idx]} grayscale")
            print(f"{class_name}: {self.class_counts[idx]} images "
                  f"({percentage:.2f}% if non-zero), "
                  f"{self.rgb_counts[idx]} RGB, {self.grayscale_counts[idx]} grayscale")

        expected_counts = {'real': 200000, 'synthetic': 200000, 'semi-synthetic': 200000} if 'train' in base_dir else {'real': 20000, 'synthetic': 20000, 'semi-synthetic': 20000}
        for class_name, count in expected_counts.items():
            actual = self.class_counts[self.class_to_idx[class_name]]
            if actual != count and actual > 0:
                logging.warning(f"Mismatch in {class_name}: Expected {count}, Got {actual}")
                print(f"Warning: Mismatch in {class_name}: Expected {count}, Got {actual}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")
        pt_file, start_idx = next((f, s) for f, s, n in self.file_indices if s <= idx < s + n)
        inner_idx = idx - start_idx
        try:
            with torch.no_grad():
                images = torch.load(pt_file, map_location="cpu")
            if inner_idx >= images.shape[0]:
                raise IndexError(f"Inner index {inner_idx} out of range for {pt_file} with {images.shape[0]} images")
            x = images[inner_idx]
            images = None
            y = self.labels[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
        except Exception as e:
            logging.error(f"Error loading image at index {idx} from {pt_file}: {e}")
            raise

    def get_class_weights(self):
        total = sum(self.class_counts)
        weights = [total / (len(self.active_classes) * count) if count > 0 else 0 for count in self.class_counts]
        return torch.tensor(weights, dtype=torch.float).to(DEVICE)

    def get_sampler_weights(self, indices=None):
        if indices is None:
            weights = [1.0 / self.class_counts[label] if self.class_counts[label] > 0 else 0 for label in self.labels]
        else:
            subset_labels = [self.labels[i] for i in indices]
            class_counts = [sum(1 for l in subset_labels if l == i) for i in range(NUM_CLASSES)]
            weights = [1.0 / class_counts[label] if class_counts[label] > 0 else 0 for label in subset_labels]
        return weights

# ==========================
# DIAGNOSTIC FUNCTION
# ==========================
def check_dataset(dataset, loader, name="Dataset"):
    logging.info(f"=== {name} Check ===")
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Class counts: {dataset.class_counts}")
    logging.info(f"RGB counts: {dataset.rgb_counts}")
    logging.info(f"Grayscale counts: {dataset.grayscale_counts}")
    
    print(f"\n=== {name} Check ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Class counts: {dataset.class_counts}")
    print(f"RGB counts: {dataset.rgb_counts}")
    print(f"Grayscale counts: {dataset.grayscale_counts}")
    
    logging.info(f"Sampling 5 images from {name}:")
    print(f"\nSampling 5 images from {name}:")
    indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    for idx in indices:
        try:
            img, label = dataset[idx]
            mean_pixel = (img[:3] / 255.0).mean().item() if img.shape[0] >= 3 else img.mean().item()
            hist = np.histogram(img.cpu().numpy().flatten() / 255.0, bins=10, range=(0, 1))[0].tolist()
            if mean_pixel < 0.078:
                logging.warning(f"Index {idx}: Very dark (mean pixel value: {mean_pixel:.3f}), histogram: {hist}")
                print(f"Index {idx}: Very dark (mean pixel value: {mean_pixel:.3f}), histogram: {hist}")
                img_np = img[:3].permute(1, 2, 0).cpu().numpy().astype(np.uint8) if img.shape[0] >= 3 else img[0].cpu().numpy().astype(np.uint8)
                Image.fromarray(img_np).save(f"debug_image_{name.lower()}_idx_{idx}.png")
            logging.info(f"Index {idx}: Shape={img.shape}, Label={label}, "
                        f"Type={'RGB' if img.shape[0] == 3 else 'Grayscale'}, "
                        f"Pre-normalization mean={mean_pixel:.3f}, histogram={hist}")
            print(f"Index {idx}: Shape={img.shape}, Label={label}, "
                  f"Type={'RGB' if img.shape[0] == 3 else 'Grayscale'}, "
                  f"Pre-normalization mean={mean_pixel:.3f}, histogram={hist}")
        except Exception as e:
            logging.error(f"Error sampling index {idx}: {e}")
            print(f"Error sampling index {idx}: {e}")

    logging.info(f"First batch from {name} loader:")
    print(f"\nFirst batch from {name} loader:")
    try:
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= 1:
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logging.info(f"Batch shape: images={x.shape}, labels={y.shape}, Device={x.device}")
            logging.info(f"Labels: {y.tolist()}")
            logging.info(f"Class distribution: real={torch.sum(y == 0)}, "
                        f"synthetic={torch.sum(y == 1)}, semi-synthetic={torch.sum(y == 2)}")
            print(f"Batch shape: images={x.shape}, labels={y.shape}, Device={x.device}")
            print(f"Labels: {y.tolist()}")
            print(f"Class distribution: real={torch.sum(y == 0)}, "
                  f"synthetic={torch.sum(y == 1)}, semi-synthetic={torch.sum(y == 2)}")
    except Exception as e:
        logging.error(f"Error loading batch from {name} loader: {e}")
        print(f"Error loading batch from {name} loader: {e}")

# ==========================
# MODEL DEFINITION
# ==========================
class ResNetViTHybrid(nn.Module):
    def __init__(self, num_classes, resnet_variant="resnet50"):
        super().__init__()
        if resnet_variant == "resnet50":
            self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        elif resnet_variant == "resnet121":
            self.resnet = models.resnet121(weights='IMAGENET1K_V2')
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
    
    per_class_mcc = []
    class_names = ['real', 'synthetic', 'semi-synthetic']
    for class_idx in range(NUM_CLASSES):
        binary_labels = (labels == class_idx).astype(int)
        binary_preds = (preds == class_idx).astype(int)
        try:
            mcc = matthews_corrcoef(binary_labels, binary_preds)
        except ValueError:
            mcc = 0.0
        per_class_mcc.append(mcc)
    
    return accuracy, overall_mcc, per_class_mcc, class_names

# ==========================
# TRAINING + VALIDATION
# ==========================
def train(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []
    accumulation_counter = 0

    for x, y in tqdm(loader, desc="Train", leave=False, postfix={"loss": 0.0, "acc": 0.0}):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast():
            outputs = model(x)
            loss = criterion(outputs, y) / ACCUMULATION_STEPS
        scaler.scale(loss).backward()

        accumulation_counter += 1
        if accumulation_counter % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accumulation_counter = 0

        running_loss += loss.item() * x.size(0) * ACCUMULATION_STEPS
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.cpu())
        labels_all.extend(y.cpu())

        batch_acc = (preds == y).float().mean().item() * 100
        tqdm_loader = tqdm(loader)
        tqdm_loader.set_postfix({"loss": running_loss / len(preds_all), "acc": batch_acc})

    if accumulation_counter > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = running_loss / len(loader.dataset)
    accuracy, overall_mcc, per_class_mcc, class_names = calculate_metrics(torch.tensor(preds_all), torch.tensor(labels_all))
    return avg_loss, accuracy, overall_mcc, per_class_mcc, class_names

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validate", leave=False, postfix={"loss": 0.0, "acc": 0.0}):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu())
            labels_all.extend(y.cpu())

            batch_acc = (preds == y).float().mean().item() * 100
            tqdm_loader = tqdm(loader)
            tqdm_loader.set_postfix({"loss": running_loss / len(preds_all), "acc": batch_acc})

    avg_loss = running_loss / len(loader.dataset)
    accuracy, overall_mcc, per_class_mcc, class_names = calculate_metrics(torch.tensor(preds_all), torch.tensor(labels_all))
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
            preds_all.extend(preds.cpu())
            labels_all.extend(y.cpu())

    accuracy, overall_mcc, per_class_mcc, class_names = calculate_metrics(torch.tensor(preds_all), torch.tensor(labels_all))
    logging.info(f"Test predicted class distribution: {np.bincount(preds_all, minlength=NUM_CLASSES).tolist()}")
    print(f"Test predicted class distribution: {np.bincount(preds_all, minlength=NUM_CLASSES).tolist()}")
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
            try:
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                logging.info(f"Best model saved to {CHECKPOINT_DIR}/best_model.pt")
                print(f"Best model saved to {CHECKPOINT_DIR}/best_model.pt")
            except Exception as e:
                logging.error(f"Error saving best model: {e}")
                print(f"Error saving best model: {e}")
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    # Load test dataset
    test_dataset = LargeTensorDataset(TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)
    check_dataset(test_dataset, test_loader, "Test")

    # Define 6 parts
    part_files = {
        'real': [list(range(i * 7, (i + 1) * 7)) for i in range(NUM_PARTS)],
        'synthetic': [list(range(i * 7, (i + 1) * 7)) for i in range(NUM_PARTS)],
        'semi-synthetic': [list(range(i * 7, (i + 1) * 7)) for i in range(NUM_PARTS)]
    }

    # Initialize model
    model = ResNetViTHybrid(NUM_CLASSES, resnet_variant=RESNET_VARIANT).to(DEVICE)
    class_weights = test_dataset.get_class_weights()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Train on each part
    for part_num in range(1, NUM_PARTS + 1):
        logging.info(f"\nStarting training for Part {part_num}")
        print(f"\nStarting training for Part {part_num}")

        # Load checkpoint from previous part
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_part_{part_num-1}.pt") if part_num > 1 else None
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                scaler.load_state_dict(checkpoint['scaler_state'])
                start_epoch = checkpoint['epoch']
                logging.info(f"Loaded checkpoint: {checkpoint_path}")
                print(f"Loaded checkpoint: {checkpoint_path}")
            except Exception as e:
                logging.error(f"Error loading checkpoint {checkpoint_path}: {e}")
                print(f"Error loading checkpoint {checkpoint_path}: {e}")

        # Load dataset for this part
        part_indices = {
            'real': part_files['real'][part_num-1],
            'synthetic': part_files['synthetic'][part_num-1],
            'semi-synthetic': part_files['semi-synthetic'][part_num-1]
        }
        part_dataset = LargeTensorDataset(TRAIN_DIR, transform=train_transforms, part_indices=part_indices)
        if len(part_dataset.active_classes) < 3:
            logging.warning(f"Part {part_num}: Missing classes {set(['real', 'synthetic', 'semi-synthetic']) - set(part_dataset.active_classes)}. Skipping.")
            print(f"Part {part_num}: Missing classes {set(['real', 'synthetic', 'semi-synthetic']) - set(part_dataset.active_classes)}. Skipping.")
            continue
        check_dataset(part_dataset, DataLoader(part_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2), f"Part {part_num}")

        # Split part into train and validation
        indices = list(range(len(part_dataset)))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, stratify=[part_dataset.labels[i] for i in indices], random_state=SEED
        )
        train_subset = Subset(part_dataset, train_idx)
        val_subset = Subset(part_dataset, val_idx)

        # Log subset distributions
        train_labels = [part_dataset.labels[i] for i in train_idx]
        val_labels = [part_dataset.labels[i] for i in val_idx]
        logging.info(f"Part {part_num} Train subset size: {len(train_subset)}")
        logging.info(f"Part {part_num} Train subset distribution: real={sum(l == 0 for l in train_labels)}, "
                    f"synthetic={sum(l == 1 for l in train_labels)}, semi-synthetic={sum(l == 2 for l in train_labels)}")
        print(f"Part {part_num} Train subset size: {len(train_subset)}")
        print(f"Part {part_num} Train subset distribution: real={sum(l == 0 for l in train_labels)}, "
              f"synthetic={sum(l == 1 for l in train_labels)}, semi-synthetic={sum(l == 2 for l in train_labels)}")
        logging.info(f"Part {part_num} Validation subset size: {len(val_subset)}")
        logging.info(f"Part {part_num} Validation subset distribution: real={sum(l == 0 for l in val_labels)}, "
                    f"synthetic={sum(l == 1 for l in val_labels)}, semi-synthetic={sum(l == 2 for l in val_labels)}")
        print(f"Part {part_num} Validation subset size: {len(val_subset)}")
        print(f"Part {part_num} Validation subset distribution: real={sum(l == 0 for l in val_labels)}, "
              f"synthetic={sum(l == 1 for l in val_labels)}, semi-synthetic={sum(l == 2 for l in val_labels)}")

        # Data loaders
        train_sampler_weights = part_dataset.get_sampler_weights(train_idx)
        train_sampler = WeightedRandomSampler(train_sampler_weights, len(train_sampler_weights), replacement=True)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=12, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)

        # Training loop for part
        early_stopping = EarlyStopping(patience=PATIENCE)
        for epoch in range(start_epoch, NUM_EPOCHS):
            logging.info(f"Part {part_num}, Epoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"\nPart {part_num}, Epoch {epoch + 1}/{NUM_EPOCHS}")

            if epoch == FINE_TUNE_AFTER:
                for param in model.resnet.layer4.parameters():
                    param.requires_grad = True
                for param in model.vit.blocks[-1].parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE / 100, weight_decay=1e-4)
                logging.info(f"Part {part_num}: Unfroze ResNet layer4 and ViT last block")
                print(f"Part {part_num}: Unfroze ResNet layer4 and ViT last block")

            train_loss, train_acc, train_mcc, train_per_class_mcc, class_names = train(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_acc, val_mcc, val_per_class_mcc, _ = validate(model, val_loader, criterion)

            logging.info(f"Part {part_num}, Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MCC: {train_mcc:.4f}")
            logging.info(f"Part {part_num}, Train Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, train_per_class_mcc)]))
            logging.info(f"Part {part_num}, Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MCC: {val_mcc:.4f}")
            logging.info(f"Part {part_num}, Val Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, val_per_class_mcc)]))
            print(f"Part {part_num}, Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MCC: {train_mcc:.4f}")
            print(f"Part {part_num}, Train Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, train_per_class_mcc)]))
            print(f"Part {part_num}, Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MCC: {val_mcc:.4f}")
            print(f"Part {part_num}, Val Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, val_per_class_mcc)]))
            scheduler.step()

            if early_stopping.step(val_mcc, model):
                logging.info(f"Part {part_num}: Early stopping triggered")
                print(f"Part {part_num}: Early stopping triggered.")
                model.load_state_dict(early_stopping.best_model)
                break

        # Save checkpoint for this part
        try:
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'epoch': epoch + 1
            }
            output_path = os.path.join(CHECKPOINT_DIR, f"model_part_{part_num}.pt")
            torch.save(checkpoint, output_path)
            logging.info(f"Part {part_num}: Checkpoint saved to {output_path}")
            print(f"Part {part_num}: Checkpoint saved to {output_path}")
        except Exception as e:
            logging.error(f"Part {part_num}: Error saving checkpoint: {e}")
            print(f"Part {part_num}: Error saving checkpoint: {e}")

    # Final test evaluation
    logging.info("Evaluating on test set")
    print("\nEvaluating on test set...")
    test_acc, test_mcc, test_per_class_mcc, class_names = test(model, test_loader)
    logging.info(f"Test Accuracy: {test_acc:.2f}%, Test MCC: {test_mcc:.4f}")
    logging.info(f"Test Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, test_per_class_mcc)]))
    print(f"Test Accuracy: {test_acc:.2f}%, Test MCC: {test_mcc:.4f}")
    print(f"Test Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, test_per_class_mcc)]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        final_path = os.path.join(CHECKPOINT_DIR, f"model_final_{timestamp}.pt")
        torch.save({'model_state_dict': model.state_dict()}, final_path)
        logging.info(f"Final model saved as {final_path}")
        print(f"Final model saved as {final_path}")
    except Exception as e:
        logging.error(f"Error saving final model: {e}")
        print(f"Error saving final model: {e}")

if __name__ == "__main__":
    main()