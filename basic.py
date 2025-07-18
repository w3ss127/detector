import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "datasets"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
BATCH_SIZE = 32
ACCUMULATION_STEPS = 2
NUM_CLASSES = 3
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 5
CHECKPOINT_PATH = "checkpoint.pt"
BEST_MODEL_PATH = "best_model.pt"
FINE_TUNE_AFTER = 5
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting training script")

# Check CUDA availability
if torch.cuda.is_available():
    logging.info(f"CUDA Available: Using {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logging.info(f"Total VRAM: {total_vram:.2f} GB")
    if total_vram < 12 and BATCH_SIZE > 16:
        logging.warning(f"Low VRAM ({total_vram:.2f} GB) for batch_size={BATCH_SIZE}. Consider reducing.")
else:
    logging.warning("CUDA not available. Using CPU.")

# ==========================
# DATA TRANSFORMS
# ==========================
train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x if x.shape[1:] == (224, 224) else transforms.Resize((224, 224))(x)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
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
        self.class_to_idx = {'real': 0, 'synthetic': 1, 'semi-synthetic': 2}
        self.tensors = []
        self.class_counts = [0] * NUM_CLASSES
        self.rgb_counts = [0] * NUM_CLASSES
        self.grayscale_counts = [0] * NUM_CLASSES
        self.active_classes = []

        for class_name in self.class_to_idx:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.exists(class_dir):
                logging.warning(f"Directory {class_dir} not found. Skipping {class_name}.")
                print(f"Warning: Directory {class_dir} not found. Skipping {class_name}.")
                continue
            pt_files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))
            if not pt_files:
                logging.warning(f"No .pt files found in {class_dir}. Skipping {class_name}.")
                print(f"Warning: No .pt files found in {class_dir}. Skipping {class_name}.")
                continue
            self.active_classes.append(class_name)
            for pt_file in tqdm(pt_files, desc=f"Loading {class_name}", leave=False):
                try:
                    images = torch.load(pt_file, map_location="cpu")
                    if len(images.shape) != 4 or images.shape[1] not in [1, 3] or images.shape[2:] != (256, 256):
                        logging.warning(f"Invalid tensor shape in {pt_file}: {images.shape}")
                        continue
                    if images.max() > 1.0 or images.min() < 0.0:
                        logging.warning(f"Image values out of range [0,1] in {pt_file}")
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
                    logging.error(f"Error loading {pt_file}: {e}")
                    continue

        if not self.image_paths:
            logging.error("No valid .pt files found in dataset.")
            raise ValueError("No valid .pt files found in dataset.")
        
        self.index_map = []
        for i, images in enumerate(self.tensors):
            length = images.shape[0]
            self.index_map.extend([(i, j) for j in range(length)])

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

        # Verify expected counts
        expected_counts = {'real': 95000, 'synthetic': 95000, 'semi-synthetic': 190000} if 'train' in base_dir else {'real': 5000, 'synthetic': 5000, 'semi-synthetic': 10000}
        for class_name, count in expected_counts.items():
            actual = self.class_counts[self.class_to_idx[class_name]]
            if actual != count and actual > 0:
                logging.warning(f"Mismatch in {class_name}: Expected {count}, Got {actual}")
                print(f"Warning: Mismatch in {class_name}: Expected {count}, Got {actual}")

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
        weights = [total / (len(self.active_classes) * count) if count > 0 else 0 for count in self.class_counts]
        return torch.tensor(weights, dtype=torch.float).to(DEVICE)

    def get_sampler_weights(self):
        weights = [1.0 / self.class_counts[label] if self.class_counts[label] > 0 else 0 for label in self.labels]
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
    indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    for idx in indices:
        img, label = dataset[idx]
        logging.info(f"Index {idx}: Shape={img.shape}, Label={label}, "
                    f"Type={'RGB' if img.shape[0] == 3 else 'Grayscale'}")
        print(f"Index {idx}: Shape={img.shape}, Label={label}, "
              f"Type={'RGB' if img.shape[0] == 3 else 'Grayscale'}")
    
    logging.info(f"First batch from {name} loader:")
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

# ==========================
# MODEL DEFINITION
# ==========================
class ResNetViTHybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = timm.create_model('resnet50', pretrained=True)
        resnet_fc_in = self.resnet.get_classifier().in_features
        self.resnet.head.fc = nn.Identity()
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
        with torch.cuda.amp.autocast():
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
        preds_all.extend(preds)
        labels_all.extend(y)

        batch_acc = (preds == y).float().mean().item() * 100
        tqdm_loader = tqdm(loader)
        tqdm_loader.set_postfix({"loss": running_loss / len(preds_all), "acc": batch_acc})

    if accumulation_counter > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = running_loss / len(loader.dataset)
    accuracy, overall_mcc, per_class_mcc, class_names = calculate_metrics(torch.stack(preds_all), torch.stack(labels_all))
    return avg_loss, accuracy, overall_mcc, per_class_mcc, class_names

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validate", leave=False, postfix={"loss": 0.0, "acc": 0.0}):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds)
            labels_all.extend(y)

            batch_acc = (preds == y).float().mean().item() * 100
            tqdm_loader = tqdm(loader)
            tqdm_loader.set_postfix({"loss": running_loss / len(preds_all), "acc": batch_acc})

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
            with torch.cuda.amp.autocast():
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
            try:
                torch.save(self.best_model, BEST_MODEL_PATH)
                logging.info(f"Best model saved to {BEST_MODEL_PATH}")
            except Exception as e:
                logging.error(f"Error saving best model: {e}")
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
    check_dataset(train_dataset, DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True), "Train")
    check_dataset(test_dataset, DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True), "Test")

    # Compute class weights
    class_weights = train_dataset.get_class_weights()
    logging.info(f"Class weights: {class_weights.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")
    sampler_weights = train_dataset.get_sampler_weights()
    sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement=True)

    # Split training data
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=[train_dataset.labels[i] for i in indices], random_state=SEED
    )
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    # Log subset distributions
    train_labels = [train_dataset.labels[i] for i in train_idx]
    val_labels = [train_dataset.labels[i] for i in val_idx]
    logging.info(f"Train subset size: {len(train_subset)}")
    logging.info(f"Train subset distribution: real={sum(l == 0 for l in train_labels)}, "
                f"synthetic={sum(l == 1 for l in train_labels)}, semi-synthetic={sum(l == 2 for l in train_labels)}")
    print(f"Train subset size: {len(train_subset)}")
    print(f"Train subset distribution: real={sum(l == 0 for l in train_labels)}, "
          f"synthetic={sum(l == 1 for l in train_labels)}, semi-synthetic={sum(l == 2 for l in train_labels)}")
    logging.info(f"Validation subset size: {len(val_subset)}")
    logging.info(f"Validation subset distribution: real={sum(l == 0 for l in val_labels)}, "
                f"synthetic={sum(l == 1 for l in val_labels)}, semi-synthetic={sum(l == 2 for l in val_labels)}")
    print(f"Validation subset size: {len(val_subset)}")
    print(f"Validation subset distribution: real={sum(l == 0 for l in val_labels)}, "
          f"synthetic={sum(l == 1 for l in val_labels)}, semi-synthetic={sum(l == 2 for l in val_labels)}")

    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Check train loader
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= 1:
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logging.info(f"Train batch shape: images={x.shape}, labels={y.shape}, Device={x.device}")
        logging.info(f"Train batch distribution: real={torch.sum(y == 0)}, "
                    f"synthetic={torch.sum(y == 1)}, semi-synthetic={torch.sum(y == 2)}")
        print(f"Train batch shape: images={x.shape}, labels={y.shape}, Device={x.device}")
        print(f"Train batch distribution: real={torch.sum(y == 0)}, "
              f"synthetic={torch.sum(y == 1)}, semi-synthetic={torch.sum(y == 2)}")

    # Initialize model
    model = ResNetViTHybrid(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Resume from checkpoint
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resumed training from epoch {start_epoch}")
            print(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            print(f"Error loading checkpoint: {e}")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        if epoch == FINE_TUNE_AFTER:
            for param in model.resnet.layer4.parameters():
                param.requires_grad = True
            for param in model.vit.blocks[-1].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE / 100, weight_decay=1e-4)
            logging.info("Unfroze ResNet layer4 and ViT last block")
            print("Unfroze ResNet layer4 and ViT last block")

        train_loss, train_acc, train_mcc, train_per_class_mcc, class_names = train(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_mcc, val_per_class_mcc, _ = validate(model, val_loader, criterion)

        logging.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MCC: {train_mcc:.4f}")
        logging.info(f"Train Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, train_per_class_mcc)]))
        logging.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MCC: {val_mcc:.4f}")
        logging.info(f"Val Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, val_per_class_mcc)]))
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MCC: {train_mcc:.4f}")
        print(f"Train Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, train_per_class_mcc)]))
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MCC: {val_mcc:.4f}")
        print(f"Val Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, val_per_class_mcc)]))
        scheduler.step(val_mcc)

        try:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_mcc': val_mcc
            }, CHECKPOINT_PATH)
            logging.info(f"Checkpoint saved to {CHECKPOINT_PATH}")
            print(f"Checkpoint saved to {CHECKPOINT_PATH}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
            print(f"Error saving checkpoint: {e}")

        if early_stopping.step(val_mcc, model):
            logging.info("Early stopping triggered")
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.best_model)
            break

    logging.info("Evaluating on test set")
    print("\nEvaluating on test set")
    test_acc, test_mcc, test_per_class_mcc, class_names = test(model, test_loader)
    logging.info(f"Test Accuracy: {test_acc:.2f}%, Test MCC: {test_mcc:.4f}")
    logging.info(f"Test Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, test_per_class_mcc)]))
    print(f"Test Accuracy: {test_acc:.2f}%, Test MCC: {test_mcc:.4f}")
    print(f"Test Per-Class MCC: " + ", ".join([f"{name}: {mcc:.4f}" for name, mcc in zip(class_names, test_per_class_mcc)]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        final_path = f"model_final_{timestamp}.pt"
        torch.save(model.state_dict(), final_path)
        logging.info(f"Final model saved as {final_path}")
        print(f"Final model saved as {final_path}")
    except Exception as e:
        logging.error(f"Error saving final model: {e}")
        print(f"Error saving final model: {e}")

if __name__ == "__main__":
    main()
