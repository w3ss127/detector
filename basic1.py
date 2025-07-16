import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from torchvision.models.vision_transformer import vit_b_16
import numpy as np
import os

# -----------------------------
# 1. Configuration
# -----------------------------
CONFIG = {
    'num_classes': 3,
    'num_epochs': 10,
    'batch_size': 64,
    'initial_lr': 1e-4,
    'unfreeze_at_epoch': 5,
    'fine_tune_lr': 1e-5,
    'tensor_path': './datasets.pt',
    'model_path': 'bitmind.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'image_size': (224, 224),
}

# -----------------------------
# 2. Data Loading & Preprocessing
# -----------------------------
def load_and_prepare_dataset(path):
    all_images = torch.load(path)  # [N, C, H, W]
    class_sizes = [50000, 50000, 50000]
    assert all_images.shape[0] == sum(class_sizes)

    fixed_images, labels = [], []
    start = 0
    for class_idx, class_size in enumerate(class_sizes):
        for img in all_images[start:start + class_size]:
            if img.shape == (1, 256, 256):
                img = img.repeat(3, 1, 1)
            if img.shape == (3, 256, 256):
                fixed_images.append(img)
                labels.append(class_idx)
        start += class_size

    all_images = torch.stack(fixed_images).float()/255
    labels = torch.tensor(labels)
    print(labels[0])
    print(labels[10000])
    print(labels[20000])

    return train_test_split(all_images, labels, test_size=0.00025, stratify=labels, random_state=42)

# -----------------------------
# 3. Model Definition
# -----------------------------
class ResNet50ViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_feature = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, 7, 7]
        self.norm = nn.BatchNorm2d(2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(2048, 768)

        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet_feature(x)  # [B, 2048, 7, 7]
        x = self.norm(x)
        x = self.pool(x).flatten(1)  # [B, 2048]
        x = self.proj(x).unsqueeze(1)  # [B, 1, 768]
        x = self.vit.encoder(x)  # [B, 1, 768]
        x = x.mean(dim=1)
        return self.classifier(x)

# -----------------------------
# 4. Utility: Per-Class MCC
# -----------------------------
def per_class_mcc(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_mccs = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
        mcc = numerator / denominator if denominator > 0 else 0
        per_class_mccs.append(mcc)
    return per_class_mccs

# -----------------------------
# 5. Training Pipeline
# -----------------------------
def train_model():
    X_train, X_test, y_train, y_test = load_and_prepare_dataset(CONFIG['tensor_path'])
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)

    model = ResNet50ViT(CONFIG['num_classes']).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['initial_lr'])

    start_epoch = 0
    best_val_acc = 0.0

    # Load checkpoint if exists
    if os.path.exists(CONFIG['model_path']):
        checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"✅ Loaded checkpoint (epoch {start_epoch}, best val_acc={best_val_acc:.4f})")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights only")

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        if epoch == CONFIG['unfreeze_at_epoch']:
            for param in model.resnet_feature.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['fine_tune_lr'])
            print(f"🔓 Unfroze ResNet backbone at epoch {epoch}")

        # Training
        model.train()
        total, correct, train_loss = 0, 0, 0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = correct / total
        train_loss /= total
        train_mcc = matthews_corrcoef(all_labels, all_preds)
        train_mccs = per_class_mcc(all_labels, all_preds, CONFIG['num_classes'])

        # Validation
        model.eval()
        total, correct, val_loss = 0, 0, 0
        test_preds, test_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        val_loss /= total
        val_mcc = matthews_corrcoef(test_labels, test_preds)
        val_mccs = per_class_mcc(test_labels, test_preds, CONFIG['num_classes'])

        print(f"\n📘 Epoch {epoch+1:02d}")
        print(f"  🔹 Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | MCC: {train_mcc:.4f}")
        for i, mcc in enumerate(train_mccs):
            print(f"    ➤ Train Class {i} MCC: {mcc:.4f}")
        print(f"  🔸 Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f}")
        for i, mcc in enumerate(val_mccs):
            print(f"    ➤ Val   Class {i} MCC: {mcc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_acc': best_val_acc,
            }, CONFIG['model_path'])
            print("💾 Saved best model!")

# -----------------------------
# 6. Main
# -----------------------------
if __name__ == "__main__":
    train_model()
