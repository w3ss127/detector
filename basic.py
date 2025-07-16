# train_resnet_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import numpy as np
import os

# ============================
# 1. Configuration
# ============================

CONFIG = {
    'num_classes': 3,
    'num_epochs': 30,
    'batch_size': 32,
    'unfreeze_at_epoch': 20,
    'initial_lr': 1e-4,
    'fine_tune_lr': 1e-5,

    'tensor_path': './datasets.pt',
    'model_weights_path': 'bitmind.pth',
    'full_model_path': 'full_bitmind.pth',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'image_size': (224, 224)
}

# ============================
# 2. Dataset Loading
# ============================

from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import io

from datasets import load_dataset
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from datasets import load_dataset
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_prepare_dataset(image_size=(224, 224), max_per_class=5000):
    dataset_paths = [
        ("bitmind/bm-real", 0),
        ("bitmind/bm-syn", 1),
        ("bitmind/bm-semi", 2),
    ]

    all_images = []
    all_labels = []

    for dataset_name, label in dataset_paths:
        ds = load_dataset(dataset_name, split='train')
        count = 0
        for item in ds:
            if count >= max_per_class:
                break
            image = item["image"]
            tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()  # [H, W, C] → [C, H, W]
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] != 3:
                continue
            all_images.append(tensor)
            all_labels.append(label)
            print(all_labels[0])
            print(all_labels[20000])
            print(all_labels[40000])
            count += 1

    X = torch.stack(all_images) / 255.0
    X = F.interpolate(X, size=image_size, mode='bilinear', align_corners=False)
    y = torch.tensor(all_labels)

    return train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)


# ============================
# 3. Model Definition
# ============================

class ResNetCustomTop(nn.Module):
    def __init__(self, backbone, out_channels, num_classes):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.norm(x)
        x = self.pool(x)
        return self.classifier(x)

# ============================
# 4. Training & Evaluation
# ============================

def preprocess_images(X_np):
    X = torch.tensor(X_np, dtype=torch.float32)
    return F.interpolate(X, size=CONFIG['image_size'], mode='bilinear', align_corners=False)

def run_training():
    X_train, X_test, y_train, y_test = load_and_prepare_dataset(image_size=CONFIG['image_size'],max_per_class=20000) # limit to 3000 per class → total 9000)


    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long) 

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)

    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(base_model.children())[:-2])
    out_channels = base_model.fc.in_features

    for param in backbone.parameters():
        param.requires_grad = False

    model = ResNetCustomTop(backbone, out_channels, CONFIG['num_classes']).to(CONFIG['device'])
    summary(model, input_size=(CONFIG['batch_size'], 3, *CONFIG['image_size']))
    
    try:
        model.load_state_dict(torch.load(CONFIG['model_weights_path'], map_location=CONFIG['device']))
    except RuntimeError as e:
        print(f"Warning: Could not load model weights due to: {e}\nProceeding with randomly initialized weights.")

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=CONFIG['initial_lr'], momentum=0.2, weight_decay=0, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['initial_lr'])

    for epoch in range(CONFIG['num_epochs']):
        if epoch == CONFIG['unfreeze_at_epoch']:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # optimizer = optim.Adam(model.parameters(), lr=CONFIG['fine_tune_lr'])
            optimizer = optim.SGD(model.parameters(), lr=CONFIG['fine_tune_lr'], momentum=0.2, weight_decay=0, nesterov=True)
            print(f"Backbone unfrozen at epoch {epoch}")

        # Train
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = train_correct / total
        train_mcc = matthews_corrcoef(all_labels, all_preds)
              
        # Validation
        model.eval()
        test_loss,test_correct, total = 0, 0, 0
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader: # Changed from val_loader to test_loader
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                test_correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_acc = test_correct / total
        test_mcc = matthews_corrcoef(test_labels, test_preds)
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | Train Loss: {train_loss/total:.4f}, Acc: {train_acc:.4f}, MCC: {train_mcc:.4f} | Val Loss: {test_loss/total:.4f}, Acc: {test_acc:.4f}, MCC: {test_mcc:.4f}")


    # Save model
    torch.save(model.state_dict(), CONFIG['model_weights_path'])
    torch.save(model, CONFIG['full_model_path'])
    print(f"Model saved to {CONFIG['model_weights_path']} and {CONFIG['full_model_path']}")


if __name__ == "__main__":
    run_training()
