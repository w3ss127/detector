# train_resnet_vit.py

import os
import yaml
import wandb
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.vision_transformer import vit_b_16, VisionTransformer_B_16_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------
# Argument Parsing
# ---------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=0, help='DDP local rank')
    return parser.parse_args()

# ---------------------------------------------
# Load YAML Config
# ---------------------------------------------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ---------------------------------------------
# Model Definition: ResNet50 + ViT
# ---------------------------------------------
class ResNet50ViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.conv_proj = nn.Conv2d(2048, 768, kernel_size=1)

        vit = vit_b_16(weights=VisionTransformer_B_16_Weights.IMAGENET1K_V1)
        self.cls_token = vit.cls_token
        self.pos_embedding = vit.encoder.pos_embedding
        self.encoder = vit.encoder
        self.dropout = vit.encoder.dropout
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.resnet(x)
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.dropout(x + self.pos_embedding[:, :x.size(1), :])
        x = self.encoder(x)
        return self.head(x[:, 0])

# ---------------------------------------------
# Dataset Preparation
# ---------------------------------------------
def load_dataset(path, image_size, class_sizes):
    all_images = torch.load(path)
    fixed_images, labels = [], []
    start = 0
    for idx, size in enumerate(class_sizes):
        for img in all_images[start:start + size]:
            if img.shape == (1, 256, 256):
                img = img.repeat(3, 1, 1)
            if img.shape == (3, 256, 256):
                fixed_images.append(img)
                labels.append(idx)
        start += size

    X = torch.stack(fixed_images).float() / 255.0
    X = F.interpolate(X, size=image_size, mode='bilinear', align_corners=False)
    y = torch.tensor(labels)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------------------------
# Metrics
# ---------------------------------------------
def compute_per_class_mcc(y_true, y_pred, num_classes):
    per_class_mcc = []
    for i in range(num_classes):
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)
        mcc = matthews_corrcoef(binary_true, binary_pred)
        per_class_mcc.append(mcc)
    return per_class_mcc

# ---------------------------------------------
# Train Loop
# ---------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc = correct / total
    mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = compute_per_class_mcc(np.array(all_labels), np.array(all_preds), num_classes)
    return total_loss / total, acc, mcc, per_class_mcc

# ---------------------------------------------
# Validation Loop
# ---------------------------------------------
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            preds = out.argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = correct / total
    mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = compute_per_class_mcc(np.array(all_labels), np.array(all_preds), num_classes)
    return total_loss / total, acc, mcc, per_class_mcc

# ---------------------------------------------
# Main Training
# ---------------------------------------------
def main():
    args = parse_args()
    config = load_config(args.config)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")

    if args.local_rank == 0:
        wandb.init(project=config['wandb_project'], config=config, name=config['run_name'])

    X_train, X_val, y_train, y_val = load_dataset(
        config['dataset_path'], tuple(config['image_size']), config['class_sizes'])

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], sampler=val_sampler)

    model = ResNet50ViT(config['num_classes']).to(device)
    model = DDP(model, device_ids=[args.local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scaler = torch.cuda.amp.GradScaler()

    best_acc, patience = 0, 0
    for epoch in range(config['epochs']):
        if epoch == config['unfreeze_at']:
            for param in model.module.resnet.parameters():
                param.requires_grad = True
            if args.local_rank == 0:
                print("🔓 Unfroze ResNet backbone")

        train_sampler.set_epoch(epoch)

        train_loss, train_acc, train_mcc, train_mcc_per_class = train_one_epoch(model, train_loader, criterion, optimizer, device, config['num_classes'])
        val_loss, val_acc, val_mcc, val_mcc_per_class = evaluate(model, val_loader, criterion, device, config['num_classes'])

        if args.local_rank == 0:
            wandb.log({
                "train_loss": train_loss, "train_acc": train_acc, "train_mcc": train_mcc,
                "val_loss": val_loss, "val_acc": val_acc, "val_mcc": val_mcc,
                **{f"train_mcc_class_{i}": m for i, m in enumerate(train_mcc_per_class)},
                **{f"val_mcc_class_{i}": m for i, m in enumerate(val_mcc_per_class)}
            }, step=epoch)

            print(f"Epoch {epoch+1}:")
            print(f"  🟦 Train Loss={train_loss:.4f} | Acc={train_acc:.4f} | MCC={train_mcc:.4f}")
            for i, mcc in enumerate(train_mcc_per_class):
                print(f"    Train MCC [Class {i}] = {mcc:.4f}")
            print(f"  🟨 Val   Loss={val_loss:.4f} | Acc={val_acc:.4f} | MCC={val_mcc:.4f}")
            for i, mcc in enumerate(val_mcc_per_class):
                print(f"    Val   MCC [Class {i}] = {mcc:.4f}")

            ckpt_name = f"{config['save_dir']}/epoch{epoch+1}_acc{val_acc:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'val_acc': val_acc
            }, ckpt_name)

            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                torch.save(model.module.state_dict(), config['best_model_path'])
            else:
                patience += 1

            if patience >= config['early_stopping']:
                break

    if args.local_rank == 0:
        print("✅ Training complete. Best accuracy:", best_acc)
        wandb.finish()

if __name__ == '__main__':
    main()