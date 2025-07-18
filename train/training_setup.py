# train/training_setup.py
import os
import torch
import wandb
from datetime import datetime
from torch import nn, optim
from torch.cuda.amp import GradScaler
from train.utils import load_checkpoint, print_model_info
from models.hybrid_model import HybridResNetViT

def setup_training(config, device, rank=0):
    # === Initialize Model ===
    model = HybridResNetViT(num_classes=config.model.num_classes, pretrained=True)
    model = model.to(device)
    print_model_info(model, config)

    # === Optimizer & Scheduler ===
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=3, verbose=(rank == 0))
    scaler = GradScaler()

    # === Loss Function ===
    criterion = nn.CrossEntropyLoss()

    # === Load Checkpoint If Needed ===
    start_epoch = 0
    best_val_acc = 0.0
    if config.train.resume_from:
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, scheduler, scaler,
                                                    config.train.resume_from, device)

    # === W&B Setup ===
    if rank == 0 and config.wandb.use:
        wandb.init(project=config.wandb.project,
                   name=config.wandb.run_name or f"{config.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                   config=config.as_dict())
        wandb.watch(model)

    return model, optimizer, scheduler, scaler, criterion, start_epoch, best_val_acc
