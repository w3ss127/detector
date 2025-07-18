# train/dataloaders.py

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import os

def load_tensor_dataset(path: str, batch_size: int = 64, val_split: float = 0.1, test_split: float = 0.1, num_workers: int = 2):
    """
    Loads a dataset from a .pt file and returns train/val/test dataloaders.
    
    Args:
        path (str): Path to .pt file containing image tensor of shape [N, 3, H, W]
        batch_size (int): Batch size
        val_split (float): Proportion for validation set
        test_split (float): Proportion for test set
        num_workers (int): DataLoader workers

    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"📂 Loading dataset from {path}")
    tensor_data = torch.load(path)  # [N, 3, H, W]
    total_size = tensor_data.size(0)

    # Dummy labels: 0 for real, 1 for syn, 2 for semi
    labels = torch.cat([
        torch.zeros(total_size // 3, dtype=torch.long),
        torch.ones(total_size // 3, dtype=torch.long),
        torch.full((total_size // 3,), 2, dtype=torch.long)
    ])

    dataset = TensorDataset(tensor_data, labels)

    # Split sizes
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"✅ Dataset loaded: {train_size} train, {val_size} val, {test_size} test")
    return train_loader, val_loader, test_loader
