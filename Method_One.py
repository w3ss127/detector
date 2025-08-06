import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from transformers import ViTModel
from sklearn.metrics import matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob
import gc
from tqdm import tqdm
import socket
import signal
import warnings
import logging
import time
import json

warnings.filterwarnings('ignore')

# CRITICAL FIX: Set multiprocessing start method to 'spawn' for CUDA compatibility
def set_multiprocessing_start_method():
    """Set the multiprocessing start method to 'spawn' for CUDA compatibility."""
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError as e:
        if "context has already been set" in str(e):
            print("Multiprocessing context already set")
        else:
            raise e

def optimize_rtx4090_settings():
    """Optimize settings specifically for RTX 4090 × 8 setup."""
    if torch.cuda.is_available():
        # Set memory fraction for each GPU to prevent OOM
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.9, i)  # Use 90% of VRAM
        
        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        
        print(f"RTX 4090 optimizations applied for {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB")

# Custom formatter for logging with rank
class RankFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'rank'):
            record.rank = 0
        return super().format(record)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = RankFormatter('%(asctime)s - %(levelname)s - [Rank %(rank)d] - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

# Set random seed for reproducibility
def set_seeds(rank=0):
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + rank)
        torch.cuda.manual_seed_all(42 + rank)

# Memory management
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_gpu_memory_info(device_id=0):
    """Get GPU memory information for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3    # GB
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
        return f"GPU {device_id}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total"
    return "No GPU available"

# Custom Dataset for loading .pt files (updated for distributed training)
class PtFileDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_classes=None, exclude_class=None, is_semi_synthetic=False):
        self.root_dir = root_dir
        self.transform = transform
        self.pt_files = []
        self.labels = []
        self.is_semi_synthetic = is_semi_synthetic

        # Define class_to_idx for training (real, synthetic)
        if not is_semi_synthetic:
            self.class_to_idx = {'real': 1, 'synthetic': 0}
            self.classes = ['real', 'synthetic']
        else:
            # For semi-synthetic, no labels needed (dummy label)
            self.class_to_idx = {'semi-synthetic': 0}
            self.classes = ['semi-synthetic']

        # Collect .pt files from specified classes
        for class_name in os.listdir(root_dir):
            if (include_classes is None or class_name in include_classes) and (exclude_class is None or class_name != exclude_class):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for pt_file in glob(os.path.join(class_dir, "*.pt")):
                        self.pt_files.append(pt_file)
                        self.labels.append(self.class_to_idx.get(class_name, 0))  # Use 0 for semi-synthetic

        # Each .pt file contains 5000 images
        self.images_per_file = 5000

    def __len__(self):
        return len(self.pt_files) * self.images_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.images_per_file
        img_idx = idx % self.images_per_file
        pt_file = self.pt_files[file_idx]
        label = self.labels[file_idx]

        try:
            # Load the .pt file
            data = torch.load(pt_file, map_location='cpu')
            img = data[img_idx]  # Shape: (C, H, W), e.g., (3, 224, 224)

            # Apply transforms if provided
            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            logger.warning(f"Error loading image at index {idx}: {e}", extra={'rank': 0})
            # Return a dummy tensor in case of error
            dummy_img = torch.zeros(3, 224, 224)
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, label

# Define custom transform classes to avoid pickling issues
class FloatNormalize:
    """Convert tensor to float and normalize to [0, 1] if needed"""
    def __call__(self, x):
        if x.max() > 1.0:
            return x.float() / 255.0
        return x.float()

# Define data transforms (for tensors)
def get_transforms():
    train_transform = transforms.Compose([
        FloatNormalize(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        FloatNormalize(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

# Define the model with memory optimizations
class ConvNeXtViTAttention(nn.Module):
    def __init__(self):
        super(ConvNeXtViTAttention, self).__init__()
        # Load pre-trained ConvNeXt
        self.convnext = torch.hub.load('pytorch/vision', 'convnext_base', weights='IMAGENET1K_V1')
        self.convnext.classifier[2] = nn.Identity()  # Remove classification head
        self.convnext_features = 1024  # ConvNeXt-base output dimension

        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_features = 768  # ViT-base output dimension

        # Project ViT features to match ConvNeXt dimension
        self.vit_projection = nn.Linear(self.vit_features, self.convnext_features)

        # Attention module to fuse ConvNeXt and ViT features
        self.attention = nn.MultiheadAttention(embed_dim=self.convnext_features, num_heads=8)
        self.fc1 = nn.Linear(self.convnext_features + self.convnext_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)  # Binary classification output (raw logits)

    def forward(self, x):
        # ConvNeXt forward
        convnext_out = self.convnext(x)  # Shape: (batch, 1024)
        convnext_out = convnext_out.unsqueeze(0)  # Shape: (1, batch, 1024) for attention

        # ViT forward
        vit_out = self.vit(pixel_values=x).last_hidden_state[:, 0, :]  # CLS token, shape: (batch, 768)
        vit_out = self.vit_projection(vit_out)  # Project to 1024, shape: (batch, 1024)
        vit_out = vit_out.unsqueeze(0)  # Shape: (1, batch, 1024)

        # Attention: use ConvNeXt features as query/key, ViT as value
        attn_output, _ = self.attention(convnext_out, convnext_out, vit_out)  # Shape: (1, batch, 1024)
        attn_output = attn_output.squeeze(0)  # Shape: (batch, 1024)

        # Concatenate ConvNeXt and attention-enhanced features
        combined = torch.cat((convnext_out.squeeze(0), attn_output), dim=1)  # Shape: (batch, 1024+1024)

        # Fully connected layers
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # No sigmoid - BCEWithLogitsLoss applies it internally
        return x

def find_free_port(start_port=12355, max_attempts=100):
    """Find a free port for distributed training."""
    port = start_port
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                s.listen(1)
                return port
        except OSError:
            port += 1
            continue
    raise RuntimeError(f"No free port found after {max_attempts} attempts")

def setup_distributed(local_rank, world_size, backend='nccl', master_addr='localhost', master_port='12355'):
    """Setup distributed training with robust NCCL configuration."""
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Set environment variables for NCCL
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['RANK'] = str(local_rank)
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            
            # NCCL configuration for better stability
            os.environ['NCCL_DEBUG'] = 'INFO'
            os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes timeout
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
            os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
            os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P
            os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback interface
            os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Use blocking wait
            os.environ['NCCL_SOCKET_NTHREADS'] = '4'  # Number of threads for socket
            os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'  # Number of sockets per thread
            
            # Set device first
            torch.cuda.set_device(local_rank)
            
            # Initialize process group
            dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)
            
            logger.info(f"Successfully initialized distributed training for rank {local_rank} (attempt {attempt + 1})", extra={'rank': local_rank})
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for rank {local_rank}: {e}", extra={'rank': local_rank})
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...", extra={'rank': local_rank})
                time.sleep(retry_delay)
                # Clean up any partial initialization
                try:
                    dist.destroy_process_group()
                except:
                    pass
            else:
                logger.error(f"Failed to setup distributed training for rank {local_rank} after {max_retries} attempts", extra={'rank': local_rank})
                return False
    
    return False

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_mcc, best_val_mcc, 
                   train_losses, val_mccs, local_rank, checkpoint_dir='binary_checkpoints'):
    """Save model checkpoint with training state."""
    if local_rank not in [-1, 0]:  # Only save on rank 0 or single GPU
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get the actual model (unwrap from DDP if needed)
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_mcc': val_mcc,
        'best_val_mcc': best_val_mcc,
        'train_losses': train_losses,
        'val_mccs': val_mccs,
        'local_rank': local_rank
    }
    
    # Save epoch checkpoint
    epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, epoch_path)
    logger.info(f'Checkpoint saved: {epoch_path}', extra={'rank': local_rank})
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    logger.info(f'Latest checkpoint updated: {latest_path}', extra={'rank': local_rank})
    
    # Save best model if this is the best validation MCC
    if val_mcc > best_val_mcc:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        logger.info(f'Best checkpoint saved: {best_path}', extra={'rank': local_rank})

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, local_rank):
    """Load model checkpoint and return training state."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f'Checkpoint not found: {checkpoint_path}', extra={'rank': local_rank})
        return None, 0, [], [], 0.0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}' if local_rank >= 0 else 'cpu')
        
        # Get the actual model (unwrap from DDP if needed)
        model_to_load = model.module if hasattr(model, 'module') else model
        
        # Load model state
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Extract training state
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        val_mccs = checkpoint.get('val_mccs', [])
        best_val_mcc = checkpoint.get('best_val_mcc', 0.0)
        
        logger.info(f'Checkpoint loaded from {checkpoint_path}', extra={'rank': local_rank})
        logger.info(f'Resuming from epoch {start_epoch + 1}', extra={'rank': local_rank})
        logger.info(f'Previous best validation MCC: {best_val_mcc:.4f}', extra={'rank': local_rank})
        
        return checkpoint, start_epoch, train_losses, val_mccs, best_val_mcc
        
    except Exception as e:
        logger.error(f'Error loading checkpoint {checkpoint_path}: {e}', extra={'rank': local_rank})
        return None, 0, [], [], 0.0

def save_training_history(train_losses, val_mccs, checkpoint_dir='binary_checkpoints'):
    """Save training history to JSON file."""
    history = {
        'train_losses': train_losses,
        'val_mccs': val_mccs,
        'epochs': list(range(1, len(train_losses) + 1))
    }
    
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f'Training history saved: {history_path}', extra={'rank': 0})

def plot_training_history(train_losses, val_mccs, checkpoint_dir='binary_checkpoints'):
    """Plot and save training history."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation MCC
    ax2.plot(epochs, val_mccs, 'r-', label='Validation MCC')
    ax2.set_title('Validation MCC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MCC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f'Training history plot saved: {plot_path}', extra={'rank': 0})

def cleanup_distributed():
    """Cleanup distributed training."""
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed", extra={'rank': 0})
    except Exception as e:
        logger.warning(f"Error during distributed cleanup: {e}", extra={'rank': 0})
    clear_gpu_memory()

def wait_for_all_processes():
    """Wait for all processes to be ready before starting training."""
    if dist.is_initialized():
        dist.barrier()
        logger.info("All processes synchronized", extra={'rank': 0})

def signal_handler(signum, frame):
    """Handle signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...", extra={'rank': 0})
    cleanup_distributed()
    exit(0)

def create_data_loaders(datasets_dir, train_transform, val_test_transform, batch_size, local_rank=-1, world_size=1):
    """Create distributed data loaders."""
    # Create datasets
    train_dataset = PtFileDataset(root_dir=datasets_dir, transform=train_transform, include_classes=['real', 'synthetic'])
    semi_synthetic_dataset = PtFileDataset(root_dir=datasets_dir, transform=val_test_transform, include_classes=['semi-synthetic'], is_semi_synthetic=True)
    
    if local_rank == 0 or local_rank == -1:
        logger.info(f"Train dataset size: {len(train_dataset)}", extra={'rank': local_rank})
        logger.info(f"Semi-synthetic dataset size: {len(semi_synthetic_dataset)}", extra={'rank': local_rank})
    
    # Split train_dataset into train/val/test (0.7/0.15/0.15)
    indices = list(range(len(train_dataset)))
    train_indices, temp_indices = train_test_split(
        indices, train_size=0.7, 
        stratify=[train_dataset.labels[i // train_dataset.images_per_file] for i in indices], 
        random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=0.5, 
        stratify=[train_dataset.labels[i // train_dataset.images_per_file] for i in temp_indices], 
        random_state=42
    )

    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    test_subset = torch.utils.data.Subset(train_dataset, test_indices)

    # Create distributed samplers if using distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(train_subset, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_subset, rank=local_rank, shuffle=False)
        test_sampler = DistributedSampler(test_subset, rank=local_rank, shuffle=False)
        semi_sampler = DistributedSampler(semi_synthetic_dataset, rank=local_rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = val_sampler = test_sampler = semi_sampler = None
        shuffle_train = True

    # Optimize num_workers for RTX 4090 × 8 setup
    # RTX 4090 has 24GB VRAM, so we can use more workers for better data loading
    if world_size > 1:
        # For distributed training, use more workers per GPU
        num_workers = 8  # Increased from 0 to 8 for better data loading
    else:
        num_workers = 16  # Single GPU can use more workers

    # Create data loaders with optimizations for RTX 4090
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, 
                             shuffle=shuffle_train, num_workers=num_workers, pin_memory=True, 
                             drop_last=True, persistent_workers=True, prefetch_factor=2)
    
    # Use smaller batch size for validation to prevent memory issues
    val_batch_size = min(batch_size, 16)  # Cap validation batch size
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, sampler=val_sampler,
                           shuffle=False, num_workers=num_workers, pin_memory=True,
                           persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_subset, batch_size=val_batch_size, sampler=test_sampler,
                            shuffle=False, num_workers=num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)
    semi_synthetic_loader = DataLoader(semi_synthetic_dataset, batch_size=val_batch_size, sampler=semi_sampler,
                                      shuffle=False, num_workers=num_workers, pin_memory=True,
                                      persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader, test_loader, semi_synthetic_loader, train_sampler

def evaluate_model(model, data_loader, criterion, device, local_rank=0):
    """Evaluate model on given data loader."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Log memory before evaluation
    if local_rank in [-1, 0]:
        logger.info(f"Starting evaluation. {get_gpu_memory_info()}", extra={'rank': local_rank})
    
    try:
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating", leave=False, disable=local_rank not in [-1, 0]):
                try:
                    # Ensure data is on the correct device and has correct shape
                    if data.dim() != 4:
                        logger.warning(f"Invalid data shape: {data.shape}, skipping batch", extra={'rank': local_rank})
                        continue
                    
                    data = data.to(device, non_blocking=True)
                    target = target.float().to(device, non_blocking=True)
                    
                    # Ensure target has correct shape
                    if target.dim() == 0:
                        target = target.unsqueeze(0)
                    
                    with autocast():
                        outputs = model(data)
                        # Handle different output shapes
                        if outputs.dim() > 1:
                            outputs = outputs.squeeze()
                        loss = criterion(outputs, target)
                    
                    total_loss += loss.item()
                    
                    # Apply sigmoid for BCEWithLogitsLoss and convert to predictions
                    pred = (torch.sigmoid(outputs) > 0.5).float()
                    
                    # Convert to numpy safely
                    all_predictions.extend(pred.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy().flatten())
                    
                    # Clear memory
                    del data, target, outputs, pred
                    clear_gpu_memory()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"GPU OOM during evaluation. Skipping batch.", extra={'rank': local_rank})
                        clear_gpu_memory()
                        continue
                    else:
                        logger.error(f"Runtime error during evaluation: {e}", extra={'rank': local_rank})
                        raise e
                except Exception as e:
                    logger.error(f"Unexpected error during evaluation: {e}", extra={'rank': local_rank})
                    continue
        
        # Calculate metrics safely
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        
        # Ensure we have predictions to calculate MCC
        if len(all_predictions) > 0 and len(all_targets) > 0:
            try:
                mcc = matthews_corrcoef(all_targets, all_predictions)
            except Exception as e:
                logger.warning(f"Error calculating MCC: {e}, using 0.0", extra={'rank': local_rank})
                mcc = 0.0
        else:
            logger.warning("No predictions generated, MCC set to 0.0", extra={'rank': local_rank})
            mcc = 0.0
        
        return avg_loss, mcc, all_predictions, all_targets
        
    except Exception as e:
        logger.error(f"Critical error in evaluation: {e}", extra={'rank': local_rank})
        return 0.0, 0.0, [], []

def train_worker(local_rank, world_size, datasets_dir, batch_size, num_epochs, master_port):
    """Main training worker function for distributed training."""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Setup distributed training
        if world_size > 1:
            success = setup_distributed(local_rank, world_size, master_port=master_port)
            if not success:
                logger.error(f"Failed to setup distributed training for rank {local_rank}", extra={'rank': local_rank})
                return
        
        # Wait for all processes to be ready
        if world_size > 1:
            wait_for_all_processes()
        
        # Set device
        device = torch.device(f'cuda:{local_rank}' if world_size > 1 else 'cuda:0')
        
        # Set seeds
        set_seeds(local_rank)
        
        # Get transforms
        train_transform, val_test_transform = get_transforms()
        
        # Create data loaders
        try:
            train_loader, val_loader, test_loader, semi_synthetic_loader, train_sampler = create_data_loaders(
                datasets_dir, train_transform, val_test_transform, batch_size, local_rank, world_size
            )
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}", extra={'rank': local_rank})
            if world_size > 1:
                cleanup_distributed()
            return

        # Initialize model, loss, and optimizer
        if local_rank == 0 or local_rank == -1:
            logger.info("Initializing model...", extra={'rank': local_rank})
        
        model = ConvNeXtViTAttention().to(device)
        
        # Wrap model with DDP for distributed training
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        scaler = GradScaler()

        # Try to load checkpoint if exists
        start_epoch = 0
        train_losses, val_mccs = [], []
        best_val_mcc = -1.0
        
        # Check for existing checkpoints
        checkpoint_paths = [
            'binary_checkpoints/checkpoint_latest.pth',
            'binary_checkpoints/checkpoint_best.pth'
        ]
        
        loaded_checkpoint = None
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                loaded_checkpoint, start_epoch, train_losses, val_mccs, best_val_mcc = load_checkpoint(
                    model, optimizer, scheduler, checkpoint_path, local_rank
                )
                if loaded_checkpoint is not None:
                    logger.info(f"Resuming training from epoch {start_epoch + 1}", extra={'rank': local_rank})
                    break
        
        if local_rank == 0 or local_rank == -1:
            if loaded_checkpoint is None:
                logger.info("Starting new training...", extra={'rank': local_rank})
            else:
                logger.info(f"Resuming training from epoch {start_epoch + 1}", extra={'rank': local_rank})
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            running_loss = 0.0
            batch_count = 0
            
            # Training progress bar (only show on rank 0)
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                             leave=False, disable=local_rank not in [-1, 0])
            
            for batch_idx, (images, labels) in enumerate(train_pbar):
                try:
                    images, labels = images.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    with autocast():
                        outputs = model(images).squeeze()
                        # Use raw logits with BCEWithLogitsLoss
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    running_loss += loss.item() * images.size(0)
                    batch_count += 1
                    
                    # Update progress bar with loss
                    if local_rank == 0 or local_rank == -1:
                        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                    
                    # Clear memory after each batch
                    del images, labels, outputs, loss
                    clear_gpu_memory()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"GPU OOM in batch {batch_idx}. Skipping batch.", extra={'rank': local_rank})
                        clear_gpu_memory()
                        continue
                    else:
                        raise e
            
            # Calculate epoch loss
            if world_size > 1:
                # Gather losses from all processes
                total_loss_tensor = torch.tensor(running_loss, device=device)
                total_batches_tensor = torch.tensor(batch_count, device=device)
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)
                epoch_loss = total_loss_tensor.item() / (len(train_loader.dataset) * world_size) if total_batches_tensor.item() > 0 else 0
            else:
                epoch_loss = running_loss / len(train_loader.dataset) if batch_count > 0 else 0
            
            train_losses.append(epoch_loss)

            # Evaluate on validation set (only on rank 0 to avoid duplication)
            val_loss, val_mcc = 0.0, 0.0  # Initialize for all ranks
            if local_rank == 0 or local_rank == -1:
                try:
                    val_loss, val_mcc, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device, local_rank)
                    val_mccs.append(val_mcc)
                    
                    # Save best model
                    if val_mcc > best_val_mcc:
                        best_val_mcc = val_mcc
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), 'best_binary_classifier.pth')
                        logger.info(f'New best model saved with Val MCC: {val_mcc:.4f}', extra={'rank': local_rank})
                    
                    logger.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}', 
                               extra={'rank': local_rank})
                except Exception as e:
                    logger.error(f"Error during validation: {e}", extra={'rank': local_rank})
                    val_loss, val_mcc = 0.0, 0.0
                    if len(val_mccs) > 0:
                        val_mccs.append(val_mccs[-1])  # Use previous MCC
                    else:
                        val_mccs.append(0.0)
            
            # Save checkpoint after every epoch (only on rank 0 to avoid conflicts)
            if local_rank in [-1, 0]:
                save_checkpoint(model, optimizer, scheduler, epoch, epoch_loss, val_mcc, best_val_mcc, 
                              train_losses, val_mccs, local_rank)
            
            scheduler.step()
            
            # Synchronize all processes
            if world_size > 1:
                dist.barrier()

        # Final evaluation (only on rank 0)
        if local_rank == 0 or local_rank == -1:
            logger.info("Evaluating on test set...", extra={'rank': local_rank})
            
            # Load best model for final evaluation
            if os.path.exists('best_binary_classifier.pth'):
                model_to_load = model.module if hasattr(model, 'module') else model
                model_to_load.load_state_dict(torch.load('best_binary_classifier.pth'))
            
            test_loss, test_mcc, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device, local_rank)
            logger.info(f'Test MCC: {test_mcc:.4f}', extra={'rank': local_rank})

            # Evaluate on semi-synthetic images
            logger.info("Evaluating on semi-synthetic images...", extra={'rank': local_rank})
            semi_synthetic_outputs = []
            
            model.eval()
            with torch.no_grad():
                for images, _ in tqdm(semi_synthetic_loader, desc='[Semi-Synthetic Evaluation]', leave=False):
                    try:
                        images = images.to(device, non_blocking=True)
                        with autocast():
                            outputs = model(images).squeeze()
                        semi_synthetic_outputs.extend(outputs.cpu().numpy())
                        
                        # Clear memory
                        del images, outputs
                        clear_gpu_memory()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning("GPU OOM during semi-synthetic evaluation. Skipping batch.", extra={'rank': local_rank})
                            clear_gpu_memory()
                            continue
                        else:
                            raise e

            # Analyze semi-synthetic outputs
            if semi_synthetic_outputs:
                mean_output = np.mean(semi_synthetic_outputs)
                std_output = np.std(semi_synthetic_outputs)
                logger.info(f'Semi-synthetic outputs: Mean = {mean_output:.4f}, Std = {std_output:.4f}', extra={'rank': local_rank})

                # Plot semi-synthetic output distribution
                plt.figure(figsize=(10, 6))
                plt.hist(semi_synthetic_outputs, bins=20, range=(0, 1), density=True)
                plt.title('Distribution of Semi-Synthetic Image Outputs')
                plt.xlabel('Model Output')
                plt.ylabel('Density')
                plt.savefig('semi_synthetic_distribution.png')
                plt.close()
                logger.info("Distribution plot saved as 'semi_synthetic_distribution.png'", extra={'rank': local_rank})

            # Save the final model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), 'final_binary_classifier.pth')
            
            # Save training history and plots
            save_training_history(train_losses, val_mccs)
            plot_training_history(train_losses, val_mccs)
            
            # Print training summary
            logger.info("\n" + "="*50, extra={'rank': local_rank})
            logger.info("TRAINING SUMMARY", extra={'rank': local_rank})
            logger.info("="*50, extra={'rank': local_rank})
            logger.info(f"Final Training Loss: {train_losses[-1]:.4f}", extra={'rank': local_rank})
            logger.info(f"Best Validation MCC: {max(val_mccs):.4f}", extra={'rank': local_rank})
            logger.info(f"Final Test MCC: {test_mcc:.4f}", extra={'rank': local_rank})
            logger.info(f"Model saved as 'final_binary_classifier.pth'", extra={'rank': local_rank})
            logger.info("Training completed successfully!", extra={'rank': local_rank})
            logger.info("="*50, extra={'rank': local_rank})

        # Cleanup
        if world_size > 1:
            cleanup_distributed()
        clear_gpu_memory()
        
    except Exception as e:
        logger.error(f"Error in train_worker for rank {local_rank}: {e}", extra={'rank': local_rank})
        if world_size > 1:
            cleanup_distributed()
        raise

def main():
    """Main function to setup and start distributed training."""
    # CRITICAL FIX: Set multiprocessing start method before any CUDA operations
    set_multiprocessing_start_method()
    
    # Optimize for RTX 4090 × 8 setup
    optimize_rtx4090_settings()
    
    # Environment setup optimized for RTX 4090 × 8
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    
    # RTX 4090 optimizations
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async CUDA operations
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Better memory management
    
    # Configuration optimized for RTX 4090 × 8
    datasets_dir = 'datasets/train'
    batch_size = 32  # Increased for RTX 4090 (24GB VRAM per GPU)
    num_epochs = 20
    
    # Check for GPU availability and setup distributed training
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Found {world_size} GPU(s)")
        
        if world_size > 1:
            # Use a fixed port for all processes
            master_port = "12355"
            print(f"Starting distributed training with {world_size} GPUs on port {master_port}")
            
            # Try distributed training first
            try:
                mp.spawn(
                    train_worker,
                    args=(world_size, datasets_dir, batch_size, num_epochs, master_port),
                    nprocs=world_size,
                    join=True
                )
            except Exception as e:
                print(f"Distributed training failed: {e}")
                print("Falling back to single GPU training...")
                # Fallback to single GPU training
                train_worker(-1, 1, datasets_dir, batch_size, num_epochs, "12355")
        else:
            print("Single GPU training")
            train_worker(-1, 1, datasets_dir, batch_size, num_epochs, "12355")
    else:
        print("No GPU available, using CPU")
        train_worker(-1, 1, datasets_dir, batch_size, num_epochs, "12355")

if __name__ == "__main__":
    main()