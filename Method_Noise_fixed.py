import os
import socket
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Tuple, Optional
import glob
import joblib
from torch.cuda.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_free_port():
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup_distributed(rank: int, world_size: int):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(find_free_port())
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_distributed():
    """Cleanup distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_tensor_data(data_dir: str, device: str = 'cpu', rank: int = 0, world_size: int = 1, max_files_per_class: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load tensor data from .pt files, keeping data on CPU"""
    classes = ['real', 'semi-synthetic', 'synthetic']
    all_tensors, all_labels = [], []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        tensor_files = glob.glob(os.path.join(class_dir, '*.pt'))
        if max_files_per_class:
            tensor_files = tensor_files[:max_files_per_class]
        
        if not tensor_files:
            logger.warning(f"Rank {rank}: No tensor files found for class {class_name}")
            continue
        
        for file in tensor_files:
            try:
                tensor = torch.load(file, map_location='cpu')  # Load to CPU
                all_tensors.append(tensor)
                all_labels.append(torch.full((tensor.size(0),), class_idx, dtype=torch.long))
            except Exception as e:
                logger.error(f"Rank {rank}: Error loading {file}: {str(e)}")
                continue
        
        logger.info(f"Rank {rank}: Loaded {len(tensor_files)} files for class {class_name}")
    
    if not all_tensors:
        raise ValueError(f"Rank {rank}: No valid tensors loaded")
    
    final_tensors = torch.cat(all_tensors, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    logger.info(f"Rank {rank}: Loaded {final_tensors.size(0)} samples with shape {final_tensors.shape}")
    
    # Shard data for distributed training
    if world_size > 1:
        samples_per_rank = len(final_tensors) // world_size
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if rank < world_size - 1 else len(final_tensors)
        final_tensors = final_tensors[start_idx:end_idx]
        final_labels = final_labels[start_idx:end_idx]
    
    return final_tensors, final_labels

def split_data(tensors: torch.Tensor, labels: torch.Tensor, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[torch.Tensor, ...]:
    """Split data into train, validation, and test sets"""
    n_samples = len(tensors)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return (tensors[train_indices], tensors[val_indices], tensors[test_indices],
            labels[train_indices], labels[val_indices], labels[test_indices])

class CBAM(nn.Module):
    """Placeholder for Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified CBAM for demonstration
        channel_att = self.channel_attention(x) * x
        spatial_att = self.spatial_attention(torch.cat([torch.mean(channel_att, dim=1, keepdim=True),
                                                      torch.max(channel_att, dim=1, keepdim=True)[0]], dim=1))
        return channel_att * spatial_att

class EnhancedDenosingAutoencoder(nn.Module):
    """Simplified Enhanced Denoising Autoencoder"""
    def __init__(self, use_attention: bool = False):
        super(EnhancedDenosingAutoencoder, self).__init__()
        self.use_attention = use_attention
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.attention = CBAM(128) if use_attention else nn.Identity()
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        if self.use_attention:
            encoded = self.attention(encoded)
        decoded = self.decoder(encoded)
        return decoded

class NoiseClassificationPipeline:
    def __init__(self, rank: int, world_size: int, device: torch.device, use_ensemble: bool = False, num_models: int = 1, use_enhanced_autoencoder: bool = True):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.use_ensemble = use_ensemble
        self.num_models = num_models
        self.use_amp = device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize autoencoder
        self.autoencoder = EnhancedDenosingAutoencoder(use_attention=False).to(self.device)
        self.autoencoder = torch.compile(self.autoencoder) if torch.__version__ >= '2.0.0' else self.autoencoder
        logger.info(f"Rank {self.rank}: Autoencoder initialized on {self.device}")
        
        # Memory usage logging
        if self.device.type == 'cuda':
            mem_info = torch.cuda.memory_reserved(self.device) / 1e9
            logger.info(f"Rank {self.rank}: Initial GPU memory used: {mem_info:.2f} GB")
    
    def train_autoencoder(self, train_tensors: torch.Tensor, val_tensors: torch.Tensor = None, 
                         epochs: int = 2, batch_size: int = 4, lr: float = 0.001, patience: int = 10,
                         resume_from_checkpoint: bool = True, early_stopping_config: dict = None,
                         train_labels: torch.Tensor = None, save_classifier_every_epoch: bool = True,
                         classifier_save_frequency: int = 1, accum_steps: int = 4) -> dict:
        """Train the denoising autoencoder with gradient accumulation"""
        mse_criterion = nn.MSELoss()
        l1_criterion = nn.L1Loss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        
        # Create DataLoader
        train_dataset = TensorDataset(train_tensors)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        best_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(epochs):
            self.autoencoder.train()
            train_loss = 0.0
            train_pbar = train_loader
            
            for batch_idx, data in enumerate(train_pbar):
                try:
                    data = data[0] if isinstance(data, (list, tuple)) else data
                    data = data.to(self.device, non_blocking=True)
                    if data.max() > 1.0:
                        data = data / 255.0
                    
                    optimizer.zero_grad(set_to_none=True)
                    for i in range(accum_steps):
                        sub_batch_size = batch_size // accum_steps
                        start_idx = i * sub_batch_size
                        end_idx = start_idx + sub_batch_size
                        if start_idx >= len(data):
                            break
                        sub_batch = data[start_idx:end_idx]
                        if len(sub_batch) == 0:
                            continue
                        
                        # Add noise
                        noise_type = np.random.choice(['gaussian', 'uniform', 'salt_pepper'])
                        if noise_type == 'gaussian':
                            noise = torch.randn_like(sub_batch, device=self.device) * 0.05
                        elif noise_type == 'uniform':
                            noise = (torch.rand_like(sub_batch, device=self.device) - 0.5) * 0.1
                        else:
                            noise = torch.zeros_like(sub_batch, device=self.device)
                            mask = torch.rand_like(sub_batch, device=self.device) < 0.02
                            noise[mask] = (torch.rand_like(sub_batch, device=self.device)[mask] - 0.5) * 1.0
                        
                        noisy_data = torch.clamp(sub_batch + noise, 0, 1)
                        
                        if self.use_amp and self.scaler is not None:
                            with autocast():
                                reconstructed = self.autoencoder(noisy_data)
                                mse_loss = mse_criterion(reconstructed, sub_batch)
                                l1_loss = l1_criterion(reconstructed, sub_batch)
                                loss = (mse_loss + 0.1 * l1_loss) / accum_steps
                            self.scaler.scale(loss).backward()
                        else:
                            reconstructed = self.autoencoder(noisy_data)
                            mse_loss = mse_criterion(reconstructed, sub_batch)
                            l1_loss = l1_criterion(reconstructed, sub_batch)
                            loss = (mse_loss + 0.1 * l1_loss) / accum_steps
                            loss.backward()
                    
                    if self.use_amp and self.scaler is not None:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    
                    train_loss += loss.item() * accum_steps
                    if self.device.type == 'cuda':
                        mem_info = torch.cuda.memory_reserved(self.device) / 1e9
                        logger.debug(f"Rank {self.rank}: Batch {batch_idx}, GPU memory: {mem_info:.2f} GB")
                
                except RuntimeError as e:
                    logger.error(f"Rank {self.rank}: Batch {batch_idx} failed: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
            
            train_loss /= len(train_loader)
            logger.info(f"Rank {self.rank}: Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
            
            # Checkpointing
            if save_classifier_every_epoch and (epoch + 1) % classifier_save_frequency == 0:
                if not self.distributed or self.rank == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.autoencoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss
                    }
                    joblib.dump(checkpoint, f'checkpoint_epoch_{epoch+1}.pkl')
                    logger.info(f"Rank {self.rank}: Saved checkpoint for epoch {epoch+1}")
            
            # Early stopping (simplified)
            if val_tensors is not None:
                val_loss = self.validate(val_tensors, batch_size)
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        logger.info(f"Rank {self.rank}: Early stopping triggered")
                        break
        
        return {'train_loss': train_loss, 'best_val_loss': best_loss}
    
    def validate(self, val_tensors: torch.Tensor, batch_size: int) -> float:
        """Validate the autoencoder"""
        self.autoencoder.eval()
        mse_criterion = nn.MSELoss()
        val_dataset = TensorDataset(val_tensors)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        val_loss = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                data = data[0] if isinstance(data, (list, tuple)) else data
                data = data.to(self.device, non_blocking=True)
                if data.max() > 1.0:
                    data = data / 255.0
                
                noise = torch.randn_like(data, device=self.device) * 0.05
                noisy_data = torch.clamp(data + noise, 0, 1)
                
                with autocast() if self.use_amp else torch.no_grad():
                    reconstructed = self.autoencoder(noisy_data)
                    loss = mse_criterion(reconstructed, data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Rank {self.rank}: Validation Loss: {val_loss:.4f}")
        return val_loss
    
    def extract_noise_features_batch(self, data: torch.Tensor) -> torch.Tensor:
        """Extract noise features (placeholder)"""
        self.autoencoder.eval()
        with torch.no_grad():
            data = data.to(self.device, non_blocking=True)
            if data.max() > 1.0:
                data = data / 255.0
            with autocast() if self.use_amp else torch.no_grad():
                reconstructed = self.autoencoder(data)
            noise_features = data - reconstructed
        return noise_features
    
    def train(self, train_tensors: torch.Tensor, train_labels: torch.Tensor, val_tensors: torch.Tensor,
              val_labels: torch.Tensor, test_tensors: torch.Tensor, test_labels: torch.Tensor,
              autoencoder_epochs: int = 2, batch_size: int = 4, resume_from_checkpoint: bool = True,
              retrain_classifier: bool = True, use_adversarial: bool = False, enable_explanations: bool = True,
              accum_steps: int = 4) -> dict:
        """Main training function"""
        results = self.train_autoencoder(
            train_tensors, val_tensors, epochs=autoencoder_epochs, batch_size=batch_size,
            resume_from_checkpoint=resume_from_checkpoint, accum_steps=accum_steps
        )
        # Placeholder for classifier training
        logger.info(f"Rank {self.rank}: Autoencoder training completed, skipping classifier for now")
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.info(f"Rank {self.rank}: Cleanup completed")

def analyze_noise_patterns(pipeline: NoiseClassificationPipeline, test_tensors: torch.Tensor, test_labels: torch.Tensor):
    """Analyze noise patterns (placeholder)"""
    logger.info(f"Rank {pipeline.rank}: Analyzing noise patterns")
    # Implement noise pattern analysis as needed

def run_complete_pipeline(rank: int, world_size: int, data_dir: str = 'datasets/train'):
    """Run the complete noise distribution classification pipeline"""
    try:
        if world_size > 1:
            try:
                setup_distributed(rank, world_size)
                distributed_mode = True
            except Exception as e:
                logger.warning(f"Rank {rank}: Distributed setup failed: {str(e)}")
                logger.warning(f"Rank {rank}: Falling back to single GPU mode")
                distributed_mode = False
                world_size = 1
                rank = 0
        else:
            distributed_mode = False
        
        device = torch.device(f'cuda:{rank}' if distributed_mode else 'cuda:0')
        tensors, labels = load_tensor_data(data_dir, device='cpu', rank=rank, world_size=world_size, max_files_per_class=2)
        train_tensors, val_tensors, test_tensors, train_labels, val_labels, test_labels = split_data(tensors, labels)
        
        pipeline = NoiseClassificationPipeline(
            rank=rank, world_size=world_size, device=device,
            use_ensemble=False, num_models=1, use_enhanced_autoencoder=True
        )
        
        results = pipeline.train(
            train_tensors, train_labels, val_tensors, val_labels, test_tensors, test_labels,
            autoencoder_epochs=2, batch_size=4, resume_from_checkpoint=True,
            retrain_classifier=True, use_adversarial=False, enable_explanations=True,
            accum_steps=4
        )
        
        if not distributed_mode or rank == 0:
            analyze_noise_patterns(pipeline, test_tensors, test_labels)
        
        pipeline.cleanup()
        return results
    
    except Exception as e:
        logger.error(f"Rank {rank}: Error in pipeline: {str(e)}")
        raise
    finally:
        if distributed_mode:
            cleanup_distributed()

if __name__ == '__main__':
    # For testing, use single GPU
    world_size = 1
    run_complete_pipeline(0, world_size)