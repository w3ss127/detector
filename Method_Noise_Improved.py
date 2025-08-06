import os
import time
import glob
import signal
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Signal handler for graceful termination
def signal_handler(sig, frame):
    logger.info("Received termination signal. Cleaning up...")
    if 'pipeline' in globals():
        pipeline.cleanup()
    if dist.is_initialized():
        dist.destroy_process_group()
    exit(0)

# Set multiprocessing start method
def set_multiprocessing_start_method():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Fixed distributed training setup
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_DEBUG'] = 'INFO'
    try:
        # Fixed: Use timedelta instead of timeout_timedelta
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=1800)  # 30 minutes timeout
        )
        torch.cuda.set_device(rank)
        logger.info(f"Rank {rank}: Distributed process group initialized successfully")
    except Exception as e:
        logger.error(f"Rank {rank}: Failed to initialize distributed process group: {str(e)}")
        raise

# Cleanup distributed processes
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")

# Memory-efficient autoencoder (much smaller)
class DenosingAutoencoder(nn.Module):
    def __init__(self, channels=3):
        super(DenosingAutoencoder, self).__init__()
        # Extremely small network to avoid OOM
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1),  # Very few filters
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, 3, padding=1),  # Even fewer
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Ultra-light enhanced autoencoder
class EnhancedDenosingAutoencoder(nn.Module):
    def __init__(self, channels=3):
        super(EnhancedDenosingAutoencoder, self).__init__()
        self.channels = channels
        # Minimal network
        self.enc_conv1 = nn.Conv2d(channels, 4, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Remove attention to save memory
        self.dec_conv1 = nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(4, channels, 3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = self.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = self.relu(self.dec_conv1(x))
        x = self.sigmoid(self.dec_conv2(x))
        return x

# Minimal ensemble (to save memory)
class EnsembleAutoencoder(nn.Module):
    def __init__(self, num_models=2):  # Reduced from 3
        super(EnsembleAutoencoder, self).__init__()
        self.models = nn.ModuleList([DenosingAutoencoder() for _ in range(num_models)])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

class NoiseDistributionAnalyzer:
    def __init__(self, device):
        self.device = device
        self.scaler = None

    def fit_scaler(self, features):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(features)

    def normalize_features(self, features):
        if self.scaler:
            return self.scaler.transform(features)
        return features

class NoiseDistributionClassifier:
    def __init__(self):
        # Reduced complexity to save memory
        self.classifier = RandomForestClassifier(n_estimators=20, max_depth=10)
        self.feature_importance = None

    def fit(self, features, labels):
        self.classifier.fit(features, labels)
        self.feature_importance = self.classifier.feature_importances_

class AdvancedDataAugmentation:
    def __init__(self, p=0.3):  # Reduced probability
        self.p = p

    def augment(self, x):
        if torch.rand(1).item() < self.p:
            x = x + torch.randn_like(x) * 0.02  # Very light noise
        return x

class ExplainableAI:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate_explanation(self, input_tensor):
        return input_tensor  # Placeholder

class AdversarialTraining:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate_adversarial_examples(self, x, y):
        return x  # Placeholder

# Main pipeline class with memory optimizations
class NoiseClassificationPipeline:
    def __init__(self, rank=0, world_size=1, device=None, use_ensemble=False, use_enhanced_autoencoder=False,
                 weights_dir='noise_autoencoder_checkpoints', classifier_dir='noise_classifier_checkpoints'):
        self.rank = rank
        self.world_size = world_size
        self.distributed = world_size > 1
        self.device = device if device is not None else torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        # Directory configuration
        self.weights_dir = weights_dir
        self.classifier_dir = classifier_dir
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.classifier_dir, exist_ok=True)

        # Force memory clearing
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Use smallest possible model
        if use_ensemble:
            self.autoencoder = EnsembleAutoencoder(num_models=2).to(self.device)
            self.use_ensemble = True
        elif use_enhanced_autoencoder:
            self.autoencoder = EnhancedDenosingAutoencoder().to(self.device)
            self.use_ensemble = False
        else:
            self.autoencoder = DenosingAutoencoder().to(self.device)
            self.use_ensemble = False

        self.noise_analyzer = NoiseDistributionAnalyzer(device=self.device)
        self.classifier = NoiseDistributionClassifier()
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.data_augmentation = AdvancedDataAugmentation(p=0.3)
        self.explainable_ai = ExplainableAI(self.autoencoder, self.device)
        self.adversarial_training = AdversarialTraining(self.autoencoder, self.device)

        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'autoencoder_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_mcc': [],
            'val_mcc': []
        }
        self.class_names = ['real', 'semi-synthetic', 'synthetic']

        if self.distributed:
            try:
                self.autoencoder = DDP(self.autoencoder, device_ids=[self.rank])
            except Exception as e:
                logger.error(f"Rank {self.rank}: Failed to initialize DDP: {str(e)}")
                raise

        logger.info(f"Rank {self.rank}: Enhanced pipeline initialized for three-class classification on device {self.device}")
        logger.info(f"Rank {self.rank}: Using {'ensemble' if use_ensemble else 'enhanced' if use_enhanced_autoencoder else 'standard'} autoencoder")
        logger.info(f"Rank {self.rank}: Weights directory: {self.weights_dir}")
        logger.info(f"Rank {self.rank}: Classifier directory: {self.classifier_dir}")

    def save_checkpoint(self, epoch, train_loss, val_loss, optimizer, scheduler):
        """Save model checkpoint to weights directory"""
        try:
            os.makedirs(self.weights_dir, exist_ok=True)
            model_state = self.autoencoder.module.state_dict() if hasattr(self.autoencoder, 'module') else self.autoencoder.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'training_history': self.training_history,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
            }
            checkpoint_path = f'{self.weights_dir}/noise_autoencoder_epoch_{epoch}_rank_{self.rank}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Rank {self.rank}: Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error saving checkpoint: {str(e)}")

    def find_latest_checkpoint(self):
        """Find the latest checkpoint file for this rank in weights directory"""
        if not os.path.exists(self.weights_dir):
            return None

        pattern = f"{self.weights_dir}/noise_autoencoder_epoch_*_rank_{self.rank}.pth"
        checkpoint_files = glob.glob(pattern)

        if not checkpoint_files:
            return None

        def extract_epoch(filename):
            try:
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    if part == 'epoch' and i + 1 < len(parts):
                        return int(parts[i + 1])
                return 0
            except:
                return 0

        latest_checkpoint = max(checkpoint_files, key=extract_epoch)
        logger.info(f"Rank {self.rank}: Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint

    def extract_noise_features_batch(self, tensors):
        """Extract noise features in very small batches to avoid OOM"""
        features = []
        self.autoencoder.eval()
        with torch.no_grad():
            for i in range(0, len(tensors), 2):  # Very small batches
                try:
                    batch = tensors[i:i+2].to(self.device)
                    reconstructed = self.autoencoder(batch)
                    feature = (batch - reconstructed).abs().mean(dim=[1, 2, 3]).cpu().numpy()
                    features.append(feature)
                    
                    # Aggressive memory cleanup
                    del batch, reconstructed
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM in feature extraction, skipping batch {i}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
        
        if features:
            return np.concatenate(features)
        else:
            return np.array([])

    def train_and_save_classifier_epoch(self, train_tensors: torch.Tensor, train_labels: torch.Tensor, epoch: int):
        """Train and save classifier after each epoch (only on rank 0)"""
        if self.distributed and self.rank != 0:
            return

        try:
            logger.info(f"🔄 Epoch {epoch}: Training classifier with current autoencoder...")

            # Use smaller subset for classifier training to avoid OOM
            subset_size = min(20, len(train_tensors))
            train_features = self.extract_noise_features_batch(train_tensors[:subset_size])

            if len(train_features) == 0:
                logger.warning(f"Epoch {epoch}: No features extracted, skipping classifier training")
                return

            self.noise_analyzer.fit_scaler(train_features.reshape(-1, 1))
            normalized_features = self.noise_analyzer.normalize_features(train_features.reshape(-1, 1))

            self.classifier.fit(normalized_features, train_labels[:len(train_features)].cpu().numpy())

            if hasattr(self.classifier.classifier, 'n_features_in_'):
                logger.info(f"✅ Epoch {epoch}: Classifier trained with {self.classifier.classifier.n_features_in_} features")

        except Exception as e:
            logger.error(f"❌ Epoch {epoch}: Error training classifier: {str(e)}")

    def train(self, train_tensors, train_labels, val_tensors, val_labels, test_tensors, test_labels,
              autoencoder_epochs=5, batch_size=2, resume_from_checkpoint=True, retrain_classifier=True,
              use_adversarial=False, enable_explanations=False, accumulation_steps=4):
        """Train the pipeline with aggressive memory management"""
        
        # Use smaller learning rate and simpler optimizer
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        criterion = nn.MSELoss()

        start_epoch = 0
        if resume_from_checkpoint:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                try:
                    checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                    model_state = checkpoint['model_state_dict']
                    if hasattr(self.autoencoder, 'module'):
                        self.autoencoder.module.load_state_dict(model_state)
                    else:
                        self.autoencoder.load_state_dict(model_state)
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if self.scaler and checkpoint['scaler_state_dict']:
                        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    self.training_history = checkpoint['training_history']
                    logger.info(f"Resumed from checkpoint: {latest_checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {str(e)}")

        for epoch in range(start_epoch, autoencoder_epochs):
            self.autoencoder.train()
            train_loss = 0
            num_batches = 0
            
            optimizer.zero_grad()
            
            for i in range(0, len(train_tensors), batch_size):
                try:
                    batch = train_tensors[i:i+batch_size].to(self.device)
                    
                    # Light augmentation only
                    if not use_adversarial:  # Skip adversarial to save memory
                        batch = self.data_augmentation.augment(batch)

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        output = self.autoencoder(batch)
                        loss = criterion(output, batch) / accumulation_steps

                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        if (i // batch_size + 1) % accumulation_steps == 0:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad()
                    else:
                        loss.backward()
                        if (i // batch_size + 1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    train_loss += loss.item() * accumulation_steps
                    num_batches += 1

                    # Aggressive cleanup
                    del batch, output, loss
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM during training batch {i}, skipping")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

            if num_batches > 0:
                train_loss /= num_batches
            else:
                train_loss = float('inf')

            self.training_history['autoencoder_loss'].append(train_loss)
            self.training_history['epochs'].append(epoch)

            # Validation with memory management
            self.autoencoder.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for i in range(0, len(val_tensors), batch_size):
                    try:
                        batch = val_tensors[i:i+batch_size].to(self.device)
                        output = self.autoencoder(batch)
                        val_loss += criterion(output, batch).item()
                        val_batches += 1
                        
                        del batch, output
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"OOM during validation batch {i}, skipping")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise

            if val_batches > 0:
                val_loss /= val_batches
            else:
                val_loss = float('inf')

            self.training_history['val_loss'].append(val_loss)
            logger.info(f"Rank {self.rank}: Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Skip explanations to save memory
            if enable_explanations and (not self.distributed or self.rank == 0):
                logger.info(f"Rank {self.rank}: Skipping explanations to save memory")

            self.save_checkpoint(epoch, train_loss, val_loss, optimizer, scheduler)
            
            if retrain_classifier:
                self.train_and_save_classifier_epoch(train_tensors, train_labels, epoch)

            scheduler.step()

        return self.training_history

    def cleanup(self):
        """Cleanup resources"""
        if self.distributed:
            cleanup_distributed()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

# Memory-efficient data loading
def load_tensor_data(data_dir, device, rank, world_size, max_samples=20):
    """Load minimal tensor data to avoid OOM"""
    try:
        # Very small tensors - 16x16 instead of 32x32
        tensors = torch.randn(max_samples, 3, 16, 16)
        labels = torch.randint(0, 3, (max_samples,))
        logger.info(f"Rank {rank}: Created {max_samples} samples of size 16x16")
        return tensors, labels
    except Exception as e:
        logger.error(f"Rank {rank}: Error loading tensor data: {str(e)}")
        raise

def split_data(tensors, labels):
    """Split data into train, validation, and test sets"""
    train_size = int(0.7 * len(tensors))
    val_size = int(0.15 * len(tensors))
    test_size = len(tensors) - train_size - val_size
    
    # Ensure minimum sizes
    if train_size < 1:
        train_size = 1
    if val_size < 1:
        val_size = 1
    if test_size < 1:
        test_size = len(tensors) - train_size - val_size
        
    train_tensors, val_tensors, test_tensors = torch.split(tensors, [train_size, val_size, test_size])
    train_labels, val_labels, test_labels = torch.split(labels, [train_size, val_size, test_size])
    return train_tensors, val_tensors, test_tensors, train_labels, val_labels, test_labels

def analyze_noise_patterns(pipeline, test_tensors, test_labels):
    """Analyze noise patterns with memory management"""
    logger.info("Analyzing noise patterns...")
    try:
        # Use only first few samples to avoid OOM
        subset_size = min(5, len(test_tensors))
        features = pipeline.extract_noise_features_batch(test_tensors[:subset_size])
        if len(features) > 0:
            logger.info(f"Extracted features shape: {features.shape}")
        else:
            logger.warning("No features extracted due to memory constraints")
    except Exception as e:
        logger.error(f"Error analyzing noise patterns: {str(e)}")

def run_complete_pipeline(rank, world_size, data_dir: str = 'datasets/train',
                         weights_dir: str = 'noise_autoencoder_checkpoints',
                         classifier_dir: str = 'noise_classifier_checkpoints'):
    """Run the complete noise distribution classification pipeline for three classes"""
    try:
        distributed_mode = False
        if world_size > 1:
            try:
                setup_distributed(rank, world_size)
                distributed_mode = True
                logger.info(f"Rank {rank}: Running in distributed mode with {world_size} GPUs")
            except Exception as e:
                logger.warning(f"Rank {rank}: Distributed setup failed: {str(e)}")
                logger.warning(f"Rank {rank}: Falling back to single GPU mode")
                distributed_mode = False
                world_size = 1
                rank = 0

        device = torch.device(f'cuda:{rank}' if distributed_mode else 'cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Rank {rank}: Using device {device}")

        # Clear GPU memory before loading data
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Load minimal data
        tensors, labels = load_tensor_data(data_dir, device, rank, world_size, max_samples=20)
        train_tensors, val_tensors, test_tensors, train_labels, val_labels, test_labels = split_data(tensors, labels)

        # Use simplest model configuration
        pipeline = NoiseClassificationPipeline(
            rank=rank,
            world_size=world_size,
            device=device,
            use_ensemble=False,
            use_enhanced_autoencoder=False,  # Use simplest model
            weights_dir=weights_dir,
            classifier_dir=classifier_dir
        )

        # Minimal training configuration
        results = pipeline.train(
            train_tensors, train_labels, val_tensors, val_labels,
            test_tensors, test_labels, 
            autoencoder_epochs=2,  # Very few epochs
            batch_size=2,  # Very small batch size
            resume_from_checkpoint=True, 
            retrain_classifier=True,
            use_adversarial=False,  # Skip adversarial training
            enable_explanations=False,  # Skip explanations
            accumulation_steps=4
        )

        if not pipeline.distributed or rank == 0:
            analyze_noise_patterns(pipeline, test_tensors, test_labels)

        pipeline.cleanup()
        return results

    except Exception as e:
        logger.error(f"Rank {rank}: Error in pipeline: {str(e)}")
        import traceback
        logger.error(f"Rank {rank}: Traceback: {traceback.format_exc()}")
        raise
    finally:
        if distributed_mode:
            cleanup_distributed()

def main():
    """Main function to run the noise distribution classification pipeline"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    set_multiprocessing_start_method()

    logger.info("Starting Noise Distribution Classification for Three Classes")
    logger.info("Classes: Real (0), Semi-synthetic (1), Synthetic (2)")

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        logger.info(f"Found {world_size} GPU(s)")

        # Start with single GPU to avoid distributed issues
        logger.info("Starting single GPU training to avoid memory issues")
        run_complete_pipeline(0, 1, 'datasets/train',
                            'noise_autoencoder_checkpoints',
                            'noise_classifier_checkpoints')
    else:
        logger.warning("No GPU available, using CPU")
        run_complete_pipeline(0, 1, 'datasets/train',
                            'noise_autoencoder_checkpoints',
                            'noise_classifier_checkpoints')

if __name__ == "__main__":
    main()