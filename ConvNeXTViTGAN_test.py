import torch
import torch.nn.functional as F
import os
from pathlib import Path
import logging
import json
import numpy as np
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Assuming the provided SuperiorConfig, SuperiorModel, and SuperiorDataset classes are available
from ConvNeXTViTGAN import SuperiorConfig, SuperiorModel, SuperiorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_checkpoint(model, checkpoint_path, device):
    """Load the model checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        raise

def get_test_transform(config):
    """Define the validation/test transform for images."""
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class TestDataset(SuperiorDataset):
    """Custom dataset for testing individual images from .pt files."""
    def __init__(self, root_dir, config):
        super().__init__(root_dir, config, transform=get_test_transform(config), is_training=False)
        self._balance_dataset = lambda: None  # Disable balancing for test dataset

    def __getitem__(self, idx):
        """Return image, label (if available), and metadata."""
        file_path, image_idx = self.file_indices[idx]
        label = self.labels[idx] if self.labels else -1
        try:
            tensor_data = self.tensor_cache.get(file_path)
            if tensor_data is None:
                tensor_data = torch.load(file_path, map_location='cpu')
                if isinstance(tensor_data, dict):
                    tensor_data = tensor_data.get('images', tensor_data.get('data', list(tensor_data.values())[0]))
                self.tensor_cache[file_path] = tensor_data
            image_tensor = tensor_data[image_idx]
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image_np = image_tensor.permute(1, 2, 0).numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            transformed = self.val_transform(image=image_np)
            image_tensor = transformed['image']
            return image_tensor, label, (file_path, image_idx)
        except Exception as e:
            logger.warning(f"Error loading image at {file_path}[{image_idx}]: {e}")
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), -1, (file_path, image_idx)

def test_individual_images(config, checkpoint_path="gan_checkpoints/best_superior_model.pth", gpu_id=None, use_cpu=False):
    """Test individual images from the test dataset and print results with per-class accuracy and MCC."""
    # Set device - try to find a GPU with available memory
    if use_cpu:
        device = torch.device("cpu")
        logger.info("Forced CPU usage")
    elif torch.cuda.is_available():
        if gpu_id is not None:
            # Use specified GPU
            if 0 <= gpu_id < torch.cuda.device_count():
                try:
                    torch.cuda.set_device(gpu_id)
                    device = torch.device(f"cuda:{gpu_id}")
                    memory_allocated = torch.cuda.memory_allocated(gpu_id)
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                    memory_free = memory_total - memory_allocated
                    logger.info(f"Using specified GPU {gpu_id} with {memory_free / 1024**3:.1f}GB free memory")
                except Exception as e:
                    logger.warning(f"Failed to use specified GPU {gpu_id}: {e}")
                    device = torch.device("cpu")
            else:
                logger.warning(f"Specified GPU {gpu_id} not available, falling back to auto-selection")
                gpu_id = None
        
        if gpu_id is None:
            # Auto-select GPU with most free memory
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_free = memory_total - memory_allocated
                    gpu_memory.append((i, memory_free))
                except:
                    gpu_memory.append((i, 0))
            
            # Sort by available memory (descending)
            gpu_memory.sort(key=lambda x: x[1], reverse=True)
            
            # Try to use the GPU with most free memory
            for gpu_id, free_memory in gpu_memory:
                if free_memory > 2 * 1024 * 1024 * 1024:  # At least 2GB free
                    try:
                        torch.cuda.set_device(gpu_id)
                        device = torch.device(f"cuda:{gpu_id}")
                        logger.info(f"Using GPU {gpu_id} with {free_memory / 1024**3:.1f}GB free memory")
                        break
                    except:
                        continue
            else:
                # If no GPU has enough memory, use CPU
                device = torch.device("cpu")
                logger.warning("No GPU with sufficient memory found, using CPU")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    
    config.DEVICE = device
    
    # Move tensors to device with error handling
    try:
        if hasattr(config, 'FOCAL_ALPHA') and config.FOCAL_ALPHA is not None:
            if not config.FOCAL_ALPHA.is_cuda:
                config.FOCAL_ALPHA = config.FOCAL_ALPHA.to(device)
        if hasattr(config, 'CLASS_WEIGHTS') and config.CLASS_WEIGHTS is not None:
            if not config.CLASS_WEIGHTS.is_cuda:
                config.CLASS_WEIGHTS = config.CLASS_WEIGHTS.to(device)
    except Exception as e:
        logger.warning(f"Failed to move tensors to device: {e}")
        # Continue without moving tensors if there's an error

    # Initialize model with memory optimization
    try:
        model = SuperiorModel(config).to(device)
        model.eval()
        load_checkpoint(model, checkpoint_path, device)
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Initialize dataset with reduced memory usage
    test_dataset = TestDataset(root_dir=config.TEST_PATH, config=config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Reduced to prevent memory issues
        pin_memory=False  # Disabled to reduce memory usage
    )

    # Lists to store predictions and true labels for metrics
    all_preds = []
    all_labels = []

    # Process each image
    logger.info("Starting individual image testing...")
    for images, labels, metadata in tqdm(test_loader, desc="Testing Images"):
        try:
            images = images.to(device)
            file_path, image_idx = metadata[0][0], metadata[1][0].item()
            
            with torch.no_grad():
                model_output = model(images)
                if len(model_output) == 3:
                    logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
                else:
                    logits, features = model_output
                    probs = F.softmax(logits, dim=1)
                    epistemic_unc, aleatoric_unc = None, None

                # Get predictions and probabilities
                pred = logits.argmax(dim=1).item()
                pred_class = config.CLASS_NAMES[pred]
                probs = probs.cpu().numpy()[0]
                prob_str = ", ".join([f"{cls}: {p:.4f}" for cls, p in zip(config.CLASS_NAMES, probs)])

                # Collect predictions and labels (if available)
                all_preds.append(pred)
                if labels.item() != -1:
                    all_labels.append(labels.item())
                else:
                    all_labels.append(-1)  # Placeholder for unlabeled data

                # Prepare output
                output = [f"File: {file_path}, Image Index: {image_idx}"]
                output.append(f"Predicted Class: {pred_class}")
                if labels.item() != -1:
                    true_class = config.CLASS_NAMES[labels.item()]
                    output.append(f"True Class: {true_class}")
                output.append(f"Probabilities: {prob_str}")
                if config.USE_UNCERTAINTY_ESTIMATION and epistemic_unc is not None:
                    epistemic_unc_np = epistemic_unc.squeeze().cpu().numpy()
                    aleatoric_unc_np = aleatoric_unc.mean(dim=1).cpu().numpy()
                    total_uncertainty = epistemic_unc_np + aleatoric_unc_np
                    epistemic_unc_val = epistemic_unc_np.item() if epistemic_unc_np.ndim == 0 else epistemic_unc_np[0]
                    aleatoric_unc_val = aleatoric_unc_np.item() if aleatoric_unc_np.ndim == 0 else aleatoric_unc_np[0]
                    total_uncertainty_val = total_uncertainty.item() if total_uncertainty.ndim == 0 else total_uncertainty[0]
                    output.append(f"Total Uncertainty: {total_uncertainty_val:.4f}")
                    output.append(f"Epistemic Uncertainty: {epistemic_unc_val:.4f}")
                    output.append(f"Aleatoric Uncertainty: {aleatoric_unc_val:.4f}")

                # Print results
                print("\n".join(output))
                print("-" * 50)
                
            # Clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing image {file_path}[{image_idx}]: {e}")
            continue

    # Calculate and display per-class accuracy and MCC
    if all_labels.count(-1) < len(all_labels):  # Ensure there are labeled samples
        valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
        valid_preds = [all_preds[i] for i in valid_indices]
        valid_labels = [all_labels[i] for i in valid_indices]

        print("\nPer-Class Metrics:")
        print("=" * 50)
        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            # Create binary labels/predictions for the current class
            binary_labels = [1 if label == class_idx else 0 for label in valid_labels]
            binary_preds = [1 if pred == class_idx else 0 for pred in valid_preds]

            # Calculate accuracy
            class_accuracy = accuracy_score(binary_labels, binary_preds)

            # Calculate MCC (handle cases with insufficient data)
            try:
                class_mcc = matthews_corrcoef(binary_labels, binary_preds)
            except ValueError:
                class_mcc = float('nan')  # MCC undefined if no positive samples or predictions

            print(f"Class: {class_name}")
            print(f"Accuracy: {class_accuracy:.4f}")
            print(f"MCC: {class_mcc:.4f}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Superior Deepfake Detection Individual Image Testing')
    parser.add_argument('--test_path', type=str, default='datasets/test', help='Path to test dataset')
    parser.add_argument('--checkpoint_path', type=str, default='gan_checkpoints/best_superior_model.pth', help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='convnext_tiny', choices=['convnext_tiny', 'convnext_small'])
    parser.add_argument('--hidden_dim', type=int, default=1536)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--no_forensics', action='store_true')
    parser.add_argument('--no_uncertainty', action='store_true')
    parser.add_argument('--no_spectral_norm', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None, help='Specific GPU ID to use (0-7)')
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()

    # Initialize config
    config = SuperiorConfig()
    config.TEST_PATH = args.test_path
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    config.NUM_WORKERS = args.num_workers
    config.CONVNEXT_BACKBONE = args.backbone
    config.HIDDEN_DIM = args.hidden_dim
    config.DROPOUT_RATE = args.dropout_rate
    config.USE_FORENSICS_MODULE = not args.no_forensics
    config.USE_UNCERTAINTY_ESTIMATION = not args.no_uncertainty
    config.USE_SPECTRAL_NORM = not args.no_spectral_norm
    config.validate()

    # Run test
    test_individual_images(config, args.checkpoint_path, args.gpu_id, args.use_cpu)

if __name__ == '__main__':
    main()