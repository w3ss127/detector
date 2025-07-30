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

# Assuming the provided SuperiorConfig, SuperiorModel, and SuperiorDataset classes are available
# Import necessary classes from the provided code
from ConvNeXTViTGAN import SuperiorConfig, SuperiorModel, SuperiorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_checkpoint(model, checkpoint_path, device):
    """Load the model checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
        # Override _balance_dataset to prevent oversampling for test data
        self._balance_dataset = lambda: None  # Disable balancing for test dataset

    def __getitem__(self, idx):
        """Return image, label (if available), and metadata."""
        file_path, image_idx = self.file_indices[idx]
        label = self.labels[idx] if self.labels else -1  # -1 indicates no label
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

def test_individual_images(config, checkpoint_path="superior_checkpoints/best_superior_model.pth"):
    """Test individual images from the test dataset and print results."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.DEVICE = device
    config.FOCAL_ALPHA = config.FOCAL_ALPHA.to(device)
    config.CLASS_WEIGHTS = config.CLASS_WEIGHTS.to(device)

    # Initialize model
    model = SuperiorModel(config).to(device)
    model.eval()
    load_checkpoint(model, checkpoint_path, device)

    # Initialize dataset
    test_dataset = TestDataset(root_dir=config.TEST_PATH, config=config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for clear output
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Process each image
    logger.info("Starting individual image testing...")
    for images, labels, metadata in tqdm(test_loader, desc="Testing Images"):
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

            # Prepare output
            output = [f"File: {file_path}, Image Index: {image_idx}"]
            output.append(f"Predicted Class: {pred_class}")
            if labels.item() != -1:
                true_class = config.CLASS_NAMES[labels.item()]
                output.append(f"True Class: {true_class}")
            output.append(f"Probabilities: {prob_str}")
            if config.USE_UNCERTAINTY_ESTIMATION and epistemic_unc is not None:
                total_uncertainty = epistemic_unc.squeeze().cpu().numpy() + aleatoric_unc.mean(dim=1).cpu().numpy()
                output.append(f"Total Uncertainty: {total_uncertainty[0]:.4f}")
                output.append(f"Epistemic Uncertainty: {epistemic_unc.squeeze().cpu().numpy()[0]:.4f}")
                output.append(f"Aleatoric Uncertainty: {aleatoric_unc.mean(dim=1).cpu().numpy()[0]:.4f}")

            # Print results
            print("\n".join(output))
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Superior Deepfake Detection Individual Image Testing')
    parser.add_argument('--test_path', type=str, default='datasets/test', help='Path to test dataset')
    parser.add_argument('--checkpoint_path', type=str, default='superior_checkpoints/best_superior_model.pth', help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='convnext_tiny', choices=['convnext_tiny', 'convnext_small'])
    parser.add_argument('--hidden_dim', type=int, default=1536)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--no_forensics', action='store_true')
    parser.add_argument('--no_uncertainty', action='store_true')
    parser.add_argument('--no_spectral_norm', action='store_true')
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
    test_individual_images(config, args.checkpoint_path)

if __name__ == '__main__':
    main()