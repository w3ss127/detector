import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm
import numpy as np
from PIL import Image
import argparse
import logging

# ============ CONFIGURATION ============
BASE_DIR = "datasets"
TEST_DIR = os.path.join(BASE_DIR, "test")
SUBFOLDERS = ["real", "synthetic", "semi-synthetic"]
CLASS_MAP = {"real": 0, "synthetic": 1, "semi-synthetic": 2}
CLASS_NAMES = ["real", "synthetic", "semi-synthetic"]
NUM_CLASSES = 3
RESNET_VARIANT = "resnet50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description="Image classifier for all subfolders")
parser.add_argument("--minimal-transforms", action="store_true", help="Use minimal transforms (skip augmentations)")
args = parser.parse_args()

# ============ CHECK DEVICE ============
logger.info(f"🔧 Using device: {DEVICE}")
print(f"🔧 Using device: {DEVICE}")

# ============ TRANSFORMS ============
if args.minimal_transforms:
    logger.info("Using minimal transforms (no augmentations)")
    test_transforms = transforms.Compose([
        # transforms.Lambda(lambda x: x / 255.0),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    test_transforms = transforms.Compose([
        # transforms.Lambda(lambda x: x / 255.0),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ============ MODEL DEFINITION ============
class ResNetViTHybrid(nn.Module):
    def __init__(self, num_classes, resnet_variant="resnet50"):
        super().__init__()
        if resnet_variant == "resnet50":
            self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        elif resnet_variant == "resnet121":
            self.resnet = models.resnet121(weights='IMAGENET1K_V2')
        else:
            raise ValueError("resnet_variant must be 'resnet50' or 'resnet121'")
        resnet_fc_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit_fc_in = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.fc1 = nn.Linear(resnet_fc_in + vit_fc_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected input shape [B, 3, 224, 224], got {x.shape}")
        res_feat = self.resnet(x)
        vit_feat = self.vit(x)
        combined = torch.cat([res_feat, vit_feat], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out)

# ============ LOAD MODEL ============
model_path = "best_model.pt"
model = ResNetViTHybrid(NUM_CLASSES, resnet_variant=RESNET_VARIANT).to(DEVICE)

if not os.path.exists(model_path):
    logger.error(f"❌ Model file {model_path} not found.")
    print(f"❌ Model file {model_path} not found.")
    sys.exit(1)

try:
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    logger.info(f"✅ Loaded model from {model_path}")
    print(f"✅ Loaded model from {model_path}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ============ PROCESS .PT FILES ============
def compute_histogram(img):
    img_np = img.cpu().numpy().flatten() / 255.0  # [0, 1]
    hist, _ = np.histogram(img_np, bins=10, range=(0, 1))
    return hist.tolist()

correct_per_class = [0] * NUM_CLASSES
total_per_class = [0] * NUM_CLASSES
correct = 0
total = 0
predictions = []
test_labels_all = []

for subfolder in SUBFOLDERS:
    subfolder_path = os.path.join(TEST_DIR, subfolder)
    logger.info(f"\n📂 Loading .pt files from: {subfolder_path}")
    print(f"\n📂 Loading .pt files from: {subfolder_path}")
    
    if not os.path.exists(subfolder_path):
        logger.warning(f"❌ Folder not found: {subfolder_path}")
        print(f"❌ Folder not found: {subfolder_path}")
        continue
    
    test_files = glob.glob(os.path.join(subfolder_path, "*.pt"))
    if not test_files:
        logger.warning(f"❌ No .pt files in {subfolder_path}")
        print(f"❌ No .pt files in {subfolder_path}")
        continue
    
    class_label = CLASS_MAP[subfolder]
    test_images = []
    test_labels = []
    
    for file in test_files:
        logger.info(f"📄 Processing file: {file}")
        print(f"📄 Processing file: {file}")
        try:
            data = torch.load(file, map_location=DEVICE)
            if not isinstance(data, torch.Tensor) or len(data.shape) != 4 or data.shape[1] not in [1, 3] or data.shape[2:4] != (256, 256):
                logger.error(f"❌ Invalid format or shape in {file}: {data.shape if isinstance(data, torch.Tensor) else type(data)}")
                print(f"❌ Invalid format or shape in {file}: {data.shape if isinstance(data, torch.Tensor) else type(data)}")
                continue
            if data.dtype != torch.uint8:
                logger.warning(f"⚠️ Expected uint8 dtype in {file}, got {data.dtype}. Converting to uint8.")
                print(f"⚠️ Expected uint8 dtype in {file}, got {data.dtype}. Converting to uint8.")
                data = data.to(torch.uint8)
            valid_images = []
            for i in range(data.shape[0]):
                img = data[i]
                if not isinstance(img, torch.Tensor) or torch.isnan(img).any() or torch.isinf(img).any():
                    logger.warning(f"⚠️ Image {i} in {file} contains invalid values, skipping")
                    print(f"⚠️ Image {i} in {file} contains invalid values, skipping")
                    continue
                mean_pixel = (img.float() / 255.0).mean().item()
                hist = compute_histogram(img)
                if mean_pixel < 0.078:
                    logger.warning(f"⚠️ Image {i} in {file} is very dark (mean pixel value: {mean_pixel:.3f}), histogram: {hist}")
                    print(f"⚠️ Image {i} in {file} is very dark (mean pixel value: {mean_pixel:.3f}), histogram: {hist}")
                    img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    Image.fromarray(img_np).save(f"debug_image_{subfolder}_{os.path.basename(file)}_{i}.png")
                img = img.float()
                img = test_transforms(img)
                valid_images.append(img)
            if valid_images:
                test_images.append(torch.stack(valid_images))
                test_labels.append(torch.tensor([class_label] * len(valid_images), dtype=torch.long))
                logger.info(f"✅ {file}: Loaded {len(valid_images)} valid images.")
                print(f"✅ {file}: Loaded {len(valid_images)} valid images.")
            else:
                logger.warning(f"⚠️ No valid images in {file}")
                print(f"⚠️ No valid images in {file}")
        except Exception as e:
            logger.error(f"❌ Failed to process {file}: {e}")
            print(f"❌ Failed to process {file}: {e}")
            continue
    
    if not test_images:
        logger.warning(f"❌ No valid test images found for {subfolder}")
        print(f"❌ No valid test images found for {subfolder}")
        continue
    
    test_images = torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    logger.info(f"📊 Total test samples for {subfolder}: {test_images.shape[0]}")
    print(f"📊 Total test samples for {subfolder}: {test_images.shape[0]}")
    
    # ============ INFERENCE ============
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(test_images):
            img = img.unsqueeze(0).to(DEVICE)
            output = model(img)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label = test_labels[i].item()
            predictions.append(pred)
            test_labels_all.append(label)
            if pred == label:
                correct += 1
                correct_per_class[class_label] += 1
            total += 1
            total_per_class[class_label] += 1
            logger.info(f"🖼️ Image {i+1}/{test_images.shape[0]} in {subfolder}: Predicted → {CLASS_NAMES[pred]}, Probabilities → {[round(p, 4) for p in probs.squeeze().tolist()]}")
            # print(f"🖼️ Image {i+1}/{test_images.shape[0]} in {subfolder}: Predicted → {CLASS_NAMES[pred]}, Probabilities → {[round(p, 4) for p in probs.squeeze().tolist()]}")
    
    torch.cuda.empty_cache()

# ============ ACCURACY ============
if total == 0:
    logger.error("❌ No images processed.")
    print("❌ No images processed.")
    sys.exit(1)

accuracy = 100 * correct / total
pred_counts = np.bincount(predictions, minlength=NUM_CLASSES)
logger.info(f"\n✅ Inference complete.")
logger.info(f"🎯 Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"\n✅ Inference complete.")
print(f"🎯 Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
for i, class_name in enumerate(CLASS_NAMES):
    class_acc = 100 * correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0
    logger.info(f"{class_name} Accuracy: {class_acc:.2f}% ({correct_per_class[i]}/{total_per_class[i]})")
    print(f"{class_name} Accuracy: {class_acc:.2f}% ({correct_per_class[i]}/{total_per_class[i]})")
logger.info(f"Predicted class distribution: real={pred_counts[0]}, synthetic={pred_counts[1]}, semi-synthetic={pred_counts[2]}")
print(f"Predicted class distribution: real={pred_counts[0]}, synthetic={pred_counts[1]}, semi-synthetic={pred_counts[2]}")