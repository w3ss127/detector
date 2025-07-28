import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle

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
np.random.seed(SEED)

# ============ CHECK DEVICE ============
print(f"🔧 Using device: {DEVICE}")

# ============ TRANSFORMS ============
test_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
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

    def forward(self, x):
        res_feat = self.resnet(x)
        vit_feat = self.vit(x)
        combined = torch.cat([res_feat, vit_feat], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out)

# ============ LOAD MODEL ============
model_path = "./checkpoints/best_model_mcc_0.9266_618116ec.pt"
model = ResNetViTHybrid(NUM_CLASSES, resnet_variant=RESNET_VARIANT).to(DEVICE)

if not os.path.exists(model_path):
    print(f"❌ Model file {model_path} not found.")
    sys.exit(1)

try:
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    print(f"✅ Loaded model from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ============ LOAD + PREPROCESS IMAGES FROM ALL SUBFOLDERS ============
test_images, test_labels = [], []

for subfolder in SUBFOLDERS:
    subfolder_path = os.path.join(TEST_DIR, subfolder)
    if not os.path.exists(subfolder_path):
        print(f"❌ Folder not found: {subfolder_path}")
        continue

    print(f"\n📂 Loading .pt files from: {subfolder_path}")
    test_files = glob.glob(os.path.join(subfolder_path, "*.pt"))
    if not test_files:
        print(f"❌ No .pt files in {subfolder_path}")
        continue

    class_label = CLASS_MAP[subfolder]
    for file in test_files:
        print(f"📄 Processing file: {file}")
        try:
            data = torch.load(file, map_location=DEVICE)
            if not isinstance(data, torch.Tensor) or len(data.shape) != 4:
                print(f"❌ Invalid format or shape in {file}")
                continue
            valid_images = []
            for i in range(data.shape[0]):
                img = data[i]
                if not isinstance(img, torch.Tensor) or torch.isnan(img).any() or torch.isinf(img).any():
                    continue
                img = img.float()
                img = test_transforms(img)
                valid_images.append(img)
            if valid_images:
                test_images.append(torch.stack(valid_images))
                test_labels.append(torch.tensor([class_label] * len(valid_images), dtype=torch.long))
                print(f"✅ {file}: Loaded {len(valid_images)} valid images.")
            else:
                print(f"⚠️ No valid images in {file}")
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")

# ============ STACK AND SHUFFLE IMAGES ============
if not test_images:
    print("❌ No valid test images found.")
    sys.exit(1)

test_images = torch.cat(test_images, dim=0)
test_labels = torch.cat(test_labels, dim=0)
print(f"\n📊 Total test samples before shuffling: {test_images.shape[0]}")

# Shuffle images and labels together
test_images_np = test_images.cpu().numpy()
test_labels_np = test_labels.cpu().numpy()
test_images_np, test_labels_np = shuffle(test_images_np, test_labels_np, random_state=SEED)
test_images = torch.tensor(test_images_np).to(DEVICE)
test_labels = torch.tensor(test_labels_np, dtype=torch.long).to(DEVICE)

print(f"🔀 Data shuffled. Total test samples: {test_images.shape[0]}")

# ============ INFERENCE ============
model.eval()
correct = 0
predictions = []
all_probs = []

with torch.no_grad():
    for i, img in enumerate(test_images):
        img = img.unsqueeze(0).to(DEVICE)
        output = model(img)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = test_labels[i].item()
        predictions.append(pred)
        all_probs.append(probs.squeeze().cpu().numpy())
        if pred == label:
            correct += 1
        print(f"🖼️ Image {i+1}: Predicted → {CLASS_NAMES[pred]}, True → {CLASS_NAMES[label]}, Probabilities → {[round(p, 4) for p in probs.squeeze().tolist()]}")

# ============ ACCURACY AND MCC PER CLASS ============
accuracy = 100 * correct / len(test_labels)
print(f"\n✅ Inference complete.")
print(f"🎯 Overall Accuracy: {accuracy:.2f}% ({correct}/{len(test_labels)})")

# Per-class accuracy and MCC
predictions = np.array(predictions)
test_labels_np = test_labels.cpu().numpy()

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_mask = test_labels_np == class_idx
    if class_mask.sum() == 0:
        print(f"\n📈 Class {class_name}: No samples in test set.")
        continue
    class_correct = (predictions[class_mask] == test_labels_np[class_mask]).sum()
    class_total = class_mask.sum()
    class_accuracy = 100 * class_correct / class_total if class_total > 0 else 0
    print(f"\n📈 Class {class_name}:")
    print(f"  Accuracy: {class_accuracy:.2f}% ({class_correct}/{class_total})")
    
    # Binarize for MCC (one-vs-rest)
    y_true_binary = (test_labels_np == class_idx).astype(int)
    y_pred_binary = (predictions == class_idx).astype(int)
    mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
    print(f"  MCC: {mcc:.4f}")

# ============ MULTI-CLASS MCC ============
multi_class_mcc = matthews_corrcoef(test_labels_np, predictions)
print(f"\n🌟 Multi-class MCC: {multi_class_mcc:.4f}")