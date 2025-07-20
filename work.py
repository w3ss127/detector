import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm

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

# ✅ Set which class subfolder to load
TARGET_SUBFOLDER = "synthetic"  # <<< CHANGE THIS to "synthetic" or "semi-synthetic"

# ============ CHECK DEVICE ============
print(f"🔧 Using device: {DEVICE}")

# ============ TRANSFORMS ============
test_transforms = transforms.Compose([
    # transforms.Lambda(lambda x: x / 255.0),
    transforms.Resize((224, 224)),
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
    print(f"❌ Model file {model_path} not found.")
    sys.exit(1)

try:
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    print(f"✅ Loaded model from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ============ CHECK SUBFOLDER ============
if TARGET_SUBFOLDER not in SUBFOLDERS:
    print(f"❌ Invalid subfolder: {TARGET_SUBFOLDER}. Choose from {SUBFOLDERS}")
    sys.exit(1)

subfolder_path = os.path.join(TEST_DIR, TARGET_SUBFOLDER)
if not os.path.exists(subfolder_path):
    print(f"❌ Folder not found: {subfolder_path}")
    sys.exit(1)

print(f"\n📂 Loading .pt files from: {subfolder_path}")
test_files = glob.glob(os.path.join(subfolder_path, "*.pt"))
if not test_files:
    print(f"❌ No .pt files in {subfolder_path}")
    sys.exit(1)

# ============ LOAD + PREPROCESS IMAGES ============
class_label = CLASS_MAP[TARGET_SUBFOLDER]
test_images, test_labels = [], []

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

# ============ STACK IMAGES ============
if not test_images:
    print("❌ No valid test images found.")
    sys.exit(1)

test_images = torch.cat(test_images, dim=0)
test_labels = torch.cat(test_labels, dim=0)
print(test_labels)
print(f"\n📊 Total test samples: {test_images.shape[0]}")

# ============ INFERENCE ============
model.eval()
correct = 0
predictions = []

with torch.no_grad():
    for i, img in enumerate(test_images):
        img = img.unsqueeze(0).to(DEVICE)
        # print(img)
        output = model(img)
        
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = test_labels[i].item()
        predictions.append(pred)
        if pred == label:
            correct += 1
        # print(f"🖼️ Image {i+1}: Predicted → {CLASS_NAMES[pred]}, Probabilities → {[round(p, 4) for p in probs.squeeze().tolist()]}")

# ============ ACCURACY ============
accuracy = 100 * correct / len(test_labels)
print(f"\n✅ Inference complete for '{TARGET_SUBFOLDER}'.")
print(f"🎯 Accuracy: {accuracy:.2f}% ({correct}/{len(test_labels)})")
