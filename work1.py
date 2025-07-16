from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vision_transformer import vit_b_16
import os
from typing import List

# ============================
# 1. Model Definition (same as training)
# ============================
class ResNetCustomTop(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_feature = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, 7, 7]
        self.norm = nn.BatchNorm2d(2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(2048, 768)

        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet_feature(x)
        x = self.norm(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x).unsqueeze(1)  # [B, 1, 768]
        x = self.vit.encoder(x) if hasattr(self.vit, "encoder") else self.vit.encoder(x)  # version-safe
        x = x.mean(dim=1)
        return self.classifier(x)

# ============================
# 2. Instantiate Model + Load Weights
# ============================
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetCustomTop(num_classes).to(device)

checkpoint = torch.load("bitmind.pth", map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Loaded full checkpoint (model_state_dict)")
else:
    model.load_state_dict(checkpoint)
    print("✅ Loaded raw model weights")

model.eval()

# ============================
# 3. Load & Preprocess Images
# ============================
def load_and_prepare_test_images(path: str) -> torch.Tensor:
    all_images = torch.load(path)
    fixed_images: List[torch.Tensor] = []
    converted, skipped = 0, 0

    for i, img in enumerate(all_images):
        if img.shape == (1, 256, 256):
            img = img.repeat(3, 1, 1)
            converted += 1
        if img.shape != (3, 256, 256):
            print(f"❌ Skipping image {i} with shape {img.shape}")
            skipped += 1
            continue
        fixed_images.append(img)

    print(f"✅ Converted: {converted}, Skipped: {skipped}, Total: {len(fixed_images)}")
    return torch.stack(fixed_images)

# Load test images
test_images = load_and_prepare_test_images("./bm_semi_images.pt")
print(f"📦 Loaded {len(test_images)} test images")

# ============================
# 4. Run Inference
# ============================
predictions = []
class0_count = 0

for i, img in enumerate(test_images):
    img = img.unsqueeze(0).float()  # shape: [1, 3, 256, 256]
    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        prob_list = probs.cpu().squeeze().tolist()

    print(f"Image {i+1}/{len(test_images)} → Class {pred}, Probabilities: {[round(p, 4) for p in prob_list]}")
    predictions.append(pred)
    if pred == 2:
        class0_count += 1
print(f"\n✅ Inference complete. Processed {len(predictions)} images.")
print(f"🟦 Class 0 Count: {class0_count}")
