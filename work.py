import torch
import torch.nn.functional as F
import os

# ============================
# 1. Model Class Definition (match training definition)
# ============================
import torchvision.models as models
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights

# Model parameters
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build backbone
backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone_out_channels = backbone.fc.in_features
backbone = nn.Sequential(*list(backbone.children())[:-2])  # remove last FC and pool

# Custom model with classifier
class ResNetCustomTop(nn.Module):
    def __init__(self, backbone, backbone_out_channels, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_feature = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, 7, 7]
        self.norm = nn.BatchNorm2d(2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(2048, 768)

        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet_feature(x)  # [B, 2048, 7, 7]
        x = self.norm(x)
        x = self.pool(x).flatten(1)  # [B, 2048]
        x = self.proj(x).unsqueeze(1)  # [B, 1, 768]
        x = self.vit.encoder(x)  # [B, 1, 768]
        x = x.mean(dim=1)
        return self.classifier(x)

# Instantiate model
model = ResNetCustomTop(backbone, backbone_out_channels, num_classes).to(device)

# ============================
# 2. Load Weights
# ============================
checkpoint = torch.load("bitmind.pth", map_location="cpu")
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
print("✅ Loaded model from bitmind.pth")

# ============================
# 3. Load & Preprocess Test Images
# ============================
def load_and_prepare_test_images(path):
    all_images = torch.load(path)
    fixed_images = []
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
    return torch.stack(fixed_images)  # <--- Convert to tensor

# Load test images
test_images = load_and_prepare_test_images("./bm_test_images.pt")
print(test_images.shape)
print(test_images[0])
# ============================
# 4. Run Inference One-by-One
# ============================
predictions = []
cnt = 0
for i, img in enumerate(test_images):
    img = img.unsqueeze(0).float()  # shape: [1, 3, 256, 256]
    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        prob_list = probs.cpu().squeeze().tolist()
        print(f"Image {i+1}/{len(test_images)}: Predicted class → {pred}, Probabilities: { [round(p, 4) for p in prob_list] }")
        if pred == 0:
            cnt += 1
        predictions.append(pred)
print(cnt)
print(f"\n✅ Inference complete. Total images processed: {len(predictions)}")
