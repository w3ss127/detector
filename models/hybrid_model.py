# models/hybrid_model.py

import torch
import torch.nn as nn
from torchvision.models import resnet50
import timm


class ResNetViTHybrid(nn.Module):
    """
    Hybrid model combining a ResNet50 backbone with a ViT encoder.
    """

    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=True):
        super(ResNetViTHybrid, self).__init__()

        # === RESNET50 BACKBONE ===
        self.backbone = resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()  # Remove classifier, keep features
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone_out_dim = 2048  # ResNet50 feature output

        # === VIT ENCODER ===
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Identity()
        self.vit_out_dim = self.vit.num_features

        # === CLASSIFICATION HEAD ===
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out_dim + self.vit_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def unfreeze_backbone(self):
        """Call this to unfreeze ResNet50 backbone after a certain epoch."""
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze_backbone = False
            print("🟢 ResNet50 backbone unfrozen.")

    def forward(self, x):
        # Run through ResNet50
        features_resnet = self.backbone(x)

        # Resize to (B, 3, 224, 224) if needed for ViT
        if x.size(-1) != 224 or x.size(-2) != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Run through ViT
        features_vit = self.vit(x)

        # Concatenate and classify
        fused = torch.cat([features_resnet, features_vit], dim=1)
        return self.classifier(fused)
