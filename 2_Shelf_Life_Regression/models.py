"""
Smart Packaging and Remaining Shelf Life (RSL) Prediction of White Button Mushrooms
==================================================================================
Step 2: Multimodal Deep Regression Architectures
  - Model A: Multimodal Swin-T (Hierarchical ViT + Treatment Metadata Fusion)
  - Model B: Multimodal ResNet50 (Deep Residual CNN + Treatment Metadata Fusion)

Inputs:
  - Image: [B, 3, 224, 224]
  - Treatment One-Hot: [B, 3] (Control, 1SKEO.2CEO, 2SKEO.1CEO)

Output:
  - Predicted Remaining Shelf Life (RSL) in Days: [B, 1]
"""

from typing import Tuple, List, Dict, Union, Optional
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    swin_t, Swin_T_Weights
)


# =============================================================================
# 1. Multimodal Swin Transformer for RSL Prediction
# =============================================================================

class MultimodalSwinRSL(nn.Module):
    """
    Multimodal Swin Transformer that fuses visual cap/gill degradation features
    with treatment packaging formulation to predict Remaining Shelf Life (RSL).
    """

    def __init__(
        self,
        pretrained: bool = True,
        treatment_dim: int = 3,
        treatment_emb_dim: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        self.model_name = "multimodal_swin_t"

        # 1. Vision Backbone: Pre-trained Swin-T
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        base_swin = swin_t(weights=weights)

        # Extract features (patch embed + 4 hierarchical stages + norm)
        self.features = base_swin.features
        self.norm = base_swin.norm
        self.permute = base_swin.permute
        self.avgpool = base_swin.avgpool
        vision_dim = 768  # Output channels of Swin-T stage 4

        # 2. Metadata Processing Branch (Treatment Embedding)
        self.treatment_net = nn.Sequential(
            nn.Linear(treatment_dim, treatment_emb_dim),
            nn.LayerNorm(treatment_emb_dim),
            nn.GELU()
        )

        # 3. Multimodal Fusion & Regression Head
        fusion_dim = vision_dim + treatment_emb_dim  # 768 + 32 = 800
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(64, 1),
            nn.ReLU()  # RSL cannot be negative
        )

    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 768-dim pooled feature vector from input image."""
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, image: torch.Tensor, treatment_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          image: [B, 3, 224, 224]
          treatment_one_hot: [B, 3]
        Returns:
          rsl_pred: [B, 1] (Continuous RSL in days)
        """
        # Visual features [B, 768]
        vis_feat = self.extract_visual_features(image)

        # Treatment features [B, 32]
        treat_feat = self.treatment_net(treatment_one_hot)

        # Concatenate multimodal features [B, 800]
        fused = torch.cat([vis_feat, treat_feat], dim=1)

        # Predict RSL
        rsl_pred = self.regression_head(fused)
        return rsl_pred


# =============================================================================
# 2. Multimodal ResNet50 for RSL Prediction
# =============================================================================

class MultimodalResNetRSL(nn.Module):
    """
    Multimodal ResNet50 architecture baseline for RSL prediction.
    """

    def __init__(
        self,
        pretrained: bool = True,
        treatment_dim: int = 3,
        treatment_emb_dim: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        self.model_name = "multimodal_resnet50"

        # 1. Vision Backbone: Pre-trained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_resnet = resnet50(weights=weights)

        # Submodules up to global avg pool
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.avgpool = base_resnet.avgpool
        vision_dim = 2048

        # 2. Metadata Processing Branch
        self.treatment_net = nn.Sequential(
            nn.Linear(treatment_dim, treatment_emb_dim),
            nn.LayerNorm(treatment_emb_dim),
            nn.GELU()
        )

        # 3. Fusion & Regression Head
        fusion_dim = vision_dim + treatment_emb_dim
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, image: torch.Tensor, treatment_one_hot: torch.Tensor) -> torch.Tensor:
        vis_feat = self.extract_visual_features(image)
        treat_feat = self.treatment_net(treatment_one_hot)
        fused = torch.cat([vis_feat, treat_feat], dim=1)
        rsl_pred = self.regression_head(fused)
        return rsl_pred


# =============================================================================
# 3. Two-Stage Fine-Tuning Helpers (Differential Learning Rates)
# =============================================================================

def freeze_backbone(model: nn.Module) -> None:
    """Freezes vision backbone for initial regression head warmup."""
    if hasattr(model, "features"):  # Swin
        for p in model.features.parameters():
            p.requires_grad = False
    elif hasattr(model, "layer1"):  # ResNet
        for layer in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]:
            for p in layer.parameters():
                p.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreezes all parameters for full end-to-end fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True


def get_rsl_parameter_groups(
    model: nn.Module,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> List[Dict]:
    """Sets distinct learning rates for vision backbone vs. multimodal fusion head."""
    backbone_params = []
    head_params = []

    head_submodules = {"treatment_net", "regression_head"}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(sm in name for sm in head_submodules):
            head_params.append(p)
        else:
            backbone_params.append(p)

    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    ]
