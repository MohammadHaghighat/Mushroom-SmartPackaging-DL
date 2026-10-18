"""
Smart Packaging and Spoilage Monitoring of White Button Mushrooms
================================================================
Step 3: Model Building & Transfer Learning
Architectures:
  1. Model A: ResNet50 (Classic Deep Residual CNN)
  2. Model B: Swin Transformer (Swin-T, Hierarchical Vision Transformer)

Both models are customized to output 4 Visual Quality Classes:
  - Class 0: Very Healthy
  - Class 1: Healthy
  - Class 2: Spoiled
  - Class 3: Very Spoiled

Includes utilities for:
  - Two-stage fine-tuning (backbone freezing / unfreezing)
  - Differential learning rates for backbone vs. classification head
  - Target layer identification for Step 4 Grad-CAM explainability
"""

from typing import Tuple, List, Dict, Union
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    swin_t, Swin_T_Weights
)


# =============================================================================
# 1. Model A: ResNet50 Builder
# =============================================================================

def build_resnet50(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """
    Constructs and adapts ResNet50 for 4-class mushroom quality classification.

    Key Architectural Details:
    - Backbone: 50 layers with bottleneck residual blocks (1x1, 3x3, 1x1 convolutions).
    - Weights: Pre-trained ImageNet-1K V2 weights (80.858% top-1 accuracy).
    - Head: Replaces original 1000-class linear layer with Dropout(0.3) + Linear(2048, num_classes).
    - Grad-CAM Target: Last bottleneck block of layer4 (`model.layer4[-1]`).

    Args:
        num_classes: Number of output classes (default 4).
        pretrained: If True, loads pre-trained ImageNet weights.
        dropout: Dropout probability before the linear classifier (prevents overfitting).

    Returns:
        nn.Module: Adapted ResNet50 model.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    # In ResNet50, final pooling produces 2048 feature channels
    in_features = model.fc.in_features

    # Replace 1000-class fc with custom head with regularization
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    # Attach model metadata for downstream training and explainability
    model.model_name = "resnet50"
    model.num_classes = num_classes
    return model


# =============================================================================
# 2. Model B: Swin Transformer (Swin-T) Builder
# =============================================================================

def build_swin_t(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """
    Constructs and adapts Swin Transformer (Swin-T) for 4-class mushroom quality classification.

    Key Architectural Details:
    - Backbone: Hierarchical Vision Transformer with Shifted Windows (W-MSA & SW-MSA).
      Processes non-overlapping local windows with cross-window connections.
    - Weights: Pre-trained ImageNet-1K V1 weights (81.474% top-1 accuracy).
    - Head: Replaces original 1000-class linear head with Dropout(0.3) + Linear(768, num_classes).
    - Grad-CAM Target: Final stage feature map norm (`model.features[7]`).

    Args:
        num_classes: Number of output classes (default 4).
        pretrained: If True, loads pre-trained ImageNet weights.
        dropout: Dropout probability before the classification head.

    Returns:
        nn.Module: Adapted Swin-T model.
    """
    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_t(weights=weights)

    # In Swin-T, stage 4 output dimension is 768
    in_features = model.head.in_features

    # Replace 1000-class head
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    # Attach metadata
    model.model_name = "swin_t"
    model.num_classes = num_classes
    return model


# =============================================================================
# 3. Unified Factory Function
# =============================================================================

def build_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """
    Unified model factory.

    Args:
        model_name: 'resnet50' or 'swin_t'.
        num_classes: Target number of classes (default 4).
        pretrained: Whether to load pre-trained weights.
        dropout: Dropout rate in classifier head.

    Returns:
        nn.Module: Configured PyTorch model.
    """
    name_clean = model_name.lower().strip()
    if name_clean in ("resnet50", "resnet", "resnet-50"):
        return build_resnet50(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif name_clean in ("swin_t", "swin", "swin_transformer", "swint"):
        return build_swin_t(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Choose from ['resnet50', 'swin_t']."
        )


# =============================================================================
# 4. Transfer Learning Utilities: Freezing, Unfreezing & Parameter Groups
# =============================================================================

def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all feature extractor backbone layers, leaving only the newly
    initialized classification head trainable. Used for Stage 1 warm-up.
    """
    model_name = getattr(model, "model_name", "").lower()

    if "resnet" in model_name:
        # Freeze all layers except model.fc
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif "swin" in model_name:
        # Freeze all layers except model.head
        for name, param in model.named_parameters():
            if "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # Generic fallback: freeze everything except the last child module
        children = list(model.named_children())
        classifier_name = children[-1][0] if children else ""
        for name, param in model.named_parameters():
            param.requires_grad = (classifier_name in name)


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreezes all parameters across the entire network for Stage 2 full fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True


def get_parameter_groups(
    model: nn.Module,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> List[Dict]:
    """
    Constructs differential parameter groups for the optimizer:
    - Backbone parameters are trained with a smaller learning rate (e.g., 1e-4)
      to carefully preserve general visual representations.
    - Classification head parameters are trained with a higher learning rate (e.g., 1e-3)
      to accelerate convergence on our specific 4 visual quality classes.

    Args:
        model: PyTorch model.
        backbone_lr: Learning rate for pre-trained feature extractor.
        head_lr: Learning rate for custom 4-class classifier head.
        weight_decay: L2 regularization coefficient.

    Returns:
        List[Dict]: Parameter group specifications for PyTorch optimizers (AdamW, SGD, etc.).
    """
    model_name = getattr(model, "model_name", "").lower()
    head_param_names = {"fc", "head"}

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if parameter belongs to the final classification head
        is_head = any(hp in name for hp in head_param_names)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay
        })

    return param_groups


def get_model_summary(model: nn.Module) -> Dict[str, Union[int, float]]:
    """
    Returns parameter count statistics for model comparison.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    }


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Returns the optimal feature activation layer for Grad-CAM explainability:
    - ResNet50: Last bottleneck block of layer4 (`model.layer4[-1]`).
    - Swin-T: Final hierarchical stage (`model.features[7]`).
    """
    model_name = getattr(model, "model_name", "").lower()
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "swin" in model_name:
        return model.features[7]
    else:
        raise ValueError(f"Unknown target layer mapping for model: {model_name}")


# =============================================================================
# 5. Standalone Model Inspection Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 3: MODEL BUILDING & TRANSFER LEARNING INSPECTION")
    print("=" * 80)

    # 1. Inspect ResNet50
    print("\n[1] Initializing ResNet50 (Adapted for 4 Quality Classes)...")
    model_resnet = build_resnet50(num_classes=4, pretrained=False)
    summary_resnet = get_model_summary(model_resnet)
    print(f"  * Classifier Head: {model_resnet.fc}")
    print(f"  * Total Parameters: {summary_resnet['total_parameters']:,}")
    print(f"  * Grad-CAM Target Layer: {get_gradcam_target_layer(model_resnet).__class__.__name__}")

    # Stage 1: Freeze Backbone
    freeze_backbone(model_resnet)
    s1_res = get_model_summary(model_resnet)
    print(f"  * Stage 1 (Warmup): Trainable Params = {s1_res['trainable_parameters']:,} ({s1_res['trainable_percentage']:.2f}%)")

    # Stage 2: Unfreeze Backbone
    unfreeze_backbone(model_resnet)
    s2_res = get_model_summary(model_resnet)
    print(f"  * Stage 2 (Fine-tune): Trainable Params = {s2_res['trainable_parameters']:,} ({s2_res['trainable_percentage']:.2f}%)")

    # Test forward pass with dummy tensor (Batch=2, Channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    out_res = model_resnet(dummy_input)
    print(f"  * Forward Pass Output Shape: {out_res.shape} (Expected: [2, 4])")
    assert out_res.shape == (2, 4), "ResNet output shape mismatch!"

    # 2. Inspect Swin Transformer (Swin-T)
    print("\n[2] Initializing Swin Transformer (swin_t, Adapted for 4 Quality Classes)...")
    model_swin = build_swin_t(num_classes=4, pretrained=False)
    summary_swin = get_model_summary(model_swin)
    print(f"  * Classifier Head: {model_swin.head}")
    print(f"  * Total Parameters: {summary_swin['total_parameters']:,}")
    print(f"  * Grad-CAM Target Layer: {get_gradcam_target_layer(model_swin).__class__.__name__}")

    # Stage 1: Freeze Backbone
    freeze_backbone(model_swin)
    s1_swin = get_model_summary(model_swin)
    print(f"  * Stage 1 (Warmup): Trainable Params = {s1_swin['trainable_parameters']:,} ({s1_swin['trainable_percentage']:.2f}%)")

    # Stage 2: Unfreeze Backbone
    unfreeze_backbone(model_swin)
    s2_swin = get_model_summary(model_swin)
    print(f"  * Stage 2 (Fine-tune): Trainable Params = {s2_swin['trainable_parameters']:,} ({s2_swin['trainable_percentage']:.2f}%)")

    # Test forward pass
    out_swin = model_swin(dummy_input)
    print(f"  * Forward Pass Output Shape: {out_swin.shape} (Expected: [2, 4])")
    assert out_swin.shape == (2, 4), "Swin output shape mismatch!"

    print("\n[3] Model Comparison Summary:")
    print(f"  {'Model':<15} | {'Backbone Paradigm':<25} | {'Total Parameters':<18} | {'Head Dimensions':<15}")
    print("-" * 80)
    print(f"  {'ResNet50':<15} | {'Deep CNN (Residual)':<25} | {summary_resnet['total_parameters']:<18,} | {'2048 -> 4':<15}")
    print(f"  {'Swin-T':<15} | {'Vision Transformer (W-MSA)':<25} | {summary_swin['total_parameters']:<18,} | {'768 -> 4':<15}")

    print("\nStep 3 Models successfully verified and ready for Training Loop (Step 4)!")
    print("=" * 80)
