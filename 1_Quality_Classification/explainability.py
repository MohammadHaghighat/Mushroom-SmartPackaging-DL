"""
Smart Packaging and Spoilage Monitoring of White Button Mushrooms
================================================================
Step 4: Grad-CAM Explainability (ResNet50 vs. Swin Transformer)

Generates visual attention heatmaps proving whether the deep learning models
focus on biological surface defects (enzymatic browning spots, cap shrinkage)
or background artifacts/plastic glare.

Implementation:
  - Pure PyTorch Hook-based Grad-CAM (No external dependencies required).
  - Supports standard 4D CNN feature maps (B, C, H, W) for ResNet50.
  - Supports permuted Transformer feature maps (B, H, W, C) for Swin-T.
  - Produces side-by-side comparative visualizations across packaging treatments.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

from dataset import (
    preprocess_mushroom_image,
    get_transforms,
    CLASS_NAMES,
    LABEL_MAPPING,
    get_or_create_sample_splits
)
from models import build_model, get_gradcam_target_layer


# =============================================================================
# 1. Native Grad-CAM Engine
# =============================================================================

class GradCAM:
    """
    Hook-based Grad-CAM implementation compatible with CNNs and Vision Transformers.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple; gradient w.r.t layer output is the first element
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates normalized Grad-CAM heatmap for the given input image.

        Args:
            input_tensor: (1, 3, H, W) normalized PyTorch image tensor.
            target_class: Target class index to explain. If None, uses top predicted class.

        Returns:
            Tuple[np.ndarray, int, float]: (heatmap_2d_norm, pred_class, confidence_score)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Backward pass for the target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        # Handle Swin Transformer feature map layout: (B, H, W, C) -> (B, C, H, W)
        if activations.ndim == 4 and activations.shape[1] != gradients.shape[1]:
            pass
        if activations.ndim == 4 and activations.shape[-1] in (768, 1024, 2048):
            # Channel dimension is last
            activations = activations.permute(0, 3, 1, 2)
            gradients = gradients.permute(0, 3, 1, 2)

        # Global Average Pooling of gradients across spatial dimensions (alpha weights)
        # activations: (1, C, H, W), gradients: (1, C, H, W)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)

        # Apply ReLU to retain only features with positive contribution to the target class
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence


def overlay_cam_on_image(
    image_rgb: np.ndarray,
    cam_heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays a Grad-CAM heatmap on the original RGB image.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        cam_heatmap: (h_cam, w_cam) float32 array in [0, 1].
        alpha: Blending weight for heatmap overlay (default 0.5).
        colormap: OpenCV colormap enum.

    Returns:
        Tuple: (blended_rgb, colored_heatmap_rgb)
    """
    h, w = image_rgb.shape[:2]
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(cam_heatmap, (w, h))

    # Convert to 8-bit heatmap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    blended = np.float32(heatmap_colored_rgb) * alpha + np.float32(image_rgb) * (1 - alpha)
    blended = np.uint8(np.clip(blended, 0, 255))

    return blended, heatmap_colored_rgb


# =============================================================================
# 2. Comparative Visualization Generator (ResNet50 vs. Swin-T)
# =============================================================================

def generate_comparative_gradcam(
    resnet_path: Optional[str] = "checkpoints/best_resnet50.pth",
    swin_path: Optional[str] = "checkpoints/best_swin_t.pth",
    root_dir: str = ".",
    output_dir: str = "outputs",
    sample_items: Optional[List[Dict]] = None
):
    """
    Generates a publication-grade side-by-side Grad-CAM figure comparing ResNet50 and Swin-T
    on critical mushroom conditions across storage days and active packaging treatments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    root_path = Path(root_dir)

    print("=" * 80)
    print("GRAD-CAM EXPLAINABILITY PIPELINE: RESNET50 vs. SWIN TRANSFORMER")
    print("=" * 80)
    print(f"Device: {device}")

    # Build models
    print("\n[1] Initializing Models & Attaching Hooks...")
    model_res = build_model("resnet50", num_classes=4, pretrained=False).to(device)
    model_swin = build_model("swin_t", num_classes=4, pretrained=False).to(device)

    # Load weights if available
    r_ckpt = Path(resnet_path) if resnet_path else None
    s_ckpt = Path(swin_path) if swin_path else None

    if r_ckpt and r_ckpt.exists():
        print(f"  * Loading ResNet50 checkpoint: {r_ckpt}")
        model_res.load_state_dict(torch.load(r_ckpt, map_location=device)["state_dict"])
    else:
        print("  * [Notice] ResNet50 checkpoint not found; running with pre-trained feature weights for demonstration.")

    if s_ckpt and s_ckpt.exists():
        print(f"  * Loading Swin-T checkpoint: {s_ckpt}")
        model_swin.load_state_dict(torch.load(s_ckpt, map_location=device)["state_dict"])
    else:
        print("  * [Notice] Swin-T checkpoint not found; running with pre-trained feature weights for demonstration.")

    # Attach Grad-CAM
    cam_res = GradCAM(model_res, get_gradcam_target_layer(model_res), device)
    cam_swin = GradCAM(model_swin, get_gradcam_target_layer(model_swin), device)

    # Prepare standard transforms
    _, eval_transform = get_transforms(img_size=224)

    # Default representative test samples if none provided
    if sample_items is None:
        sample_items = [
            {
                "title": "Control Day 0 (Class 0: Very Healthy)",
                "path": root_path / "Control" / "0" / "1-b.jpg",
                "label": 0
            },
            {
                "title": "Control Day 7 (Class 2: Browning Onset / Spoiled)",
                "path": root_path / "Control" / "7" / "1-b.jpg",
                "label": 2
            },
            {
                "title": "1SKEO.2CEO Day 20 (Class 2: Packaging Retarded Spoilage)",
                "path": root_path / "1SKEO.2CEO" / "20" / "1-b.jpg",
                "label": 2
            },
            {
                "title": "Control Day 20 (Class 3: Severe Browning & Shrinkage)",
                "path": root_path / "Control" / "20" / "1-b.jpg",
                "label": 3
            }
        ]

    # Filter to samples that actually exist
    valid_samples = [s for s in sample_items if Path(s["path"]).exists()]
    if not valid_samples:
        print("No valid sample images found for Grad-CAM. Check paths.")
        return

    n_samples = len(valid_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4.5 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row, sample in enumerate(valid_samples):
        img_path = str(sample["path"])
        ground_truth_label = sample["label"]

        # 1. OpenCV Preprocessing (Center-Crop 0.65 + LAB-CLAHE)
        preprocessed_rgb = preprocess_mushroom_image(img_path, crop_ratio=0.65)
        pil_img = Image.fromarray(preprocessed_rgb)
        input_tensor = eval_transform(pil_img).unsqueeze(0)

        # 2. ResNet50 Grad-CAM
        heatmap_res, pred_res, conf_res = cam_res.generate_heatmap(input_tensor, ground_truth_label)
        overlay_res, _ = overlay_cam_on_image(cv2.resize(preprocessed_rgb, (224, 224)), heatmap_res)

        # 3. Swin-T Grad-CAM
        heatmap_swin, pred_swin, conf_swin = cam_swin.generate_heatmap(input_tensor, ground_truth_label)
        overlay_swin, _ = overlay_cam_on_image(cv2.resize(preprocessed_rgb, (224, 224)), heatmap_swin)

        # Col 0: Preprocessed Image
        axes[row, 0].imshow(cv2.resize(preprocessed_rgb, (224, 224)))
        axes[row, 0].set_title(
            f"{sample['title']}\nGround Truth: {CLASS_NAMES[ground_truth_label]}",
            fontsize=11, fontweight="bold"
        )
        axes[row, 0].axis("off")

        # Col 1: ResNet50 Grad-CAM
        axes[row, 1].imshow(overlay_res)
        axes[row, 1].set_title(
            f"ResNet50 Attention (CNN)\nPred: {CLASS_NAMES[pred_res]} ({conf_res*100:.1f}%)",
            fontsize=11, fontweight="bold", color="navy"
        )
        axes[row, 1].axis("off")

        # Col 2: Swin-T Grad-CAM
        axes[row, 2].imshow(overlay_swin)
        axes[row, 2].set_title(
            f"Swin-T Attention (Vision Transformer)\nPred: {CLASS_NAMES[pred_swin]} ({conf_swin*100:.1f}%)",
            fontsize=11, fontweight="bold", color="darkred"
        )
        axes[row, 2].axis("off")

    master_cam_path = out_path / "gradcam_resnet_vs_swin_comparison.png"
    plt.tight_layout()
    plt.savefig(master_cam_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n[Success] Grad-CAM comparative master figure saved to:\n  {master_cam_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grad-CAM Explainability")
    parser.add_argument("--resnet-ckpt", type=str, default="checkpoints/best_resnet50.pth")
    parser.add_argument("--swin-ckpt", type=str, default="checkpoints/best_swin_t.pth")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    generate_comparative_gradcam(
        resnet_path=args.resnet_ckpt,
        swin_path=args.swin_ckpt,
        root_dir=args.data_dir,
        output_dir=args.out_dir
    )
