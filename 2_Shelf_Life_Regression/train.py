"""
Smart Packaging and Remaining Shelf Life (RSL) Prediction of White Button Mushrooms
==================================================================================
Step 3 & 4: Multimodal Regression Training Engine & Validation Pipeline

Supports:
  - Multimodal Swin-T (Hierarchical ViT)
  - Multimodal ResNet50 (Deep CNN)

Metrics:
  - Mean Absolute Error (MAE in days)
  - Root Mean Squared Error (RMSE in days)
  - Coefficient of Determination (R^2 Score)
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    torch = None

from dataset import create_rsl_dataloaders
from models import (
    MultimodalSwinRSL,
    MultimodalResNetRSL,
    freeze_backbone,
    unfreeze_backbone,
    get_rsl_parameter_groups
)


# =============================================================================
# 1. Regression Metrics Calculator
# =============================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, and R2-score."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# =============================================================================
# 2. Training and Evaluation Steps
# =============================================================================

def train_one_epoch_rsl(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (mean_loss, train_mae)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    for images, treatments, targets in loader:
        images = images.to(device, non_blocking=True)
        treatments = treatments.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, treatments)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / total_samples
    y_pred = np.vstack(all_preds).flatten()
    y_true = np.vstack(all_targets).flatten()
    metrics = compute_regression_metrics(y_true, y_pred)

    return epoch_loss, metrics["mae"]


@torch.no_grad()
def evaluate_epoch_rsl(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Runs evaluation on validation or test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    for images, treatments, targets in loader:
        images = images.to(device, non_blocking=True)
        treatments = treatments.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images, treatments)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / total_samples
    y_pred = np.vstack(all_preds).flatten()
    y_true = np.vstack(all_targets).flatten()
    metrics = compute_regression_metrics(y_true, y_pred)

    return epoch_loss, metrics, y_true, y_pred


# =============================================================================
# 3. Two-Stage Fine-Tuning Orchestrator
# =============================================================================

def train_multimodal_rsl(
    model_name: str = "swin_t",
    root_dir: str = ".",
    batch_size: int = 16,
    warmup_epochs: int = 5,
    finetune_epochs: int = 30,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    checkpoint_dir: str = "checkpoints_rsl",
    output_dir: str = "outputs_rsl",
    strong_aug: bool = True
):
    """
    Executes complete Two-Stage training for RSL Multimodal Regression.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"RSL TRAINING ENGINE | Device: {device} | Model: {model_name}")
    print("=" * 70)

    # 1. Build DataLoaders with heavy augmentation
    train_loader, val_loader, test_loader, datasets = create_rsl_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        strong_aug=strong_aug
    )
    print(f"Dataset Partitions: Train={len(datasets['train'])} | Val={len(datasets['val'])} | Test={len(datasets['test'])}")

    # 2. Build Model
    if model_name.lower() in ("swin", "swin_t"):
        model = MultimodalSwinRSL(pretrained=True)
    elif model_name.lower() in ("resnet", "resnet50"):
        model = MultimodalResNetRSL(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model = model.to(device)

    # Huber Loss (SmoothL1Loss) is more robust to label noise and outliers than pure MSE
    criterion = nn.SmoothL1Loss(beta=1.0)

    # Output paths
    ckpt_path = Path(checkpoint_dir)
    out_path = Path(output_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_file = ckpt_path / f"best_{model_name}_rsl.pth"

    history = {
        "train_loss": [], "val_loss": [],
        "train_mae": [], "val_mae": [], "val_rmse": [], "val_r2": []
    }

    best_val_mae = float("inf")
    epochs_no_improve = 0

    # -------------------------------------------------------------------------
    # STAGE 1: Warmup Fusion Head (Frozen Backbone)
    # -------------------------------------------------------------------------
    print(f"\n>>> Stage 1: Warmup Multimodal Head ({warmup_epochs} epochs, Frozen Backbone) <<<")
    freeze_backbone(model)
    warmup_optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=head_lr,
        weight_decay=weight_decay
    )

    for epoch in range(1, warmup_epochs + 1):
        t0 = time.time()
        tr_loss, tr_mae = train_one_epoch_rsl(model, train_loader, criterion, warmup_optimizer, device)
        val_loss, val_metrics, _, _ = evaluate_epoch_rsl(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])

        elapsed = time.time() - t0
        print(f"  [Warmup {epoch:02d}/{warmup_epochs:02d}] "
              f"Train Loss: {tr_loss:.4f}, MAE: {tr_mae:.2f}d || "
              f"Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.2f}d, R2: {val_metrics['r2']:.3f} "
              f"({elapsed:.1f}s)")

    # -------------------------------------------------------------------------
    # STAGE 2: End-to-End Fine-Tuning (Differential LRs)
    # -------------------------------------------------------------------------
    print(f"\n>>> Stage 2: Full End-to-End Fine-Tuning ({finetune_epochs} epochs) <<<")
    unfreeze_backbone(model)
    param_groups = get_rsl_parameter_groups(
        model,
        backbone_lr=backbone_lr,
        head_lr=head_lr * 0.5,
        weight_decay=weight_decay
    )
    optimizer = AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-7)

    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        tr_loss, tr_mae = train_one_epoch_rsl(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate_epoch_rsl(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])

        elapsed = time.time() - t0
        is_best = val_metrics["mae"] < best_val_mae
        marker = " [* BEST]" if is_best else ""

        print(f"  [Epoch {epoch:02d}/{finetune_epochs:02d}] "
              f"Train Loss: {tr_loss:.4f}, MAE: {tr_mae:.2f}d || "
              f"Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.2f}d, R2: {val_metrics['r2']:.3f}{marker} "
              f"({elapsed:.1f}s)")

        if is_best:
            best_val_mae = val_metrics["mae"]
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "val_mae": best_val_mae,
                "val_metrics": val_metrics,
                "history": history
            }, best_ckpt_file)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print(f"\nTraining Complete! Best Validation MAE: {best_val_mae:.2f} days")
    print(f"Best checkpoint saved at: {best_ckpt_file}")

    # -------------------------------------------------------------------------
    # TEST SET EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING FINAL TEST SET EVALUATION (Held-Out Samples)")
    print("=" * 70)
    ckpt = torch.load(best_ckpt_file, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics, y_test_true, y_test_pred = evaluate_epoch_rsl(
        model, test_loader, criterion, device
    )

    print(f"Test MAE:  {test_metrics['mae']:.2f} days")
    print(f"Test RMSE: {test_metrics['rmse']:.2f} days")
    print(f"Test R^2:  {test_metrics['r2']:.4f}")

    # Plot & Save Actual vs. Predicted RSL
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_true, y_test_pred, color="#1f77b4", alpha=0.7, edgecolors="k", s=60, label="Test Samples")
    max_val = max(float(y_test_true.max()), float(y_test_pred.max()), 20.0)
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Ideal (y = x)")
    plt.xlabel("Actual RSL (Days)", fontsize=12)
    plt.ylabel("Predicted RSL (Days)", fontsize=12)
    plt.title(f"Remaining Shelf Life Prediction ({model_name.upper()})\nMAE: {test_metrics['mae']:.2f}d | R²: {test_metrics['r2']:.3f}", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    pred_plot_path = out_path / f"{model_name}_rsl_actual_vs_pred.png"
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"Regression scatter plot saved to: {pred_plot_path}")

    return history, test_metrics
