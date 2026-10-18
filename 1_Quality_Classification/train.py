"""
Smart Packaging and Spoilage Monitoring of White Button Mushrooms
================================================================
Step 4: Training Loop with Two-Stage Fine-Tuning & Early Stopping

Supports:
  - ResNet50 (Classic CNN)
  - Swin-T (Vision Transformer)

Features:
  - Two-Stage Transfer Learning:
      Stage 1 (Warmup): Frozen backbone, train head only (5 epochs).
      Stage 2 (Fine-Tuning): Full model fine-tuning with differential learning rates.
  - Cosine Annealing Learning Rate Scheduler.
  - Early Stopping based on validation loss / macro F1.
  - Automatic checkpointing of best model weights.
  - Exports training curves (Loss and Accuracy).
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from dataset import create_dataloaders, CLASS_NAMES
from models import (
    build_model,
    freeze_backbone,
    unfreeze_backbone,
    get_parameter_groups
)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (mean_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Runs validation evaluation and returns (mean_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def run_training(
    model_name: str = "resnet50",
    root_dir: str = ".",
    preprocessed_dir: str = None,
    total_epochs: int = 40,
    warmup_epochs: int = 5,
    batch_size: int = 16,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"TRAINING PIPELINE: {model_name.upper()}")
    print("=" * 80)
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(f"GPU Name:             {torch.cuda.get_device_name(0)}")
    print(f"Total Epochs:         {total_epochs} (Warmup Head: {warmup_epochs}, Full Fine-Tune: {total_epochs - warmup_epochs})")
    print(f"Batch Size:           {batch_size}")
    print(f"Learning Rates:       Backbone = {backbone_lr}, Classifier Head = {head_lr}")
    print(f"Early Stopping:       Patience = {patience} epochs")

    ckpt_path = Path(checkpoint_dir)
    out_path = Path(output_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Create DataLoaders
    print("\n[1] Initializing DataLoaders...")
    train_loader, val_loader, test_loader, datasets = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        img_size=224,
        num_workers=0
    )
    print(f"  * Train Batches: {len(train_loader)} ({len(datasets['train'])} images)")
    print(f"  * Val   Batches: {len(val_loader)} ({len(datasets['val'])} images)")
    print(f"  * Test  Batches: {len(test_loader)} ({len(datasets['test'])} images)")

    # 2. Build Model
    print(f"\n[2] Initializing Pre-trained {model_name}...")
    model = build_model(model_name, num_classes=4, pretrained=True).to(device)

    # 3. Loss & Optimizer Setup
    criterion = nn.CrossEntropyLoss()

    # --- STAGE 1: WARMUP CLASSIFICATION HEAD ---
    if warmup_epochs > 0:
        print(f"\n>>> STAGE 1: Warmup Classifier Head ({warmup_epochs} epochs with frozen backbone) <<<")
        freeze_backbone(model)
        warmup_optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=head_lr,
            weight_decay=weight_decay
        )

        for epoch in range(1, warmup_epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, warmup_optimizer, device)
            v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device)
            print(f"  [Warmup Epoch {epoch:02d}/{warmup_epochs:02d}] "
                  f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% || "
                  f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")

    # --- STAGE 2: FULL NETWORK FINE-TUNING ---
    print(f"\n>>> STAGE 2: Unfreezing Full Network for End-to-End Fine-Tuning <<<")
    unfreeze_backbone(model)

    param_groups = get_parameter_groups(
        model,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay
    )
    optimizer = AdamW(param_groups)
    fine_tune_epochs = total_epochs - warmup_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-6)

    # Tracking metrics
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    best_checkpoint_file = ckpt_path / f"best_{model_name}.pth"

    start_time = time.time()

    for epoch in range(1, fine_tune_epochs + 1):
        current_epoch = warmup_epochs + epoch
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        # Check for best model
        is_best = v_loss < best_val_loss
        if is_best:
            best_val_loss = v_loss
            best_val_acc = v_acc
            patience_counter = 0
            # Save weights
            torch.save({
                "epoch": current_epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "class_names": CLASS_NAMES
            }, best_checkpoint_file)
            best_mark = " [* BEST SAVED]"
        else:
            patience_counter += 1
            best_mark = ""

        print(f"  [Epoch {current_epoch:02d}/{total_epochs:02d}] "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% || "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%{best_mark}")

        # Early Stopping check
        if patience_counter >= patience:
            print(f"\n[Early Stopping Triggered] No validation loss improvement for {patience} consecutive epochs.")
            break

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Training completed in {elapsed / 60:.2f} minutes.")
    print(f"Best Validation Loss:     {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved Checkpoint:         {best_checkpoint_file}")

    # Plot & Save Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    axes[0].set_title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Fine-Tuning Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_acc"], "b-o", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs_range, history["val_acc"], "g--s", label="Val Accuracy", linewidth=2)
    axes[1].set_title(f"{model_name.upper()} - Classification Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Fine-Tuning Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    curve_path = out_path / f"training_curves_{model_name}.png"
    plt.tight_layout()
    plt.savefig(curve_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {curve_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spoilage Classifier")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "swin_t"], help="Model architecture")
    parser.add_argument("--data-dir", type=str, default=".", help="Root dataset directory")
    parser.add_argument("--epochs", type=int, default=35, help="Total training epochs")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup epochs with frozen backbone")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    args = parser.parse_args()

    run_training(
        model_name=args.model,
        root_dir=args.data_dir,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup,
        batch_size=args.batch_size,
        patience=args.patience
    )
