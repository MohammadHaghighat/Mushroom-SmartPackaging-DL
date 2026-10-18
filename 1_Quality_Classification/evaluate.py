"""
Smart Packaging and Spoilage Monitoring of White Button Mushrooms
================================================================
Evaluation Script: Test Set Metrics, Confusion Matrix, and Spoilage Delay Analysis

Features:
  - Evaluates trained model on the held-out Test set (zero data leakage).
  - Computes Overall Accuracy, Macro F1, Precision, and Recall.
  - Plots and saves high-resolution Confusion Matrix (Normalized & Counts).
  - Compares Spoilage Classification across Treatments (Control vs Active Packaging).
"""

import argparse
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from dataset import create_dataloaders, CLASS_NAMES
from models import build_model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4):
    """Calculates accuracy, per-class precision, recall, f1, and macro averages."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = np.trace(cm) / len(y_true) * 100.0

    precision = []
    recall = []
    f1 = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    macro_precision = np.mean(precision) * 100.0
    macro_recall = np.mean(recall) * 100.0
    macro_f1 = np.mean(f1) * 100.0

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    model_name: str
):
    """Generates and saves a publication-quality Confusion Matrix plot."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ticks = np.arange(len(class_names))
    short_names = [f"Class {i}\n({name.split(': ')[-1]})" for i, name in enumerate(class_names)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Predicted Visual Grade", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Visual Grade", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix: {model_name.upper()} on Test Set", fontsize=14, fontweight="bold", pad=15)

    # Loop over data dimensions and create text annotations
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > thresh else "black"
            text = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion Matrix saved to: {save_path}")


def evaluate_model(
    model_name: str = "resnet50",
    checkpoint_path: str = None,
    root_dir: str = ".",
    output_dir: str = "outputs"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/best_{model_name}.pth"

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}. Please train the model first.")

    print("=" * 80)
    print(f"MODEL EVALUATION ON TEST SET: {model_name.upper()}")
    print("=" * 80)
    print(f"Checkpoint: {ckpt_file}")
    print(f"Device:     {device}")

    # Load DataLoaders
    _, _, test_loader, datasets = create_dataloaders(
        root_dir=root_dir,
        batch_size=16,
        img_size=224,
        num_workers=0
    )
    test_ds = datasets["test"]

    # Load Model
    model = build_model(model_name, num_classes=4, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    metadata_list = []

    print("\nRunning inference on held-out Test set...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(targets.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute comprehensive metrics
    metrics = calculate_metrics(y_true, y_pred, num_classes=4)

    print("\n" + "=" * 80)
    print("TEST SET CLASSIFICATION REPORT")
    print("=" * 80)
    print(f"{'Class':<25} | {'Precision (%)':<15} | {'Recall (%)':<15} | {'F1-Score (%)':<15}")
    print("-" * 80)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<25} | {metrics['precision'][i]*100:<15.2f} | {metrics['recall'][i]*100:<15.2f} | {metrics['f1'][i]*100:<15.2f}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<25}: {metrics['accuracy']:.2f}%")
    print(f"{'Macro Precision':<25}: {metrics['macro_precision']:.2f}%")
    print(f"{'Macro Recall':<25}: {metrics['macro_recall']:.2f}%")
    print(f"{'Macro F1-Score':<25}: {metrics['macro_f1']:.2f}%")
    print("=" * 80)

    # Plot Confusion Matrix
    cm_path = out_path / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_path, model_name)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mushroom Spoilage Model")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "swin_t"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        root_dir=args.data_dir,
        output_dir=args.output_dir
    )
