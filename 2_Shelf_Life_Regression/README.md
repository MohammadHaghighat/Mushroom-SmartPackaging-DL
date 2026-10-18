# ⏱️ Phase 2: Remaining Shelf Life (RSL) Multimodal Regression

This directory contains the deep regression pipeline for predicting the continuous **Remaining Shelf Life (RSL in days)** of active bio-nanocomposite packaged white button mushrooms.

---

## 🎯 Objectives
- Predict exact remaining days before reaching commercial unacceptability ($RSL = \max(0, \text{Threshold} - \text{Current Day})$).
- Overcome the single-image ambiguity problem by **Multimodal Fusion**:
  - **Vision Branch:** Deep cap browning and shrinkage features (Swin-T / ResNet50).
  - **Metadata Branch:** One-hot encoded packaging treatment formulation (Control vs. 1SKEO.2CEO vs. 2SKEO.1CEO).
- Prevent overfitting on small biological cohorts using an **Advanced Data Augmentation Pipeline**:
  - Random Flips (Horizontal & Vertical)
  - Random Rotation ($\pm 25^\circ$)
  - Random Affine Scaling & Translation
  - ColorJitter (compensates for lighting and reflection variations)
  - Random Erasing (forces global cap feature attention)
- Continuous evaluation metrics: **MAE (days)**, **RMSE (days)**, and **$R^2$ Score**.

---

## 📁 Files Description
- **`dataset.py`:** PyTorch Dataset with continuous RSL calculation, One-Hot treatment encoding, and heavy augmentation pipeline.
- **`models.py`:** Multimodal Swin-T and Multimodal ResNet50 architectures with feature concatenation and deep MLP regression head.
- **`train.py`:** Two-stage regression training engine utilizing **Smooth L1 (Huber Loss)** for robust outlier handling.
- **`kaggle_mushroom_rsl_pipeline.ipynb`:** Self-contained, ready-to-run Kaggle notebook with GPU acceleration.
- **`splits.json`:** Deterministic Sample-ID split ensuring zero data leakage.

---

## 🚀 How to Run

### 1. Training Multimodal Swin Transformer
```bash
python train.py --model swin_t --epochs 35 --batch-size 16
```

### 2. Training Multimodal ResNet50
```bash
python train.py --model resnet50 --epochs 35 --batch-size 16
```
