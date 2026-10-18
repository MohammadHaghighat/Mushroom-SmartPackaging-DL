# 🔍 Phase 1: Visual Quality Grade Classification & Explainability (Grad-CAM)

This directory contains the complete deep learning pipeline for **4-Class Biological Spoilage Classification** of white button mushrooms (*Agaricus bisporus*).

---

## 🎯 Objectives
- Classify mushroom images into 4 biological quality classes:
  - **Class 0:** Very Healthy (Fresh, white, tight veil)
  - **Class 1:** Healthy (Minor physiological aging, acceptable)
  - **Class 2:** Spoiled (Enzymatic browning onset, surface softening)
  - **Class 3:** Very Spoiled (Severe decay, dark brown, tissue collapse)
- Prevent data leakage by strict **Sample-ID Partitioning (1 to 22)**.
- Suppress packaging plastic glare using **LAB Color Space CLAHE**.
- Compare **ResNet-50 (CNN)** vs. **Swin Transformer (Hierarchical ViT)**.
- Provide visual explanations of model focus via **Grad-CAM**.

---

## 📁 Files Description
- **`dataset.py`:** PyTorch Dataset with dynamic label mapping, Center-Crop, and LAB-CLAHE preprocessing.
- **`models.py`:** Model builders for ResNet50 and Swin-T with custom classification heads and two-stage freeze/unfreeze utilities.
- **`train.py`:** Two-stage training engine with Cosine Annealing scheduler and Early Stopping.
- **`evaluate.py`:** Generates classification reports and confusion matrices for the held-out test set.
- **`explainability.py`:** Computes Grad-CAM activation maps for visual validation against plastic reflections.
- **`kaggle_mushroom_pipeline.ipynb`:** Self-contained Jupyter Notebook for Kaggle execution with GPU acceleration.
- **`splits.json`:** Deterministic sample partition ensuring zero data leakage.

---

## 🚀 How to Run

### 1. Training
```bash
python train.py --model swin_t --epochs 35 --batch-size 16
```

### 2. Evaluation on Test Set
```bash
python evaluate.py --model swin_t --checkpoint checkpoints/best_swin_t.pth
```

### 3. Grad-CAM Visualization
```bash
python explainability.py --checkpoint checkpoints/best_swin_t.pth
```
