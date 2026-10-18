"""
Smart Packaging and Remaining Shelf Life (RSL) Prediction of White Button Mushrooms
==================================================================================
Dataset, Multimodal Metadata Integration, and Advanced Augmentation Pipeline.

Key Concepts:
  1. Shelf-Life Thresholds (End of Commercial Usability):
     - Control: Spoilage threshold is Day 7  (Shelf-life ~ 7 days)
     - 1SKEO.2CEO: Spoilage threshold is Day 20 (Shelf-life ~ 20 days)
     - 2SKEO.1CEO: Spoilage threshold is Day 20 (Shelf-life ~ 20 days)

  2. Target Variable (RSL in Days):
     RSL = max(0, Spoilage_Day - Current_Day)

  3. Treatment Metadata Encoding (Multimodal input):
     - Control:    0 (or One-Hot [1, 0, 0])
     - 1SKEO.2CEO: 1 (or One-Hot [0, 1, 0])
     - 2SKEO.1CEO: 2 (or One-Hot [0, 0, 1])

  4. Advanced Data Augmentation:
     Flips, Rotations, Affine transforms, ColorJitter, and RandomErasing to boost
     generalization on small sample sizes.
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    transforms = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

try:
    import cv2
except ImportError:
    cv2 = None


def preprocess_mushroom_image(
    image_input: Union[str, Path, np.ndarray],
    crop_ratio: float = 0.65,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Center-crop + LAB-CLAHE enhancement for glare reduction."""
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for preprocess_mushroom_image")
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {image_input}")
    else:
        img_bgr = image_input

    h, w = img_bgr.shape[:2]
    crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
    sx, sy = max(0, (w - crop_w) // 2), max(0, (h - crop_h) // 2)
    cropped = img_bgr[sy : sy + crop_h, sx : sx + crop_w]

    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enh = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2RGB)


def get_or_create_sample_splits(
    root_dir: Union[str, Path],
    split_file: Optional[Union[str, Path]] = None,
    train_count: int = 16,
    val_count: int = 3,
    test_count: int = 3,
    random_seed: int = 42
) -> Dict[str, List[int]]:
    """Sample-ID level partitioning to guarantee zero data leakage."""
    root_path = Path(root_dir)
    split_path = Path(split_file) if split_file else root_path / "splits.json"

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback deterministic split
    return {
        "train": [2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22],
        "val": [8, 9, 18],
        "test": [1, 4, 21]
    }

# =============================================================================
# 1. RSL Formulation & Metadata Mapping
# =============================================================================

# Spoilage arrival day for each treatment (reaching biological Class 2)
SPOILAGE_THRESHOLDS: Dict[str, float] = {
    "Control": 7.0,
    "1SKEO.2CEO": 20.0,
    "2SKEO.1CEO": 20.0,
}

TREATMENT_TO_INDEX: Dict[str, int] = {
    "Control": 0,
    "1SKEO.2CEO": 1,
    "2SKEO.1CEO": 2,
}

INDEX_TO_TREATMENT = {v: k for k, v in TREATMENT_TO_INDEX.items()}


def calculate_rsl(treatment: str, current_day: int) -> float:
    """Calculates Remaining Shelf Life (RSL) in days."""
    threshold = SPOILAGE_THRESHOLDS.get(treatment, 7.0)
    return float(max(0.0, threshold - current_day))


# =============================================================================
# 2. Advanced Augmentation Pipelines
# =============================================================================

def get_rsl_transforms(
    img_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    strong_aug: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns high-performance transforms with heavy data augmentation for training
    and deterministic transforms for validation/testing.
    """
    if strong_aug:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.06, 0.06),
                scale=(0.92, 1.08)
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
                hue=0.03
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
                value="random"
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


# =============================================================================
# 3. Multimodal RSL PyTorch Dataset
# =============================================================================

class MushroomRSLDataset(Dataset):
    """
    Dataset that outputs:
      - image: Augmented/Processed PyTorch Tensor [3, 224, 224]
      - treatment_feat: One-hot encoded or scalar treatment ID Tensor
      - rsl: Target continuous remaining shelf life (in days) Tensor [1]
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        split_file: Optional[Union[str, Path]] = "splits.json",
        transform=None,
        use_preprocessing: bool = True,
        preprocessed_dir: Optional[Union[str, Path]] = "preprocessed_224",
        normalize_rsl: bool = False,
        max_rsl: float = 20.0
    ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform
        self.use_preprocessing = use_preprocessing
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir and Path(preprocessed_dir).exists() else None
        self.normalize_rsl = normalize_rsl
        self.max_rsl = max_rsl

        # Load Sample ID splits to guarantee zero data leakage
        self.splits = get_or_create_sample_splits(
            root_dir=self.root_dir,
            split_file=split_file,
            random_seed=42
        )

        if self.split == "all":
            self.allowed_sample_ids = set(
                self.splits["train"] + self.splits["val"] + self.splits["test"]
            )
        else:
            self.allowed_sample_ids = set(self.splits[self.split])

        self.samples = self._index_dataset()

    def _index_dataset(self) -> List[Dict]:
        samples = []
        name_pattern = re.compile(r"^(\d+)-([bBsS])\.", re.IGNORECASE)

        # Iterate over raw structure: Treatment / Day / Image
        for img_path in sorted(self.root_dir.glob("*/*/*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            treatment_name = img_path.parent.parent.name
            day_str = img_path.parent.name

            if treatment_name not in TREATMENT_TO_INDEX or not day_str.isdigit():
                continue

            day = int(day_str)
            match = name_pattern.match(img_path.name)
            if not match:
                continue

            sample_id = int(match.group(1))
            view = "back" if match.group(2).lower() == "b" else "side"

            if sample_id in self.allowed_sample_ids:
                rsl = calculate_rsl(treatment_name, day)
                samples.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "treatment": treatment_name,
                    "treatment_idx": TREATMENT_TO_INDEX[treatment_name],
                    "day": day,
                    "sample_id": sample_id,
                    "view": view,
                    "rsl": rsl
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        meta = self.samples[idx]

        # 1. Load image (from preprocessed cache if available, else process)
        image = None
        if self.preprocessed_dir is not None:
            cached_path = self.preprocessed_dir / meta["treatment"] / str(meta["day"]) / meta["filename"]
            if cached_path.exists():
                image = Image.open(cached_path).convert("RGB")

        if image is None:
            if self.use_preprocessing:
                img_rgb = preprocess_mushroom_image(meta["path"])
                image = Image.fromarray(img_rgb)
            else:
                image = Image.open(meta["path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # 2. Encode treatment as One-Hot vector [3]
        treatment_one_hot = torch.zeros(3, dtype=torch.float32)
        treatment_one_hot[meta["treatment_idx"]] = 1.0

        # 3. Target RSL (Remaining Shelf Life in days)
        rsl_val = meta["rsl"]
        if self.normalize_rsl:
            rsl_val = rsl_val / self.max_rsl

        target_rsl = torch.tensor([rsl_val], dtype=torch.float32)

        return image, treatment_one_hot, target_rsl

    def get_metadata(self, idx: int) -> Dict:
        return self.samples[idx]


# =============================================================================
# 4. DataLoader Factory for RSL
# =============================================================================

def create_rsl_dataloaders(
    root_dir: Union[str, Path] = ".",
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 0,
    preprocessed_dir: Optional[str] = "preprocessed_224",
    strong_aug: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, MushroomRSLDataset]]:
    """Builds Train, Val, and Test DataLoaders for RSL Prediction."""
    train_tf, eval_tf = get_rsl_transforms(img_size=img_size, strong_aug=strong_aug)

    train_ds = MushroomRSLDataset(
        root_dir=root_dir,
        split="train",
        transform=train_tf,
        preprocessed_dir=preprocessed_dir
    )
    val_ds = MushroomRSLDataset(
        root_dir=root_dir,
        split="val",
        transform=eval_tf,
        preprocessed_dir=preprocessed_dir
    )
    test_ds = MushroomRSLDataset(
        root_dir=root_dir,
        split="test",
        transform=eval_tf,
        preprocessed_dir=preprocessed_dir
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    return train_loader, val_loader, test_loader, datasets


if __name__ == "__main__":
    print("Testing MushroomRSLDataset...")
    train_loader, val_loader, test_loader, datasets = create_rsl_dataloaders(
        root_dir=".",
        batch_size=8
    )
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Val samples:   {len(datasets['val'])}")
    print(f"Test samples:  {len(datasets['test'])}")

    images, treatments, rsls = next(iter(train_loader))
    print(f"Batch Image Shape:     {images.shape}")
    print(f"Batch Treatment Shape: {treatments.shape}")
    print(f"Batch Target RSL:      {rsls.squeeze().tolist()[:4]} days")
    print("RSL Dataset loaded successfully!")
