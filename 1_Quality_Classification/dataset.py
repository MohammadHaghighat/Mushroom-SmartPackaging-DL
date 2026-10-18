"""
Smart Packaging and Spoilage Monitoring of White Button Mushrooms
================================================================
Step 1 & Step 2: Custom PyTorch Dataset, Dynamic Biological Age Labeling,
Data Leakage Prevention via Sample-ID Splitting, and OpenCV Preprocessing.

Dataset Structure:
Root Directory:
  - Control / 1SKEO.2CEO / 2SKEO.1CEO (Treatments)
    - 0, 3, 7, 11, 20, 30 (Storage Days)
      - {SampleID}-{View}.jpg (e.g., 1-b.jpg, 1-s.jpg up to 22-b.jpg, 22-s.jpg)

Biological Quality Classes (Target Labels):
  - Class 0 (Very Healthy): Control Day 0 | 1SKEO.2CEO Days 0, 3 | 2SKEO.1CEO Days 0, 3
  - Class 1 (Healthy):      Control Day 3 | 1SKEO.2CEO Days 7, 11 | 2SKEO.1CEO Days 7, 11
  - Class 2 (Spoiled):      Control Days 7, 11 | 1SKEO.2CEO Day 20 | 2SKEO.1CEO Day 20
  - Class 3 (Very Spoiled):  Control Days 20, 30 | 1SKEO.2CEO Day 30 | 2SKEO.1CEO Day 30
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError:
    # PyTorch will be imported when installed in the user's environment
    torch = None
    Dataset = object
    DataLoader = None
    transforms = None


# =============================================================================
# 1. Biological Quality Mapping & Constants
# =============================================================================

LABEL_MAPPING: Dict[Tuple[str, int], int] = {
    # Class 0 (Very Healthy): 5 conditions (220 images)
    ('Control', 0): 0,
    ('1SKEO.2CEO', 0): 0,
    ('1SKEO.2CEO', 3): 0,
    ('2SKEO.1CEO', 0): 0,
    ('2SKEO.1CEO', 3): 0,

    # Class 1 (Healthy): 5 conditions (220 images)
    ('Control', 3): 1,
    ('1SKEO.2CEO', 7): 1,
    ('1SKEO.2CEO', 11): 1,
    ('2SKEO.1CEO', 7): 1,
    ('2SKEO.1CEO', 11): 1,

    # Class 2 (Spoiled): 4 conditions (176 images)
    ('Control', 7): 2,
    ('Control', 11): 2,
    ('1SKEO.2CEO', 20): 2,
    ('2SKEO.1CEO', 20): 2,

    # Class 3 (Very Spoiled): 4 conditions (176 images)
    ('Control', 20): 3,
    ('Control', 30): 3,
    ('1SKEO.2CEO', 30): 3,
    ('2SKEO.1CEO', 30): 3,
}

CLASS_NAMES: List[str] = [
    "Class 0: Very Healthy",
    "Class 1: Healthy",
    "Class 2: Spoiled",
    "Class 3: Very Spoiled"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# 2. OpenCV Image Preprocessing (Center Crop + LAB-CLAHE)
# =============================================================================

def preprocess_mushroom_image(
    image_input: Union[str, Path, np.ndarray],
    crop_ratio: float = 0.65,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies domain-specific OpenCV preprocessing to mushroom package images:
    1. Center Crop: Removes external lightbox borders and background noise.
    2. LAB Color Space CLAHE on L-Channel:
       - Suppresses specular plastic film glare.
       - Enhances enzymatic browning defects and cap surface shrinkage.
       - Preserves chromatic 'a' and 'b' channels intact to prevent hue distortion.

    Args:
        image_input: File path (str/Path) or pre-loaded BGR image array (np.ndarray).
        crop_ratio: Ratio of central image region to retain (default 0.65).
        clip_limit: CLAHE threshold for contrast limiting (default 2.0).
        tile_grid_size: CLAHE grid partition (default 8x8).

    Returns:
        np.ndarray: Preprocessed RGB image ready for PyTorch transforms or display.
    """
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image from: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input
    else:
        raise TypeError("image_input must be a file path or numpy array")

    h, w = img_bgr.shape[:2]

    # --- Step A: Center Crop ---
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)

    cropped_bgr = img_bgr[start_y : start_y + crop_h, start_x : start_x + crop_w]

    # --- Step B: LAB Color Space Conversion & CLAHE on L-Channel ---
    lab = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

    # Convert back to standard RGB for PyTorch / PIL
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return enhanced_rgb


# =============================================================================
# 3. Sample-ID Partitioning & Data Leakage Prevention
# =============================================================================

def get_or_create_sample_splits(
    root_dir: Union[str, Path],
    split_file: Optional[Union[str, Path]] = None,
    train_count: int = 16,
    val_count: int = 3,
    test_count: int = 3,
    random_seed: int = 42,
    force_recreate: bool = False
) -> Dict[str, List[int]]:
    """
    Splits the 22 physical mushroom samples into Train (16), Val (3), and Test (3).

    CRITICAL LEAKAGE PREVENTION:
    Splitting is performed strictly at the physical Sample-ID level (1 to 22),
    NOT randomly per image. All days (0..30), all treatments (Control, Active),
    and both camera angles (-b, -s) for a given sample ID are confined to the
    same split partition.

    Args:
        root_dir: Dataset root directory.
        split_file: Path to splits.json file. Defaults to `root_dir / 'splits.json'`.
        train_count: Number of sample IDs in Train split (default 16 ~ 72.7%).
        val_count: Number of sample IDs in Val split (default 3 ~ 13.6%).
        test_count: Number of sample IDs in Test split (default 3 ~ 13.6%).
        random_seed: Seed for deterministic, reproducible shuffling.
        force_recreate: If True, re-generates and overwrites splits.json.

    Returns:
        Dict[str, List[int]]: Dictionary with 'train', 'val', 'test' sample ID lists.
    """
    root_path = Path(root_dir)
    if split_file is None:
        split_file = root_path / "splits.json"
    else:
        split_file = Path(split_file)

    if split_file.exists() and not force_recreate:
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        return splits

    # Discover all unique Sample IDs across the dataset
    sample_ids = set()
    pattern = re.compile(r"^(\d+)-[bBsS]\.")

    for img_path in root_path.glob("*/*/*.*"):
        match = pattern.match(img_path.name)
        if match:
            sample_ids.add(int(match.group(1)))

    sorted_ids = sorted(list(sample_ids))
    if not sorted_ids:
        # Fallback to standard 1..22 if glob found no files (e.g., custom root path)
        sorted_ids = list(range(1, 23))

    total_samples = len(sorted_ids)
    assert train_count + val_count + test_count == total_samples, (
        f"Sum of splits ({train_count} + {val_count} + {test_count}) "
        f"must equal total samples ({total_samples})"
    )

    rng = random.Random(random_seed)
    shuffled_ids = sorted_ids.copy()
    rng.shuffle(shuffled_ids)

    splits = {
        "metadata": {
            "random_seed": random_seed,
            "total_samples": total_samples,
            "train_ratio": train_count / total_samples,
            "val_ratio": val_count / total_samples,
            "test_ratio": test_count / total_samples,
        },
        "train": sorted(shuffled_ids[:train_count]),
        "val": sorted(shuffled_ids[train_count : train_count + val_count]),
        "test": sorted(shuffled_ids[train_count + val_count :]),
    }

    # Save to JSON for absolute reproducibility across experiments
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    return splits


# =============================================================================
# 4. PyTorch Custom Dataset
# =============================================================================

class MushroomDataset(Dataset):
    """
    PyTorch Dataset for Mushroom Spoilage Visual Quality Classification.

    Features:
    - Dynamic mapping of (Treatment, Day) to 4 Biological Quality Grades.
    - Zero data leakage: strict Sample-ID subset filtering.
    - Built-in OpenCV Center-Crop + CLAHE pipeline.
    - Multi-view handling: Back (-b) and Side (-s) views are individual samples
      sharing the same biological label.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        split_file: Optional[Union[str, Path]] = None,
        transform=None,
        use_preprocessing: bool = True,
        crop_ratio: float = 0.65,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        random_seed: int = 42,
        preprocessed_dir: Optional[Union[str, Path]] = None
    ):
        """
        Args:
            root_dir: Path to raw dataset root folder.
            split: 'train', 'val', 'test', or 'all'.
            split_file: Optional path to splits.json file.
            transform: PyTorch torchvision transform pipeline.
            use_preprocessing: If True, applies Center-Crop + CLAHE via OpenCV.
            crop_ratio: Center crop ratio.
            clip_limit: CLAHE clipLimit parameter.
            tile_grid_size: CLAHE tileGridSize parameter.
            random_seed: Seed for sample splitting.
            preprocessed_dir: Optional directory with offline-preprocessed images.
        """
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform
        self.use_preprocessing = use_preprocessing
        self.crop_ratio = crop_ratio
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        if self.split not in ("train", "val", "test", "all"):
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', 'test', or 'all'.")

        # Load or generate Sample ID split partition
        self.splits = get_or_create_sample_splits(
            root_dir=self.root_dir,
            split_file=split_file,
            random_seed=random_seed
        )

        if self.split == "all":
            self.allowed_sample_ids = set(
                self.splits["train"] + self.splits["val"] + self.splits["test"]
            )
        else:
            self.allowed_sample_ids = set(self.splits[self.split])

        # Scan and index all valid samples matching the split
        self.samples = self._index_dataset()

    def _index_dataset(self) -> List[Dict]:
        """Scans dataset files, parses metadata, maps biological classes, and filters by split."""
        samples = []
        name_pattern = re.compile(r"^(\d+)-([bBsS])\.", re.IGNORECASE)

        for img_path in sorted(self.root_dir.glob("*/*/*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            treatment_name = img_path.parent.parent.name
            day_str = img_path.parent.name

            if not day_str.isdigit():
                continue
            day = int(day_str)

            # Determine Biological Quality Class (0, 1, 2, 3)
            key = (treatment_name, day)
            if key not in LABEL_MAPPING:
                continue
            target_class = LABEL_MAPPING[key]

            # Parse Sample ID and Camera View from filename
            match = name_pattern.match(img_path.name)
            if not match:
                continue

            sample_id = int(match.group(1))
            view_char = match.group(2).lower()
            view = "back" if view_char == "b" else "side"

            # Filter strictly by physical Sample ID to prevent data leakage
            if sample_id in self.allowed_sample_ids:
                samples.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "treatment": treatment_name,
                    "day": day,
                    "sample_id": sample_id,
                    "view": view,
                    "label": target_class,
                    "class_name": CLASS_NAMES[target_class]
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_meta = self.samples[idx]
        img_path = sample_meta["path"]
        target = sample_meta["label"]

        if self.preprocessed_dir is not None:
            pre_path = self.preprocessed_dir / sample_meta["treatment"] / str(sample_meta["day"]) / sample_meta["filename"]
            if pre_path.exists():
                image = Image.open(pre_path).convert("RGB")
            else:
                # Fallback to on-the-fly preprocessing if cached file missing
                img_rgb = preprocess_mushroom_image(
                    img_path,
                    crop_ratio=self.crop_ratio,
                    clip_limit=self.clip_limit,
                    tile_grid_size=self.tile_grid_size
                )
                image = Image.fromarray(img_rgb)
        elif self.use_preprocessing:
            # Apply OpenCV Center-Crop and LAB CLAHE enhancement on the fly
            img_rgb = preprocess_mushroom_image(
                img_path,
                crop_ratio=self.crop_ratio,
                clip_limit=self.clip_limit,
                tile_grid_size=self.tile_grid_size
            )
            image = Image.fromarray(img_rgb)
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def get_metadata(self, idx: int) -> Dict:
        """Returns the full metadata dictionary for a given index."""
        return self.samples[idx]

    def get_class_distribution(self) -> Dict[int, int]:
        """Calculates the sample counts for each of the 4 biological quality classes."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in self.samples:
            counts[s["label"]] += 1
        return counts


# =============================================================================
# 5. Transforms & DataLoaders Factory
# =============================================================================

def get_transforms(
    img_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD
):
    """
    Returns standard training and validation/test torchvision transform pipelines.

    Training Transforms include data augmentation tailored for defect recognition:
    - Random Horizontal and Vertical Flips (orientation invariance)
    - Random Rotation (±15 degrees)
    - Gentle ColorJitter (brightness and contrast variations)
    """
    if transforms is None:
        raise ImportError("torchvision is required for get_transforms()")

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


def create_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 0,
    random_seed: int = 42,
    use_preprocessing: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, MushroomDataset]]:
    """
    Constructs Train, Validation, and Test PyTorch DataLoaders with strict Sample-ID splits.

    Returns:
        (train_loader, val_loader, test_loader, datasets_dict)
    """
    if torch is None:
        raise ImportError("PyTorch is required for create_dataloaders()")

    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = MushroomDataset(
        root_dir=root_dir,
        split="train",
        transform=train_transform,
        use_preprocessing=use_preprocessing,
        random_seed=random_seed
    )

    val_dataset = MushroomDataset(
        root_dir=root_dir,
        split="val",
        transform=eval_transform,
        use_preprocessing=use_preprocessing,
        random_seed=random_seed
    )

    test_dataset = MushroomDataset(
        root_dir=root_dir,
        split="test",
        transform=eval_transform,
        use_preprocessing=use_preprocessing,
        random_seed=random_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

    return train_loader, val_loader, test_loader, datasets


# =============================================================================
# 6. Verification & Self-Test Script
# =============================================================================

if __name__ == "__main__":
    import sys
    print("=" * 80)
    print("MUSHROOM SPOILAGE DATASET VALIDATION (STEP 1 & STEP 2)")
    print("=" * 80)

    # Use current script directory as workspace root
    workspace_root = Path(__file__).parent.resolve()
    print(f"Dataset Root: {workspace_root}")

    # 1. Generate / Verify Sample-ID Partitioning
    splits = get_or_create_sample_splits(workspace_root, random_seed=42)
    print("\n[1] Physical Sample-ID Partitions (Leakage-Free Splitting):")
    print(f"  * Train Sample IDs ({len(splits['train'])}): {splits['train']}")
    print(f"  * Val   Sample IDs ({len(splits['val'])}):   {splits['val']}")
    print(f"  * Test  Sample IDs ({len(splits['test'])}):  {splits['test']}")

    # Check for zero leakage
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])
    assert train_set.isdisjoint(val_set), "LEAKAGE ERROR: Train and Val overlap!"
    assert train_set.isdisjoint(test_set), "LEAKAGE ERROR: Train and Test overlap!"
    assert val_set.isdisjoint(test_set), "LEAKAGE ERROR: Val and Test overlap!"
    print("  -> Zero Data Leakage Check: PASSED (All sets strictly mutually disjoint)")

    # 2. Instantiate Datasets without PyTorch transforms (to test OpenCV indexing & counts)
    train_ds = MushroomDataset(workspace_root, split="train", use_preprocessing=False)
    val_ds = MushroomDataset(workspace_root, split="val", use_preprocessing=False)
    test_ds = MushroomDataset(workspace_root, split="test", use_preprocessing=False)

    print("\n[2] Dataset Image Distribution across Partitions:")
    print(f"  * Train Images: {len(train_ds)} (Expected: 16 samples * 36 images = 576)")
    print(f"  * Val Images:   {len(val_ds)}   (Expected:  3 samples * 36 images = 108)")
    print(f"  * Test Images:  {len(test_ds)}  (Expected:  3 samples * 36 images = 108)")
    print(f"  * Total Images: {len(train_ds) + len(val_ds) + len(test_ds)} (Expected: 792)")

    # 3. Class Counts Summary
    print("\n[3] Biological Quality Class Distribution:")
    header = f"{'Class':<25} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    t_counts = train_ds.get_class_distribution()
    v_counts = val_ds.get_class_distribution()
    te_counts = test_ds.get_class_distribution()

    for c_idx, c_name in enumerate(CLASS_NAMES):
        total_c = t_counts[c_idx] + v_counts[c_idx] + te_counts[c_idx]
        print(f"{c_name:<25} | {t_counts[c_idx]:<8} | {v_counts[c_idx]:<8} | {te_counts[c_idx]:<8} | {total_c:<8}")

    # 4. Verify OpenCV Preprocessing on a real sample
    print("\n[4] Testing OpenCV Preprocessing (Center-Crop 0.65 + LAB-CLAHE)...")
    sample_item = train_ds.get_metadata(0)
    print(f"  Sample Image: {sample_item['path']}")
    processed_rgb = preprocess_mushroom_image(sample_item["path"])
    print(f"  Raw Image Loaded -> Preprocessed Shape: {processed_rgb.shape}, Dtype: {processed_rgb.dtype}")
    print("  -> Preprocessing Pipeline Check: PASSED")

    print("\nStep 1 & Step 2 implementation successfully verified!")
    print("=" * 80)
