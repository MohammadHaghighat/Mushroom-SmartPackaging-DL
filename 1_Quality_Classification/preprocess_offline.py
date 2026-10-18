"""
Smart Packaging and Spoilage Monitoring of White Button Mushrooms
================================================================
High-Speed Parallel Offline Preprocessing Script

Features:
  - Multi-process parallel execution across all CPU cores (Intel i5-12500H 12-core / 16-thread).
  - Center-Crop (0.65) to eliminate lightbox background borders.
  - LAB-CLAHE on L-Channel to eliminate plastic glare and enhance browning spots.
  - Downscales to target resolution (224x224 px) to reduce 1.7 GB down to ~11.5 MB (99.3% reduction).
  - Automatically copies splits.json and packages the entire dataset into a lightweight ZIP archive.
"""

import os
import time
import shutil
import argparse
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from tqdm import tqdm

from dataset import preprocess_mushroom_image


def _process_single_image(
    img_path_str: str,
    raw_root_str: str,
    out_root_str: str,
    crop_ratio: float,
    clip_limit: float,
    target_size: Optional[int]
) -> bool:
    """
    Worker function executed in parallel processes.
    Reads raw image, applies center-crop + LAB-CLAHE, resizes, and writes to destination.
    """
    img_path = Path(img_path_str)
    raw_root = Path(raw_root_str)
    out_root = Path(out_root_str)

    rel_path = img_path.relative_to(raw_root)
    dest_path = out_root / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        return True

    try:
        # 1. Apply Center-Crop and LAB CLAHE on L-channel
        enhanced_rgb = preprocess_mushroom_image(
            img_path,
            crop_ratio=crop_ratio,
            clip_limit=clip_limit
        )

        # 2. Resize to target dimension (e.g. 224x224)
        if target_size is not None and target_size > 0:
            enhanced_rgb = cv2.resize(
                enhanced_rgb,
                (target_size, target_size),
                interpolation=cv2.INTER_AREA
            )

        # 3. Save as RGB (OpenCV expects BGR to write to disk)
        bgr_to_save = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dest_path), bgr_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def run_parallel_preprocessing(
    raw_root: Path,
    out_root: Path,
    crop_ratio: float = 0.65,
    clip_limit: float = 2.0,
    target_size: int = 224,
    workers: Optional[int] = None,
    create_zip: bool = True
) -> Path:
    start_time = time.time()
    num_cpus = os.cpu_count() or 4
    num_workers = workers if workers and workers > 0 else num_cpus

    print("=" * 80)
    print("PARALLEL OFFLINE PREPROCESSING & COMPRESSION PIPELINE")
    print("=" * 80)
    print(f"Raw Source Directory:   {raw_root}")
    print(f"Destination Directory:  {out_root}")
    print(f"Target Square Size:     {target_size} x {target_size} px")
    print(f"Center Crop Ratio:      {crop_ratio}")
    print(f"CLAHE Clip Limit:       {clip_limit}")
    print(f"Parallel Worker Cores:  {num_workers} (System logical cores: {num_cpus})")

    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Collect all valid raw images
    all_images = sorted([
        str(p) for p in raw_root.glob("*/*/*.*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and not p.name.startswith((".", "_"))
        and not "preprocessed" in p.parts
    ])

    total_images = len(all_images)
    print(f"Discovered {total_images} total images across treatments and days.")

    # 2. Process in parallel using ProcessPoolExecutor
    print(f"\nProcessing {total_images} images with {num_workers} processes in parallel...")
    success_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _process_single_image,
                img_path_str=img_str,
                raw_root_str=str(raw_root),
                out_root_str=str(out_root),
                crop_ratio=crop_ratio,
                clip_limit=clip_limit,
                target_size=target_size
            )
            for img_str in all_images
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            if future.result():
                success_count += 1

    proc_duration = time.time() - start_time
    print(f"\nProcessed {success_count}/{total_images} images in {proc_duration:.2f} seconds ({total_images / proc_duration:.1f} img/sec)!")

    # 3. Copy splits.json into destination if it exists
    splits_src = raw_root / "splits.json"
    if splits_src.exists():
        shutil.copy2(splits_src, out_root / "splits.json")
        print(f"Copied splits.json into {out_root / 'splits.json'}")

    # 4. Create ZIP archive
    zip_path = None
    if create_zip:
        zip_base_name = raw_root / f"mushroom_preprocessed_{target_size}"
        print(f"\nCreating compressed ZIP archive: {zip_base_name}.zip ...")
        zip_file = shutil.make_archive(
            base_name=str(zip_base_name),
            format="zip",
            root_dir=str(out_root)
        )
        zip_path = Path(zip_file)
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Archive successfully created: {zip_path.name}")
        print(f"Archive size: {zip_size_mb:.2f} MB (Reduced by {(1 - (zip_size_mb / 1694)) * 100:.1f}% from 1.7 GB!)")

    total_duration = time.time() - start_time
    print(f"\nAll operations completed in {total_duration:.2f} seconds.")
    print("=" * 80)
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Mushroom Dataset Offline Preprocessor")
    parser.add_argument("--raw-dir", type=str, default=".", help="Raw dataset root directory")
    parser.add_argument("--out-dir", type=str, default="preprocessed_224", help="Output directory for cached images")
    parser.add_argument("--crop-ratio", type=float, default=0.65, help="Center crop ratio (default: 0.65)")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--target-size", type=int, default=224, help="Target square size in pixels (default: 224)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--zip", action="store_true", default=True, help="Create zip archive of preprocessed folder")

    args = parser.parse_args()
    raw_path = Path(args.raw_dir).resolve()
    out_path = Path(args.out_dir).resolve()

    run_parallel_preprocessing(
        raw_root=raw_path,
        out_root=out_path,
        crop_ratio=args.crop_ratio,
        clip_limit=args.clip_limit,
        target_size=args.target_size,
        workers=args.workers,
        create_zip=args.zip
    )
