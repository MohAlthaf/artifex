#!/usr/bin/env python3
"""
ARTIFEX — Dataset Preprocessing Pipeline

Steps: discover raw Van Gogh images, resize to 512×512, split train/val/test
(70/15/15, seed=42), distribute masks, augment masks, create damaged images.

Usage:
  python scripts/preprocess_dataset.py \\
      --raw-images  datasets/VanGoghPaintingsData \\
      --mask-dir    datasets/mask \\
      --output-dir  data/processed
"""
import argparse
import json
import os
import random
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.ndimage import binary_erosion, binary_dilation
from tqdm import tqdm


# ============================================================================
# 1. IMAGE DISCOVERY
# ============================================================================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def discover_images(root_dir):
    """Recursively find all image files under *root_dir*."""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in VALID_EXTENSIONS:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


# ============================================================================
# 2. RESIZE & SPLIT
# ============================================================================

def preprocess_image(input_path, output_path, target_size=(512, 512)):
    """Resize a single image to *target_size* via LANCZOS and save as PNG."""
    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        img.save(output_path, "PNG")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_images(image_paths, save_dir, prefix):
    """Batch-resize a list of image paths into *save_dir*."""
    os.makedirs(save_dir, exist_ok=True)
    success = 0
    for idx, img_path in enumerate(tqdm(image_paths, desc=f"  {prefix}")):
        out = os.path.join(save_dir, f"{prefix}_{idx:04d}.png")
        if preprocess_image(img_path, out):
            success += 1
    print(f"  Processed {success}/{len(image_paths)} images → {save_dir}")


def split_and_process(all_image_paths, output_dir, seed, target_size):
    """Shuffle, split 70/15/15, resize and save to train/val/test."""
    rng = random.Random(seed)
    paths = list(all_image_paths)
    rng.shuffle(paths)

    n = len(paths)
    t_end = int(n * 0.70)
    v_end = t_end + int(n * 0.15)

    splits = {
        "train": paths[:t_end],
        "val": paths[t_end:v_end],
        "test": paths[v_end:],
    }
    for name, sp in splits.items():
        print(f"  {name}: {len(sp)} images")

    for name, sp in splits.items():
        save_dir = os.path.join(output_dir, name, "original")
        prefix = f"vangogh_{name}"
        process_images(sp, save_dir, prefix)


# ============================================================================
# 3. MASK DISTRIBUTION
# ============================================================================

def distribute_masks(mask_dir, output_dir, seed):
    """Copy base masks into train/val/test with a 70/15/15 split."""
    all_masks = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    print(f"\n  Total base masks: {len(all_masks)}")

    rng = random.Random(seed)
    rng.shuffle(all_masks)

    t_end = int(len(all_masks) * 0.70)
    v_end = t_end + int(len(all_masks) * 0.15)
    split_map = {
        "train": all_masks[:t_end],
        "val": all_masks[t_end:v_end],
        "test": all_masks[v_end:],
    }

    for name, masks in split_map.items():
        dest = os.path.join(output_dir, name, "masks")
        os.makedirs(dest, exist_ok=True)
        for m in masks:
            shutil.copy(os.path.join(mask_dir, m), os.path.join(dest, m))
        print(f"  {name} masks: {len(masks)}")


# ============================================================================
# 4. MASK AUGMENTATION
# ============================================================================

def load_mask(path, size=(512, 512)):
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    mask = np.array(mask)
    return (mask > 128).astype(np.uint8) * 255


def save_mask(mask, path):
    Image.fromarray(mask.astype(np.uint8)).save(path)


def horizontal_flip(img):
    return np.fliplr(img)


def vertical_flip(img):
    return np.flipud(img)


def rotate(img, angle):
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)


def scale(img, factor):
    h, w = img.shape
    new_size = (int(w * factor), int(h * factor))
    scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
    if factor < 1.0:
        pad_w = (w - new_size[0]) // 2
        pad_h = (h - new_size[1]) // 2
        padded = np.zeros((h, w), dtype=np.uint8)
        padded[pad_h : pad_h + new_size[1], pad_w : pad_w + new_size[0]] = scaled
        return padded
    else:
        sx = (new_size[0] - w) // 2
        sy = (new_size[1] - h) // 2
        return scaled[sy : sy + h, sx : sx + w]


def elastic_deformation(image, alpha, sigma):
    rs = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((rs.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rs.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    distorted = map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)
    return (distorted > 128).astype(np.uint8) * 255


def adjust_mask_coverage(mask, target_coverage, tol=0.02, max_iter=50):
    h, w = mask.shape
    total = h * w
    adjusted = mask.copy()
    for _ in range(max_iter):
        current = np.sum(adjusted > 0) / total
        if abs(current - target_coverage) < tol:
            break
        if current > target_coverage:
            adjusted = binary_erosion(adjusted, structure=np.ones((3, 3))).astype(np.uint8) * 255
        else:
            adjusted = binary_dilation(adjusted, structure=np.ones((3, 3))).astype(np.uint8) * 255
    return adjusted


def augment_mask(mask):
    aug = [mask, horizontal_flip(mask), vertical_flip(mask)]
    for a in [90, 180, 270]:
        aug.append(rotate(mask, a))
    for s in [0.8, 1.2]:
        aug.append(scale(mask, s))
    aug.append(elastic_deformation(mask, alpha=8, sigma=4))
    return aug


def augment_all_masks(input_folder, output_folder, coverage_targets=(0.15, 0.30, 0.45)):
    """Augment base masks and adjust to target coverage levels."""
    masks = sorted(f for f in os.listdir(input_folder) if f.endswith(".png"))
    os.makedirs(output_folder, exist_ok=True)
    idx = 0
    for mask_file in tqdm(masks, desc=f"  Augmenting {os.path.basename(input_folder)}"):
        mask = load_mask(os.path.join(input_folder, mask_file))
        for aug_idx, aug_mask in enumerate(augment_mask(mask)):
            for cov in coverage_targets:
                adj = adjust_mask_coverage(aug_mask, cov, tol=0.02)
                out_name = f"{os.path.splitext(mask_file)[0]}_aug{aug_idx}_cov{int(cov * 100)}.png"
                save_mask(adj, os.path.join(output_folder, out_name))
                idx += 1
    print(f"  Generated {idx} augmented masks → {output_folder}")


# ============================================================================
# 5. DAMAGED IMAGE CREATION
# ============================================================================

def create_damaged_images(output_dir, split, seed):
    """Apply randomly-shifted masks to originals, producing corrupted images."""
    original_dir = os.path.join(output_dir, split, "original")
    masked_dir = os.path.join(output_dir, split, "masked")
    aug_mask_dir = os.path.join(output_dir, split, "augmented_masks")
    mask_save_dir = os.path.join(output_dir, split, "masks")
    metadata_path = os.path.join(output_dir, split, "metadata.json")

    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    image_files = sorted(f for f in os.listdir(original_dir) if f.endswith(".png"))
    aug_masks = sorted(f for f in os.listdir(aug_mask_dir) if f.endswith(".png"))
    if not aug_masks:
        print(f"  WARNING: no augmented masks found in {aug_mask_dir}")
        return

    # Pre-load masks
    mask_cache = {}
    for mf in tqdm(aug_masks, desc=f"  Loading masks ({split})"):
        mask_cache[mf] = load_mask(os.path.join(aug_mask_dir, mf))

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    metadata = []
    for img_file in tqdm(image_files, desc=f"  Damaging ({split})"):
        original = np.array(Image.open(os.path.join(original_dir, img_file)).convert("RGB")) / 255.0

        mask_file = rng.choice(aug_masks)
        mask = mask_cache[mask_file]

        # Random spatial shift
        dx = rng.randint(-50, 50)
        dy = rng.randint(-50, 50)
        h, w = mask.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        corrupted = original.copy()
        mask_bool = shifted > 0
        corrupted[mask_bool] = 0  # standard removal corruption

        Image.fromarray((corrupted * 255).astype(np.uint8)).save(
            os.path.join(masked_dir, img_file)
        )
        save_mask(shifted, os.path.join(mask_save_dir, img_file))

        coverage = float(np.sum(mask_bool) / (512 * 512) * 100)
        metadata.append({
            "image_name": img_file,
            "mask_used": mask_file,
            "coverage_percent": round(coverage, 2),
            "dx": dx,
            "dy": dy,
        })

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {split}: {len(image_files)} damaged images created")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="ARTIFEX dataset preprocessing pipeline")
    p.add_argument("--raw-images", required=True,
                   help="Root directory of raw Van Gogh images (e.g. datasets/VanGoghPaintingsData)")
    p.add_argument("--mask-dir", required=True,
                   help="Directory of base mask PNGs (e.g. datasets/mask)")
    p.add_argument("--output-dir", default="data/processed",
                   help="Output directory for processed data (default: data/processed)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--target-size", type=int, default=512, help="Image size (default: 512)")
    p.add_argument("--coverage-targets", nargs="+", type=float, default=[0.15, 0.30, 0.45],
                   help="Mask coverage targets (default: 0.15 0.30 0.45)")
    p.add_argument("--skip-resize", action="store_true",
                   help="Skip image resize step (if already done)")
    p.add_argument("--skip-masks", action="store_true",
                   help="Skip mask distribution + augmentation")
    p.add_argument("--skip-damage", action="store_true",
                   help="Skip damaged image creation")
    args = p.parse_args()

    size = (args.target_size, args.target_size)

    # Step 1: Create folder structure
    print("=" * 60)
    print("ARTIFEX PREPROCESSING PIPELINE")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        for sub in ["original", "masked", "masks", "augmented_masks"]:
            os.makedirs(os.path.join(args.output_dir, split, sub), exist_ok=True)

    # Step 2: Discover, resize, split images
    if not args.skip_resize:
        print("\n[1/4] Discovering raw images...")
        all_paths = discover_images(args.raw_images)
        print(f"  Found {len(all_paths)} images")
        print("\n[2/4] Resizing and splitting (70/15/15)...")
        split_and_process(all_paths, args.output_dir, args.seed, size)
    else:
        print("\n[1-2/4] Skipping image resize (--skip-resize)")

    # Step 3: Distribute and augment masks
    if not args.skip_masks:
        print("\n[3/4] Distributing base masks...")
        distribute_masks(args.mask_dir, args.output_dir, args.seed)
        print("\n  Augmenting masks...")
        for split in ["train", "val", "test"]:
            base_dir = os.path.join(args.output_dir, split, "masks")
            aug_dir = os.path.join(args.output_dir, split, "augmented_masks")
            if os.path.isdir(base_dir) and os.listdir(base_dir):
                augment_all_masks(base_dir, aug_dir, tuple(args.coverage_targets))
    else:
        print("\n[3/4] Skipping mask distribution (--skip-masks)")

    # Step 4: Create damaged images
    if not args.skip_damage:
        print("\n[4/4] Creating damaged images...")
        split_seeds = {"train": args.seed, "val": args.seed + 81, "test": args.seed + 414}
        for split in ["train", "val", "test"]:
            create_damaged_images(args.output_dir, split, split_seeds[split])
    else:
        print("\n[4/4] Skipping damage creation (--skip-damage)")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
