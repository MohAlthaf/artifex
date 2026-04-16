#!/usr/bin/env python3
"""
ARTIFEX — Brushstroke Feature Extraction (H5)

Extracts per-image brushstroke priors from preprocessed Van Gogh images
and stores them in a single HDF5 file with train/val/test groups.

Usage:
  python scripts/extract_brushstroke_h5.py \\
      --processed-dir data/processed \\
      --output        brushstroke_features.h5
"""
import argparse
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

NUM_BINS = 8  # 8 bins × 22.5° = [0, 180) coverage


# ============================================================================
# CORE FEATURE EXTRACTORS
# ============================================================================

def compute_orientation_and_coherence(image_gray, sigma=3.0, flow_offset_deg=90.0):
    """
    Structure tensor orientation + coherence.

    Args:
        image_gray      : (H, W) float32/64 in [0, 1]
        sigma           : Gaussian smoothing for structure tensor components
        flow_offset_deg : add 90° to rotate gradient-normal into stroke FLOW direction

    Returns:
        orientation : (H, W) float32, strictly in [0, 180)
        coherence   : (H, W) float32, in [0, 1]
    """
    Ix = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = gaussian_filter(Ix * Ix, sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma)

    theta_rad = 0.5 * np.arctan2(2.0 * Ixy, (Ixx - Iyy))
    theta_deg = np.degrees(theta_rad)
    theta_deg = (theta_deg + float(flow_offset_deg)) % 180.0

    diff = Ixx - Iyy
    numerator = np.sqrt(diff * diff + 4.0 * Ixy * Ixy)
    denominator = Ixx + Iyy + 1e-8
    coherence = np.clip(numerator / denominator, 0.0, 1.0)

    return theta_deg.astype(np.float32), coherence.astype(np.float32)


def compute_edge_strength_map(image_gray):
    """
    Sobel magnitude, normalized per image via 99th percentile to [0,1].
    """
    sx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    p99 = np.percentile(mag, 99)
    mag = np.clip(mag / (p99 + 1e-8), 0.0, 1.0)
    return mag.astype(np.float32)


def compute_spatial_angular_histograms(orientation_map, coherence_map, grid_size,
                                       coherence_thresh=0.05):
    """
    Coherence-weighted orientation histograms over a grid_size×grid_size partition.

    Returns:
        histograms : (grid_size², 8) float32, each row sums to 1
    """
    h, w = orientation_map.shape
    rh, rw = h // grid_size, w // grid_size
    n_regions = grid_size * grid_size

    histograms = np.zeros((n_regions, NUM_BINS), dtype=np.float64)
    bin_edges = np.linspace(0.0, 180.0, NUM_BINS + 1)

    for i in range(grid_size):
        for j in range(grid_size):
            region_idx = i * grid_size + j
            sh, eh = i * rh, (i + 1) * rh
            sw, ew = j * rw, (j + 1) * rw

            theta = orientation_map[sh:eh, sw:ew].ravel()
            coh = coherence_map[sh:eh, sw:ew].ravel()

            m = coh >= coherence_thresh
            if not np.any(m):
                histograms[region_idx] = 1.0 / NUM_BINS
                continue

            hist, _ = np.histogram(theta[m], bins=bin_edges, weights=coh[m])
            total = hist.sum()
            histograms[region_idx] = hist / total if total > 0 else 1.0 / NUM_BINS

    return histograms.astype(np.float32)


# ============================================================================
# FLOAT16 SAFETY
# ============================================================================

def orientation_float16_safe(orient_f32):
    """Ensure orientation stays strictly in [0,180) after float16 cast."""
    orient_f32 = orient_f32.astype(np.float32, copy=False)
    max_below_180 = np.nextafter(np.float32(180.0), np.float32(0.0))
    orient_f32 = np.minimum(orient_f32, max_below_180)
    orient_f16 = orient_f32.astype(np.float16)
    if np.any(orient_f16 == np.float16(180.0)):
        orient_f16 = orient_f16.copy()
        orient_f16[orient_f16 == np.float16(180.0)] = np.float16(0.0)
    return orient_f16


# ============================================================================
# H5 WRITER
# ============================================================================

def build_brushstroke_h5(proc_dir, h5_path, fast_dev_run=False,
                         sigma=3.0, flow_offset_deg=90.0,
                         coherence_thresh=0.05, compression_level=4):
    """
    Extract priors for train/val/test and save to HDF5.

    Maps:  float16 + gzip
    Hists: float32
    """
    split_dirs = {
        "train": os.path.join(proc_dir, "train/original"),
        "val":   os.path.join(proc_dir, "val/original"),
        "test":  os.path.join(proc_dir, "test/original"),
    }

    print("=" * 70)
    print("BRUSHSTROKE PRIOR EXTRACTION → HDF5")
    print("=" * 70)
    print(f"  Output      : {h5_path}")
    print(f"  fast_dev_run: {fast_dev_run}")
    print(f"  sigma       : {sigma}")
    print(f"  flow_offset : {flow_offset_deg}°")
    print(f"  hist coh th : {coherence_thresh}")
    print()

    with h5py.File(h5_path, "w") as f:
        for split, split_dir in split_dirs.items():
            if not os.path.exists(split_dir):
                print(f"  WARNING: {split_dir} not found — skipping {split}")
                continue

            grp = f.create_group(split)
            files = sorted(fn for fn in os.listdir(split_dir) if fn.endswith(".png"))
            if fast_dev_run:
                files = files[:5]

            print(f"  {split}: {len(files)} images")
            for fn in tqdm(files, desc=f"  {split}"):
                path = os.path.join(split_dir, fn)
                img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
                H, W = img.shape

                orient, coh = compute_orientation_and_coherence(
                    img, sigma=sigma, flow_offset_deg=flow_offset_deg
                )
                edge = compute_edge_strength_map(img)
                h4 = compute_spatial_angular_histograms(orient, coh, 4, coherence_thresh)
                h8 = compute_spatial_angular_histograms(orient, coh, 8, coherence_thresh)

                ig = grp.create_group(fn)

                orient16 = orientation_float16_safe(orient).reshape(H, W, 1)
                coh16 = coh.astype(np.float16).reshape(H, W, 1)
                edge16 = edge.astype(np.float16).reshape(H, W, 1)

                for key, arr in [
                    ("orientation_field", orient16),
                    ("coherence_map", coh16),
                    ("edge_strength_map", edge16),
                ]:
                    ig.create_dataset(
                        key, data=arr,
                        chunks=(H, W, 1),
                        compression="gzip",
                        compression_opts=int(compression_level),
                    )

                ig.create_dataset("spatial_hist_4x4", data=h4.astype(np.float32))
                ig.create_dataset("spatial_hist_8x8", data=h8.astype(np.float32))

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"\nH5 saved → {h5_path}  ({size_mb:.1f} MB)")


# ============================================================================
# H5 VERIFICATION
# ============================================================================

def verify_h5(h5_path, proc_dir):
    """Validate H5 contents: splits, counts, shapes, ranges, histogram sums."""
    print("=" * 70)
    print("H5 VERIFICATION")
    print("=" * 70)

    all_ok = True
    with h5py.File(h5_path, "r") as f:
        print(f"Splits: {list(f.keys())}")

        for split in ["train", "val", "test"]:
            if split not in f:
                print(f"\n  {split}: MISSING FROM H5")
                all_ok = False
                continue

            grp = f[split]
            h5_names = set(grp.keys())
            disk_dir = os.path.join(proc_dir, f"{split}/original")
            disk_names = (
                set(fn for fn in os.listdir(disk_dir) if fn.endswith(".png"))
                if os.path.exists(disk_dir) else set()
            )

            print(f"\n  {split.upper()}:")
            print(f"    Disk images : {len(disk_names)}")
            print(f"    H5 images   : {len(h5_names)}")

            missing = disk_names - h5_names
            extra = h5_names - disk_names
            if missing:
                print(f"    FAIL: {len(missing)} disk images missing from H5")
                all_ok = False
            if extra:
                print(f"    WARN: {len(extra)} H5 entries with no matching disk image")
            if not missing and not extra:
                print("    Alignment: PERFECT MATCH")

            if not h5_names:
                continue

            first = sorted(h5_names)[0]
            ig = grp[first]

            required = {
                "orientation_field", "coherence_map", "edge_strength_map",
                "spatial_hist_4x4", "spatial_hist_8x8",
            }
            if not required.issubset(set(ig.keys())):
                print(f"    FAIL: missing keys in {first}")
                all_ok = False
                continue

            o = ig["orientation_field"][:].astype(np.float32)
            c = ig["coherence_map"][:].astype(np.float32)
            e = ig["edge_strength_map"][:].astype(np.float32)
            h4 = ig["spatial_hist_4x4"][:]
            h8 = ig["spatial_hist_8x8"][:]

            if o.ndim != 3 or o.shape[2] != 1:
                print(f"    FAIL: orientation_field bad shape {o.shape}")
                all_ok = False
            if c.shape != o.shape or e.shape != o.shape:
                print(f"    FAIL: map shape mismatch o={o.shape} c={c.shape} e={e.shape}")
                all_ok = False
            if h4.shape != (16, 8) or h8.shape != (64, 8):
                print(f"    FAIL: hist shapes h4={h4.shape} h8={h8.shape}")
                all_ok = False

            has_180 = bool((o >= 180.0).any())
            if o.min() < 0.0 or has_180:
                print(f"    FAIL: orientation range [{o.min():.4f}, {o.max():.4f}] (180={has_180})")
                all_ok = False
            else:
                print(f"    orientation_field: [{o.min():.4f}, {o.max():.4f}] OK")

            if c.min() < 0.0 or c.max() > 1.001:
                print(f"    FAIL: coherence [{c.min():.4f}, {c.max():.4f}]")
                all_ok = False
            else:
                print(f"    coherence_map:     [{c.min():.4f}, {c.max():.4f}] OK")

            if e.min() < 0.0 or e.max() > 1.001:
                print(f"    FAIL: edge [{e.min():.4f}, {e.max():.4f}]")
                all_ok = False
            else:
                print(f"    edge_strength_map: [{e.min():.4f}, {e.max():.4f}] OK")

            if not np.allclose(h4.sum(axis=1), 1.0, atol=1e-4):
                print("    FAIL: spatial_hist_4x4 rows do not sum to 1")
                all_ok = False
            if not np.allclose(h8.sum(axis=1), 1.0, atol=1e-4):
                print("    FAIL: spatial_hist_8x8 rows do not sum to 1")
                all_ok = False
            else:
                print("    histograms: row sums OK")

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"\n  File size: {size_mb:.1f} MB")
    print("H5 verification:", "ALL CHECKS PASSED" if all_ok else "FAILURES FOUND")
    print("=" * 70)
    return all_ok


# ============================================================================
# SANITY CHECK (visual)
# ============================================================================

def run_sanity_checks(proc_dir, sigma=3.0, flow_offset_deg=90.0, coherence_thresh=0.05):
    sample_path = os.path.join(proc_dir, "train/original/vangogh_train_0000.png")
    if not os.path.exists(sample_path):
        print("Sample not found — skipping sanity check")
        return

    print("=" * 70)
    print("SANITY CHECK — vangogh_train_0000.png")
    print("=" * 70)

    img = np.array(Image.open(sample_path).convert("L"), dtype=np.float32) / 255.0
    orient, coh = compute_orientation_and_coherence(img, sigma=sigma, flow_offset_deg=flow_offset_deg)
    edge = compute_edge_strength_map(img)
    h4 = compute_spatial_angular_histograms(orient, coh, 4, coherence_thresh)
    h8 = compute_spatial_angular_histograms(orient, coh, 8, coherence_thresh)

    assert orient.min() >= 0.0, f"FAIL: orientation min = {orient.min()}"
    assert orient.max() < 180.0, f"FAIL: orientation max = {orient.max()}"
    n_unique = len(np.unique(orient))
    assert n_unique > 10_000, f"FAIL: only {n_unique} unique values — looks quantized"
    assert 0.0 <= coh.min() and coh.max() <= 1.0
    assert 0.0 <= edge.min() and edge.max() <= 1.001
    assert h4.shape == (16, 8) and h8.shape == (64, 8)
    assert np.allclose(h4.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(h8.sum(axis=1), 1.0, atol=1e-5)

    print(f"  orientation  : [{orient.min():.4f}, {orient.max():.4f}], {n_unique} unique  OK")
    print(f"  coherence    : [{coh.min():.4f}, {coh.max():.4f}]  OK")
    print(f"  edge_strength: [{edge.min():.4f}, {edge.max():.4f}]  OK")
    print(f"  hist_4x4     : {h4.shape}, row sums OK")
    print(f"  hist_8x8     : {h8.shape}, row sums OK")
    print("All sanity checks PASSED\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="ARTIFEX brushstroke feature extraction → HDF5")
    p.add_argument("--processed-dir", required=True,
                   help="Processed data dir (e.g. data/processed)")
    p.add_argument("--output", default="brushstroke_features.h5",
                   help="Output H5 path (default: brushstroke_features.h5)")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Structure tensor Gaussian sigma (default: 3.0)")
    p.add_argument("--flow-offset", type=float, default=90.0,
                   help="Orientation offset degrees (default: 90 = stroke flow)")
    p.add_argument("--coherence-thresh", type=float, default=0.05,
                   help="Histogram coherence threshold (default: 0.05)")
    p.add_argument("--compression", type=int, default=4,
                   help="Gzip compression level 1-9 (default: 4)")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Process only 5 images per split (for testing)")
    p.add_argument("--sanity-check", action="store_true",
                   help="Run visual sanity check on first train image")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify an existing H5 file, don't build")
    args = p.parse_args()

    if args.verify_only:
        verify_h5(args.output, args.processed_dir)
        return

    if args.sanity_check:
        run_sanity_checks(args.processed_dir, sigma=args.sigma,
                          flow_offset_deg=args.flow_offset,
                          coherence_thresh=args.coherence_thresh)
        return

    build_brushstroke_h5(
        proc_dir=args.processed_dir,
        h5_path=args.output,
        fast_dev_run=args.fast_dev_run,
        sigma=args.sigma,
        flow_offset_deg=args.flow_offset,
        coherence_thresh=args.coherence_thresh,
        compression_level=args.compression,
    )
    verify_h5(args.output, args.processed_dir)


if __name__ == "__main__":
    main()
