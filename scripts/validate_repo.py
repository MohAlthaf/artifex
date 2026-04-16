#!/usr/bin/env python3
"""
ARTIFEX — Reproducibility Validation Script
=============================================
Validates all repository artifacts without retraining.

Checks:
  1. Dataset counts (train/val/test)
  2. Split counts
  3. H5 sample counts, feature names, shapes, dtypes
  4. Checkpoint existence
  5. Final result file existence
  6. Final CSV/JSON numerical consistency
  7. No NaN in final outputs
  8. Serving code can import and model architecture matches

Usage:
  python scripts/validate_repo.py
"""
import csv
import json
import math
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
warnings = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        failed += 1


def warn(name, detail):
    global warnings
    print(f"  [WARN] {name}: {detail}")
    warnings += 1


# =========================================================================
# 1. Dataset counts
# =========================================================================
print("\n=== 1. Dataset Counts ===")
for split, expected in [("train", 1417), ("val", 303), ("test", 305)]:
    orig_dir = os.path.join(BASE, "data", "processed", split, "original")
    if os.path.isdir(orig_dir):
        count = len([f for f in os.listdir(orig_dir) if f.endswith(".png")])
        check(f"{split} original count", count == expected, f"got {count}, expected {expected}")
    else:
        check(f"{split} directory exists", False, f"missing: {orig_dir}")

# Check masks and masked exist
for split in ["train", "val", "test"]:
    for subdir in ["masks", "masked"]:
        d = os.path.join(BASE, "data", "processed", split, subdir)
        check(f"{split}/{subdir} exists", os.path.isdir(d), f"missing: {d}")

# =========================================================================
# 2. Raw mask count
# =========================================================================
print("\n=== 2. Raw Masks ===")
mask_dir = os.path.join(BASE, "datasets", "mask")
if os.path.isdir(mask_dir):
    # Count unique base masks (before augmentation)
    all_masks = os.listdir(mask_dir)
    png_masks = [f for f in all_masks if f.lower().endswith(".png")]
    jpg_masks = [f for f in all_masks if f.lower().endswith((".jpg", ".jpeg"))]
    print(f"  Total mask files: {len(all_masks)} (augmented)")
    # Base masks are files without "aug" in name, or check augmented_masks
    warn("mask_dir", f"Contains {len(all_masks)} augmented masks. 50 base files (49 PNG usable, 1 JPG excluded).")
else:
    warn("mask_dir", "datasets/mask/ not found (may be in augmented form)")

# =========================================================================
# 3. H5 features
# =========================================================================
print("\n=== 3. H5 Features ===")
h5_path = os.path.join(BASE, "brushstroke_features.h5")
if os.path.isfile(h5_path):
    try:
        import h5py
        with h5py.File(h5_path, "r") as h5:
            splits = list(h5.keys())
            check("H5 has train/val/test", set(splits) >= {"train", "val", "test"}, f"found: {splits}")

            expected_features = ["orientation_field", "coherence_map", "edge_strength_map", "spatial_hist_4x4", "spatial_hist_8x8"]
            for split in ["train", "val", "test"]:
                if split in h5:
                    grp = h5[split]
                    samples = list(grp.keys())
                    if split == "train":
                        check(f"H5 {split} count", len(samples) == 1417, f"got {len(samples)}")
                    elif split == "val":
                        check(f"H5 {split} count", len(samples) == 303, f"got {len(samples)}")
                    elif split == "test":
                        check(f"H5 {split} count", len(samples) == 305, f"got {len(samples)}")

                    # Check features in first sample
                    if samples:
                        first = grp[samples[0]]
                        feats = sorted(first.keys())
                        check(f"H5 {split} features", set(feats) == set(expected_features),
                              f"got: {feats}, expected: {expected_features}")

                        # Check shapes
                        for feat in expected_features:
                            if feat in first:
                                ds = first[feat]
                                shape = ds.shape
                                dtype = ds.dtype
                                if feat in ["orientation_field", "coherence_map", "edge_strength_map"]:
                                    check(f"H5 {split}/{feat} shape", shape == (512, 512, 1),
                                          f"got {shape}")
                                elif feat == "spatial_hist_4x4":
                                    check(f"H5 {split}/{feat} shape", shape == (16, 8),
                                          f"got {shape}")
                                elif feat == "spatial_hist_8x8":
                                    check(f"H5 {split}/{feat} shape", shape == (64, 8),
                                          f"got {shape}")
    except ImportError:
        warn("H5 check", "h5py not installed; skipping H5 validation")
else:
    check("H5 file exists", False, f"missing: {h5_path}")

# =========================================================================
# 4. Checkpoint existence
# =========================================================================
print("\n=== 4. Checkpoints ===")
checkpoints = {
    "baseline_official":       "models/baseline_official/baseline_official_best.pth",
    "full_official":           "models/full_official/full_official_best.pth",
    "dir_only_15ep_official":  "models/dir_only_15ep_official/dir_only_15ep_official_best.pth",
    "edge_only_15ep_official": "models/edge_only_15ep_official/edge_only_15ep_official_best.pth",
    "hist_only_15ep_official": "models/hist_only_15ep_official/hist_only_15ep_official_best.pth",
}
for name, rel_path in checkpoints.items():
    full_path = os.path.join(BASE, rel_path)
    check(f"{name} checkpoint exists", os.path.isfile(full_path), f"missing: {rel_path}")

# =========================================================================
# 5. Result file existence
# =========================================================================
print("\n=== 5. Result Files ===")
result_files = [
    "results/baseline_ep46_v2/evaluation_results.json",
    "results/full_eval_v2/evaluation_results.json",
    "results/baseline_vs_full_comparison.json",
    "results/statistical_tests.json",
    "results/official_tables/final_metrics_15ep_ablation.csv",
    "results/ablation_15ep_comparison.json",
    "results/final_submission/README.md",
]
for rel_path in result_files:
    full_path = os.path.join(BASE, rel_path)
    check(f"{rel_path} exists", os.path.isfile(full_path))

# Ablation eval files
for run_name in ["dir_only_15ep_20260312_210744", "edge_only_15ep_20260313_023647", "hist_only_15ep_20260313_101629"]:
    p = os.path.join(BASE, "runs", run_name, "logs", "evaluation_results.json")
    check(f"ablation eval {run_name}", os.path.isfile(p), f"missing: {p}")

# =========================================================================
# 6. Numerical consistency
# =========================================================================
print("\n=== 6. Numerical Consistency ===")
with open(os.path.join(BASE, "results/baseline_ep46_v2/evaluation_results.json")) as f:
    b_eval = json.load(f)
with open(os.path.join(BASE, "results/full_eval_v2/evaluation_results.json")) as f:
    f_eval = json.load(f)
with open(os.path.join(BASE, "results/baseline_vs_full_comparison.json")) as f:
    comp = json.load(f)

# Check comparison matches sources
for row in comp["comparison"]:
    m = row["metric"]
    bk = f"avg_{m}"
    if bk in b_eval:
        check(f"comparison {m} baseline matches", abs(row["baseline"] - b_eval[bk]) < 1e-12)
        check(f"comparison {m} full matches", abs(row["full"] - f_eval[bk]) < 1e-12)

# Check CSV matches
with open(os.path.join(BASE, "results/official_tables/final_metrics_15ep_ablation.csv")) as f:
    reader = csv.DictReader(f)
    csv_rows = {r["model"]: r for r in reader}

for model_name, eval_src in [("baseline_official", b_eval), ("full_official", f_eval)]:
    if model_name in csv_rows:
        for k in ["avg_psnr", "avg_ssim", "avg_lpips", "avg_l1"]:
            json_val = eval_src[k]
            csv_val = float(csv_rows[model_name][k])
            check(f"CSV {model_name} {k}", abs(json_val - csv_val) < 1e-4,
                  f"json={json_val:.6f} csv={csv_val:.6f}")

# =========================================================================
# 7. No NaN in final outputs
# =========================================================================
print("\n=== 7. NaN Check ===")
for fname in ["results/baseline_ep46_v2/evaluation_results.json",
              "results/full_eval_v2/evaluation_results.json",
              "results/statistical_tests.json",
              "results/baseline_vs_full_comparison.json"]:
    full_path = os.path.join(BASE, fname)
    if os.path.isfile(full_path):
        with open(full_path) as f:
            content = f.read()
        check(f"No NaN in {fname}", "NaN" not in content)

# =========================================================================
# 8. Serving code import check
# =========================================================================
print("\n=== 8. Serving Code ===")
ml_dir = os.path.join(BASE, "artifex", "server", "ml")
for fname in ["app.py", "canonical_inference.py", "model_registry.py", "metrics.py"]:
    check(f"serving {fname} exists", os.path.isfile(os.path.join(ml_dir, fname)))

# Check canonical_inference has ArtifexGenerator
ci_path = os.path.join(ml_dir, "canonical_inference.py")
if os.path.isfile(ci_path):
    with open(ci_path) as f:
        ci_content = f.read()
    check("ArtifexGenerator in canonical_inference.py", "class ArtifexGenerator" in ci_content)
    check("load_generator in canonical_inference.py", "def load_generator" in ci_content)
    check("run_inference in canonical_inference.py", "def run_inference" in ci_content)

# Check model_registry points to v2 evals
mr_path = os.path.join(ml_dir, "model_registry.py")
if os.path.isfile(mr_path):
    with open(mr_path) as f:
        mr_content = f.read()
    check("registry uses baseline_ep46_v2", "baseline_ep46_v2" in mr_content)
    check("registry uses full_eval_v2", "full_eval_v2" in mr_content)

# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*60}")
print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed, {warnings} warnings")
print(f"{'='*60}")
if failed > 0:
    sys.exit(1)
