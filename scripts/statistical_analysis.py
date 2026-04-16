#!/usr/bin/env python3
"""
ARTIFEX — Statistical Significance Analysis

Paired Wilcoxon signed-rank tests on per-image metrics across model pairs.
Brushstroke metrics tested for baseline vs full only.

Usage:
  python scripts/statistical_analysis.py [--output results/statistical_tests.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Evaluation JSONs — all must have a "per_image" dict
EVAL_PATHS = {
    "baseline":  os.path.join(BASE, "results/baseline_ep46_v2/evaluation_results.json"),
    "full":      os.path.join(BASE, "results/full_eval_v2/evaluation_results.json"),
    "dir_only":  None,  # resolved by scanning runs/
    "edge_only": None,
    "hist_only": None,
}

# V1 evaluation files with per-image brushstroke data (direction, edge_strength)
# These are the ONLY source of per-image brushstroke data.
# Their direction/edge_strength averages match v2 exactly.
BRUSHSTROKE_EVAL_PATHS = {
    "baseline": os.path.join(BASE, "results/baseline_ep46/evaluation_results.json"),
    "full":     os.path.join(BASE, "results/full_eval/full_best/evaluation_results.json"),
}

METRICS = ["psnr", "ssim", "lpips", "l1"]

# Brushstroke metrics testable from v1 per-image data (baseline vs full only)
BRUSHSTROKE_METRICS = ["direction", "edge_strength"]

# Pairs to compare for standard metrics
COMPARISONS = [
    ("baseline", "full"),
    ("baseline", "dir_only"),
    ("baseline", "edge_only"),
    ("baseline", "hist_only"),
    ("full", "dir_only"),
    ("full", "edge_only"),
    ("full", "hist_only"),
]

# Pairs testable for brushstroke metrics (per-image data available)
BRUSHSTROKE_COMPARISONS = [
    ("baseline", "full"),
]


def resolve_ablation_paths():
    """Find the actual run directories for 15-ep ablation models."""
    runs_dir = os.path.join(BASE, "runs")
    if not os.path.isdir(runs_dir):
        return
    for entry in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, entry, "logs", "evaluation_results.json")
        if not os.path.exists(path):
            continue
        if "dir_only_15ep" in entry and EVAL_PATHS["dir_only"] is None:
            EVAL_PATHS["dir_only"] = path
        elif "edge_only_15ep" in entry and EVAL_PATHS["edge_only"] is None:
            EVAL_PATHS["edge_only"] = path
        elif "hist_only_15ep" in entry and EVAL_PATHS["hist_only"] is None:
            EVAL_PATHS["hist_only"] = path


def load_per_image(path):
    """Load per-image metrics dict from evaluation JSON."""
    with open(path) as f:
        data = json.load(f)
    pi = data.get("per_image")
    if pi is None:
        print(f"  WARNING: no per_image key in {path}")
        return {}
    return pi


def extract_metric_array(per_image, metric, image_keys):
    """Extract metric values in consistent image order."""
    vals = []
    for k in image_keys:
        entry = per_image.get(k, {})
        v = entry.get(metric)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vals.append(np.nan)
        else:
            vals.append(float(v))
    return np.array(vals)


def main():
    p = argparse.ArgumentParser(description="ARTIFEX statistical significance tests")
    p.add_argument("--output", default=os.path.join(BASE, "results/statistical_tests.json"),
                   help="Output JSON path")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (default: 0.05)")
    args = p.parse_args()

    resolve_ablation_paths()

    # Validate paths
    for name, path in EVAL_PATHS.items():
        if path is None or not os.path.exists(path):
            print(f"ERROR: evaluation file not found for {name}: {path}")
            sys.exit(1)

    # Load all per-image data (standard metrics)
    print("Loading per-image metrics (standard: PSNR, SSIM, LPIPS, L1)...")
    all_data = {}
    for name, path in EVAL_PATHS.items():
        all_data[name] = load_per_image(path)
        print(f"  {name:12s}: {len(all_data[name])} images from {os.path.relpath(path, BASE)}")

    # Load brushstroke per-image data (from v1 evals)
    print("\nLoading per-image brushstroke metrics (direction, edge_strength from v1 evals)...")
    brushstroke_data = {}
    for name, path in BRUSHSTROKE_EVAL_PATHS.items():
        if os.path.exists(path):
            brushstroke_data[name] = load_per_image(path)
            print(f"  {name:12s}: {len(brushstroke_data[name])} images from {os.path.relpath(path, BASE)}")
        else:
            print(f"  {name:12s}: MISSING — {path}")

    # Find common image keys across all models (standard metrics)
    common_keys = None
    for name, pi in all_data.items():
        keys = set(pi.keys())
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = sorted(common_keys)
    print(f"\nCommon images across all models (standard): {len(common_keys)}")

    if len(common_keys) < 10:
        print("ERROR: too few common images for statistical testing")
        sys.exit(1)

    # Find common keys for brushstroke testing
    brushstroke_common = None
    for name, pi in brushstroke_data.items():
        keys = set(pi.keys())
        brushstroke_common = keys if brushstroke_common is None else brushstroke_common & keys
    brushstroke_common = sorted(brushstroke_common) if brushstroke_common else []
    print(f"Common images for brushstroke testing: {len(brushstroke_common)}")

    # Total comparisons for Bonferroni: 7 standard pairs * 4 metrics + brushstroke tests
    n_brushstroke_tests = len(BRUSHSTROKE_COMPARISONS) * len(BRUSHSTROKE_METRICS)
    total_tests = len(COMPARISONS) * len(METRICS) + n_brushstroke_tests
    bonferroni_factor = total_tests

    # Run paired tests — standard metrics
    print(f"\n{'='*80}")
    print("PAIRED WILCOXON SIGNED-RANK TESTS")
    print(f"{'='*80}")
    print(f"N = {len(common_keys)} paired observations per test")
    print(f"Significance level: alpha = {args.alpha}")
    print(f"Correction: Bonferroni ({total_tests} total tests)")
    print()

    results = {"n_images": len(common_keys), "alpha": args.alpha,
               "bonferroni_factor": bonferroni_factor,
               "total_tests": total_tests,
               "standard_metric_tests": len(COMPARISONS) * len(METRICS),
               "brushstroke_metric_tests": n_brushstroke_tests,
               "tests": []}

    for metric in METRICS:
        print(f"--- {metric.upper()} ---")
        higher_better = metric in ("psnr", "ssim")

        for model_a, model_b in COMPARISONS:
            a_vals = extract_metric_array(all_data[model_a], metric, common_keys)
            b_vals = extract_metric_array(all_data[model_b], metric, common_keys)

            valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
            a_clean = a_vals[valid]
            b_clean = b_vals[valid]

            if len(a_clean) < 10:
                print(f"  {model_a} vs {model_b}: insufficient valid pairs ({len(a_clean)})")
                continue

            diff = b_clean - a_clean
            mean_a = float(np.mean(a_clean))
            mean_b = float(np.mean(b_clean))
            mean_diff = float(np.mean(diff))

            try:
                stat_val, p_raw = stats.wilcoxon(a_clean, b_clean, alternative="two-sided")
            except ValueError:
                stat_val, p_raw = 0.0, 1.0

            p_adj = min(p_raw * bonferroni_factor, 1.0)
            sig = p_adj < args.alpha

            n_valid = len(a_clean)
            r_effect = 1 - (2 * stat_val) / (n_valid * (n_valid + 1)) if n_valid > 0 else 0.0

            if higher_better:
                winner = model_b if mean_diff > 0 else model_a
            else:
                winner = model_b if mean_diff < 0 else model_a

            flag = "***" if sig else "   "
            print(f"  {model_a:12s} vs {model_b:12s}: "
                  f"mean_a={mean_a:.4f} mean_b={mean_b:.4f} "
                  f"diff={mean_diff:+.4f} p_adj={p_adj:.4e} {flag}")

            results["tests"].append({
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "n_valid": int(n_valid),
                "mean_a": round(mean_a, 6),
                "mean_b": round(mean_b, 6),
                "mean_diff": round(mean_diff, 6),
                "wilcoxon_stat": round(float(stat_val), 4),
                "p_raw": round(float(p_raw), 8),
                "p_bonferroni": round(float(p_adj), 8),
                "significant": bool(sig),
                "effect_size_r": round(float(r_effect), 4),
                "winner": winner,
                "higher_is_better": higher_better,
            })

        print()

    # Run paired tests — brushstroke metrics (baseline vs full only)
    if brushstroke_common and len(brushstroke_common) >= 10:
        print(f"--- BRUSHSTROKE METRICS (baseline vs full only, N={len(brushstroke_common)}) ---")
        for metric in BRUSHSTROKE_METRICS:
            higher_better = False  # lower is better for all brushstroke error metrics

            for model_a, model_b in BRUSHSTROKE_COMPARISONS:
                a_vals = extract_metric_array(brushstroke_data[model_a], metric, brushstroke_common)
                b_vals = extract_metric_array(brushstroke_data[model_b], metric, brushstroke_common)

                valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
                a_clean = a_vals[valid]
                b_clean = b_vals[valid]

                if len(a_clean) < 10:
                    print(f"  {model_a} vs {model_b} [{metric}]: insufficient valid pairs ({len(a_clean)})")
                    continue

                mean_a = float(np.mean(a_clean))
                mean_b = float(np.mean(b_clean))
                mean_diff = float(np.mean(b_clean - a_clean))

                try:
                    stat_val, p_raw = stats.wilcoxon(a_clean, b_clean, alternative="two-sided")
                except ValueError:
                    stat_val, p_raw = 0.0, 1.0

                p_adj = min(p_raw * bonferroni_factor, 1.0)
                sig = p_adj < args.alpha

                n_valid = len(a_clean)
                r_effect = 1 - (2 * stat_val) / (n_valid * (n_valid + 1)) if n_valid > 0 else 0.0

                winner = model_b if mean_diff < 0 else model_a

                flag = "***" if sig else "   "
                print(f"  {model_a:12s} vs {model_b:12s} [{metric:15s}]: "
                      f"mean_a={mean_a:.4f} mean_b={mean_b:.4f} "
                      f"diff={mean_diff:+.4f} p_adj={p_adj:.4e} {flag}")

                results["tests"].append({
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_valid": int(n_valid),
                    "mean_a": round(mean_a, 6),
                    "mean_b": round(mean_b, 6),
                    "mean_diff": round(mean_diff, 6),
                    "wilcoxon_stat": round(float(stat_val), 4),
                    "p_raw": round(float(p_raw), 8),
                    "p_bonferroni": round(float(p_adj), 8),
                    "significant": bool(sig),
                    "effect_size_r": round(float(r_effect), 4),
                    "winner": winner,
                    "higher_is_better": higher_better,
                    "data_source": "v1_evaluation_per_image",
                    "data_source_note": "Per-image values from v1 evaluation (avg matches v2). Tested for baseline vs full only.",
                })
        print()
    else:
        print("WARNING: brushstroke per-image data insufficient for testing")
        print()

    # Add untested metrics note
    results["untested_metrics"] = {
        "histogram": "Not statistically tested. V1 and v2 evaluations use different histogram computation methods (v1 avg=0.057, v2 avg=0.328). Per-image v1 values do not correspond to the v2 metric scale.",
        "direction_ablations": "Direction tested only for baseline vs full. Ablation models lack per-image brushstroke data.",
        "edge_strength_ablations": "Edge strength tested only for baseline vs full. Ablation models lack per-image brushstroke data.",
    }

    # Summary table
    print(f"{'='*80}")
    print("SUMMARY: Significant improvements over baseline (Bonferroni-corrected)")
    print(f"{'='*80}")
    for t in results["tests"]:
        if t["model_a"] == "baseline" and t["significant"]:
            direction = "better" if (
                (t["higher_is_better"] and t["mean_diff"] > 0) or
                (not t["higher_is_better"] and t["mean_diff"] < 0)
            ) else "worse"
            print(f"  {t['model_b']:12s} {t['metric']:6s}: {direction} (p={t['p_bonferroni']:.2e}, r={t['effect_size_r']:.3f})")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
