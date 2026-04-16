# Final Submission Results Package

This folder contains the **sole authoritative source of truth** for all thesis results.

## Contents

| File | Description |
|------|-------------|
| `baseline_evaluation.json` | Baseline model evaluation (v2, VGG LPIPS, 305 images) |
| `full_evaluation.json` | Full model evaluation (v2, VGG LPIPS, 305 images) |
| `baseline_vs_full_comparison.json` | Head-to-head comparison (regenerated from v2 sources) |
| `statistical_tests.json` | Paired Wilcoxon tests with Bonferroni correction |
| `final_metrics_15ep_ablation.csv` | All 5 models summary metrics |
| `ablation_15ep_comparison.json` | Full 5-model comparison with per-image data |

## Authoritative Source Files

- Baseline eval: `results/baseline_ep46_v2/evaluation_results.json`
- Full eval: `results/full_eval_v2/evaluation_results.json`
- Ablation evals: `runs/{dir,edge,hist}_only_15ep_*/logs/evaluation_results.json`

## What NOT to use

- `results/baseline_ep46/` — v1, LPIPS=NaN
- `results/full_eval/` — v1, LPIPS=NaN
- `results/baseline_ep60/` — superseded by ep46

## Statistical Testing Coverage

| Metric | Tested? | Pairs | Notes |
|--------|---------|-------|-------|
| PSNR | Yes | All 7 pairs | Bonferroni-corrected |
| SSIM | Yes | All 7 pairs | Bonferroni-corrected |
| LPIPS | Yes | All 7 pairs | Bonferroni-corrected |
| L1 | Yes | All 7 pairs | Bonferroni-corrected |
| direction | Yes | baseline vs full only | From v1 per-image data (avg matches v2) |
| edge_strength | Yes | baseline vs full only | From v1 per-image data (avg matches v2) |
| histogram | **No** | — | V1/v2 use different computation; cannot test |
