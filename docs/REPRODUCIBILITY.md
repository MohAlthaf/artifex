# ARTIFEX — Reproducibility Guide

## Environment Setup

```bash
# Python 3.13 on macOS Apple Silicon (M2)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Device:** MPS (Apple Silicon). The notebook auto-detects the device in cell 2. For CUDA, install the appropriate PyTorch build.

## Data

Place the preprocessed dataset at `data/processed/` with structure:

```
data/processed/
  train/{original,masked,masks}/
  val/{original,masked,masks}/
  test/{original,masked,masks}/
```

Place the brushstroke feature file at `brushstroke_features.h5` in the project root.

## Canonical Notebook

All training, evaluation, and ablation work is done inside:

```
artifex_FULLY_FIXED_M2_PATHS.ipynb
```

### Cell execution order for full reproduction

1. **Cells 1–14:** Environment, config, data loading, model definitions, losses
2. **Cells 15–16:** Training loop definitions (`train_epoch`, `validate_epoch`, `train_full_model`)
3. **Cell 17:** Start baseline training (already completed — see `models/baseline_official/`)
4. **Cell 18:** LPIPS setup (VGG backbone — see LPIPS note below)
5. **Cell 19:** Evaluation function (`evaluate_on_test_set`)
6. **Cells 27–29:** Ablation runner, baseline warm-start
7. **Cell 42:** `verify_ablation_setup()` — pre-train sanity checks

### 15-Epoch Short Ablation Suite (cells 50–59)

8.  **Cell 51:** Re-evaluate `baseline_official` and `full_official` with fixed LPIPS
9.  **Cell 52:** dir_only 15-ep resume prep (epoch 5 → 15)
10. **Cell 53:** dir_only training call
11. **Cell 54:** edge_only 15-ep prep + warm-start + verify
12. **Cell 55:** edge_only training call
13. **Cell 56:** hist_only 15-ep prep + warm-start + verify
14. **Cell 57:** hist_only training call
15. **Cell 58:** Evaluate & freeze all three 15-ep ablation checkpoints
16. **Cell 59:** Generate comparison JSON + CSV artifacts

## LPIPS Note

Cell 18 uses `lpips.LPIPS(net='vgg')` instead of the more common `net='alex'`. This is because the AlexNet weights (~233 MB) are not cached locally and the download consistently times out. VGG16 weights (~528 MB) are already present at `~/.cache/torch/hub/checkpoints/vgg16-397923af.pth`.

The sanity check uses a 64×64 zero-tensor (VGG-LPIPS requires input ≥ 16×16 due to 5 pooling layers). An earlier version used 8×8, which silently crashed and set `LPIPS_AVAILABLE = False`, producing NaN everywhere.

**Important:** VGG-based LPIPS values are on a different numerical scale than AlexNet-based values. All comparisons in this project use the VGG backbone consistently.

## Frozen Official Artifacts

These checkpoints must **not** be modified:

| Artifact                 | Path                                                                     | Notes                       |
| ------------------------ | ------------------------------------------------------------------------ | --------------------------- |
| Baseline official        | `models/baseline_official/baseline_official_best.pth`                    | Epoch 46, val_loss=0.300352 |
| Full official            | `models/full_official/full_official_best.pth`                            | Epoch 3, metric wins 7/10   |
| dir_only 15-ep official  | `models/dir_only_15ep_official/dir_only_15ep_official_best.pth`          | Epoch 15 (resumed from 5)   |
| edge_only 15-ep official | `models/edge_only_15ep_official/edge_only_15ep_official_best.pth`        | Epoch 5, baseline warm-start|
| hist_only 15-ep official | `models/hist_only_15ep_official/hist_only_15ep_official_best.pth`        | Epoch 5, baseline warm-start|
| Baseline eval (original) | `results/baseline_ep46/evaluation_results.json`                          | LPIPS=NaN (AlexNet)         |
| Full eval (original)     | `results/full_eval/full_best/evaluation_results.json`                    | LPIPS=NaN (AlexNet)         |
| Baseline eval (v2)       | `results/baseline_ep46_v2/evaluation_results.json`                       | LPIPS valid (VGG)           |
| Full eval (v2)           | `results/full_eval_v2/evaluation_results.json`                           | LPIPS valid (VGG)           |
| 15-ep comparison         | `results/ablation_15ep_comparison.json`                                  | All 5 models, LPIPS valid   |
| 15-ep CSV table          | `results/official_tables/final_metrics_15ep_ablation.csv`                | 6 rows, all metrics         |

## Ablation Naming Convention

- `*_15ep_*` — 15-epoch short ablation (local M2)
- `*_official_best.pth` — frozen checkpoint selected for comparison
- `selection_record.json` — provenance record for each frozen checkpoint

## Duration Mismatch Warning

The baseline was trained for 100 epochs (best at epoch 46). The full model was trained for ~50 epochs (best at epoch 3). The 15-epoch ablations are intentionally short and are **not** duration-matched. They are useful for comparing relative effects of individual brushstroke loss components, not for absolute performance claims.
