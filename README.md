# ARTIFEX — Brushstroke-Aware Deep Inpainting for Damaged Painting Restoration

Final-year thesis implementation. This repository contains the implemented
research pipeline, trained model checkpoints, evaluation results, and a working
demo prototype.

## Project Overview

ARTIFEX investigates whether incorporating brushstroke-level priors — orientation,
edge strength, and spatial histogram distributions — as auxiliary loss constraints
can improve both restoration quality and brushstroke-faithfulness when inpainting
damaged paintings. The system is developed and evaluated on a Van Gogh painting
dataset (2025 images, 512×512, 70/15/15 split).

The generator architecture is a dual-stream design (BiSCCFormer texture stream

- FocalBlock structure stream) with attention-based fusion and a U-Net decoder.
  Training uses a composite loss combining L1, VGG perceptual, style, adversarial,
  and three brushstroke-aware terms (direction, edge strength, histogram).

## Repository Purpose

This repository serves as the primary implementation and evidence base for the
thesis. It contains the research pipeline, trained checkpoints, evaluation
outputs, and the demo prototype used for the final submission.

Specifically, it includes:

- Preprocessing and brushstroke prior extraction code
- Model architecture and training code (in a Jupyter notebook)
- 5 trained model checkpoints (baseline + full + 3 ablations)
- Evaluation results for all models on 305 test images
- Statistical significance testing (30 paired Wilcoxon tests, Bonferroni-corrected)
- A web-based demo prototype (Flask backend + Next.js frontend)

## Canonical Workflow

The end-to-end pipeline follows these stages:

```
1. Preprocess dataset          → scripts/preprocess_dataset.py
   (discover, resize 512×512, split, distribute masks, augment, create damaged images)

2. Extract brushstroke priors   → artifex_v2.ipynb  OR  scripts/extract_brushstroke_h5.py
   (structure tensor orientation, coherence, Sobel edge strength, spatial angular histograms)
   Output: brushstroke_features.h5

3. Train models                 → artifex_training.ipynb
   (baseline with λ_brushstroke=0, full model, 3 single-loss ablations)

4. Evaluate                     → scripts/evaluate_full_model.py, scripts/evaluate_ablation.py
   (10 metrics × 305 test images per model)

5. Statistical analysis         → scripts/statistical_analysis.py
   (30 paired Wilcoxon tests with Bonferroni correction)

6. Validate repo                → scripts/validate_repo.py
   (8-category reproducibility check)
```

## Repository Structure

```
implementation/
├── README.md                                  ← This file
├── requirements.txt                           ← Python dependencies (PyTorch 2.3.0, etc.)
├── .gitignore
│
├── artifex_v2.ipynb                           ← Preprocessing / H5 generation notebook
├── artifex_training.ipynb                      ← Canonical training / evaluation notebook
├── brushstroke_features.h5                    ← Extracted priors (gitignored, ~2.5 GB)
│
├── scripts/
│   ├── preprocess_dataset.py                  ← Full data pipeline
│   ├── extract_brushstroke_h5.py              ← Standalone H5 extraction
│   ├── evaluate_full_model.py                 ← Full model evaluation
│   ├── evaluate_ablation.py                   ← Ablation evaluation
│   ├── statistical_analysis.py                ← Significance testing
│   └── validate_repo.py                       ← Reproducibility validation
│
├── models/                                    ← Official frozen checkpoints (gitignored)
│   ├── baseline_official/                     ← Baseline (epoch 46, no brushstroke losses)
│   ├── full_official/                         ← Full model (all 3 brushstroke losses)
│   ├── dir_only_15ep_official/                ← Ablation: direction loss only
│   ├── edge_only_15ep_official/               ← Ablation: edge strength loss only
│   ├── hist_only_15ep_official/               ← Ablation: histogram loss only
│   └── saved_models/                          ← Pre-trained VGG19 for perceptual loss
│
├── runs/                                      ← Training run logs and checkpoints (gitignored)
│   ├── baseline_20260226_092117/
│   ├── full_20260302_070353/
│   ├── dir_only_15ep_20260312_210744/
│   ├── edge_only_15ep_20260313_023647/
│   └── hist_only_15ep_20260313_101629/
│
├── results/
│   ├── final_submission/                      ← AUTHORITATIVE final results package
│   ├── baseline_ep46_v2/                      ← Authoritative baseline evaluation
│   ├── full_eval_v2/                          ← Authoritative full model evaluation
│   ├── official_tables/                       ← Final ablation metrics CSV
│   ├── statistical_tests.json                 ← Significance test results
│   ├── ablation_15ep_comparison.json          ← 5-model comparison
│   ├── baseline_vs_full_comparison.json       ← Baseline vs full comparison
│   ├── baseline_ep46/                         ← Reference only — see note below
│   ├── full_eval/                             ← Reference only — see note below
│   └── outputs/                               ← Misc output images
│
├── data/                                      ← Processed dataset (gitignored)
├── datasets/                                  ← Raw dataset (gitignored)
│
├── artifex/                                   ← Demo prototype
│   ├── README.md
│   ├── start.sh                               ← Launch script (Flask + Next.js)
│   ├── test_model.py                          ← Standalone inference test script
│   ├── server/ml/                             ← Flask backend (API, inference, metrics)
│   ├── client-next/                           ← Next.js 15 frontend
│   ├── damage paintng/                        ← Sample damaged images for demo (existing folder name)
│   └── test_outputs/                          ← Sample inference outputs
│
└── _legacy/                                   ← Archived earlier iterations (see _legacy/README.md)
```

## Main Files and Their Roles

| File                                       | Role                                                                                                                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `artifex_v2.ipynb`                         | Preprocessing and brushstroke prior generation notebook. This notebook created `brushstroke_features.h5`. It is not the training notebook.                                                 |
| `artifex_training.ipynb`                   | Canonical training, ablation, and evaluation notebook. This notebook contains the final training, ablation, checkpoint-selection, and evaluation workflow used for the implemented models. |
| `scripts/preprocess_dataset.py`            | Automated data pipeline: image discovery, resizing, train/val/test splitting, mask distribution, mask augmentation, and synthetic damage creation.                                         |
| `scripts/extract_brushstroke_h5.py`        | Standalone brushstroke prior extraction (mirrors the H5-generation code in `artifex_v2.ipynb`).                                                                                            |
| `scripts/evaluate_full_model.py`           | Evaluates full model on 305 test images across 10 metrics. Performs checkpoint selection between best-val and final-epoch.                                                                 |
| `scripts/evaluate_ablation.py`             | Evaluates ablation models with head-to-head comparison.                                                                                                                                    |
| `scripts/statistical_analysis.py`          | Runs 30 paired Wilcoxon signed-rank tests with Bonferroni correction across all model pairs.                                                                                               |
| `scripts/validate_repo.py`                 | Checks 8 reproducibility categories (data, H5, models, runs, results, scripts, architecture, demo).                                                                                        |
| `artifex/server/ml/canonical_inference.py` | Inference module used by the demo. Contains `ArtifexGenerator` (identical to the notebook's `SGRGANGenerator`).                                                                            |

## Training and Evaluation Evidence

**Dataset:** 2025 Van Gogh paintings — 1417 train / 303 val / 305 test (512×512).

**Models trained (5):**

| Model          | Epochs | Brushstroke Losses           | Best Epoch |
| -------------- | ------ | ---------------------------- | ---------- |
| Baseline       | 60/100 | None (all λ=0)               | 46         |
| Full           | 50     | Direction + Edge + Histogram | 3          |
| Direction-only | 15     | Direction only               | 15         |
| Edge-only      | 15     | Edge strength only           | 5          |
| Histogram-only | 15     | Histogram only               | 5          |

**Evaluation:** Each model evaluated on all 305 test images. Metrics: PSNR, SSIM, LPIPS, L1, L2, perceptual, style, direction, edge strength, histogram.

**Statistical testing:** 30 paired Wilcoxon signed-rank tests with Bonferroni correction (α=0.05, correction factor=30). Results in `results/statistical_tests.json`.

Each model checkpoint includes a `selection_record.json` documenting the selection rationale.

## Demo / Prototype

The `artifex/` folder contains a working web-based demo:

- **Backend:** Flask API (`artifex/server/ml/app.py`) — loads all 5 official checkpoints, provides inference endpoints.
- **Frontend:** Next.js 15 / React 19 (`artifex/client-next/`) — single-page thesis demo with upload, model comparison, and benchmark evidence display.

To run the demo:

```bash
cd artifex
./start.sh
# Opens at http://localhost:3000
```

Prerequisites: Python 3.10+ with dependencies from `requirements.txt`, Node.js 18+, and `npm install` in `client-next/`.

## Important Notes for Examiners

1. **Two notebooks, distinct roles.** `artifex_v2.ipynb` handles preprocessing and H5 generation. `artifex_training.ipynb` handles all training and evaluation. Both are required.
2. **Final results are in `results/final_submission/`.** This folder is the main final submission evidence package for the reported final metrics and tables. See its own `README.md` for contents.
3. **The `_legacy/` folder** contains archived earlier iterations of the codebase. It is kept for provenance and is not part of the final implementation. See `_legacy/README.md` for details.
4. **Large files are gitignored.** Model checkpoints (~276 MB each), the H5 file (~2.5 GB), and processed data are excluded from version control. They must be present on disk to run training or evaluation.
5. **All scripts derive paths from `__file__`.** The scripts in `scripts/` are portable and do not contain hardcoded absolute paths.
6. **Run notebooks from the repository root.** Both notebooks resolve data and output paths relative to the current working directory. Open and run them with `implementation/` as `cwd`.

## Known Limitations

- **Histogram similarity metric** was implemented but not verified with a standalone numerical test. It is used in training and produces plausible gradients, but no isolated unit test confirms its correctness.
- **LPIPS trade-off.** The full model achieves better brushstroke fidelity scores than the baseline but shows a slight increase in LPIPS (perceptual distance). This is documented and discussed in the thesis as an expected trade-off of the multi-loss formulation.
- **Ablation training duration.** Ablation variants were trained for 15 epochs (vs. 50 for baseline/full) due to compute constraints. This is noted wherever ablation results are reported.
- **Full model best epoch.** The full model's best validation loss occurred at epoch 3 (0-indexed: 2). Training continued to epoch 50 but validation loss increased monotonically afterward. The saved checkpoint is from epoch 3.

## Stale / Reference-Only Artifacts

The following result folders contain **v1 evaluation outputs** with known issues (LPIPS=NaN due to an AlexNet download failure in the original evaluation run). They are superseded by the v2 folders listed above but are retained because `scripts/statistical_analysis.py` reads per-image brushstroke data (direction, edge strength) from them for significance testing.

| Folder                   | Status                | Why Kept                                                     |
| ------------------------ | --------------------- | ------------------------------------------------------------ |
| `results/baseline_ep46/` | Stale (v1, LPIPS=NaN) | Per-image brushstroke data used by `statistical_analysis.py` |
| `results/full_eval/`     | Stale (v1, LPIPS=NaN) | Per-image brushstroke data used by `statistical_analysis.py` |

Both folders contain a `_STALE_README.md` file explaining their status.

**Do not treat these folders as final results.** Use `results/final_submission/` instead.

## Setup Notes

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# For the demo frontend
cd artifex/client-next
npm install
cd ../..
```

- Developed on macOS Apple Silicon (M2) with Python 3.12.7 and PyTorch 2.3.0.
- For CUDA environments, install the appropriate PyTorch build from https://pytorch.org/.

## What to Read First

1. This README
2. `results/final_submission/README.md` — final results package guide
3. `artifex_training.ipynb` — canonical training notebook
4. `scripts/statistical_analysis.py` — significance testing
5. `artifex_v2.ipynb` — preprocessing / H5 generation (if verifying data pipeline)
