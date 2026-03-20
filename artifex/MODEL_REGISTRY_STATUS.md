# MODEL REGISTRY STATUS

Operational status of all ARTIFEX thesis models.
Last verified: **20 March 2026** (all 5 models evaluated with valid LPIPS).

---

## Current Models

| Model ID             | Status       | Live Restore | Benchmark | Checkpoint Path                                                          | Epoch          | Params |
| -------------------- | ------------ | ------------ | --------- | ------------------------------------------------------------------------ | -------------- | ------ |
| `baseline_official`  | **Official** | Enabled      | Enabled   | `models/baseline_official/baseline_official_best.pth`                    | 45 (0-indexed) | 20.6M  |
| `full_official`      | **Official** | Enabled      | Enabled   | `models/full_official/full_official_best.pth`                            | 2 (0-indexed)  | 20.6M  |
| `dir_only_official`  | **Official** | Enabled      | Enabled   | `models/dir_only_15ep_official/dir_only_15ep_official_best.pth`          | 14 (0-indexed) | 20.6M  |
| `edge_only_official` | **Official** | Enabled      | Enabled   | `models/edge_only_15ep_official/edge_only_15ep_official_best.pth`        | 4 (0-indexed)  | 20.6M  |
| `hist_only_official` | **Official** | Enabled      | Enabled   | `models/hist_only_15ep_official/hist_only_15ep_official_best.pth`        | 4 (0-indexed)  | 20.6M  |

---

## Evaluation Data

| Model ID             | Eval JSON                                                              | Per-Image Metrics | LPIPS    | Pre-Computed Images |
| -------------------- | ---------------------------------------------------------------------- | ----------------- | -------- | ------------------- |
| `baseline_official`  | `results/baseline_ep46_v2/evaluation_results.json` (305 images)        | Yes               | 0.1397   | No                  |
| `full_official`      | `results/full_eval_v2/evaluation_results.json` (305 images)            | Yes               | 0.1437   | Yes (305 images)    |
| `dir_only_official`  | `runs/dir_only_15ep_*/logs/evaluation_results.json` (305 images)       | Yes               | 0.1405   | No                  |
| `edge_only_official` | `runs/edge_only_15ep_*/logs/evaluation_results.json` (305 images)      | Yes               | 0.1383   | No                  |
| `hist_only_official` | `runs/hist_only_15ep_*/logs/evaluation_results.json` (305 images)      | Yes               | 0.1399   | No                  |

---

## Checkpoint Source Paths (Absolute)

These are the **official thesis checkpoints** used by the app:

```
/Users/althafali/Downloads/ARTIFEX/implementation/models/baseline_official/baseline_official_best.pth
/Users/althafali/Downloads/ARTIFEX/implementation/models/full_official/full_official_best.pth
/Users/althafali/Downloads/ARTIFEX/implementation/models/dir_only_15ep_official/dir_only_15ep_official_best.pth
/Users/althafali/Downloads/ARTIFEX/implementation/models/edge_only_15ep_official/edge_only_15ep_official_best.pth
/Users/althafali/Downloads/ARTIFEX/implementation/models/hist_only_15ep_official/hist_only_15ep_official_best.pth
```

The old prototype checkpoints in `artifex/model/` are **NOT used**:

```
artifex/model/baseline_best.pth   (OLD — incompatible architecture)
artifex/model/full_best.pth       (OLD — incompatible architecture)
```

---

## Last Verified Working Flow

**Date:** 4 March 2026

| Check                           | Result                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------- |
| `baseline_official` strict load | PASS (162 state_dict keys, no missing/unexpected)                                |
| `full_official` strict load     | PASS (162 state_dict keys, no missing/unexpected)                                |
| Composition step active         | PASS (`comp = corrupted * (1-mask) + restored * mask`)                           |
| Normalization: [0, 1]           | PASS (ToTensor only, no ImageNet mean/std)                                       |
| Mask: 1=damaged, 0=intact       | PASS                                                                             |
| Input size: 512x512             | PASS                                                                             |
| Output range                    | PASS ([0.0002, 0.9995])                                                          |
| PSNR vs GT (vangogh_test_0000)  | baseline=15.50 dB, full=15.87 dB                                                 |
| Debug artifacts saved           | PASS (input_image, input_mask, raw_model_output, composed_output, metadata.json) |
| Inference time (CPU, M-series)  | ~1.2-1.4s per model                                                              |

---

## Known Issues

| Issue                     | Severity  | Notes                                                                                                             |
| ------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------- |
| Double `.png` extension   | Handled   | Pre-computed images in `full_eval/full_best/restored_images/` use `*.png.png`. API tries both naming conventions. |
| MPS OOM                   | Handled   | Apple Silicon MPS backend causes MultiheadAttention buffer OOM. Device forced to CPU.                             |
| No per-upload metrics     | By design | Live inference shows only official test-set averages, clearly labelled.                                           |
| Old NaN eval JSONs        | Archived  | `results/baseline_ep46/` and `results/full_eval/` contain NaN LPIPS. The `*_v2/` versions have valid LPIPS.      |
| Legacy model directories  | Archived  | `models/dir_only_official/`, `edge_only_official/`, `hist_only_official/` are superseded by `*_15ep_official/`.   |
| Duration mismatch         | By design | 15-ep ablations vs 100-ep baseline / ~50-ep full — not directly comparable in absolute terms.                    |

---

## How to Add a New Model

1. Train the model and save checkpoint to `models/<model_id>/<model_id>_best.pth`
2. Run evaluation to create `results/<model_id>/evaluation_results.json`
3. The model will be **automatically discovered** by `model_registry.py` on next Flask startup
4. No code changes needed — registry config already includes all 5 planned models
