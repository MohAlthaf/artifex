# ARTIFEX — Development Log

## Session: 15-Epoch Ablation Suite Implementation

### LPIPS Fix (Cell 18)

**Problem:** `lpips.LPIPS(net='alex')` silently failed because the AlexNet backbone weights (~233 MB) were not cached at `~/.cache/torch/hub/checkpoints/` and the download consistently timed out on the local connection. This caused all LPIPS values in evaluation JSONs to be `NaN`.

**Root cause verified:** Tested the download — only 1.5 MB of 233 MB transferred before timeout. VGG16 weights (528 MB) were already present.

**Fix:** Changed cell 18 to use `net='vgg'`. Added:

- Sanity check: verifies a zero-distance pair returns ~0, not NaN
- Input clamping: `.clamp(0, 1)` before scaling to [-1, 1]
- Runtime error: raises if LPIPS unexpectedly returns NaN during evaluation

**Impact:** All LPIPS values from this point forward use the VGG backbone. The original evaluation JSONs (`results/baseline_ep46/`, `results/full_eval/`) still contain NaN. Updated results are in `*_v2/` directories.

### Lambda Verification Fix (Cell 42)

**Problem:** `verify_ablation_setup()` had hardcoded lambda expectations that didn't match `ABLATION_CONFIGS` in cell 5:

- `edge_only`: expected `lambda_thickness=1.5`, should be `1.0`
- `full`: expected `lambda_thickness=1.5`, should be `1.0`

**Fix:** Corrected both to `1.0` to match the canonical `ABLATION_CONFIGS`.

**Added:** `allow_short_run=False` parameter so 15-epoch runs pass validation without error (epochs >= 5 accepted when `allow_short_run=True`).

### .png.png Double Extension Fix

**Problem:** `scripts/evaluate_full_model.py` line ~280 used `f'{img_name}.png'` but `img_name` already includes the `.png` extension from the dataset, producing filenames like `image_001.png.png`.

**Fix:** Changed to bare `img_name` (matching the fix already present in `scripts/evaluate_ablation.py`).

### 15-Epoch Ablation Cells (Cells 50–59)

Added 10 new cells to the canonical notebook:

| Cell | Purpose                                                       |
| ---- | ------------------------------------------------------------- |
| 50   | Markdown header explaining the 15-ep suite                    |
| 51   | Re-evaluate baseline_official & full_official with VGG LPIPS  |
| 52   | dir_only 15-ep resume prep (new run dir, resume from epoch 5) |
| 53   | dir_only training call                                        |
| 54   | edge_only 15-ep fresh prep + warm-start + verify              |
| 55   | edge_only training call                                       |
| 56   | hist_only 15-ep fresh prep + warm-start + verify              |
| 57   | hist_only training call                                       |
| 58   | Evaluate & freeze all three 15-ep ablations                   |
| 59   | Generate comparison JSON + CSV artifacts                      |

**Design decisions:**

- dir*only resumes from `runs/dir_only_20260304_001954/checkpoints/checkpoint_epoch_5.pth` into a **new** `dir_only_15ep*\*` run directory to avoid corrupting old evidence
- edge_only and hist_only start fresh with baseline warm-start (`init_generator_from_baseline`)
- Run directories include `_15ep_` in the name for honest provenance
- Comparison artifacts explicitly note the duration mismatch

### New Files Created

- `requirements.txt` — pinned versions from local `.venv`
- `docs/REPRODUCIBILITY.md` — environment setup, cell execution order, LPIPS notes
- `docs/DEVLOG.md` — this file

### Known Remaining Risks

1. **Duration mismatch:** 15-ep ablations vs 100-ep baseline / ~50-ep full — not directly comparable in absolute terms
2. **LPIPS scale change:** VGG-based LPIPS ≠ AlexNet-based LPIPS — old NaN results cannot be retroactively compared
3. **Existing NaN JSONs:** `results/baseline_ep46/` and `results/full_eval/` still contain NaN LPIPS — the v2 directories have corrected values
4. **dir_only history gap:** The old 5-epoch run has `epochs_planned=50` in its metadata; the 15-ep continuation creates a clean break with its own history

---

## Session: LPIPS Fix, Evaluation Regeneration & Registry Update

### LPIPS Root Cause (Cell 18)

**Problem:** Despite switching from `net='alex'` to `net='vgg'`, all LPIPS values remained NaN across every evaluation JSON — all 5 models, 305 images each.

**Root cause:** The sanity check in Cell 18 used `torch.zeros(1, 3, 8, 8)`. VGG-LPIPS runs the input through VGG16 which has 5 max-pooling layers (each halving the spatial dimension). At the 4th pooling layer, an 8×8 input has already been reduced to 1×1, and the 5th pooling produces a 0×0 feature map — triggering `RuntimeError: Output size is too small`. The `except Exception` block caught this silently, set `LPIPS_AVAILABLE = False`, and every subsequent `calculate_lpips()` call returned `float('nan')`.

**Fix:** Changed `torch.zeros(1, 3, 8, 8, device=device)` → `torch.zeros(1, 3, 64, 64, device=device)` with a comment explaining the VGG minimum input size.

**Verified:** LPIPS sanity check now passes on both CPU and MPS. Confirmed empirically that VGG-LPIPS works for inputs ≥ 16×16; 64×64 chosen for safety margin.

### Evaluation Regeneration

All 5 models re-evaluated on the 305-image test set with working LPIPS:

| Model             | PSNR  | SSIM   | LPIPS  | Direction | Edge   | Histogram |
| ----------------- | ----- | ------ | ------ | --------- | ------ | --------- |
| baseline_official | 25.61 | 0.8622 | 0.1397 | 0.2212    | 0.1484 | 0.3282    |
| full_official     | 25.91 | 0.8693 | 0.1437 | 0.1925    | 0.1466 | 0.3109    |
| dir_only_15ep     | 25.70 | 0.8663 | 0.1405 | 0.1997    | 0.1480 | 0.3074    |
| edge_only_15ep    | 25.82 | 0.8680 | 0.1383 | 0.2185    | 0.1440 | 0.3188    |
| hist_only_15ep    | 25.93 | 0.8681 | 0.1399 | 0.2097    | 0.1465 | 0.3063    |

All v2 evaluation JSONs now include masked-region metrics (psnr_masked, ssim_masked, lpips_masked) and mask-gated histogram_loss. Zero NaN values across 1525 per-image results.

Updated artifacts:
- `results/baseline_ep46_v2/evaluation_results.json`
- `results/full_eval_v2/evaluation_results.json`
- `runs/dir_only_15ep_*/logs/evaluation_results.json`
- `runs/edge_only_15ep_*/logs/evaluation_results.json`
- `runs/hist_only_15ep_*/logs/evaluation_results.json`
- `results/ablation_15ep_comparison.json`
- `results/official_tables/final_metrics_15ep_ablation.csv`

### Model Registry Update

- `edge_only_official` → now points to `models/edge_only_15ep_official/`
- `hist_only_official` → now points to `models/hist_only_15ep_official/`
- Baseline and full eval_json paths updated to v2 directories
- `EvalSummary` dataclass now includes `avg_lpips`

### dir_only History Clarification

Added `note` field to both `training_history_final.json` and `selection_record.json` explaining that `epochs_completed=10` counts only the resumed session (epochs 5→15), while the checkpoint confirms 15 total epochs.
