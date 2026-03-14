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

| Cell | Purpose |
|------|---------|
| 50 | Markdown header explaining the 15-ep suite |
| 51 | Re-evaluate baseline_official & full_official with VGG LPIPS |
| 52 | dir_only 15-ep resume prep (new run dir, resume from epoch 5) |
| 53 | dir_only training call |
| 54 | edge_only 15-ep fresh prep + warm-start + verify |
| 55 | edge_only training call |
| 56 | hist_only 15-ep fresh prep + warm-start + verify |
| 57 | hist_only training call |
| 58 | Evaluate & freeze all three 15-ep ablations |
| 59 | Generate comparison JSON + CSV artifacts |

**Design decisions:**
- dir_only resumes from `runs/dir_only_20260304_001954/checkpoints/checkpoint_epoch_5.pth` into a **new** `dir_only_15ep_*` run directory to avoid corrupting old evidence
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
