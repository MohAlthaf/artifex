# SYSTEM ARCHITECTURE PROGRESS

Changelog and milestone tracker for the ARTIFEX thesis demo system.

---

## v3 — Next.js Migration + Per-Upload Evaluation (4 March 2026)

### Summary

Complete frontend migration from React+Vite+Express to **Next.js App Router + Tailwind CSS**, with the Express proxy layer removed entirely. Flask ML backend now serves the frontend directly via CORS. Major product change: single thesis-demo page with per-upload evaluation.

### Architecture Change

**Before (v2):** React 19 + Vite (5173) → Express proxy (3001) → Flask ML (5001)
**After (v3):** Next.js 15 App Router (3000) → Flask ML (5001) directly

### Key Changes

| Change | File(s) | Details |
|---|---|---|
| New Next.js app | `client-next/` | App Router, Tailwind CSS, single page concept |
| Single thesis demo page | `client-next/app/page.jsx` | Replaces two-tab SPA (Live Restore + Benchmark Explorer) |
| Per-upload evaluation endpoint | `server/ml/app.py` | `POST /api/restore-with-eval` — accepts optional ground_truth file |
| Metric computation module | `server/ml/metrics.py` | PSNR, SSIM, L1, L2, Perceptual, Style (VGG-based) |
| Express proxy removed | `start.sh` | Only 2 servers now: Flask (5001) + Next.js (3000) |
| Tailwind CSS | `client-next/app/globals.css` | Premium dark theme with Van Gogh gold accents |
| Upload with GT support | `client-next/components/UploadZone.jsx` | 3 upload slots: damaged + mask + ground truth |
| Honest metric policy | Frontend + Backend | Metrics ONLY when GT provided |
| Benchmark evidence section | `client-next/components/BenchmarkEvidence.jsx` | Comparison table + sample browser, same page |

### Metric Honesty Policy

| Scenario | Behaviour |
|---|---|
| No ground truth | Restoration runs. No per-upload metrics shown. |
| With ground truth | PSNR, SSIM, L1, L2, Perceptual, Style computed per model vs GT |
| Benchmark section | Official test-set averages from saved JSONs (always visible) |

### Files Created

| File | Purpose |
|---|---|
| `client-next/package.json` | Next.js 15 + React 19 + Tailwind CSS |
| `client-next/next.config.mjs` | Flask backend URL config |
| `client-next/app/layout.jsx` | Root layout with Google Fonts |
| `client-next/app/page.jsx` | Single thesis demo page |
| `client-next/app/globals.css` | Tailwind v4 + Van Gogh theme |
| `client-next/lib/api.js` | API client (direct Flask calls) |
| `client-next/components/UploadZone.jsx` | 3-slot upload (damaged + mask + GT) |
| `client-next/components/ModelResultCard.jsx` | Per-model result card |
| `client-next/components/BeforeAfterSlider.jsx` | Drag slider comparison |
| `client-next/components/MetricBadge.jsx` | Metric display badge |
| `client-next/components/BenchmarkEvidence.jsx` | Benchmark table + sample browser |
| `server/ml/metrics.py` | Per-upload metric computation |

### Files Modified

| File | Change |
|---|---|
| `server/ml/app.py` | Added `POST /api/restore-with-eval` endpoint, torch import, metrics import |
| `start.sh` | Removed Express, launches Flask + Next.js |
| `README.md` | Rewritten for v3 architecture |
| `SYSTEM_ARCHITECTURE_PROGRESS.md` | This section |

### Legacy Preserved

| File/Directory | Status |
|---|---|
| `client/` | Preserved — v2 React+Vite frontend |
| `server/server.js` | Preserved — v2 Express proxy |
| `server/ml/app_legacy.py` | Preserved — v1 prototype |
| `server/ml/model_legacy.py` | Preserved — v1 architecture |

---

## v2.1 — Inference Pipeline Fix (4 March 2026)

### Summary

Critical inference pipeline audit and fix. The v2 architecture was correctly wired to official checkpoints, but the live inference postprocessing was **missing the canonical composition step**, causing visually poor outputs.

### Critical Bug Fixed

**Problem:** `run_inference()` in `canonical_inference.py` returned raw model output without applying the composition step. During training, the final output is:

```python
comp = corrupted * (1 - mask) + restored * mask
```

This ensures intact regions keep original pixels and the model output is only used inside the damaged mask. Without this step, intact areas showed artifacts because the model was not trained to reproduce them perfectly.

**Fix:** Added canonical composition step to `run_inference()`. The composed output now exactly matches the training-time evaluation pipeline used to produce the official benchmark results.

### Other Changes

| Change                       | File                                        | Details                                                                                                             |
| ---------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Composition step added       | `server/ml/canonical_inference.py`          | `composed = img * (1 - mask) + output * mask` applied before postprocessing                                         |
| Debug artifact saving        | `server/ml/canonical_inference.py`          | Each request saves input_image, input_mask, raw_model_output, composed_output + metadata.json to `debug_artifacts/` |
| Inference metadata in API    | `server/ml/app.py`                          | `/api/restore-all` now returns `inference_time_s`, `mask_coverage_pct`, `checkpoint_path` per model                 |
| Registry fields expanded     | `server/ml/model_registry.py`               | Added `status`, `enabled_for_live_restore`, `enabled_for_benchmark`, `selection_record_path` fields                 |
| Frontend: inference metadata | `client/src/components/ModelResultCard.jsx` | Shows inference time, mask coverage %, checkpoint path for each result                                              |
| Model Registry Status doc    | `MODEL_REGISTRY_STATUS.md`                  | New operational status doc tracking live/benchmark status, last verified flow                                       |

### Validation Results (4 March 2026)

| Check                           | Result                                                          |
| ------------------------------- | --------------------------------------------------------------- |
| `baseline_official` strict load | PASS (162 keys)                                                 |
| `full_official` strict load     | PASS (162 keys)                                                 |
| Composition step                | PASS — output matches training-time evaluation                  |
| PSNR vs GT (vangogh_test_0000)  | baseline=15.50 dB, full=15.87 dB (matches official eval: 15.48) |
| Debug artifacts                 | PASS — saved to `server/ml/debug_artifacts/`                    |
| dir/edge/hist models            | Correctly marked unavailable (status=missing)                   |

---

## v2 — Thesis-Compatible Upgrade (4 March 2026)

### Summary

Full system audit and rewrite to make the demo compatible with the canonical thesis-trained models (`baseline_official`, `full_official`) instead of the old prototype checkpoints. Adds two major new UI modes: Live Restore (multi-model side-by-side inference) and Benchmark Explorer (305-image test-set browser with real metrics).

---

### Files Created

| File                                         | Purpose                                                                                                         |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `server/ml/canonical_inference.py`           | Exact canonical SGRGANGenerator architecture (with skip connections) + load/preprocess/inference helpers        |
| `server/ml/model_registry.py`                | Dynamic checkpoint discovery + eval JSON loading. Reads real thesis artifact paths.                             |
| `client/src/pages/LiveRestorePage.jsx`       | Multi-model live restore page. Calls `/api/restore-all`. Shows official test-set metrics with clear disclaimer. |
| `client/src/pages/BenchmarkExplorerPage.jsx` | Test-set browser. Paginated sample viewer, comparison table, per-image metrics from real evaluation JSONs.      |
| `client/src/components/ModelResultCard.jsx`  | Reusable card: handles loading / unavailable / error / result states. BeforeAfterSlider + MetricBadge.          |
| `README.md`                                  | Architecture overview, setup instructions, API reference                                                        |
| `SYSTEM_ARCHITECTURE_PROGRESS.md`            | This file                                                                                                       |

### Files Rewritten

| File                 | Change                                                                                              |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| `server/ml/app.py`   | Complete rewrite: 10 new endpoints, model registry, benchmark data API, NaN-safe JSON serialisation |
| `server/server.js`   | Updated proxy routes for all new Flask endpoints; added `_forwardUpload()` + `_proxyGet()` helpers  |
| `client/src/App.jsx` | Rewritten from single-page to two-tab navigation (Live Restore / Benchmark Explorer)                |
| `start.sh`           | Removed hardcoded pyenv Python path; now uses `python3` from PATH or `PYTHON` env var               |

### Files Renamed (Preserved as Legacy)

| Old Name             | New Name                    | Reason                                                                          |
| -------------------- | --------------------------- | ------------------------------------------------------------------------------- |
| `server/ml/model.py` | `server/ml/model_legacy.py` | Old architecture (no skip connections) — incompatible with official checkpoints |
| `server/ml/app.py`   | `server/ml/app_legacy.py`   | Old single-model prototype Flask app                                            |

---

### Root Issue Fixed

**Problem:** The original `model.py` implemented a U-Net generator WITHOUT skip connection projection layers (`skip_conv1`, `skip_conv2`, `skip_conv3`). The official thesis checkpoints (`baseline_official_best.pth`, `full_official_best.pth`) were trained with the canonical architecture that INCLUDES these layers. Loading official checkpoints with the old `model.py` would raise a `RuntimeError: unexpected key(s) in state_dict`.

**Fix:** `canonical_inference.py` implements the exact canonical `SGRGANGenerator` architecture, verified against the state-dict keys in both official checkpoints.

---

### Known Limitations (v2)

| Issue                        | Status    | Notes                                                                                                                                                                      |
| ---------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ablation checkpoints missing | ❌ Open   | `dir_only_official`, `edge_only_official`, `hist_only_official` — not yet trained. Registry marks them unavailable. UI shows "Not available — checkpoint not yet trained." |
| Live inference metrics       | By design | Live inference does not compute PSNR/SSIM/etc. — only official test-set averages are shown, clearly labelled.                                                              |
| Double `.png` extension      | Handled   | Pre-computed restored images in `full_eval/full_best/restored_images/` are named `vangogh_test_XXXX.png.png`. The benchmark API tries both naming conventions.             |
| MPS OOM                      | Handled   | Apple Silicon MPS backend causes MultiheadAttention OOM at 512×512. Device is forced to `cpu`. Inference is ~5–15s per model on M-series chips.                            |
| `full_official` epoch        | Known     | `full_official_best.pth` is epoch 3 — an early stopping artifact. It still outperforms baseline on 8/9 metrics per `baseline_vs_full_comparison.json`.                     |

---

## v1 — Prototype (pre-March 2026)

### Summary

Initial proof-of-concept demo built during early development. Used prototype model checkpoints and a different (simpler) generator architecture without skip connection projection layers.

### Architecture

- Single-model Flask service loading `model/final_model.pth` or `model/best_model.pth` (247 MB each)
- `server/ml/model.py` — `GeneratorUNet` without `skip_conv1/2/3`
- `server/ml/app.py` — single `/predict` endpoint, single model
- `client/src/App.jsx` — single page (upload → single model result)

### Why v1 was replaced

1. Prototype checkpoints were trained on an earlier architecture and are NOT compatible with the thesis-final training code
2. Single-model demo cannot show ablation study comparisons
3. No access to official benchmark metrics in the UI
4. Hardcoded Python paths made it non-portable

---

## Metrics Reference

All metrics computed on 305-image test set from `data/processed/test/`:

| Metric                | Higher is better    | Measures                             |
| --------------------- | ------------------- | ------------------------------------ |
| PSNR                  | ✅                  | Pixel-level fidelity (dB)            |
| SSIM                  | ✅                  | Structural similarity (0–1)          |
| MS-SSIM               | ✅                  | Multi-scale structural similarity    |
| LPIPS                 | ❌ (lower = better) | Perceptual similarity (VGG features) |
| Edge Coherence        | ✅                  | Edge preservation                    |
| Brushstroke Direction | ✅                  | Directional stroke consistency       |
| Brushstroke Texture   | ✅                  | Texture stroke consistency           |
| Color Harmony         | ✅                  | Color palette consistency            |
| Style Transfer        | ✅                  | Overall style fidelity               |

Reference: `results/baseline_vs_full_comparison.json` — full_official wins 8/9 metrics. Verdict: `full_improves_all_brushstroke`.

---

## Next Milestones

- [ ] Train `dir_only_official`, `edge_only_official`, `hist_only_official` ablation models and add checkpoints to `models/`
- [ ] Add `results/<ablation>/evaluation_results.json` for each ablation model
- [ ] Extend `model_registry.py` if new checkpoints use a different state-dict key format
- [ ] Consider switching to `mps` device once PyTorch MPS MultiheadAttention is stable
- [ ] Production build: `npm run build` in `client/`, serve static files from Express
