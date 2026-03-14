# ARTIFEX — Van Gogh Art Restoration Demo

> **Thesis Project Demo (v3)**
> Single-page research prototype: upload a damaged Van Gogh painting + damage mask + optional clean ground truth. Run it through all available official SGRGAN thesis models. See real per-upload metrics ONLY when ground truth is provided — no fake metrics, ever. Browse official benchmark evidence (305-image test set).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser (port 3000)                           │
│              Next.js 15 — App Router, Tailwind CSS              │
│                                                                  │
│   Single thesis demo page:                                       │
│     1. Upload: damaged image + mask + optional ground truth      │
│     2. → All-model inference (all available official models)     │
│     3. → Per-upload metrics (ONLY when GT provided)              │
│     4. Official benchmark evidence section (n=305)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (direct, no Express proxy)
┌───────────────────────────▼─────────────────────────────────────┐
│                Flask ML Service (port 5001)                      │
│  canonical_inference.py — SGRGANGenerator (exact thesis arch)    │
│  model_registry.py      — Dynamic checkpoint discovery           │
│  metrics.py             — Per-upload metric computation           │
│  app.py                 — Flask endpoints + CORS                  │
│                                                                  │
│  Models loaded at startup:                                       │
│    ✅ baseline_official   (models/baseline_official/)             │
│    ✅ full_official       (models/full_official/)                 │
│    ❌ dir/edge/hist ablations — not yet trained                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decision: No Express Proxy

v3 removes the Express proxy layer. Next.js calls Flask directly via CORS.
This simplifies the stack and reduces failure modes.

---

## Metric Honesty Policy

| Scenario                              | What is shown                                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Upload: damaged + mask (no GT)        | Restoration runs. **No per-upload metrics** — because computing PSNR/SSIM without ground truth would be misleading.       |
| Upload: damaged + mask + ground truth | Restoration runs. **Real per-upload metrics** (PSNR, SSIM, L1, L2, Perceptual, Style) computed against YOUR ground truth. |
| Benchmark section                     | **Official test-set averages** from saved evaluation JSONs (n=305). Always visible.                                       |

Brushstroke-specific metrics (direction, edge strength, histogram) require pre-extracted feature maps and are available only for official benchmark images.

---

## Official Model Registry

| Model ID             | Status             | Checkpoint                                                    | Eval Results                                          |
| -------------------- | ------------------ | ------------------------------------------------------------- | ----------------------------------------------------- |
| `baseline_official`  | ✅ Available       | `models/baseline_official/baseline_official_best.pth` (ep 46) | `results/baseline_ep46/evaluation_results.json`       |
| `full_official`      | ✅ Available       | `models/full_official/full_official_best.pth` (ep 3)          | `results/full_eval/full_best/evaluation_results.json` |
| `dir_only_official`  | ❌ Not yet trained | —                                                             | —                                                     |
| `edge_only_official` | ❌ Not yet trained | —                                                             | —                                                     |
| `hist_only_official` | ❌ Not yet trained | —                                                             | —                                                     |

---

## Setup

### Prerequisites

- Python 3.10+ with: `torch torchvision flask flask-cors numpy Pillow`
- Node.js 18+ and npm

### Install dependencies (first time only)

```bash
pip install torch torchvision flask flask-cors numpy Pillow

cd artifex/client-next && npm install
```

### Git hook setup (one-time, per clone)

Enable the repo-local pre-commit hook that auto-updates tracking docs on every commit:

```bash
git config core.hooksPath .githooks
```

This will run `scripts/update_task_tracker.py` and `scripts/update_devlog_from_staged_diff.py` before each commit, then stage the updated `docs/IMPLEMENTATION_TASK_TRACKER.md`, `docs/WORK_DONE_SNAPSHOT.md`, and `docs/DEVLOG.md`.

### Run

```bash
cd artifex
chmod +x start.sh
./start.sh
```

Open **http://localhost:3000**

---

## API Endpoints

### Primary (v3)

| Method | Endpoint                 | Description                                                                 |
| ------ | ------------------------ | --------------------------------------------------------------------------- |
| `POST` | `/api/restore-with-eval` | Upload image + mask + optional GT → all-model results + conditional metrics |
| `GET`  | `/api/models`            | Full model registry                                                         |
| `GET`  | `/health`                | Health check                                                                |

### Benchmark

| Method | Endpoint                    | Description                 |
| ------ | --------------------------- | --------------------------- |
| `GET`  | `/api/benchmark/models`     | Models with evaluation data |
| `GET`  | `/api/benchmark/samples`    | Paginated sample list       |
| `GET`  | `/api/benchmark/sample/:id` | Single sample detail        |
| `GET`  | `/api/benchmark/comparison` | Baseline vs Full comparison |

### Legacy (preserved)

| Method | Endpoint           | Description                       |
| ------ | ------------------ | --------------------------------- |
| `POST` | `/api/restore-all` | v2 all-model restore (no GT eval) |
| `POST` | `/predict`         | v1 single-model restore           |

If `python3` is not on PATH (e.g. pyenv):

```bash
PYTHON=~/.pyenv/versions/3.10.9/bin/python3 ./start.sh
```

### Manual start (separate terminals)

```bash
# Terminal 1 — Flask ML Service
cd server/ml && python3 app.py

# Terminal 2 — Express API
cd server && npm run dev

# Terminal 3 — React Frontend
cd client && npm run dev
```

---

## Key Files

```
artifex/
  start.sh                          Startup script (all three servers)
  README.md                         This file
  SYSTEM_ARCHITECTURE_PROGRESS.md   Changelog and architecture notes

  client/src/
    App.jsx                         Root — two-tab navigation
    pages/LiveRestorePage.jsx        Upload + multi-model inference
    pages/BenchmarkExplorerPage.jsx  Test-set browser + comparison table
    components/ModelResultCard.jsx   Per-model result card with metrics

  server/
    server.js                       Express proxy
    ml/
      app.py                        Flask ML service (canonical)
      canonical_inference.py        SGRGANGenerator architecture + inference
      model_registry.py             Dynamic checkpoint discovery
      app_legacy.py                 Old prototype Flask app (preserved)
      model_legacy.py               Old prototype model (no skip connections)
```

---

## Benchmark Data

```
data/processed/test/
  original/   305 ground-truth images (vangogh_test_0000.png … 0304.png)
  masked/     305 damaged input images
  masks/      305 binary damage masks

results/
  baseline_ep46/evaluation_results.json        per-image metrics, baseline_official
  full_eval/full_best/evaluation_results.json  per-image metrics, full_official
  baseline_vs_full_comparison.json             aggregate comparison (full wins 8/9)
```

**Metric disclaimer:** Metrics in the Live Restore tab are official test-set averages (n=305). They are clearly labelled and are NOT computed per-upload.

---

## Tech Stack

- **Frontend:** React 19 + Vite 7
- **API:** Node.js + Express 4
- **ML:** Python 3.10 + Flask + PyTorch (CPU; MPS forced off due to MultiheadAttention OOM at 512×512)
