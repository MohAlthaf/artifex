# ARTIFEX — Van Gogh Art Restoration Demo

> **Thesis Project Demo (v3)**
ARTIFEX is a thesis demonstration system for brushstroke-aware painting restoration. The demo allows a user to upload a damaged painting, an optional damage mask, and an optional clean ground-truth image. The system then runs the image through the available ARTIFEX model variants and displays the restoration outputs.

Per-upload metrics are only shown when a clean ground-truth image is provided. This avoids showing misleading scores when there is no valid reference image for comparison. The demo also includes an official benchmark evidence section based on the fixed 305-image Van Gogh test set.

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
│  canonical_inference.py — ArtifexGenerator (exact thesis arch)   │
│  model_registry.py      — Dynamic checkpoint discovery           │
│  metrics.py             — Per-upload metric computation           │
│  app.py                 — Flask endpoints + CORS                  │
│                                                                  │
│  Models loaded at startup:                                       │
│     baseline_official   (models/baseline_official/)             │
│     full_official       (models/full_official/)                 │
│     dir_only_official   (models/dir_only_15ep_official/)        │
│     edge_only_official  (models/edge_only_15ep_official/)       │
│     hist_only_official  (models/hist_only_15ep_official/)       │
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
| `baseline_official`  |  Available | `models/baseline_official/baseline_official_best.pth` (ep 46)             | `results/baseline_ep46_v2/evaluation_results.json`                  |
| `full_official`      |  Available | `models/full_official/full_official_best.pth` (ep 3)                      | `results/full_eval_v2/evaluation_results.json`                      |
| `dir_only_official`  |  Available | `models/dir_only_15ep_official/dir_only_15ep_official_best.pth` (ep 15)   | `runs/dir_only_15ep_20260312_210744/logs/evaluation_results.json`   |
| `edge_only_official` | Available | `models/edge_only_15ep_official/edge_only_15ep_official_best.pth` (ep 5)  | `runs/edge_only_15ep_20260313_023647/logs/evaluation_results.json`  |
| `hist_only_official` |  Available | `models/hist_only_15ep_official/hist_only_15ep_official_best.pth` (ep 5)  | `runs/hist_only_15ep_20260313_101629/logs/evaluation_results.json`  |

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

# Terminal 2 — Next.js Frontend
cd client-next && npm run dev
```

---

## Key Files

```
artifex/
  start.sh                          Startup script (Flask + Next.js)
  README.md                         This file
  test_model.py                     Standalone inference test

  client-next/                      Next.js 15 frontend
    app/page.jsx                    Single-page thesis demo
    components/UploadZone.jsx       Drag-and-drop upload
    components/ModelResultCard.jsx  Per-model result card with metrics
    components/BenchmarkEvidence.jsx Benchmark evidence section

  server/ml/
    app.py                          Flask ML service (canonical)
    canonical_inference.py          ArtifexGenerator architecture + inference
    model_registry.py               Dynamic checkpoint discovery
    metrics.py                      Per-upload metric computation
```

---

## Benchmark Data

```
data/processed/test/
  original/   305 ground-truth images (vangogh_test_0000.png … 0304.png)
  masked/     305 damaged input images
  masks/      305 binary damage masks

results/
  baseline_ep46_v2/evaluation_results.json     per-image metrics, baseline_official (authoritative)
  full_eval_v2/evaluation_results.json         per-image metrics, full_official (authoritative)
  baseline_vs_full_comparison.json             aggregate comparison
  final_submission/                            authoritative results package
```

Per-upload metrics (PSNR, SSIM, L1, etc.) are computed only when ground truth is provided. Official test-set averages (n=305) are always shown from saved evaluation JSONs.

---

## Tech Stack

- **Frontend:** Next.js 15 / React 19 (App Router, Tailwind CSS)
- **ML Backend:** Python 3.10+ / Flask + PyTorch (CPU; MPS forced off due to MultiheadAttention OOM at 512×512)
- **No proxy layer:** Next.js calls Flask directly via CORS

