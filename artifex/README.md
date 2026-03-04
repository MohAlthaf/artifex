# ARTIFEX — Van Gogh Art Restoration Demo

> **Thesis Project Demo**
> Interactive web application for exploring the SGRGAN-based painting restoration system trained for the thesis. Upload a damaged Van Gogh painting and run it through all available official thesis checkpoints simultaneously, or browse the 305-image official test-set benchmark results.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Browser (port 5173)                       │
│              React 19 + Vite — Two-tab SPA                      │
│   ┌──────────────────────┐  ┌──────────────────────────────┐   │
│   │  ✨ Live Restore      │  │  📊 Benchmark Explorer       │   │
│   │  Upload → all-model  │  │  Browse 305-image test set   │   │
│   │  inference side-by-  │  │  with ground truth + metrics │   │
│   │  side with metrics   │  │  from saved evaluation JSONs │   │
│   └──────────────────────┘  └──────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│                  Express API Server (port 3001)                  │
│              Node.js — Proxy + static serving                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (localhost)
┌───────────────────────────▼─────────────────────────────────────┐
│                Flask ML Service (port 5001)                      │
│  canonical_inference.py — SGRGANGenerator (exact thesis arch)    │
│  model_registry.py      — Dynamic checkpoint discovery           │
│  app.py                 — Flask endpoints                        │
│                                                                  │
│  Models loaded at startup:                                       │
│    ✅ baseline_official  (models/baseline_official/)             │
│    ✅ full_official       (models/full_official/)                 │
│    ❌ dir/edge/hist ablations — not yet trained                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Official Model Registry

| Model ID             | Status             | Checkpoint                                                    | Eval Results                                          |
| -------------------- | ------------------ | ------------------------------------------------------------- | ----------------------------------------------------- |
| `baseline_official`  | ✅ Available       | `models/baseline_official/baseline_official_best.pth` (ep 46) | `results/baseline_ep46/evaluation_results.json`       |
| `full_official`      | ✅ Available       | `models/full_official/full_official_best.pth` (ep 3)          | `results/full_eval/full_best/evaluation_results.json` |
| `dir_only_official`  | ❌ Not yet trained | —                                                             | —                                                     |
| `edge_only_official` | ❌ Not yet trained | —                                                             | —                                                     |
| `hist_only_official` | ❌ Not yet trained | —                                                             | —                                                     |

> **Note:** The `model/` directory contains old prototype checkpoints (247 MB each, no skip connections). These are **not used** by this app — they are preserved for audit purposes only.

---

## Setup

### Prerequisites

- Python 3.10+ with: `torch torchvision flask flask-cors numpy Pillow h5py`
- Node.js 18+ and npm

### Install dependencies (first time only)

```bash
pip install torch torchvision flask flask-cors numpy Pillow h5py

cd artifex/server && npm install
cd artifex/client && npm install
```

### Run

```bash
cd artifex
chmod +x start.sh
./start.sh
```

Open **http://localhost:5173**

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
