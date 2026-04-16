# ARTIFEX

ARTIFEX is a deep learning system for restoring damaged paintings while preserving the artist's brushstroke style. It was built as a final-year thesis project, focused specifically on Van Gogh paintings. The system uses brushstroke-level priors (orientation, edge strength, spatial histograms) as auxiliary training signals to encourage the inpainting model to produce restorations that look structurally faithful to the original painter's technique.

The repository includes everything needed to reproduce the research: a data preprocessing pipeline, brushstroke feature extraction, model training (via Jupyter notebooks), evaluation scripts, statistical significance testing, and a working web demo with a Flask backend and Next.js frontend.

## What This Project Does

Given a damaged painting and a binary mask showing where the damage is, ARTIFEX fills in the missing regions. Unlike generic inpainting, it tries to match the brushstroke patterns of the surrounding area, not just the colour and texture.

The generator uses a dual-stream architecture (BiSCCFormer for texture, FocalBlock for structure) with attention-based fusion and a U-Net decoder. The training loss combines standard terms (L1, VGG perceptual, style, adversarial) with three brushstroke-aware terms that penalise deviations in stroke direction, edge sharpness, and angular distribution.

Five model variants were trained and evaluated on a 305-image test set:

- Baseline (no brushstroke losses)
- Full model (all three brushstroke losses)
- Three single-loss ablations (direction-only, edge-only, histogram-only)

## Key Features

- Complete preprocessing pipeline that resizes, splits, augments masks, and generates synthetic damage
- Brushstroke prior extraction (structure tensor orientation, coherence, Sobel edge strength, spatial angular histograms) stored in a single HDF5 file
- Five trained model checkpoints with per-image evaluation results
- Statistical significance testing across all model pairs (paired Wilcoxon, Bonferroni-corrected)
- Web demo with multi-model side-by-side comparison and optional per-upload metrics when ground truth is provided
- Repository integrity validation script that checks 8 reproducibility categories

## Quick Start

If you have the model checkpoints already in place and just want to run the demo:

```bash
# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install frontend dependencies
cd artifex/client-next && npm install && cd ../..

# Start both servers
cd artifex
chmod +x start.sh
./start.sh
```

Open http://localhost:3000 in your browser. The Flask backend runs on port 5001.

If you are starting completely from scratch (no checkpoints, no processed data), follow the full Installation section below.

## Prerequisites

- **Python 3.10 or newer** (developed on 3.12.7, macOS Apple Silicon M2)
- **Node.js 18 or newer** with npm (for the web demo frontend only)
- **About 10 GB of free disk space** for the processed dataset, HDF5 file (~2.5 GB), and model checkpoints (~276 MB each)
- **Git** for cloning
- A CUDA GPU is helpful for training but not required. The demo runs inference on CPU.

## Installation

### 1. Clone and create a virtual environment

```bash
git clone <repository-url>
cd implementation

python3 -m venv .venv
source .venv/bin/activate
```

On Windows, replace the activate line with `.venv\Scripts\activate`.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins PyTorch 2.3.0 for macOS MPS. If you are on a CUDA machine, install the right PyTorch build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining packages normally.

### 3. Obtain the raw dataset

Place the Van Gogh painting images and base mask templates under `datasets/`:

```
datasets/
  VanGoghPaintingsData/
    VincentVanGogh/         # raw painting image files
    VanGoghPaintings.csv    # metadata CSV
  mask/
    *.png                   # base mask templates
```

This directory is gitignored. You need to supply the data yourself.

### 4. Preprocess the dataset

```bash
python scripts/preprocess_dataset.py \
    --raw-images datasets/VanGoghPaintingsData \
    --mask-dir datasets/mask \
    --output-dir data/processed
```

This discovers images, resizes them to 512x512, splits them 70/15/15 into train/val/test (seed 42), distributes and augments masks, and creates synthetic damaged images. Output goes to `data/processed/`.

### 5. Extract brushstroke priors

```bash
python scripts/extract_brushstroke_h5.py \
    --processed-dir data/processed \
    --output brushstroke_features.h5
```

Alternatively, run the equivalent cells in `artifex_v2.ipynb`.

This produces `brushstroke_features.h5` (~2.5 GB) containing per-image orientation fields, coherence maps, edge strength maps, and spatial angular histograms for all splits.

### 6. Train models

Open `artifex_training.ipynb` from the repository root directory and run the cells in order. Both notebooks resolve paths relative to the working directory, so always launch Jupyter from `implementation/`.

```bash
jupyter notebook artifex_training.ipynb
```

Training produces run directories under `runs/`. After training, copy final checkpoints to the `models/` directory.

### 7. Install frontend dependencies (for the web demo)

```bash
cd artifex/client-next
npm install
cd ../..
```

## Running the Project

### Option A: One-command start (recommended)

```bash
cd artifex
chmod +x start.sh
./start.sh
```

This starts the Flask ML service on port 5001 and the Next.js frontend on port 3000. Press Ctrl+C to stop both.

If `python3` is not on your PATH (for example, you use pyenv):

```bash
PYTHON=/path/to/python3 ./start.sh
```

### Option B: Start services separately

Terminal 1 -- backend:

```bash
cd artifex/server/ml
python3 app.py
```

Terminal 2 -- frontend:

```bash
cd artifex/client-next
npm run dev
```

The backend loads all five model checkpoints at startup, which takes a few seconds. It forces CPU inference to avoid MPS out-of-memory issues on Apple Silicon at 512x512.

## How to Use

1. Open http://localhost:3000 in your browser.
2. Upload a damaged painting image.
3. Optionally add a binary mask showing the damaged regions.
4. Optionally add a clean ground-truth version of the image.
5. Click to run restoration. The system processes the image through all available model variants.
6. Compare the restoration outputs side by side.
7. If ground truth was provided, per-upload metrics (PSNR, SSIM, L1, L2, perceptual, style) are shown. Without ground truth, metrics are intentionally hidden because they would be meaningless.
8. The benchmark evidence section shows official test-set results (n=305) regardless of what you upload.

## Project Structure

```
implementation/
  requirements.txt                   # Python dependencies
  artifex_v2.ipynb                   # Preprocessing and brushstroke H5 generation
  artifex_training.ipynb             # Model training, ablation, and evaluation
  brushstroke_features.h5            # Extracted brushstroke priors (~2.5 GB, gitignored)

  scripts/
    preprocess_dataset.py            # Data pipeline: resize, split, mask, damage
    extract_brushstroke_h5.py        # Standalone brushstroke feature extraction
    evaluate_full_model.py           # Evaluate baseline and full models
    evaluate_ablation.py             # Evaluate ablation variants
    statistical_analysis.py          # Paired Wilcoxon tests with Bonferroni correction
    validate_repo.py                 # 8-category reproducibility check

  models/                            # Frozen checkpoints (gitignored)
    baseline_official/               # Epoch 46, no brushstroke losses
    full_official/                   # Epoch 3, all brushstroke losses
    dir_only_15ep_official/          # Direction loss only
    edge_only_15ep_official/         # Edge strength loss only
    hist_only_15ep_official/         # Histogram loss only
    saved_models/                    # Pre-trained VGG19 for perceptual loss

  data/processed/                    # Processed dataset (gitignored)
    train/                           # 1417 images
    val/                             # 303 images
    test/                            # 305 images
      original/                      # Clean ground-truth images
      masked/                        # Damaged images
      masks/                         # Binary damage masks

  datasets/                          # Raw dataset source (gitignored)

  results/
    final_submission/                # Authoritative final results package
    baseline_ep46_v2/                # Baseline evaluation (authoritative)
    full_eval_v2/                    # Full model evaluation (authoritative)
    official_tables/                 # Ablation metrics CSV
    statistical_tests.json           # Significance test output
    ablation_15ep_comparison.json    # 5-model comparison
    baseline_vs_full_comparison.json # Baseline vs full comparison

  runs/                              # Training run logs and checkpoints (gitignored)

  artifex/                           # Web demo
    start.sh                         # Launch script (Flask + Next.js)
    test_model.py                    # Standalone inference test
    server/ml/
      app.py                         # Flask API (port 5001)
      canonical_inference.py         # Generator architecture and inference
      model_registry.py              # Checkpoint discovery and metadata
      metrics.py                     # Per-upload metric computation
    client-next/                     # Next.js 15 / React 19 frontend
      app/page.jsx                   # Main demo page
      components/                    # UI components
```

## Configuration

There are no `.env` files to configure. The important defaults are:

- **Flask backend port:** 5001 (hardcoded in `artifex/server/ml/app.py`)
- **Next.js frontend port:** 3000 (Next.js default)
- **API URL:** The frontend reads `NEXT_PUBLIC_API_URL` from the environment, defaulting to `http://localhost:5001`. You can override it:
  ```bash
  NEXT_PUBLIC_API_URL=http://some-other-host:5001 npm run dev
  ```
- **Model checkpoint paths:** Resolved automatically by `model_registry.py` relative to the repository root. No manual path configuration needed as long as checkpoints are in `models/`.
- **Processed data paths:** Scripts use `__file__`-relative paths. No hardcoded absolute paths.

## Models and Checkpoints

The system expects five model checkpoints in the `models/` directory:

| Directory                         | Checkpoint file                    | Description                         |
| --------------------------------- | ---------------------------------- | ----------------------------------- |
| `models/baseline_official/`       | `baseline_official_best.pth`       | Baseline, best at epoch 46          |
| `models/full_official/`           | `full_official_best.pth`           | Full model, best at epoch 3         |
| `models/dir_only_15ep_official/`  | `dir_only_15ep_official_best.pth`  | Direction-only ablation, epoch 15   |
| `models/edge_only_15ep_official/` | `edge_only_15ep_official_best.pth` | Edge-only ablation, best at epoch 5 |
| `models/hist_only_15ep_official/` | `hist_only_15ep_official_best.pth` | Histogram-only ablation, epoch 5    |

Each checkpoint is roughly 276 MB. They are gitignored and must be present on disk for the demo or evaluation scripts to work. The Flask server logs which models it successfully loads at startup.

A pre-trained VGG19 used for perceptual loss during training should be in `models/saved_models/`.

Each model directory also contains a `selection_record.json` documenting why that particular checkpoint was chosen (best validation loss epoch).

## Evaluation and Results

All evaluation was done on the 305-image test split across 10 metrics: PSNR, SSIM, LPIPS, L1, L2, perceptual loss, style loss, direction similarity, edge strength similarity, and histogram similarity.

To re-run evaluation:

```bash
python scripts/evaluate_full_model.py
python scripts/evaluate_ablation.py
```

To run statistical significance testing (30 paired Wilcoxon signed-rank tests with Bonferroni correction):

```bash
python scripts/statistical_analysis.py
```

Final results live in `results/final_submission/`. See `results/final_submission/README.md` for what each file contains.

The folders `results/baseline_ep46/` and `results/full_eval/` are stale v1 outputs with LPIPS=NaN from an earlier evaluation run. They are kept because `statistical_analysis.py` reads per-image brushstroke data from them. Do not use them as final figures.

## Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
Your virtual environment is not activated. Run `source .venv/bin/activate` first.

**Flask fails to start or models do not load**
Check that the `.pth` checkpoint files are actually present in `models/`. They are gitignored and will not be there after a fresh clone. The server logs which models it finds and which it fails to load.

**Port 5001 or 3000 already in use**
Kill the existing process:

```bash
lsof -ti:5001 | xargs kill
lsof -ti:3000 | xargs kill
```

**Next.js cannot reach the Flask backend**
Make sure the Flask server is running and healthy:

```bash
curl http://localhost:5001/health
```

If the backend is on a different host or port, set `NEXT_PUBLIC_API_URL` before starting the frontend.

**FileNotFoundError for dataset or H5 file**
The processed data and `brushstroke_features.h5` are gitignored. You need to run `scripts/preprocess_dataset.py` and `scripts/extract_brushstroke_h5.py` first, or obtain them separately.

**LPIPS shows NaN in some result files**
Use the `*_v2` result folders or `results/final_submission/`. The original v1 evaluation had an AlexNet download failure that produced NaN LPIPS values.

**PyTorch MPS warnings on Apple Silicon**
Expected. The demo forces CPU inference because PyTorch's MultiheadAttention runs out of memory on MPS at 512x512 input size.

**Verifying your setup is correct**

```bash
python scripts/validate_repo.py
```

This checks data, H5, models, runs, results, scripts, architecture, and demo in 8 categories.

## Limitations

- The full model's best validation loss was at epoch 3 out of 50. Validation loss increased monotonically after that. The saved checkpoint is from epoch 3, which limits what the model learned.
- Ablation models were trained for only 15 epochs due to compute constraints, compared to 50-60 for the baseline and full model. This makes the ablation comparison uneven.
- The full model improves brushstroke fidelity scores but slightly increases LPIPS compared to the baseline. This trade-off is inherent to the multi-loss formulation.
- The histogram similarity metric was implemented and used in training but was never validated with a standalone numerical unit test.
- Inference runs on CPU. It works but is not fast.

## Intended Use

This is a thesis research project. It is intended for academic evaluation and as a proof of concept for brushstroke-aware inpainting. It is not a production-ready restoration tool and should not be used as one. The dataset is limited to Van Gogh paintings at 512x512 resolution.

## Contributing

This repository was developed as a solo thesis project. If you find issues or have suggestions, feel free to open an issue. Pull requests are welcome but there is no formal contribution workflow in place.
