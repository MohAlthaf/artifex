"""
app.py — ARTIFEX ML Service

Flask server (port 5001) for model discovery, live restoration,
and benchmark image/metric serving. Loads only official thesis
checkpoints; forces CPU to avoid MPS OOM on Apple Silicon.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import threading
from typing import Any, Dict, List, Optional

import torch
from flask import Flask, abort, jsonify, request, send_file
from flask_cors import CORS

from canonical_inference import (
    ArtifexGenerator,
    load_generator,
    run_inference,
    select_device,
    preprocess_image,
    postprocess_image,
)
from model_registry import (
    BENCHMARK_MASKED_DIR,
    BENCHMARK_MASKS_DIR,
    BENCHMARK_ORIGINAL_DIR,
    ModelInfo,
    build_registry,
    get_available_model_ids,
    load_comparison_json,
)
from metrics import compute_per_upload_metrics

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Global state — initialised once at startup
# ---------------------------------------------------------------------------

DEVICE = select_device()
print(f"[ARTIFEX] Using device: {DEVICE}")

# Model registry: {model_id: ModelInfo}
_registry: Dict[str, ModelInfo] = {}

# Loaded generators (only available models)
_generators: Dict[str, ArtifexGenerator] = {}
_generators_lock = threading.Lock()

# Benchmark per-image metrics cache: {model_id: {image_name: {metric: value}}}
_benchmark_metrics: Dict[str, Dict[str, Any]] = {}

# Benchmark sample list (shared across models — uses test split)
_benchmark_samples: List[str] = []


def _init():
    """Load registry, generators, and benchmark data at startup."""
    global _registry, _generators, _benchmark_metrics, _benchmark_samples

    print("[ARTIFEX] Building model registry …")
    _registry = build_registry()

    available = get_available_model_ids(_registry)
    print(f"[ARTIFEX] Available models ({len(available)}): {available}")
    unavailable = [m for m in _registry if m not in available]
    print(f"[ARTIFEX] Unavailable models ({len(unavailable)}): {unavailable}")

    # Load generators for available models
    for model_id in available:
        info = _registry[model_id]
        print(f"[ARTIFEX] Loading {model_id} from {info.checkpoint_path} …")
        try:
            gen = load_generator(info.checkpoint_path, device=DEVICE)
            _generators[model_id] = gen
            print(f"[ARTIFEX]   ✓ {model_id} loaded")
        except Exception as e:
            print(f"[ARTIFEX]   ✗ {model_id} FAILED to load: {e}")

    # Load benchmark per-image metrics
    for model_id, info in _registry.items():
        if info.eval_summary is None:
            continue
        eval_json_path = info.eval_summary  # we need the raw path…
        # Re-read from registry config to get per_image data
        for cfg_id, cfg in _get_registry_configs().items():
            if cfg_id == model_id and cfg.get("eval_json"):
                _load_per_image_metrics(model_id, cfg["eval_json"])

    # Build benchmark sample list from original dir
    if os.path.isdir(BENCHMARK_ORIGINAL_DIR):
        _benchmark_samples = sorted(
            f for f in os.listdir(BENCHMARK_ORIGINAL_DIR)
            if f.lower().endswith(".png")
        )
        print(f"[ARTIFEX] Benchmark samples: {len(_benchmark_samples)} images")
    else:
        print(f"[ARTIFEX] WARNING: benchmark original dir not found: {BENCHMARK_ORIGINAL_DIR}")


def _get_registry_configs():
    """Re-import registry config to access eval_json paths."""
    from model_registry import _REGISTRY_CONFIG
    return _REGISTRY_CONFIG


def _load_per_image_metrics(model_id: str, eval_json_path: str):
    if not os.path.isfile(eval_json_path):
        return
    try:
        with open(eval_json_path) as f:
            raw = json.load(f)
        per_image = raw.get("per_image", {})
        # Sanitise NaN
        cleaned = {}
        for img_name, metrics in per_image.items():
            cleaned[img_name] = {
                k: (None if (v is not None and isinstance(v, float) and math.isnan(v)) else v)
                for k, v in metrics.items()
            }
        _benchmark_metrics[model_id] = cleaned
        print(f"[ARTIFEX]   ✓ Loaded {len(cleaned)} per-image metrics for {model_id}")
    except Exception as e:
        print(f"[ARTIFEX]   ✗ Failed to load benchmark metrics for {model_id}: {e}")


# ---------------------------------------------------------------------------
# Helper: image to base64 PNG
# ---------------------------------------------------------------------------

def _bytes_to_b64_png(image_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode()


def _file_to_b64_png(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return _bytes_to_b64_png(f.read())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "available_models": list(_generators.keys()),
    })


@app.route("/info")
def info():
    """System info and version endpoint for thesis demo."""
    return jsonify({
        "system": "ARTIFEX ML Service",
        "version": "2.0",
        "architecture": "Flask + PyTorch (CPU inference)",
        "model_architecture": "ArtifexGenerator (dual-stream: BiSCCFormer + FocalModulation, attention fusion, U-Net decoder)",
        "input_channels": 4,
        "input_size": "512x512",
        "device": str(DEVICE),
        "available_models": list(_generators.keys()),
        "total_registered_models": len(_registry),
        "benchmark_samples": len(_benchmark_samples),
        "official_eval_source": "results/baseline_ep46_v2 and results/full_eval_v2",
    })


# ---------------------------
# Model discovery
# ---------------------------

@app.route("/api/models")
def list_models():
    """Full registry with availability, metadata, and eval summaries."""
    return jsonify({
        "models": {mid: info.to_dict() for mid, info in _registry.items()},
        "available_model_ids": get_available_model_ids(_registry),
    })


# ---------------------------
# Restore endpoints
# ---------------------------

@app.route("/api/restore-all", methods=["POST"])
def restore_all():
    """Run inference through all available models. Form-data: image, mask (optional)."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = request.files["image"].read()
    mask_bytes = request.files["mask"].read() if "mask" in request.files else None

    results = []
    for model_id in sorted(_registry.keys(), key=lambda m: _registry[m].order):
        info = _registry[model_id]
        if not info.available:
            results.append({
                "model_id":       model_id,
                "display_name":   info.display_name,
                "available":      False,
                "restored_image_b64": None,
                "error":          "Checkpoint not yet available — training in progress",
                "official_eval_summary": None,
            })
            continue

        gen = _generators.get(model_id)
        if gen is None:
            results.append({
                "model_id":     model_id,
                "display_name": info.display_name,
                "available":    False,
                "restored_image_b64": None,
                "error":        "Model failed to load at startup",
                "official_eval_summary": None,
            })
            continue

        try:
            restored_bytes, run_info = run_inference(
                gen, image_bytes, mask_bytes,
                device=DEVICE,
                model_name=model_id,
                checkpoint_path=info.checkpoint_path,
                save_debug=True,
            )
            eval_summary_dict = (info.eval_summary.to_dict()
                                 if info.eval_summary else None)
            results.append({
                "model_id":         model_id,
                "display_name":     info.display_name,
                "available":        True,
                "restored_image_b64": _bytes_to_b64_png(restored_bytes),
                "error":            None,
                "checkpoint_path":  info.checkpoint_path,
                "checkpoint_metadata": info.checkpoint_metadata,
                "official_eval_summary": eval_summary_dict,
                "inference_time_s":  run_info.get("inference_time_s"),
                "mask_coverage_pct": run_info.get("mask_coverage_pct"),
            })
        except Exception as e:
            results.append({
                "model_id":     model_id,
                "display_name": info.display_name,
                "available":    True,
                "restored_image_b64": None,
                "error":        str(e),
                "official_eval_summary": None,
            })

    return jsonify({
        "results": results,
        "metric_disclaimer": (
            "Metrics shown are official test-set averages computed over the "
            "Van Gogh benchmark (n=305 images with ground truth). "
            "They are NOT computed for your uploaded image. "
            "Use the Benchmark Explorer to view per-image metrics."
        ),
    })


@app.route("/api/restore-one", methods=["POST"])
def restore_one():
    """Restore with a single model. Returns PNG binary."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    model_id = request.form.get("model_id", "baseline_official")

    info = _registry.get(model_id)
    if info is None:
        return jsonify({"error": f"Unknown model_id: {model_id}"}), 400
    if not info.available:
        return jsonify({"error": "Checkpoint not yet available"}), 503

    gen = _generators.get(model_id)
    if gen is None:
        return jsonify({"error": "Model failed to load at startup"}), 503

    image_bytes = request.files["image"].read()
    mask_bytes = request.files["mask"].read() if "mask" in request.files else None

    try:
        restored_bytes, _info = run_inference(
            gen, image_bytes, mask_bytes,
            device=DEVICE,
            model_name=model_id,
            checkpoint_path=info.checkpoint_path,
        )
        return send_file(
            io.BytesIO(restored_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name=f"restored_{model_id}.png",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# v3: Per-upload evaluation (thesis demo)
# ---------------------------

# @app.route("/api/restore-with-eval", methods=["POST"])
# def restore_with_eval():
#     """Restore all models + compute per-upload metrics if ground truth provided."""
#     import time as _time

#     if "image" not in request.files:
#         return jsonify({"error": "No image provided"}), 400

#     image_bytes = request.files["image"].read()
#     mask_bytes = request.files["mask"].read() if "mask" in request.files else None
#     gt_bytes = (
#         request.files["ground_truth"].read()
#         if "ground_truth" in request.files
#         else None
#     )

#     has_gt = gt_bytes is not None

#     # Pre-process ground truth if provided (for metric computation)
#     gt_tensor = None
#     if has_gt:
#         from PIL import Image as _PIL
#         import numpy as _np

#         pil_gt = _PIL.open(io.BytesIO(gt_bytes)).convert("RGB")
#         pil_gt_resized = pil_gt.resize((512, 512), _PIL.LANCZOS)
#         gt_arr = _np.array(pil_gt_resized, dtype=_np.float32) / 255.0
#         gt_tensor = (
#             torch.from_numpy(gt_arr)
#             .permute(2, 0, 1)
#             .unsqueeze(0)
#             .to(DEVICE)
#         )

#     results = []
#     for model_id in sorted(_registry.keys(), key=lambda m: _registry[m].order):
#         info = _registry[model_id]

#         if not info.available:
#             results.append({
#                 "model_id":       model_id,
#                 "display_name":   info.display_name,
#                 "available":      False,
#                 "restored_image_b64": None,
#                 "error":          "Checkpoint not yet available — training in progress",
#                 "inference_time_s": None,
#                 "mask_coverage_pct": None,
#                 "checkpoint_path": None,
#                 "checkpoint_metadata": None,
#                 "official_eval_summary": None,
#                 "per_upload_metrics": None,
#             })
#             continue

#         gen = _generators.get(model_id)
#         if gen is None:
#             results.append({
#                 "model_id":       model_id,
#                 "display_name":   info.display_name,
#                 "available":      False,
#                 "restored_image_b64": None,
#                 "error":          "Model failed to load at startup",
#                 "inference_time_s": None,
#                 "mask_coverage_pct": None,
#                 "checkpoint_path": None,
#                 "checkpoint_metadata": None,
#                 "official_eval_summary": None,
#                 "per_upload_metrics": None,
#             })
#             continue

#         try:
#             restored_bytes, run_info = run_inference(
#                 gen, image_bytes, mask_bytes,
#                 device=DEVICE,
#                 model_name=model_id,
#                 checkpoint_path=info.checkpoint_path,
#                 save_debug=True,
#             )

#             # Compute per-upload metrics if GT provided
#             per_upload_metrics = None
#             if has_gt and gt_tensor is not None:
#                 try:
#                     # Re-process the composed output as a tensor for metrics
#                     from canonical_inference import preprocess_image as _preprocess
#                     from PIL import Image as _PIL
#                     import numpy as _np

#                     pil_restored = _PIL.open(io.BytesIO(restored_bytes)).convert("RGB")
#                     pil_restored_resized = pil_restored.resize((512, 512), _PIL.LANCZOS)
#                     r_arr = _np.array(pil_restored_resized, dtype=_np.float32) / 255.0
#                     restored_tensor = (
#                         torch.from_numpy(r_arr)
#                         .permute(2, 0, 1)
#                         .unsqueeze(0)
#                         .to(DEVICE)
#                     )
#                     _, metric_mask_tensor, _ = preprocess_image(image_bytes, mask_bytes, device=DEVICE)

#                     per_upload_metrics = compute_per_upload_metrics(
#                         restored_tensor,
#                         gt_tensor,
#                         DEVICE,
#                         mask_tensor=metric_mask_tensor,
#                     )
#                 except Exception as metric_err:
#                     print(f"[ARTIFEX] Metric computation failed for {model_id}: {metric_err}")
#                     per_upload_metrics = {"error": str(metric_err)}

#             eval_summary_dict = (
#                 info.eval_summary.to_dict() if info.eval_summary else None
#             )

#             results.append({
#                 "model_id":         model_id,
#                 "display_name":     info.display_name,
#                 "available":        True,
#                 "restored_image_b64": _bytes_to_b64_png(restored_bytes),
#                 "error":            None,
#                 "inference_time_s":  run_info.get("inference_time_s"),
#                 "mask_coverage_pct": run_info.get("mask_coverage_pct"),
#                 "checkpoint_path":   info.checkpoint_path,
#                 "checkpoint_metadata": info.checkpoint_metadata,
#                 "official_eval_summary": eval_summary_dict,
#                 "per_upload_metrics": per_upload_metrics,
#             })
#         except Exception as e:
#             results.append({
#                 "model_id":     model_id,
#                 "display_name": info.display_name,
#                 "available":    True,
#                 "restored_image_b64": None,
#                 "error":        str(e),
#                 "inference_time_s": None,
#                 "mask_coverage_pct": None,
#                 "checkpoint_path": None,
#                 "checkpoint_metadata": None,
#                 "official_eval_summary": None,
#                 "per_upload_metrics": None,
#             })

#     # Metric policy message
#     if has_gt:
#         policy = (
#             "Per-upload metrics (PSNR, SSIM, L1, L2, Perceptual, Style) are "
#             "computed against YOUR provided ground truth for this specific upload. "
#             "Brushstroke-specific metrics (direction, edge strength, histogram) "
#             "require pre-extracted feature maps and are available only for "
#             "official benchmark images."
#         )
#     else:
#         policy = (
#             "No ground truth provided — per-upload metrics are NOT shown. "
#             "This is by design: computing PSNR/SSIM without valid ground truth "
#             "would be misleading. Upload a clean ground-truth image alongside "
#             "the damaged image to see real per-upload metrics."
#         )

#     return jsonify({
#         "has_ground_truth": has_gt,
#         "results": results,
#         "metric_policy": policy,
#     })

@app.route("/api/restore-with-eval", methods=["POST"])
def restore_with_eval():
    """Restore all models + compute per-upload metrics if ground truth provided."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = request.files["image"].read()
    mask_bytes = request.files["mask"].read() if "mask" in request.files else None
    gt_bytes = (
        request.files["ground_truth"].read()
        if "ground_truth" in request.files
        else None
    )

    has_gt = gt_bytes is not None

    gt_tensor = None
    if has_gt:
        from PIL import Image as _PIL
        import numpy as _np

        pil_gt = _PIL.open(io.BytesIO(gt_bytes)).convert("RGB")
        pil_gt_resized = pil_gt.resize((512, 512), _PIL.LANCZOS)
        gt_arr = _np.array(pil_gt_resized, dtype=_np.float32) / 255.0
        gt_tensor = (
            torch.from_numpy(gt_arr)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

    metric_mask_tensor = None
    if has_gt and mask_bytes is not None:
        try:
            _, metric_mask_tensor, _ = preprocess_image(
                image_bytes,
                mask_bytes,
                device=DEVICE,
            )
        except Exception as mask_err:
            print(f"[ARTIFEX] Failed to preprocess mask for upload metrics: {mask_err}")
            metric_mask_tensor = None

    results = []

    for model_id in sorted(_registry.keys(), key=lambda m: _registry[m].order):
        info = _registry[model_id]

        if not info.available:
            results.append({
                "model_id": model_id,
                "display_name": info.display_name,
                "available": False,
                "restored_image_b64": None,
                "error": "Checkpoint not yet available — training in progress",
                "inference_time_s": None,
                "mask_coverage_pct": None,
                "checkpoint_path": None,
                "checkpoint_metadata": None,
                "official_eval_summary": None,
                "per_upload_metrics": None,
            })
            continue

        gen = _generators.get(model_id)
        if gen is None:
            results.append({
                "model_id": model_id,
                "display_name": info.display_name,
                "available": False,
                "restored_image_b64": None,
                "error": "Model failed to load at startup",
                "inference_time_s": None,
                "mask_coverage_pct": None,
                "checkpoint_path": None,
                "checkpoint_metadata": None,
                "official_eval_summary": None,
                "per_upload_metrics": None,
            })
            continue

        try:
            restored_bytes, run_info = run_inference(
                gen,
                image_bytes,
                mask_bytes,
                device=DEVICE,
                model_name=model_id,
                checkpoint_path=info.checkpoint_path,
                save_debug=True,
            )

            per_upload_metrics = None
            if has_gt and gt_tensor is not None:
                try:
                    from PIL import Image as _PIL
                    import numpy as _np

                    pil_restored = _PIL.open(io.BytesIO(restored_bytes)).convert("RGB")
                    pil_restored_resized = pil_restored.resize((512, 512), _PIL.LANCZOS)
                    r_arr = _np.array(pil_restored_resized, dtype=_np.float32) / 255.0

                    restored_tensor = (
                        torch.from_numpy(r_arr)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )

                    per_upload_metrics = compute_per_upload_metrics(
                        restored_tensor,
                        gt_tensor,
                        DEVICE,
                        mask_tensor=metric_mask_tensor,
                    )
                except Exception as metric_err:
                    print(f"[ARTIFEX] Metric computation failed for {model_id}: {metric_err}")
                    per_upload_metrics = {"error": str(metric_err)}

            eval_summary_dict = (
                info.eval_summary.to_dict() if info.eval_summary else None
            )

            results.append({
                "model_id": model_id,
                "display_name": info.display_name,
                "available": True,
                "restored_image_b64": _bytes_to_b64_png(restored_bytes),
                "error": None,
                "inference_time_s": run_info.get("inference_time_s"),
                "mask_coverage_pct": run_info.get("mask_coverage_pct"),
                "checkpoint_path": info.checkpoint_path,
                "checkpoint_metadata": info.checkpoint_metadata,
                "official_eval_summary": eval_summary_dict,
                "per_upload_metrics": per_upload_metrics,
            })
        except Exception as e:
            results.append({
                "model_id": model_id,
                "display_name": info.display_name,
                "available": True,
                "restored_image_b64": None,
                "error": str(e),
                "inference_time_s": None,
                "mask_coverage_pct": None,
                "checkpoint_path": None,
                "checkpoint_metadata": None,
                "official_eval_summary": None,
                "per_upload_metrics": None,
            })

    if has_gt:
        policy = (
            "Per-upload metrics are computed against your provided ground truth for this specific upload. "
            "PSNR, SSIM, L1, L2, Perceptual, and Style are full-image metrics. "
            "Direction, Edge Strength, and Histogram are computed live from the upload using the active mask."
        )
    else:
        policy = (
            "No ground truth provided — per-upload metrics are NOT shown. "
            "This is by design: computing PSNR/SSIM without valid ground truth "
            "would be misleading. Upload a clean ground-truth image alongside "
            "the damaged image to see real per-upload metrics."
        )

    return jsonify({
        "has_ground_truth": has_gt,
        "results": results,
        "metric_policy": policy,
    })
# ---------------------------
# Benchmark endpoints
# ---------------------------

@app.route("/api/benchmark/models")
def benchmark_models():
    """Models that have eval data (benchmark-capable)."""
    bm_models = {}
    for mid, info in _registry.items():
        bm_models[mid] = {
            "model_id":       mid,
            "display_name":   info.display_name,
            "available":      info.available,
            "has_eval":       info.eval_summary is not None,
            "has_per_image_metrics": mid in _benchmark_metrics,
            "has_restored_images": info.has_restored_images,
            "eval_summary":   info.eval_summary.to_dict() if info.eval_summary else None,
        }
    return jsonify(bm_models)


@app.route("/api/benchmark/samples")
def benchmark_samples():
    """Paginated list of benchmark test sample names."""
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    total = len(_benchmark_samples)
    start = (page - 1) * per_page
    end = start + per_page
    batch = _benchmark_samples[start:end]
    return jsonify({
        "samples": batch,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": math.ceil(total / per_page) if total else 0,
    })


@app.route("/api/benchmark/sample/<sample_id>")
def benchmark_sample(sample_id: str):
    """All images and per-image metrics for a single benchmark sample."""
    # Sanitise sample_id
    if not sample_id.endswith(".png"):
        sample_id = sample_id + ".png"
    if "/" in sample_id or ".." in sample_id:
        abort(400)

    damaged_path   = os.path.join(BENCHMARK_MASKED_DIR,   sample_id)
    mask_path      = os.path.join(BENCHMARK_MASKS_DIR,    sample_id)
    original_path  = os.path.join(BENCHMARK_ORIGINAL_DIR, sample_id)

    def _b64(path: str) -> Optional[str]:
        return _file_to_b64_png(path)

    response: Dict[str, Any] = {
        "sample_id":         sample_id,
        "damaged_b64":       _b64(damaged_path),
        "mask_b64":          _b64(mask_path),
        "ground_truth_b64":  _b64(original_path),
        "restored": {},
        "per_image_metrics": {},
    }

    # Add pre-computed restored images
    for model_id, info in _registry.items():
        if info.has_restored_images and info.restored_images_dir:
            # Files saved as "<sample_id>.png" — but eval script adds .png.png
            # Try both naming conventions
            candidates = [
                os.path.join(info.restored_images_dir, sample_id),
                os.path.join(info.restored_images_dir, sample_id + ".png"),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    response["restored"][model_id] = _b64(cand)
                    break

    # Add per-image metrics
    for model_id, per_image in _benchmark_metrics.items():
        if sample_id in per_image:
            response["per_image_metrics"][model_id] = per_image[sample_id]

    return jsonify(response)


@app.route("/api/benchmark/metrics/<model_id>")
def benchmark_metrics(model_id: str):
    """Aggregate eval summary + per-image metrics for a model."""
    info = _registry.get(model_id)
    if info is None:
        return jsonify({"error": f"Unknown model_id: {model_id}"}), 404

    per_image_data = _benchmark_metrics.get(model_id, {})
    return jsonify({
        "model_id":       model_id,
        "display_name":   info.display_name,
        "eval_summary":   info.eval_summary.to_dict() if info.eval_summary else None,
        "per_image":      per_image_data,
        "num_images":     len(per_image_data),
    })


@app.route("/api/benchmark/comparison")
def benchmark_comparison():
    """Saved baseline vs full comparison JSON."""
    data = load_comparison_json()
    if data is None:
        return jsonify({"error": "Comparison JSON not yet generated"}), 404
    return jsonify(data)


# ---------------------------------------------------------------------------
# Legacy compatibility — keep /predict for any old clients
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_legacy():
    """Legacy endpoint: kept for backward compatibility. Uses baseline_official."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    gen = _generators.get("baseline_official") or next(iter(_generators.values()), None)
    if gen is None:
        return jsonify({"error": "No models loaded"}), 503

    model_id = "baseline_official" if "baseline_official" in _generators else next(iter(_generators.keys()))
    image_bytes = request.files["image"].read()
    mask_bytes = request.files["mask"].read() if "mask" in request.files else None

    try:
        restored_bytes, _info = run_inference(
            gen, image_bytes, mask_bytes,
            device=DEVICE,
            model_name=model_id,
            checkpoint_path=_registry[model_id].checkpoint_path,
        )
        return send_file(io.BytesIO(restored_bytes), mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
