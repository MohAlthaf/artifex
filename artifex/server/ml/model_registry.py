"""
model_registry.py
=================
Official thesis model registry for the ARTIFEX serving system.

Responsibilities:
  1. Declare the full *intended* set of official models.
  2. Dynamically detect which checkpoints actually exist on disk.
  3. Load checkpoint metadata (epoch, val_loss, config) from checkpoint files.
  4. Load evaluation summaries from saved evaluation JSON files.
  5. Expose a clean ModelInfo dataclass to the rest of the application.

No model weights are loaded here — weight loading is done lazily in app.py.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

# app.py lives at:  <artifex>/server/ml/app.py
# Thesis root is 2 levels up from this file:
#   <artifex>/server/ml/ → <artifex>/server/ → <artifex>/
# Then one more level for the implementation root:
#   <artifex>/ → implementation/
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_ARTIFEX_DIR = os.path.abspath(os.path.join(_ML_DIR, "..", ".."))
THESIS_ROOT = os.path.abspath(os.path.join(_ARTIFEX_DIR, ".."))

# Benchmark data paths (used by benchmark API)
BENCHMARK_ORIGINAL_DIR = os.path.join(THESIS_ROOT, "data", "processed", "test", "original")
BENCHMARK_MASKED_DIR   = os.path.join(THESIS_ROOT, "data", "processed", "test", "masked")
BENCHMARK_MASKS_DIR    = os.path.join(THESIS_ROOT, "data", "processed", "test", "masks")

# Comparison JSON
COMPARISON_JSON = os.path.join(THESIS_ROOT, "results", "baseline_vs_full_comparison.json")


# ---------------------------------------------------------------------------
# Official model registry definition
# ---------------------------------------------------------------------------

# Each entry defines the *intended* model.  The system will dynamically check
# checkpoint existence and mark unavailable models explicitly.
_REGISTRY_CONFIG: Dict[str, Dict[str, Any]] = {
    "baseline_official": {
        "display_name":   "Baseline SGRGAN",
        "description":    "Baseline model — no brushstroke losses. "
                          "Trained for 60 epochs; best checkpoint at epoch 46.",
        "ablation_losses": [],
        "checkpoint_path": os.path.join(
            THESIS_ROOT, "models", "baseline_official",
            "baseline_official_best.pth",
        ),
        "selection_record": os.path.join(
            THESIS_ROOT, "models", "baseline_official", "selection_record.json"
        ),
        "eval_json": os.path.join(
            THESIS_ROOT, "results", "baseline_ep46_v2", "evaluation_results.json"
        ),
        "restored_images_dir": None,   # Not pre-computed
        "order": 1,
    },
    "full_official": {
        "display_name":   "Full SGRGAN",
        "description":    "Full model — all brushstroke losses active "
                          "(direction + edge strength + histogram). "
                          "Best checkpoint at epoch 3.",
        "ablation_losses": ["direction", "edge_strength", "histogram"],
        "checkpoint_path": os.path.join(
            THESIS_ROOT, "models", "full_official", "full_official_best.pth"
        ),
        "selection_record": os.path.join(
            THESIS_ROOT, "models", "full_official", "selection_record.json"
        ),
        "eval_json": os.path.join(
            THESIS_ROOT, "results", "full_eval_v2",
            "evaluation_results.json",
        ),
        "restored_images_dir": os.path.join(
            THESIS_ROOT, "results", "full_eval", "full_best", "restored_images"
        ),
        "order": 2,
    },
    "dir_only_official": {
        "display_name":   "Direction-Only SGRGAN",
        "description":    "Ablation — direction loss only (λ_direction=2.0). "
                          "15-epoch short suite; best at epoch 15.",
        "ablation_losses": ["direction"],
        "checkpoint_path": os.path.join(
            THESIS_ROOT, "models", "dir_only_15ep_official",
            "dir_only_15ep_official_best.pth",
        ),
        "selection_record": os.path.join(
            THESIS_ROOT, "models", "dir_only_15ep_official", "selection_record.json"
        ),
        "eval_json": os.path.join(
            THESIS_ROOT, "results", "ablation_15ep_comparison.json",
            
        ),
        # /Users/althafali/Downloads/ARTIFEX/implementation/results/ablation_15ep_comparison.json
        "restored_images_dir": None,
        "order": 3,
    },
    "edge_only_official": {
        "display_name":   "Edge-Only SGRGAN",
        "description":    "Ablation — edge-strength loss only (λ_edge=2.0). "
                          "15-epoch short suite; best at epoch 5.",
        "ablation_losses": ["edge_strength"],
        "checkpoint_path": os.path.join(
            THESIS_ROOT, "models", "edge_only_15ep_official",
            "edge_only_15ep_official_best.pth",
        ),
        "selection_record": os.path.join(
            THESIS_ROOT, "models", "edge_only_15ep_official", "selection_record.json"
        ),
        "eval_json": os.path.join(
            THESIS_ROOT, "results", "ablation_15ep_comparison.json",
        ),
        "restored_images_dir": None,
        "order": 4,
    },
    "hist_only_official": {
        "display_name":   "Histogram-Only SGRGAN",
        "description":    "Ablation — histogram loss only (λ_hist=2.0). "
                          "15-epoch short suite; best at epoch 5.",
        "ablation_losses": ["histogram"],
        "checkpoint_path": os.path.join(
            THESIS_ROOT, "models", "hist_only_15ep_official",
            "hist_only_15ep_official_best.pth",
        ),
        "selection_record": os.path.join(
            THESIS_ROOT, "models", "hist_only_15ep_official", "selection_record.json"
        ),
        "eval_json": os.path.join(
            THESIS_ROOT, "results", "ablation_15ep_comparison.json",
        ),
        "restored_images_dir": None,
        "order": 5,
    },
}


# ---------------------------------------------------------------------------
# ModelInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalSummary:
    """Aggregate evaluation metrics for a model over the official test set."""
    num_images: int = 0
    avg_psnr: Optional[float] = None
    avg_ssim: Optional[float] = None
    avg_lpips: Optional[float] = None
    avg_l1: Optional[float] = None
    avg_l2: Optional[float] = None
    avg_direction: Optional[float] = None
    avg_edge_strength: Optional[float] = None
    avg_histogram: Optional[float] = None
    avg_perceptual: Optional[float] = None
    avg_style: Optional[float] = None
    # std fields
    std_psnr: Optional[float] = None
    std_ssim: Optional[float] = None
    label: Optional[str] = None
    epoch: Optional[int] = None
    val_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (None if (v is not None and isinstance(v, float) and math.isnan(v)) else v)
                for k, v in self.__dict__.items()}


@dataclass
class ModelInfo:
    """Complete information about one official model."""
    model_id: str
    display_name: str
    description: str
    ablation_losses: list
    order: int
    available: bool                           # checkpoint file exists on disk
    checkpoint_path: str
    status: str = "missing"                   # 'official', 'experimental', 'missing'
    enabled_for_live_restore: bool = False
    enabled_for_benchmark: bool = False
    selection_record_path: Optional[str] = None
    checkpoint_metadata: Dict[str, Any] = field(default_factory=dict)
    eval_summary: Optional[EvalSummary] = None
    has_restored_images: bool = False
    restored_images_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model_id":             self.model_id,
            "display_name":         self.display_name,
            "description":          self.description,
            "ablation_losses":      self.ablation_losses,
            "order":                self.order,
            "available":            self.available,
            "checkpoint_path":      self.checkpoint_path,
            "status":               self.status,
            "enabled_for_live_restore": self.enabled_for_live_restore,
            "enabled_for_benchmark":    self.enabled_for_benchmark,
            "checkpoint_metadata":  self.checkpoint_metadata,
            "eval_summary":         self.eval_summary.to_dict() if self.eval_summary else None,
            "has_restored_images":  self.has_restored_images,
        }
        return d


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_checkpoint_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load lightweight metadata from a checkpoint without loading model weights.
    Returns a dict with epoch, val_loss, config, model_type etc.
    """
    import torch
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta: Dict[str, Any] = {}
        if isinstance(ckpt, dict):
            for key in ("epoch", "val_loss", "best_val_loss", "config",
                        "model_type", "model_architecture"):
                if key in ckpt:
                    meta[key] = ckpt[key]
        return meta
    except Exception as e:
        return {"load_error": str(e)}


def _load_eval_summary(eval_json_path: Optional[str]) -> Optional[EvalSummary]:
    """Parse the evaluation JSON produced by evaluate_full_model.py."""
    if eval_json_path is None or not os.path.isfile(eval_json_path):
        return None
    try:
        with open(eval_json_path, "r") as f:
            data: Dict[str, Any] = json.load(f)

        def _safe(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                v = float(val)
                return None if math.isnan(v) else v
            except (TypeError, ValueError):
                return None

        return EvalSummary(
            num_images=data.get("num_test_images", len(data.get("per_image", {}))),
            label=data.get("label"),
            epoch=data.get("epoch"),
            val_loss=_safe(data.get("val_loss") or data.get("best_val_loss")),
            avg_psnr=_safe(data.get("avg_psnr")),
            std_psnr=_safe(data.get("std_psnr")),
            avg_ssim=_safe(data.get("avg_ssim")),
            std_ssim=_safe(data.get("std_ssim")),
            avg_l1=_safe(data.get("avg_l1")),
            avg_l2=_safe(data.get("avg_l2")),
            avg_direction=_safe(data.get("avg_direction")),
            avg_edge_strength=_safe(data.get("avg_edge_strength")),
            avg_histogram=_safe(data.get("avg_histogram")),
            avg_perceptual=_safe(data.get("avg_perceptual")),
            avg_style=_safe(data.get("avg_style")),
        )
    except Exception:
        return None


def _has_restored_images(restored_dir: Optional[str]) -> bool:
    if not restored_dir:
        return False
    return os.path.isdir(restored_dir) and bool(os.listdir(restored_dir))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_registry() -> Dict[str, ModelInfo]:
    """
    Build and return the full model registry.

    Checkpoint existence is checked at call time.  Metadata and eval summaries
    are loaded lazily only for models whose checkpoints exist.
    This is called once at Flask startup and again on /api/models.
    """
    registry: Dict[str, ModelInfo] = {}
    for model_id, cfg in sorted(_REGISTRY_CONFIG.items(),
                                key=lambda x: x[1]["order"]):
        ckpt_path = cfg["checkpoint_path"]
        available = os.path.isfile(ckpt_path)

        meta: Dict[str, Any] = {}
        eval_summary: Optional[EvalSummary] = None
        has_restored = _has_restored_images(cfg.get("restored_images_dir"))
        has_eval = False

        if available:
            meta = _load_checkpoint_metadata(ckpt_path)
            eval_summary = _load_eval_summary(cfg.get("eval_json"))
            has_eval = eval_summary is not None

        # Determine status
        if available:
            status = "official"
        else:
            status = "missing"

        registry[model_id] = ModelInfo(
            model_id=model_id,
            display_name=cfg["display_name"],
            description=cfg["description"],
            ablation_losses=cfg["ablation_losses"],
            order=cfg["order"],
            available=available,
            checkpoint_path=ckpt_path,
            status=status,
            enabled_for_live_restore=available,
            enabled_for_benchmark=available and has_eval,
            selection_record_path=cfg.get("selection_record"),
            checkpoint_metadata=meta,
            eval_summary=eval_summary,
            has_restored_images=has_restored,
            restored_images_dir=cfg.get("restored_images_dir"),
        )

    return registry


def get_available_model_ids(registry: Dict[str, ModelInfo]) -> list:
    """Return model_ids that are available (checkpoint exists), ordered."""
    return [
        m.model_id
        for m in sorted(registry.values(), key=lambda x: x.order)
        if m.available
    ]


def load_comparison_json() -> Optional[Dict[str, Any]]:
    """Load the baseline vs full comparison JSON if it exists."""
    if not os.path.isfile(COMPARISON_JSON):
        return None
    try:
        with open(COMPARISON_JSON) as f:
            raw = json.load(f)
        # Sanitise NaN → null for JSON serialisation
        import math

        def _clean(obj: Any) -> Any:
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(i) for i in obj]
            return obj

        return _clean(raw)
    except Exception:
        return None
