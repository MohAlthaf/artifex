"""
metrics.py — Per-upload metric computation for ARTIFEX
======================================================
Computes image quality metrics between a restored image and ground truth.
Used ONLY when the user provides a clean ground-truth image.

Metrics computed:
  - PSNR (dB)          — Peak Signal-to-Noise Ratio
  - SSIM               — Structural Similarity Index
  - L1                 — Mean Absolute Error
  - L2                 — Mean Squared Error
  - Perceptual         — VGG-based perceptual loss (if available)
  - Style              — VGG-based style loss (if available)

Note: Brushstroke-specific metrics (direction, edge_strength, histogram)
require pre-extracted feature maps from the H5 dataset and are only
available for the official benchmark test set, not for user uploads.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Basic image metrics (tensor-based, no extra deps)
# ---------------------------------------------------------------------------

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    PSNR between two images.
    Args: img1, img2 — (C, H, W) tensors in [0, 1]
    """
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> float:
    """
    SSIM between two images.
    Args: img1, img2 — (C, H, W) tensors in [0, 1]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Add batch dim
    x = img1.unsqueeze(0)
    y = img2.unsqueeze(0)
    channels = x.shape[1]

    # Gaussian window
    sigma = 1.5
    gauss = torch.Tensor(
        [
            math.exp(-((i - window_size // 2) ** 2) / (2 * sigma ** 2))
            for i in range(window_size)
        ]
    )
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()
    window = window.to(x.device)

    mu1 = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(y, window, padding=window_size // 2, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu2_sq
    )
    sigma12 = (
        F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def calculate_l1(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean Absolute Error. Args: (C,H,W) tensors in [0,1]."""
    return F.l1_loss(img1, img2).item()


def calculate_l2(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean Squared Error. Args: (C,H,W) tensors in [0,1]."""
    return F.mse_loss(img1, img2).item()


# ---------------------------------------------------------------------------
# VGG-based metrics (perceptual + style)
# ---------------------------------------------------------------------------

_vgg_features = None


def _load_vgg(device: torch.device):
    """Load VGG19 feature extractor (first time only)."""
    global _vgg_features
    if _vgg_features is not None:
        return _vgg_features

    try:
        from torchvision import models
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        vgg = vgg.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        _vgg_features = vgg
        print("[METRICS] VGG19 loaded for perceptual/style metrics")
        return vgg
    except Exception as e:
        print(f"[METRICS] VGG19 not available: {e}")
        return None


def _extract_vgg_features(
    vgg: torch.nn.Module, x: torch.Tensor, layers: Tuple[int, ...] = (3, 8, 17, 26, 35)
) -> list:
    """Extract features from specified VGG layers."""
    features = []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in layers:
            features.append(x)
    return features


def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for style loss."""
    b, c, h, w = x.shape
    f = x.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)


def calculate_perceptual_loss(
    img1: torch.Tensor, img2: torch.Tensor, device: torch.device
) -> Optional[float]:
    """
    VGG-based perceptual loss.
    Args: img1, img2 — (C, H, W) tensors in [0, 1]
    """
    vgg = _load_vgg(device)
    if vgg is None:
        return None

    x = img1.unsqueeze(0).to(device)
    y = img2.unsqueeze(0).to(device)

    # Normalize to ImageNet stats for VGG
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    x = (x - mean) / std
    y = (y - mean) / std

    feat_x = _extract_vgg_features(vgg, x)
    feat_y = _extract_vgg_features(vgg, y)

    loss = sum(F.l1_loss(fx, fy) for fx, fy in zip(feat_x, feat_y))
    return loss.item()


def calculate_style_loss(
    img1: torch.Tensor, img2: torch.Tensor, device: torch.device
) -> Optional[float]:
    """
    VGG-based style loss (Gram matrix comparison).
    Args: img1, img2 — (C, H, W) tensors in [0, 1]
    """
    vgg = _load_vgg(device)
    if vgg is None:
        return None

    x = img1.unsqueeze(0).to(device)
    y = img2.unsqueeze(0).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    x = (x - mean) / std
    y = (y - mean) / std

    feat_x = _extract_vgg_features(vgg, x)
    feat_y = _extract_vgg_features(vgg, y)

    loss = sum(
        F.l1_loss(_gram_matrix(fx), _gram_matrix(fy))
        for fx, fy in zip(feat_x, feat_y)
    )
    return loss.item()


# ---------------------------------------------------------------------------
# Public API: compute all available metrics
# ---------------------------------------------------------------------------

def compute_per_upload_metrics(
    restored_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Compute all available metrics between restored and ground truth.

    Args:
        restored_tensor : (1, 3, H, W) in [0, 1] — composed model output
        gt_tensor       : (1, 3, H, W) in [0, 1] — user-provided ground truth

    Returns:
        Dict with metric values. Keys match thesis metric names.
        None values indicate the metric was not computable.
    """
    # Squeeze batch dim for per-image metrics
    r = restored_tensor.squeeze(0)  # (3, H, W)
    g = gt_tensor.squeeze(0)        # (3, H, W)

    metrics: Dict[str, Any] = {}

    # Core metrics — always computable
    metrics["psnr"] = round(calculate_psnr(r, g), 4)
    metrics["ssim"] = round(calculate_ssim(r, g), 6)
    metrics["l1"] = round(calculate_l1(r, g), 6)
    metrics["l2"] = round(calculate_l2(r, g), 6)

    # VGG-based metrics — require torchvision
    with torch.no_grad():
        perceptual = calculate_perceptual_loss(r, g, device)
        style = calculate_style_loss(r, g, device)

    metrics["perceptual"] = round(perceptual, 6) if perceptual is not None else None
    metrics["style"] = round(style, 6) if style is not None else None

    # Brushstroke metrics — NOT available for user uploads
    # These require pre-extracted orientation/coherence/edge/histogram feature maps
    # from the H5 dataset, which are only available for the benchmark test set.
    metrics["direction"] = None
    metrics["edge_strength"] = None
    metrics["histogram"] = None
    metrics["_brushstroke_note"] = (
        "Brushstroke metrics (direction, edge_strength, histogram) require "
        "pre-extracted feature maps and are available only for benchmark images."
    )

    return metrics
