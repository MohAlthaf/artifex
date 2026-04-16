# """
# metrics.py — Per-upload image quality metrics.

# Computes PSNR, SSIM, L1, L2, perceptual, and style loss when the user
# provides a ground-truth image. Brushstroke metrics (direction, edge, histogram)
# are only available for official benchmark images via pre-extracted H5 features.
# """

# from __future__ import annotations

# import math
# from typing import Any, Dict, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn.functional as F

# # ---------------------------------------------------------------------------
# # Basic image metrics (tensor-based, no extra deps)
# # ---------------------------------------------------------------------------

# def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
#     """PSNR in dB. Inputs: (C,H,W) in [0,1]."""
#     mse = F.mse_loss(img1, img2).item()
#     if mse < 1e-10:
#         return 100.0
#     return 10.0 * math.log10(1.0 / mse)


# def calculate_ssim(
#     img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
# ) -> float:
#     """SSIM. Inputs: (C,H,W) in [0,1]."""
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     # Add batch dim
#     x = img1.unsqueeze(0)
#     y = img2.unsqueeze(0)
#     channels = x.shape[1]

#     # Gaussian window
#     sigma = 1.5
#     gauss = torch.Tensor(
#         [
#             math.exp(-((i - window_size // 2) ** 2) / (2 * sigma ** 2))
#             for i in range(window_size)
#         ]
#     )
#     gauss = gauss / gauss.sum()
#     _1d = gauss.unsqueeze(1)
#     _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
#     window = _2d.expand(channels, 1, window_size, window_size).contiguous()
#     window = window.to(x.device)

#     mu1 = F.conv2d(x, window, padding=window_size // 2, groups=channels)
#     mu2 = F.conv2d(y, window, padding=window_size // 2, groups=channels)
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = (
#         F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu1_sq
#     )
#     sigma2_sq = (
#         F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu2_sq
#     )
#     sigma12 = (
#         F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu1_mu2
#     )

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
#         (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
#     )
#     return ssim_map.mean().item()


# def calculate_l1(img1: torch.Tensor, img2: torch.Tensor) -> float:
#     """Mean Absolute Error. Args: (C,H,W) tensors in [0,1]."""
#     return F.l1_loss(img1, img2).item()


# def calculate_l2(img1: torch.Tensor, img2: torch.Tensor) -> float:
#     """Mean Squared Error. Args: (C,H,W) tensors in [0,1]."""
#     return F.mse_loss(img1, img2).item()


# # ---------------------------------------------------------------------------
# # VGG-based metrics (perceptual + style)
# # ---------------------------------------------------------------------------

# _vgg_features = None


# def _load_vgg(device: torch.device):
#     """Load VGG19 features (cached after first call)."""
#     global _vgg_features
#     if _vgg_features is not None:
#         return _vgg_features

#     try:
#         from torchvision import models
#         vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
#         vgg = vgg.to(device).eval()
#         for p in vgg.parameters():
#             p.requires_grad = False
#         _vgg_features = vgg
#         print("[METRICS] VGG19 loaded for perceptual/style metrics")
#         return vgg
#     except Exception as e:
#         print(f"[METRICS] VGG19 not available: {e}")
#         return None


# def _extract_vgg_features(
#     vgg: torch.nn.Module, x: torch.Tensor, layers: Tuple[int, ...] = (1, 6, 11, 20, 29)
# ) -> list:
#     """Extract features from specified VGG layers."""
#     features = []
#     for i, layer in enumerate(vgg):
#         x = layer(x)
#         if i in layers:
#             features.append(x)
#     return features


# def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
#     """Compute Gram matrix for style loss."""
#     b, c, h, w = x.shape
#     f = x.view(b, c, h * w)
#     g = torch.bmm(f, f.transpose(1, 2))
#     return g / (c * h * w)


# def calculate_perceptual_loss(
#     img1: torch.Tensor, img2: torch.Tensor, device: torch.device
# ) -> Optional[float]:
#     """VGG-based perceptual loss. Inputs: (C,H,W) in [0,1]."""
#     vgg = _load_vgg(device)
#     if vgg is None:
#         return None

#     x = img1.unsqueeze(0).to(device)
#     y = img2.unsqueeze(0).to(device)

#     # Normalize to ImageNet stats for VGG
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     x = (x - mean) / std
#     y = (y - mean) / std

#     feat_x = _extract_vgg_features(vgg, x)
#     feat_y = _extract_vgg_features(vgg, y)

#     loss = sum(F.l1_loss(fx, fy) for fx, fy in zip(feat_x, feat_y))
#     return loss.item()


# def calculate_style_loss(
#     img1: torch.Tensor, img2: torch.Tensor, device: torch.device
# ) -> Optional[float]:
#     """VGG-based style loss (Gram matrix). Inputs: (C,H,W) in [0,1]."""
#     vgg = _load_vgg(device)
#     if vgg is None:
#         return None

#     x = img1.unsqueeze(0).to(device)
#     y = img2.unsqueeze(0).to(device)

#     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     x = (x - mean) / std
#     y = (y - mean) / std

#     feat_x = _extract_vgg_features(vgg, x)
#     feat_y = _extract_vgg_features(vgg, y)

#     loss = sum(
#         F.l1_loss(_gram_matrix(fx), _gram_matrix(fy))
#         for fx, fy in zip(feat_x, feat_y)
#     )
#     return loss.item()


# # ---------------------------------------------------------------------------
# # Public API: compute all available metrics
# # ---------------------------------------------------------------------------

# def compute_per_upload_metrics(
#     restored_tensor: torch.Tensor,
#     gt_tensor: torch.Tensor,
#     device: torch.device,
#     mask_tensor: Optional[torch.Tensor] = None,
# ) -> Dict[str, Any]:
#     """All available metrics between restored output and ground truth."""
#     # Squeeze batch dim for per-image metrics
#     r = restored_tensor.squeeze(0)  # (3, H, W)
#     g = gt_tensor.squeeze(0)        # (3, H, W)

#     metrics: Dict[str, Any] = {}

#     # Core metrics — always computable
#     metrics["psnr"] = round(calculate_psnr(r, g), 4)
#     metrics["ssim"] = round(calculate_ssim(r, g), 6)
#     metrics["l1"] = round(calculate_l1(r, g), 6)
#     metrics["l2"] = round(calculate_l2(r, g), 6)

#     # VGG-based metrics — require torchvision
#     with torch.no_grad():
#         perceptual = calculate_perceptual_loss(r, g, device)
#         style = calculate_style_loss(r, g, device)

#     metrics["perceptual"] = round(perceptual, 6) if perceptual is not None else None
#     metrics["style"] = round(style, 6) if style is not None else None

#     # Brushstroke metrics — NOT available for user uploads
#     # These require pre-extracted orientation/coherence/edge/histogram feature maps
#     # from the H5 dataset, which are only available for the benchmark test set.
#     metrics["direction"] = None
#     metrics["edge_strength"] = None
#     metrics["histogram"] = None
#     metrics["_brushstroke_note"] = (
#         "Brushstroke metrics (direction, edge_strength, histogram) require "
#         "pre-extracted feature maps and are available only for benchmark images."
#     )

#     return metrics
"""
metrics.py — Per-upload image quality metrics.

Computes PSNR, SSIM, L1, L2, perceptual, and style loss when the user
provides a ground-truth image. Also computes live brushstroke metrics
(direction, edge strength, histogram) from the uploaded image, ground truth,
and damage mask.
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
    """PSNR in dB. Inputs: (C,H,W) in [0,1]."""
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> float:
    """SSIM. Inputs: (C,H,W) in [0,1]."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    x = img1.unsqueeze(0)
    y = img2.unsqueeze(0)
    channels = x.shape[1]

    sigma = 1.5
    gauss = torch.tensor(
        [
            math.exp(-((i - window_size // 2) ** 2) / (2 * sigma ** 2))
            for i in range(window_size)
        ],
        dtype=x.dtype,
        device=x.device,
    )
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()

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

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
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
    """Load VGG19 features (cached after first call)."""
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
    vgg: torch.nn.Module, x: torch.Tensor, layers: Tuple[int, ...] = (1, 6, 11, 20, 29)
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
    """VGG-based perceptual loss. Inputs: (C,H,W) in [0,1]."""
    vgg = _load_vgg(device)
    if vgg is None:
        return None

    x = img1.unsqueeze(0).to(device)
    y = img2.unsqueeze(0).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = (y - mean) / std

    feat_x = _extract_vgg_features(vgg, x)
    feat_y = _extract_vgg_features(vgg, y)

    loss = sum(F.l1_loss(fx, fy) for fx, fy in zip(feat_x, feat_y))
    return loss.item()


def calculate_style_loss(
    img1: torch.Tensor, img2: torch.Tensor, device: torch.device
) -> Optional[float]:
    """VGG-based style loss (Gram matrix). Inputs: (C,H,W) in [0,1]."""
    vgg = _load_vgg(device)
    if vgg is None:
        return None

    x = img1.unsqueeze(0).to(device)
    y = img2.unsqueeze(0).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=device).view(1, 3, 1, 1)
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
# Brushstroke helpers for live per-upload metrics
# ---------------------------------------------------------------------------

_NUM_BINS = 8

_SOBEL_X = torch.tensor(
    [[[-1.0, 0.0, 1.0],
      [-2.0, 0.0, 2.0],
      [-1.0, 0.0, 1.0]]],
    dtype=torch.float32,
).unsqueeze(0)

_SOBEL_Y = torch.tensor(
    [[[-1.0, -2.0, -1.0],
      [0.0, 0.0, 0.0],
      [1.0, 2.0, 1.0]]],
    dtype=torch.float32,
).unsqueeze(0)


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """
    Input:  (B,3,H,W) or (3,H,W) in [0,1]
    Output: (B,1,H,W)
    """
    x = _ensure_bchw(x)
    if x.shape[1] == 1:
        return x
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _gaussian_kernel(
    size: int = 7,
    sigma: float = 3.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size // 2)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, size, size)


def _sobel_xy(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = _ensure_bchw(gray)
    kx = _SOBEL_X.to(device=gray.device, dtype=gray.dtype)
    ky = _SOBEL_Y.to(device=gray.device, dtype=gray.dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return gx, gy


def _extract_orientation_and_coherence(
    image_rgb: torch.Tensor,
    sigma: float = 3.0,
    flow_offset_deg: float = 90.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Structure-tensor orientation in [0,180) and coherence in [0,1].
    """
    gray = _rgb_to_gray(image_rgb)
    gx, gy = _sobel_xy(gray)

    ixx = gx * gx
    ixy = gx * gy
    iyy = gy * gy

    gk = _gaussian_kernel(
        size=7,
        sigma=sigma,
        device=gray.device,
        dtype=gray.dtype,
    )

    ixx = F.conv2d(ixx, gk, padding=3)
    ixy = F.conv2d(ixy, gk, padding=3)
    iyy = F.conv2d(iyy, gk, padding=3)

    theta_rad = 0.5 * torch.atan2(2.0 * ixy, ixx - iyy)
    theta_deg = (torch.rad2deg(theta_rad) + flow_offset_deg) % 180.0

    diff = ixx - iyy
    numerator = torch.sqrt(diff * diff + 4.0 * ixy * ixy + 1e-8)
    denominator = ixx + iyy + 1e-8
    coherence = torch.clamp(numerator / denominator, 0.0, 1.0)

    return theta_deg, coherence


def _extract_edge_strength(image_rgb: torch.Tensor) -> torch.Tensor:
    """
    Sobel magnitude, normalized per image by the 99th percentile to [0,1].
    """
    gray = _rgb_to_gray(image_rgb)
    gx, gy = _sobel_xy(gray)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)

    flat = mag.flatten(start_dim=1)
    p99 = torch.quantile(flat, 0.99, dim=1, keepdim=True).view(-1, 1, 1, 1)
    mag = torch.clamp(mag / (p99 + 1e-8), 0.0, 1.0)
    return mag


def _prepare_mask(mask_tensor: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if mask_tensor is None:
        return None
    m = _ensure_bchw(mask_tensor).to(device=ref.device, dtype=ref.dtype)
    if m.shape[1] != 1:
        m = m[:, :1]
    return (m > 0.5).to(dtype=ref.dtype)


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Optional[float]:
    if weights is None:
        weights = torch.ones_like(mask)
    w = mask * weights
    denom = w.sum()
    if denom.item() <= 1e-8:
        return None
    return float(((values * w).sum() / denom).item())


def _compute_spatial_histograms(
    orientation_map: torch.Tensor,
    weight_map: torch.Tensor,
    mask: torch.Tensor,
    grid_size: int,
    coherence_thresh: float = 0.05,
) -> Optional[np.ndarray]:
    """
    Coherence-weighted 8-bin orientation histograms over masked regions only.
    Returns shape: (n_valid_cells, 8)
    """
    theta = orientation_map.squeeze().detach().cpu().numpy()
    weights = weight_map.squeeze().detach().cpu().numpy()
    mask_np = (mask.squeeze().detach().cpu().numpy() > 0.5)

    h, w = theta.shape
    cell_h = max(h // grid_size, 1)
    cell_w = max(w // grid_size, 1)
    bins = np.linspace(0.0, 180.0, _NUM_BINS + 1)

    rows = []
    for i in range(grid_size):
        for j in range(grid_size):
            y0 = i * cell_h
            x0 = j * cell_w
            y1 = h if i == grid_size - 1 else (i + 1) * cell_h
            x1 = w if j == grid_size - 1 else (j + 1) * cell_w

            region_theta = theta[y0:y1, x0:x1].reshape(-1)
            region_weights = weights[y0:y1, x0:x1].reshape(-1)
            region_mask = mask_np[y0:y1, x0:x1].reshape(-1)

            valid = region_mask & (region_weights >= coherence_thresh)
            if not np.any(valid):
                continue

            hist, _ = np.histogram(
                region_theta[valid],
                bins=bins,
                weights=region_weights[valid],
            )
            total = hist.sum()
            if total <= 1e-8:
                continue

            rows.append((hist / (total + 1e-8)).astype(np.float32))

    if not rows:
        return None

    return np.stack(rows, axis=0)


def _chi_square_hist_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.shape != b.shape or a.size == 0:
        return None
    eps = 1e-8
    dist = 0.5 * np.sum(((a - b) ** 2) / (a + b + eps), axis=1)
    return float(np.mean(dist))


# ---------------------------------------------------------------------------
# Public API: compute all available metrics
# ---------------------------------------------------------------------------

def compute_per_upload_metrics(
    restored_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    device: torch.device,
    mask_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Per-upload metrics between composed output and ground truth.

    Standard metrics are full-image metrics.
    Brushstroke metrics are computed only inside the supplied mask.
    """
    restored_tensor = _ensure_bchw(restored_tensor).to(device)
    gt_tensor = _ensure_bchw(gt_tensor).to(device)

    r = restored_tensor.squeeze(0)
    g = gt_tensor.squeeze(0)

    metrics: Dict[str, Any] = {}

    metrics["psnr"] = round(calculate_psnr(r, g), 4)
    metrics["ssim"] = round(calculate_ssim(r, g), 6)
    metrics["l1"] = round(calculate_l1(r, g), 6)
    metrics["l2"] = round(calculate_l2(r, g), 6)

    with torch.no_grad():
        perceptual = calculate_perceptual_loss(r, g, device)
        style = calculate_style_loss(r, g, device)

    metrics["perceptual"] = round(perceptual, 6) if perceptual is not None else None
    metrics["style"] = round(style, 6) if style is not None else None

    mask = _prepare_mask(mask_tensor, restored_tensor)
    if mask is None:
        metrics["direction"] = None
        metrics["edge_strength"] = None
        metrics["histogram"] = None
        metrics["_brushstroke_note"] = "No valid mask supplied for brushstroke metrics."
        return metrics

    with torch.no_grad():
        pred_orient, _pred_coh = _extract_orientation_and_coherence(restored_tensor)
        gt_orient, gt_coh = _extract_orientation_and_coherence(gt_tensor)

        pred_edge = _extract_edge_strength(restored_tensor)
        gt_edge = _extract_edge_strength(gt_tensor)

        delta_deg = pred_orient - gt_orient
        circular_error = 1.0 - torch.cos(2.0 * torch.deg2rad(delta_deg))
        direction_val = _masked_weighted_mean(circular_error, mask, gt_coh)

        edge_abs = torch.abs(pred_edge - gt_edge)
        edge_val = _masked_weighted_mean(edge_abs, mask)

        pred_hist_4 = _compute_spatial_histograms(pred_orient, gt_coh, mask, grid_size=4)
        gt_hist_4 = _compute_spatial_histograms(gt_orient, gt_coh, mask, grid_size=4)
        pred_hist_8 = _compute_spatial_histograms(pred_orient, gt_coh, mask, grid_size=8)
        gt_hist_8 = _compute_spatial_histograms(gt_orient, gt_coh, mask, grid_size=8)

        hist_4 = _chi_square_hist_distance(pred_hist_4, gt_hist_4)
        hist_8 = _chi_square_hist_distance(pred_hist_8, gt_hist_8)

    metrics["direction"] = round(direction_val, 6) if direction_val is not None else None
    metrics["edge_strength"] = round(edge_val, 6) if edge_val is not None else None

    if hist_4 is not None and hist_8 is not None:
        metrics["histogram"] = round((hist_4 + hist_8) / 2.0, 6)
    elif hist_4 is not None:
        metrics["histogram"] = round(hist_4, 6)
    elif hist_8 is not None:
        metrics["histogram"] = round(hist_8, 6)
    else:
        metrics["histogram"] = None

    metrics["_brushstroke_note"] = (
        "Direction, edge_strength, and histogram are live upload metrics "
        "computed from the composed output, provided ground truth, and active mask."
    )

    return metrics