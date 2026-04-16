"""
canonical_inference.py — ARTIFEX generator architecture + inference.

Exact architecture from the thesis notebook, extracted for serving.
Keep in sync with artifex_training.ipynb for checkpoint compat.
"""

from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """CUDA if available, else CPU. MPS skipped (attention OOM at 512×512)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# BiSCCFormer components  (texture encoder)
# ============================================================================

class BiLevelRoutingAttention(nn.Module):
    """Windowed bi-level routing attention (avoids O(N²) memory)."""

    def __init__(self, dim: int, num_heads: int = 8,
                 qkv_bias: bool = False, topk: int = 8, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk = topk
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    # ------------------------------------------------------------------
    def _window_partition(self, x: torch.Tensor, ws: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)

    def _window_reverse(self, windows: torch.Tensor, ws: int,
                        H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / ws / ws))
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws = min(self.window_size, H, W)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        x_win = self._window_partition(x, ws).view(-1, ws * ws, C)

        Bw, Nw = x_win.shape[:2]
        qkv = self.qkv(x_win).reshape(Bw, Nw, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(Bw, Nw, C)
        out = self.proj(out).view(-1, ws, ws, C)
        out = self._window_reverse(out, ws, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :].contiguous()
        return out


class SCConv(nn.Module):
    """Depthwise spatial conv + pointwise channel conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels, in_channels,
                                     kernel_size=3, padding=1,
                                     groups=in_channels)
        self.channel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_conv(self.spatial_conv(x))


class BiSCCFormerBlock(nn.Module):
    """Attention + SCConv block for the texture encoder."""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BiLevelRoutingAttention(dim, num_heads,
                                           window_size=window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.scconv = SCConv(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Attention branch
        x_a = x.permute(0, 2, 3, 1)
        x_a = x_a + self.attn(self.norm1(x_a))
        x_a = x_a.permute(0, 3, 1, 2)
        # SCConv branch
        x_out = x_a + self.scconv(
            self.norm2(x_a.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        )
        return x_out


# ============================================================================
# Focal components  (structure encoder)
# ============================================================================

class FocalModulation(nn.Module):
    """Multi-scale dilated convolutions for hierarchical context."""

    def __init__(self, dim: int):
        super().__init__()
        self.focal_layers = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim),
        ])
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.modulation = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [layer(x) for layer in self.focal_layers]
        fused = self.fusion(torch.cat(feats, dim=1))
        return x * self.modulation(fused) + fused


class FocalBlock(nn.Module):
    """FocalModulation + FFN block for the structure encoder."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.focal_mod = FocalModulation(dim)
        self.norm2 = nn.GroupNorm(8, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.focal_mod(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# ArtifexGenerator  (canonical — matches official checkpoint keys)
# ============================================================================

class ArtifexGenerator(nn.Module):
    """
    Dual-stream generator: BiSCCFormer (texture) + Focal (structure),
    attention fusion, U-Net decoder with skip connections.

    Input:  (B,3,H,W) damaged image + (B,1,H,W) mask (1=damaged)
    Output: (B,3,H,W) restored image in [0,1]
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()

        # === TEXTURE ENCODER ===
        self.texture_enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            BiSCCFormerBlock(64, num_heads=4),
        )
        self.texture_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            BiSCCFormerBlock(128, num_heads=4),
        )
        self.texture_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            BiSCCFormerBlock(256, num_heads=8),
        )
        self.texture_enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            BiSCCFormerBlock(512, num_heads=8),
        )

        # === STRUCTURE ENCODER ===
        self.structure_enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            FocalBlock(64),
        )
        self.structure_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            FocalBlock(128),
        )
        self.structure_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            FocalBlock(256),
        )
        self.structure_enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            FocalBlock(512),
        )

        # === FUSION MODULE ===
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=False
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # === DECODER with U-Net skip connections ===
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # 32 → 64
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # 256 + 256 (t3) + 256 (s3) = 768 → 256
        self.skip_conv1 = nn.Sequential(
            nn.Conv2d(768, 256, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 64 → 128
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # 128 + 128 (t2) + 128 (s2) = 384 → 128
        self.skip_conv2 = nn.Sequential(
            nn.Conv2d(384, 128, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 128 → 256
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 64 + 64 (t1) + 64 (s1) = 192 → 64
        self.skip_conv3 = nn.Sequential(
            nn.Conv2d(192, 64, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # 256 → 512
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # === OUTPUT ===
        self.output = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Texture encoder
        inp = torch.cat([x, mask], dim=1)          # (B, 4, H, W)
        t1 = self.texture_enc1(inp)                 # (B,  64, H/2, W/2)
        t2 = self.texture_enc2(t1)                  # (B, 128, H/4, W/4)
        t3 = self.texture_enc3(t2)                  # (B, 256, H/8, W/8)
        t4 = self.texture_enc4(t3)                  # (B, 512, H/16, W/16)

        # Structure encoder
        s1 = self.structure_enc1(mask)              # (B,  64, H/2, W/2)
        s2 = self.structure_enc2(s1)                # (B, 128, H/4, W/4)
        s3 = self.structure_enc3(s2)                # (B, 256, H/8, W/8)
        s4 = self.structure_enc4(s3)                # (B, 512, H/16, W/16)

        # Attention fusion
        B, C, H, W = t4.shape
        t4_flat = t4.view(B, C, -1).permute(2, 0, 1)   # (H*W, B, C)
        s4_flat = s4.view(B, C, -1).permute(2, 0, 1)
        fused_flat, _ = self.fusion_attention(t4_flat, s4_flat, s4_flat)
        fused = fused_flat.permute(1, 2, 0).view(B, C, H, W)
        combined = torch.cat([t4, fused], dim=1)        # (B, 1024, ...)
        fused_features = self.fusion_conv(combined)     # (B, 512, ...)

        # Decoder with skip connections
        d1 = self.skip_conv1(torch.cat([self.dec1(fused_features), t3, s3], dim=1))
        d2 = self.skip_conv2(torch.cat([self.dec2(d1), t2, s2], dim=1))
        d3 = self.skip_conv3(torch.cat([self.dec3(d2), t1, s1], dim=1))
        d4 = self.dec4(d3)

        return self.output(d4)


# Backward-compat alias (notebook + eval scripts reference SGRGANGenerator by name)
SGRGANGenerator = ArtifexGenerator


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_generator(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> ArtifexGenerator:
    """Load a generator from an official thesis checkpoint (.pth)."""
    if device is None:
        device = select_device()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)

    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Raw state dict
        state_dict = checkpoint

    model = ArtifexGenerator(in_channels=4, out_channels=3)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Preprocessing & postprocessing
# ============================================================================

TARGET_SIZE = (512, 512)


def preprocess_image(
    image_bytes: bytes,
    mask_bytes: Optional[bytes] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Resize image + mask to 512×512 tensors. Returns (img, mask, original_size)."""
    if device is None:
        device = select_device()

    # --- Image ---
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = pil_img.size   # (W, H)
    pil_img_resized = pil_img.resize(TARGET_SIZE, Image.LANCZOS)
    img_arr = np.array(pil_img_resized, dtype=np.float32) / 255.0
    img_tensor = (
        torch.from_numpy(img_arr)
        .permute(2, 0, 1)          # H,W,C → C,H,W
        .unsqueeze(0)              # add batch dim
        .to(device)
    )

    # --- Mask ---
    if mask_bytes is not None:
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        pil_mask = pil_mask.resize(TARGET_SIZE, Image.NEAREST)
        mask_arr = (np.array(pil_mask, dtype=np.float32) / 255.0 > 0.5).astype(
            np.float32
        )
    else:
        # Auto-detect damaged regions from black pixels in the image
        gray = np.array(pil_img_resized.convert("L"), dtype=np.float32) / 255.0
        mask_arr = (gray < 0.05).astype(np.float32)

    mask_tensor = (
        torch.from_numpy(mask_arr)
        .unsqueeze(0).unsqueeze(0)  # 1,1,H,W
        .to(device)
    )

    return img_tensor, mask_tensor, original_size


def postprocess_image(
    output_tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> bytes:
    """Tensor (1,3,H,W) → PNG bytes, resized to original_size."""
    arr = output_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil_out = Image.fromarray(arr, mode="RGB")
    if pil_out.size != original_size:
        pil_out = pil_out.resize(original_size, Image.LANCZOS)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return buf.getvalue()


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1,C,H,W) tensor → PIL Image."""
    arr = t.squeeze(0).cpu().detach().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)  # C,H,W → H,W,C
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Debug artifact directory
# ---------------------------------------------------------------------------
_DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_artifacts")


def _save_debug_artifacts(
    request_id: str,
    img_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    raw_output: torch.Tensor,
    composed: torch.Tensor,
    original_size: Tuple[int, int],
    model_name: str,
    checkpoint_path: str,
    mask_coverage: float,
) -> str:
    """Save debug images + metadata. Returns the output directory."""
    req_dir = os.path.join(_DEBUG_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    _tensor_to_pil(img_tensor).save(os.path.join(req_dir, "input_image.png"))
    _tensor_to_pil(mask_tensor).save(os.path.join(req_dir, "input_mask.png"))
    _tensor_to_pil(raw_output).save(os.path.join(req_dir, "raw_model_output.png"))
    _tensor_to_pil(composed).save(os.path.join(req_dir, "composed_output.png"))

    # Save metadata
    import json as _json
    meta = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "original_size_WxH": list(original_size),
        "inference_size": "512x512",
        "mask_coverage_pct": round(mask_coverage * 100, 2),
        "output_range": [float(raw_output.min()), float(raw_output.max())],
    }
    with open(os.path.join(req_dir, "metadata.json"), "w") as f:
        _json.dump(meta, f, indent=2)

    return req_dir


def run_inference(
    model: ArtifexGenerator,
    image_bytes: bytes,
    mask_bytes: Optional[bytes] = None,
    device: Optional[torch.device] = None,
    model_name: str = "unknown",
    checkpoint_path: str = "",
    save_debug: bool = True,
) -> Tuple[bytes, dict]:
    """
    Full inference pipeline: raw bytes → model → composed PNG bytes.

    Composition: keep intact pixels, fill damaged region with model output.
    Returns (png_bytes, {inference_time_s, mask_coverage_pct, debug_dir}).
    """
    import time as _time

    if device is None:
        device = next(model.parameters()).device

    img_tensor, mask_tensor, original_size = preprocess_image(
        image_bytes, mask_bytes, device=device
    )

    # Mask coverage
    mask_coverage = float(mask_tensor.mean().item())

    t0 = _time.time()
    with torch.no_grad():
        raw_output = model(img_tensor, mask_tensor)
        # ============ CANONICAL COMPOSITION STEP ============
        # During training the loss is computed on:
        #   comp = corrupted * (1 - mask) + restored * mask
        # This keeps original pixels in intact areas and uses the
        # generator output only inside the damaged region.
        composed = img_tensor * (1.0 - mask_tensor) + raw_output * mask_tensor
    inference_time = _time.time() - t0

    # Debug artifacts
    debug_dir = None
    if save_debug:
        try:
            import datetime
            rid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{model_name}"
            debug_dir = _save_debug_artifacts(
                rid, img_tensor, mask_tensor, raw_output, composed,
                original_size, model_name, checkpoint_path, mask_coverage,
            )
        except Exception as e:
            print(f"[DEBUG] Failed to save debug artifacts: {e}")

    png_bytes = postprocess_image(composed, original_size)

    info = {
        "inference_time_s": round(inference_time, 2),
        "mask_coverage_pct": round(mask_coverage * 100, 2),
        "debug_dir": debug_dir,
    }
    return png_bytes, info
