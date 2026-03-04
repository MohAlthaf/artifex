"""
Test script for full_best.pth — SGRGAN Van Gogh Art Restoration Model
=====================================================================
Loads the checkpoint, runs inference on damaged paintings with their
corresponding masks, and saves restored results.

Usage:
    python test_model.py                              # restore all damaged paintings
    python test_model.py --image path/to/damaged.png --mask path/to/mask.png
    python test_model.py --no-save                    # display only, don't save
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Import model architecture from server/ml/model.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server", "ml"))
from model import SGRGANGenerator  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "full_best.pth")
DAMAGED_DIR = os.path.join(SCRIPT_DIR, "damage paintng")
MASKS_DIR = os.path.join(SCRIPT_DIR, "damage paintng", "masks")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_outputs")
TARGET_SIZE = (512, 512)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("cpu")  # MPS can OOM on attention layers
    return torch.device("cpu")


def load_checkpoint(path, device):
    """Load checkpoint and return generator + metadata dict."""
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        print(f"  Checkpoint keys : {list(checkpoint.keys())}")
        if "epoch" in checkpoint:
            print(f"  Trained epochs  : {checkpoint['epoch'] + 1}")
        if "val_loss" in checkpoint:
            print(f"  Best val loss   : {checkpoint['val_loss']:.4f}")
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            print(f"  Model type      : {cfg.get('model_type', 'N/A')}")
    else:
        print("  Checkpoint is a raw state_dict (no metadata)")

    model = SGRGANGenerator(in_channels=4, out_channels=3)
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["generator_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        state = checkpoint if not isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Generator loaded — {param_count:,} parameters on {device}")
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def load_image(image_path, device):
    """Load an RGB image, resize to 512×512, return tensor + original size."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, original_size


def load_mask(mask_path, device):
    """Load a grayscale mask, resize to 512×512, binarise, return (1,1,H,W) tensor."""
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(TARGET_SIZE, Image.NEAREST)
    arr = np.array(mask, dtype=np.float32) / 255.0
    # Binarise: white (1) = damaged region, black (0) = intact
    arr = (arr > 0.5).astype(np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def tensor_to_image(tensor, target_size=None):
    """Convert (1, C, H, W) tensor in [0,1] to PIL Image."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return img


def add_label(image, text, font_size=18):
    """Return a new image with a label bar on top."""
    w, h = image.size
    bar_h = font_size + 12
    canvas = Image.new("RGB", (w, h + bar_h), (30, 30, 30))
    canvas.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, 4), text, fill=(255, 255, 255), font=font)
    return canvas


def compute_metrics(restored, original):
    """Compute PSNR, SSIM (approx), L1, L2 between two (1,3,H,W) tensors."""
    l1 = F.l1_loss(restored, original).item()
    l2 = F.mse_loss(restored, original).item()
    psnr = 10 * np.log10(1.0 / (l2 + 1e-10))

    mu_r = restored.mean(dim=[2, 3])
    mu_o = original.mean(dim=[2, 3])
    sigma_r = restored.var(dim=[2, 3])
    sigma_o = original.var(dim=[2, 3])
    sigma_ro = ((restored - mu_r[:, :, None, None]) *
                (original - mu_o[:, :, None, None])).mean(dim=[2, 3])
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_val = (((2 * mu_r * mu_o + C1) * (2 * sigma_ro + C2)) /
                ((mu_r ** 2 + mu_o ** 2 + C1) * (sigma_r + sigma_o + C2)))
    ssim_avg = ssim_val.mean().item()

    return {"PSNR (dB)": psnr, "SSIM": ssim_avg, "L1": l1, "L2 (MSE)": l2}


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------
@torch.no_grad()
def restore_painting(model, image_path, mask_path, device, save_dir=None):
    """Restore a single damaged painting using its mask."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n{'─' * 60}")
    print(f"  Damaged image : {image_path}")
    print(f"  Mask          : {mask_path}")

    # Load inputs
    img_tensor, orig_size = load_image(image_path, device)
    mask_tensor = load_mask(mask_path, device)
    mask_coverage = mask_tensor.mean().item() * 100
    print(f"  Original size : {orig_size[0]}×{orig_size[1]} → resized to 512×512")
    print(f"  Mask coverage : {mask_coverage:.1f}%")

    # Create the corrupted input: zero out masked (damaged) regions
    corrupted = img_tensor * (1 - mask_tensor)

    # Forward pass
    t0 = time.time()
    restored = model(corrupted, mask_tensor)
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.3f}s")

    # Composite: keep intact pixels, fill damaged regions with model output
    composite = img_tensor * (1 - mask_tensor) + restored * mask_tensor

    # Metrics (composite vs original damaged input for reference)
    metrics = compute_metrics(composite, img_tensor)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    # Save outputs
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Individual images
        damaged_img = tensor_to_image(img_tensor)
        damaged_img.save(os.path.join(save_dir, f"{basename}_damaged.png"))

        corrupted_img = tensor_to_image(corrupted)
        corrupted_img.save(os.path.join(save_dir, f"{basename}_corrupted.png"))

        mask_vis = tensor_to_image(mask_tensor.repeat(1, 3, 1, 1))
        mask_vis.save(os.path.join(save_dir, f"{basename}_mask.png"))

        restored_img = tensor_to_image(restored)
        restored_img.save(os.path.join(save_dir, f"{basename}_restored_raw.png"))

        comp_img = tensor_to_image(composite, orig_size)
        comp_img.save(os.path.join(save_dir, f"{basename}_restored.png"))

        # Side-by-side comparison with labels
        panels = [
            ("Damaged Input", damaged_img.resize((512, 512))),
            ("Mask", mask_vis.resize((512, 512))),
            ("Corrupted (masked)", corrupted_img.resize((512, 512))),
            ("Restored Output", tensor_to_image(composite).resize((512, 512))),
        ]
        labeled = [add_label(im, lbl) for lbl, im in panels]
        gap = 6
        total_w = sum(p.size[0] for p in labeled) + gap * (len(labeled) - 1)
        total_h = labeled[0].size[1]
        canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
        x = 0
        for p in labeled:
            canvas.paste(p, (x, 0))
            x += p.size[0] + gap
        canvas.save(os.path.join(save_dir, f"{basename}_comparison.png"))

        print(f"  Saved → {save_dir}/{basename}_*.png")

    return composite


# ---------------------------------------------------------------------------
# Pair matching
# ---------------------------------------------------------------------------
def find_image_mask_pairs(damaged_dir, masks_dir):
    """Match damaged images to their masks by filename."""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = sorted(
        f for f in os.listdir(damaged_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    )
    pairs = []
    for fname in images:
        img_path = os.path.join(damaged_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        if os.path.isfile(mask_path):
            pairs.append((img_path, mask_path))
        else:
            # Try common mask naming conventions
            base = os.path.splitext(fname)[0]
            for suffix in ["_mask.png", "_mask.jpg", ".png", ".jpg"]:
                alt = os.path.join(masks_dir, base + suffix)
                if os.path.isfile(alt):
                    pairs.append((img_path, alt))
                    break
            else:
                print(f"  Warning: no mask found for {fname}, skipping")
    return pairs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Restore damaged Van Gogh paintings using full_best.pth"
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="Path to .pth checkpoint")
    parser.add_argument("--image", default=None,
                        help="Path to a single damaged image")
    parser.add_argument("--mask", default=None,
                        help="Path to the mask for --image")
    parser.add_argument("--damaged-dir", default=DAMAGED_DIR,
                        help="Directory with damaged paintings")
    parser.add_argument("--masks-dir", default=MASKS_DIR,
                        help="Directory with mask images")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory to save restored results")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving output images")
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device}\n")

    # Load model
    model, meta = load_checkpoint(args.checkpoint, device)

    save_dir = None if args.no_save else args.output_dir

    if args.image:
        # --- Single image mode ---
        if not os.path.isfile(args.image):
            print(f"Error: image not found — {args.image}"); sys.exit(1)
        if not args.mask or not os.path.isfile(args.mask):
            print(f"Error: --mask is required for single image mode"); sys.exit(1)
        restore_painting(model, args.image, args.mask, device, save_dir)
    else:
        # --- Batch mode: all damaged paintings ---
        if not os.path.isdir(args.damaged_dir):
            print(f"Error: damaged paintings directory not found — {args.damaged_dir}")
            sys.exit(1)
        if not os.path.isdir(args.masks_dir):
            print(f"Error: masks directory not found — {args.masks_dir}")
            sys.exit(1)

        pairs = find_image_mask_pairs(args.damaged_dir, args.masks_dir)
        if not pairs:
            print("No image–mask pairs found."); sys.exit(1)

        print(f"\nFound {len(pairs)} damaged painting(s) with masks")
        print(f"  Damaged dir : {args.damaged_dir}")
        print(f"  Masks dir   : {args.masks_dir}")

        for img_path, mask_path in pairs:
            restore_painting(model, img_path, mask_path, device, save_dir)

    print(f"\n{'═' * 60}")
    print("Restoration complete.")
    if save_dir:
        print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
