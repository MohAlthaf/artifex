#!/usr/bin/env python3
"""
ARTIFEX — Full Model Evaluation Pipeline
=========================================
Evaluates both full_best.pth and checkpoint_epoch_50.pth on the test set,
selects the official full checkpoint, and creates the baseline-vs-full comparison.

This script extracts the necessary code from the canonical notebook
(artifex_FULLY_FIXED_M2_PATHS.ipynb) to ensure consistency, then runs
the evaluation pipeline.

Usage:
    python3 scripts/evaluate_full_model.py

Outputs:
    results/full_eval/full_best/evaluation_results.json
    results/full_eval/full_best/visual_comparison.png
    results/full_eval/full_best/brushstroke_analysis.png
    results/full_eval/epoch_50/evaluation_results.json
    results/full_eval/epoch_50/visual_comparison.png
    results/full_eval/epoch_50/brushstroke_analysis.png
    models/full_official/full_official_best.pth
    models/full_official/selection_record.json
    results/baseline_vs_full_comparison.json
"""

import os, sys, json, math, shutil, datetime, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

# ---- Project root ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_PATH = PROJECT_ROOT / 'artifex_FULLY_FIXED_M2_PATHS.ipynb'

# ---- Paths ----
FULL_RUN_DIR = PROJECT_ROOT / 'runs' / 'full_20260302_070353'
FULL_BEST_CKPT = FULL_RUN_DIR / 'checkpoints' / 'full_best.pth'
FULL_EP50_CKPT = FULL_RUN_DIR / 'checkpoints' / 'checkpoint_epoch_50.pth'
BASELINE_EVAL_JSON = PROJECT_ROOT / 'results' / 'baseline_ep46' / 'evaluation_results.json'
BASELINE_OFFICIAL_CKPT = PROJECT_ROOT / 'models' / 'baseline_official' / 'baseline_official_best.pth'

# Output directories
EVAL_DIR_BEST = PROJECT_ROOT / 'results' / 'full_eval' / 'full_best'
EVAL_DIR_EP50 = PROJECT_ROOT / 'results' / 'full_eval' / 'epoch_50'
FULL_OFFICIAL_DIR = PROJECT_ROOT / 'models' / 'full_official'

# Fixed visual sample indices (same deterministic seed as baseline evaluation)
_rng = np.random.RandomState(42)
VISUAL_INDICES = sorted(_rng.choice(305, size=8, replace=False).tolist())

print("=" * 80)
print("ARTIFEX — Full Model Evaluation Pipeline")
print("=" * 80)
print(f"Project root: {PROJECT_ROOT}")
print(f"Full best checkpoint: {FULL_BEST_CKPT}")
print(f"Full ep50 checkpoint: {FULL_EP50_CKPT}")
print(f"Baseline eval JSON:   {BASELINE_EVAL_JSON}")
print(f"Visual sample indices: {VISUAL_INDICES}")
print()

# ---- Verify inputs exist ----
for p, desc in [
    (FULL_BEST_CKPT, "full_best.pth"),
    (FULL_EP50_CKPT, "checkpoint_epoch_50.pth"),
    (BASELINE_EVAL_JSON, "baseline evaluation JSON"),
    (BASELINE_OFFICIAL_CKPT, "baseline official checkpoint"),
    (NOTEBOOK_PATH, "canonical notebook"),
]:
    assert p.exists(), f"Missing {desc}: {p}"
    print(f"  ✓ {desc}")
print()

# ============================================================================
# Step 1: Extract and execute notebook cells to set up the environment
# ============================================================================
print("Setting up environment from notebook cells...")

with open(NOTEBOOK_PATH) as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

def get_cell_source(cell_idx_1based):
    """Get source code of a 1-based cell index from notebook."""
    cell = nb['cells'][cell_idx_1based - 1]
    assert cell['cell_type'] == 'code', f"Cell {cell_idx_1based} is not code"
    return ''.join(cell['source'])

# We need cells 2-14, 18 (skipping 15-16 training functions, 17 auto-run)
# Cell 2: imports + device
# Cell 3: path config
# Cell 4: sys check
# Cell 5: CONFIG + seeds
# Cell 6: H5 audit + data viz (skip heavy viz, just need data setup)
# Cell 7: VanGoghDataset
# Cell 8: mask check
# Cell 9: BiSCCFormer
# Cell 10: FocalBlock
# Cell 11: SGRGANGenerator
# Cell 12: Discriminators (don't need for eval, but may be needed for loading)
# Cell 13: BrushstrokeLosses
# Cell 14: VGG + loss criteria
# Cell 18: metric functions

# Build a namespace dict to exec cells into
ns = {'__builtins__': __builtins__, '__name__': '__main__'}

# Execute cells in order, with some patches
cells_to_exec = [2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]

for cell_num in cells_to_exec:
    src = get_cell_source(cell_num)
    # Skip pip install lines
    src = '\n'.join(line for line in src.split('\n') if not line.strip().startswith('!pip'))
    # Skip plt.show() calls during headless execution
    src = src.replace('plt.show()', 'plt.close("all")')
    try:
        exec(compile(src, f'<cell_{cell_num}>', 'exec'), ns)
        print(f"  ✓ Cell {cell_num} executed")
    except Exception as e:
        print(f"  ✗ Cell {cell_num} failed: {e}")
        # Some cells may fail due to missing vars from skipped cells
        # We'll handle this by providing fallbacks

# Cell 6 is data-heavy (creates split_lists, does H5 audit + plots).
# We need it for the CONFIG paths but can skip the heavy audit.
# Let's exec it too since it sets up CONFIG fully.
src6 = get_cell_source(6)
src6 = src6.replace('plt.show()', 'plt.close("all")')
src6 = '\n'.join(line for line in src6.split('\n') if not line.strip().startswith('!pip'))
try:
    exec(compile(src6, '<cell_6>', 'exec'), ns)
    print(f"  ✓ Cell 6 executed (data setup)")
except Exception as e:
    print(f"  ✗ Cell 6 failed: {e}")

# Cell 8 (mask check)
try:
    src8 = get_cell_source(8)
    exec(compile(src8, '<cell_8>', 'exec'), ns)
    print(f"  ✓ Cell 8 executed (mask check)")
except Exception as e:
    print(f"  ! Cell 8 skipped: {e}")

# Cell 18: metric functions
src18 = get_cell_source(18)
# Remove pip install line
src18 = '\n'.join(line for line in src18.split('\n') if not line.strip().startswith('!pip'))
try:
    exec(compile(src18, '<cell_18>', 'exec'), ns)
    print(f"  ✓ Cell 18 executed (metrics)")
except Exception as e:
    print(f"  ✗ Cell 18 failed: {e}")

# Import key objects from namespace
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

device = ns['device']
CONFIG = ns['CONFIG']
SGRGANGenerator = ns['SGRGANGenerator']
VanGoghDataset = ns['VanGoghDataset']
brushstroke_losses = ns['brushstroke_losses']
criterion_perceptual = ns['criterion_perceptual']
criterion_style = ns['criterion_style']
calculate_psnr = ns['calculate_psnr']
calculate_ssim = ns['calculate_ssim']
calculate_lpips = ns['calculate_lpips']
calculate_l1_error = ns['calculate_l1_error']
calculate_l2_error = ns['calculate_l2_error']

print(f"\nDevice: {device}")
print(f"LPIPS available: {ns.get('LPIPS_AVAILABLE', False)}")

# ============================================================================
# Step 2: Create test dataset and dataloader
# ============================================================================
print("\n" + "=" * 80)
print("Creating test dataset...")

from torch.utils.data import DataLoader
from tqdm import tqdm

test_dataset = VanGoghDataset(
    CONFIG['test_original_dir'], CONFIG['test_masked_dir'],
    CONFIG['test_masks_dir'], CONFIG['brushstroke_h5'], split='test'
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"Test dataset: {len(test_dataset)} images")
print(f"Visual sample indices: {VISUAL_INDICES}")


# ============================================================================
# Step 3: Evaluation function (reused for both checkpoints)
# ============================================================================

def evaluate_checkpoint(label, ckpt_path, out_dir, test_dataset, test_loader,
                        visual_indices, save_restored=True):
    """
    Evaluate a single checkpoint on the full test set.

    Returns: (avg_metrics_dict, per_image_dict)
    Saves: evaluation_results.json, visual_comparison.png, brushstroke_analysis.png
    """
    ckpt_path = str(ckpt_path)
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    restored_dir = os.path.join(out_dir, 'restored_images')
    if save_restored:
        os.makedirs(restored_dir, exist_ok=True)

    # Load checkpoint into a FRESH generator
    gen = SGRGANGenerator(in_channels=4, out_channels=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['generator_state_dict'])
    gen.eval()

    epoch_num = ckpt.get('epoch', -1)
    if epoch_num >= 0:
        epoch_num += 1  # stored as 0-indexed
    val_loss = ckpt.get('val_loss', float('nan'))
    best_val_loss = ckpt.get('best_val_loss', val_loss)

    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {label}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Epoch: {epoch_num}, val_loss={val_loss:.6f}")
    print(f"  Output dir: {out_dir}")
    print(f"{'=' * 80}")

    # ---- Full quantitative evaluation ----
    all_metrics = {
        'psnr': [], 'ssim': [], 'lpips': [], 'l1': [], 'l2': [],
        'style': [], 'perceptual': [],
        'direction': [], 'edge_strength': [], 'histogram': [],
        'image_names': []
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing [{label}]")):
            corrupted = batch['corrupted'].to(device)
            mask = batch['mask'].to(device)
            original = batch['original'].to(device)
            orientation_field = batch['orientation_field'].to(device)
            coherence_map = batch['coherence_map'].to(device)
            edge_strength_map = batch['edge_strength_map'].to(device)
            spatial_hist_4x4 = batch['spatial_hist_4x4'].to(device)
            spatial_hist_8x8 = batch['spatial_hist_8x8'].to(device)
            img_name = batch['image_name'][0]

            restored = gen(corrupted, mask)
            comp = corrupted * (1 - mask) + restored * mask

            all_metrics['psnr'].append(calculate_psnr(comp[0], original[0]))
            all_metrics['ssim'].append(calculate_ssim(comp[0], original[0]))
            all_metrics['lpips'].append(calculate_lpips(comp[0], original[0]))
            all_metrics['l1'].append(calculate_l1_error(comp[0], original[0]))
            all_metrics['l2'].append(calculate_l2_error(comp[0], original[0]))
            all_metrics['style'].append(criterion_style(comp, original).item())
            all_metrics['perceptual'].append(criterion_perceptual(comp, original).item())
            all_metrics['direction'].append(
                brushstroke_losses.direction_loss(comp, orientation_field, coherence_map, mask).item())
            all_metrics['edge_strength'].append(
                brushstroke_losses.edge_strength_loss(comp, edge_strength_map, mask).item())
            all_metrics['histogram'].append(
                brushstroke_losses.histogram_loss(comp, spatial_hist_4x4, spatial_hist_8x8, coherence_map).item())
            all_metrics['image_names'].append(img_name)

            # Save restored image
            if save_restored:
                from torchvision.utils import save_image
                # img_name already has .png extension — don't double it
                save_image(comp[0], os.path.join(restored_dir, img_name))

    # Compute averages
    avg_metrics = {}
    for key in all_metrics:
        if key != 'image_names':
            vals = [v for v in all_metrics[key] if not (isinstance(v, float) and math.isnan(v))]
            avg_metrics[f'avg_{key}'] = float(np.mean(vals)) if vals else float('nan')
            avg_metrics[f'std_{key}'] = float(np.std(vals)) if vals else float('nan')

    # Print summary
    print(f"\n  Results for {label}:")
    print(f"  PSNR:       {avg_metrics['avg_psnr']:.2f} ± {avg_metrics['std_psnr']:.2f} dB")
    print(f"  SSIM:       {avg_metrics['avg_ssim']:.4f} ± {avg_metrics['std_ssim']:.4f}")
    lp = avg_metrics.get('avg_lpips', float('nan'))
    print(f"  LPIPS:      {lp:.4f}" if not math.isnan(lp) else "  LPIPS:      N/A (library unavailable)")
    print(f"  L1:         {avg_metrics['avg_l1']:.4f} ± {avg_metrics['std_l1']:.4f}")
    print(f"  L2:         {avg_metrics['avg_l2']:.6f} ± {avg_metrics['std_l2']:.6f}")
    print(f"  Perceptual: {avg_metrics['avg_perceptual']:.4f} ± {avg_metrics['std_perceptual']:.4f}")
    print(f"  Style:      {avg_metrics['avg_style']:.6f} ± {avg_metrics['std_style']:.6f}")
    print(f"  Direction:  {avg_metrics['avg_direction']:.4f} ± {avg_metrics['std_direction']:.4f}")
    print(f"  Edge:       {avg_metrics['avg_edge_strength']:.4f} ± {avg_metrics['std_edge_strength']:.4f}")
    print(f"  Histogram:  {avg_metrics['avg_histogram']:.4f} ± {avg_metrics['std_histogram']:.4f}")

    # Save evaluation_results.json
    results_json = {
        'label': label,
        'checkpoint': ckpt_path,
        'epoch': epoch_num,
        'val_loss': float(val_loss),
        'best_val_loss': float(best_val_loss),
        'num_test_images': len(test_dataset),
        'timestamp': datetime.datetime.now().isoformat(),
        **avg_metrics,
        'per_image': {
            name: {k: float(all_metrics[k][i]) for k in all_metrics if k != 'image_names'}
            for i, name in enumerate(all_metrics['image_names'])
        }
    }
    json_path = os.path.join(out_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✓ Metrics saved: {json_path}")

    # ---- Qualitative: side-by-side on fixed samples ----
    n = len(visual_indices)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(visual_indices):
            sample = test_dataset[idx]
            corrupted_s = sample['corrupted'].unsqueeze(0).to(device)
            mask_s = sample['mask'].unsqueeze(0).to(device)
            original_s = sample['original'].unsqueeze(0).to(device)
            img_name_s = sample['image_name']

            restored_s = gen(corrupted_s, mask_s)
            comp_s = corrupted_s * (1 - mask_s) + restored_s * mask_s

            corr_np = corrupted_s[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            comp_np = comp_s[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            mask_np = mask_s[0, 0].cpu().numpy()
            orig_np = original_s[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)

            psnr_v = calculate_psnr(comp_s[0], original_s[0])
            ssim_v = calculate_ssim(comp_s[0], original_s[0])

            axes[row, 0].imshow(corr_np)
            axes[row, 0].set_title(f'Corrupted\n{img_name_s}', fontsize=9)
            axes[row, 0].axis('off')
            axes[row, 1].imshow(mask_np, cmap='gray')
            axes[row, 1].set_title('Mask', fontsize=9)
            axes[row, 1].axis('off')
            axes[row, 2].imshow(comp_np)
            axes[row, 2].set_title(f'Restored ({label})\nPSNR={psnr_v:.2f} SSIM={ssim_v:.4f}', fontsize=9)
            axes[row, 2].axis('off')
            axes[row, 3].imshow(orig_np)
            axes[row, 3].set_title('Ground Truth', fontsize=9)
            axes[row, 3].axis('off')

    fig.suptitle(f'Full Model {label} — Visual Results', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    vis_path = os.path.join(out_dir, 'visual_comparison.png')
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Visual comparison saved: {vis_path}")

    # ---- Brushstroke analysis on first 3 fixed samples ----
    bs_indices = visual_indices[:3]
    fig2, axes2 = plt.subplots(len(bs_indices), 5, figsize=(25, 5 * len(bs_indices)))
    if len(bs_indices) == 1:
        axes2 = axes2[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(bs_indices):
            sample = test_dataset[idx]
            corrupted_s = sample['corrupted'].unsqueeze(0).to(device)
            mask_s = sample['mask'].unsqueeze(0).to(device)
            original_s = sample['original'].unsqueeze(0).to(device)
            gt_orient = sample['orientation_field'].unsqueeze(0).to(device)

            restored_s = gen(corrupted_s, mask_s)
            comp_s = corrupted_s * (1 - mask_s) + restored_s * mask_s

            pred_orient = brushstroke_losses._extract_orientation(comp_s)
            pred_edge = brushstroke_losses._sobel_magnitude(comp_s)

            comp_np = comp_s[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            orig_np = original_s[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            pred_orient_np = pred_orient[0, 0].cpu().numpy()
            gt_orient_np = (gt_orient[0, 0].cpu().numpy()
                            if gt_orient.shape[1] == 1
                            else gt_orient[0].argmax(0).cpu().numpy() * 22.5)
            pred_edge_np = pred_edge[0, 0].cpu().numpy()

            axes2[row, 0].imshow(comp_np)
            axes2[row, 0].set_title('Restored', fontsize=10)
            axes2[row, 0].axis('off')
            axes2[row, 1].imshow(orig_np)
            axes2[row, 1].set_title('Ground Truth', fontsize=10)
            axes2[row, 1].axis('off')
            axes2[row, 2].imshow(pred_orient_np, cmap='hsv', vmin=0, vmax=180)
            axes2[row, 2].set_title('Pred Orientation', fontsize=10)
            axes2[row, 2].axis('off')
            axes2[row, 3].imshow(gt_orient_np, cmap='hsv', vmin=0, vmax=180)
            axes2[row, 3].set_title('GT Orientation', fontsize=10)
            axes2[row, 3].axis('off')
            axes2[row, 4].imshow(pred_edge_np, cmap='hot')
            axes2[row, 4].set_title('Pred Edge Strength', fontsize=10)
            axes2[row, 4].axis('off')

    fig2.suptitle(f'Brushstroke Analysis — {label}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    bs_path = os.path.join(out_dir, 'brushstroke_analysis.png')
    fig2.savefig(bs_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ Brushstroke analysis saved: {bs_path}")

    # Free GPU memory
    del gen, ckpt
    import gc
    gc.collect()
    if device.type == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return avg_metrics, results_json


# ============================================================================
# Step 4: Evaluate both checkpoints
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATING FULL MODEL CHECKPOINTS")
print("=" * 80)

avg_best, results_best = evaluate_checkpoint(
    label='full_best',
    ckpt_path=FULL_BEST_CKPT,
    out_dir=EVAL_DIR_BEST,
    test_dataset=test_dataset,
    test_loader=test_loader,
    visual_indices=VISUAL_INDICES,
    save_restored=True
)

avg_ep50, results_ep50 = evaluate_checkpoint(
    label='epoch_50',
    ckpt_path=FULL_EP50_CKPT,
    out_dir=EVAL_DIR_EP50,
    test_dataset=test_dataset,
    test_loader=test_loader,
    visual_indices=VISUAL_INDICES,
    save_restored=True
)


# ============================================================================
# Step 5: Head-to-head comparison & checkpoint selection
# ============================================================================
print("\n" + "=" * 80)
print("HEAD-TO-HEAD: full_best vs epoch_50")
print("=" * 80)

comparison_keys = ['avg_psnr', 'avg_ssim', 'avg_lpips', 'avg_l1', 'avg_l2',
                   'avg_perceptual', 'avg_style',
                   'avg_direction', 'avg_edge_strength', 'avg_histogram']
higher_better = {'avg_psnr', 'avg_ssim'}  # all others: lower is better

print(f"\n{'Metric':<22} {'full_best':>14} {'epoch_50':>14} {'Delta':>12} {'Winner':>10}")
print("-" * 74)

wins = {'full_best': 0, 'epoch_50': 0, 'tie': 0}
rows_for_json = []

for key in comparison_keys:
    a_val = avg_best.get(key, float('nan'))
    b_val = avg_ep50.get(key, float('nan'))

    # Skip NaN comparisons (e.g., LPIPS)
    if math.isnan(a_val) and math.isnan(b_val):
        rows_for_json.append({'metric': key.replace('avg_', ''), 'full_best': None, 'epoch_50': None,
                              'delta': None, 'winner': 'N/A'})
        print(f"  {key.replace('avg_', ''):<20} {'N/A':>14} {'N/A':>14} {'N/A':>12} {'N/A':>10}")
        continue

    delta = b_val - a_val
    name = key.replace('avg_', '')

    if key in higher_better:
        winner = 'full_best' if a_val >= b_val else 'epoch_50'
    else:
        winner = 'full_best' if a_val <= b_val else 'epoch_50'

    wins[winner] += 1

    fmt = '.2f' if 'psnr' in key else ('.6f' if key in ['avg_l2', 'avg_style'] else '.4f')
    a_s = f"{a_val:{fmt}}"
    b_s = f"{b_val:{fmt}}"
    d_s = f"{delta:+{fmt}}"
    print(f"  {name:<20} {a_s:>14} {b_s:>14} {d_s:>12} {winner:>10}")
    rows_for_json.append({'metric': name, 'full_best': a_val, 'epoch_50': b_val,
                          'delta': delta, 'winner': winner})

print("-" * 74)
print(f"  Wins:  full_best={wins['full_best']}  epoch_50={wins['epoch_50']}  tie={wins['tie']}")

# ---- Selection logic ----
# Priority: brushstroke metrics (direction, edge_strength, histogram) matter most
# Then check PSNR/SSIM are reasonable
brushstroke_keys = ['avg_direction', 'avg_edge_strength', 'avg_histogram']
brushstroke_wins = {'full_best': 0, 'epoch_50': 0}
for key in brushstroke_keys:
    a_val = avg_best.get(key, float('nan'))
    b_val = avg_ep50.get(key, float('nan'))
    if not math.isnan(a_val) and not math.isnan(b_val):
        if a_val <= b_val:
            brushstroke_wins['full_best'] += 1
        else:
            brushstroke_wins['epoch_50'] += 1

print(f"\n  Brushstroke wins: full_best={brushstroke_wins['full_best']}  epoch_50={brushstroke_wins['epoch_50']}")

# Decide winner
if brushstroke_wins['full_best'] > brushstroke_wins['epoch_50']:
    selected = 'full_best'
    selected_ckpt = str(FULL_BEST_CKPT)
    selected_epoch = results_best['epoch']
elif brushstroke_wins['epoch_50'] > brushstroke_wins['full_best']:
    selected = 'epoch_50'
    selected_ckpt = str(FULL_EP50_CKPT)
    selected_epoch = results_ep50['epoch']
else:
    # Tie on brushstroke — use overall wins
    if wins['full_best'] >= wins['epoch_50']:
        selected = 'full_best'
        selected_ckpt = str(FULL_BEST_CKPT)
        selected_epoch = results_best['epoch']
    else:
        selected = 'epoch_50'
        selected_ckpt = str(FULL_EP50_CKPT)
        selected_epoch = results_ep50['epoch']

selected_metrics = avg_best if selected == 'full_best' else avg_ep50

# Build reason string
reason_parts = [
    f"Selected {selected} as official full model.",
    f"Brushstroke metric wins: full_best={brushstroke_wins['full_best']}, epoch_50={brushstroke_wins['epoch_50']}.",
    f"Overall metric wins: full_best={wins['full_best']}, epoch_50={wins['epoch_50']}.",
    f"PSNR: full_best={avg_best.get('avg_psnr', 0):.2f}, epoch_50={avg_ep50.get('avg_psnr', 0):.2f}.",
    f"Direction: full_best={avg_best.get('avg_direction', 0):.4f}, epoch_50={avg_ep50.get('avg_direction', 0):.4f}.",
]
selection_reason = ' '.join(reason_parts)

print(f"\n  >>> SELECTED: {selected} (epoch {selected_epoch})")
print(f"  Reason: {selection_reason}")


# ============================================================================
# Step 6: Freeze official full checkpoint
# ============================================================================
print("\n" + "=" * 80)
print("FREEZING OFFICIAL FULL CHECKPOINT")
print("=" * 80)

os.makedirs(str(FULL_OFFICIAL_DIR), exist_ok=True)
frozen_path = FULL_OFFICIAL_DIR / 'full_official_best.pth'

shutil.copy2(selected_ckpt, str(frozen_path))
print(f"  ✓ Copied {selected_ckpt}")
print(f"     → {frozen_path}")

# Write selection record
selection_record = {
    'selected_checkpoint': selected_ckpt,
    'selected_label': selected,
    'frozen_copy': str(frozen_path),
    'epoch': selected_epoch,
    'run_folder': str(FULL_RUN_DIR),
    'selection_timestamp': datetime.datetime.now().isoformat(),
    'selection_reason': selection_reason,
    'candidates': {
        'full_best': {
            'checkpoint': str(FULL_BEST_CKPT),
            'epoch': results_best['epoch'],
            'val_loss': results_best['val_loss'],
            'avg_psnr': avg_best.get('avg_psnr'),
            'avg_ssim': avg_best.get('avg_ssim'),
            'avg_l1': avg_best.get('avg_l1'),
            'avg_direction': avg_best.get('avg_direction'),
            'avg_edge_strength': avg_best.get('avg_edge_strength'),
            'avg_histogram': avg_best.get('avg_histogram'),
            'avg_perceptual': avg_best.get('avg_perceptual'),
            'avg_style': avg_best.get('avg_style'),
        },
        'epoch_50': {
            'checkpoint': str(FULL_EP50_CKPT),
            'epoch': results_ep50['epoch'],
            'val_loss': results_ep50['val_loss'],
            'avg_psnr': avg_ep50.get('avg_psnr'),
            'avg_ssim': avg_ep50.get('avg_ssim'),
            'avg_l1': avg_ep50.get('avg_l1'),
            'avg_direction': avg_ep50.get('avg_direction'),
            'avg_edge_strength': avg_ep50.get('avg_edge_strength'),
            'avg_histogram': avg_ep50.get('avg_histogram'),
            'avg_perceptual': avg_ep50.get('avg_perceptual'),
            'avg_style': avg_ep50.get('avg_style'),
        },
    },
    'comparison': rows_for_json,
    'wins': wins,
    'brushstroke_wins': brushstroke_wins,
}

record_path = FULL_OFFICIAL_DIR / 'selection_record.json'
with open(str(record_path), 'w') as f:
    json.dump(selection_record, f, indent=2)
print(f"  ✓ Selection record saved: {record_path}")


# ============================================================================
# Step 7: Baseline vs Full comparison
# ============================================================================
print("\n" + "=" * 80)
print("BASELINE vs FULL COMPARISON")
print("=" * 80)

# Load baseline evaluation
with open(str(BASELINE_EVAL_JSON)) as f:
    baseline_eval = json.load(f)

# Use the selected full checkpoint's metrics
full_eval = avg_best if selected == 'full_best' else avg_ep50

# Build comparison
metric_names = ['psnr', 'ssim', 'lpips', 'l1', 'l2', 'perceptual', 'style',
                'direction', 'edge_strength', 'histogram']
higher_is_better = {'psnr', 'ssim'}
lower_is_better = {'l1', 'l2', 'perceptual', 'style', 'direction', 'edge_strength', 'histogram', 'lpips'}

comparison_rows = []
baseline_wins_count = 0
full_wins_count = 0

print(f"\n{'Metric':<22} {'Baseline':>14} {'Full':>14} {'Delta':>12} {'Winner':>10}")
print("-" * 74)

for metric in metric_names:
    b_val = baseline_eval.get(f'avg_{metric}', float('nan'))
    f_val = full_eval.get(f'avg_{metric}', float('nan'))

    if math.isnan(b_val) or math.isnan(f_val):
        winner = 'N/A'
        delta = None
    else:
        delta = f_val - b_val
        if metric in higher_is_better:
            winner = 'full' if f_val > b_val else ('baseline' if b_val > f_val else 'tie')
        else:
            winner = 'full' if f_val < b_val else ('baseline' if b_val < f_val else 'tie')

    if winner == 'baseline':
        baseline_wins_count += 1
    elif winner == 'full':
        full_wins_count += 1

    fmt = '.2f' if metric == 'psnr' else ('.6f' if metric in ['l2', 'style'] else '.4f')
    b_s = f"{b_val:{fmt}}" if not math.isnan(b_val) else 'N/A'
    f_s = f"{f_val:{fmt}}" if not math.isnan(f_val) else 'N/A'
    d_s = f"{delta:+{fmt}}" if delta is not None else 'N/A'
    print(f"  {metric:<20} {b_s:>14} {f_s:>14} {d_s:>12} {winner:>10}")

    comparison_rows.append({
        'metric': metric,
        'baseline': float(b_val) if not math.isnan(b_val) else None,
        'full': float(f_val) if not math.isnan(f_val) else None,
        'delta': float(delta) if delta is not None else None,
        'winner': winner,
    })

print("-" * 74)
print(f"  Wins:  baseline={baseline_wins_count}  full={full_wins_count}")

# Verdict
brushstroke_improved = sum(
    1 for r in comparison_rows
    if r['metric'] in ['direction', 'edge_strength', 'histogram'] and r['winner'] == 'full'
)
brushstroke_total = sum(
    1 for r in comparison_rows
    if r['metric'] in ['direction', 'edge_strength', 'histogram'] and r['winner'] != 'N/A'
)

if brushstroke_improved == brushstroke_total and brushstroke_total > 0:
    verdict = "full_improves_all_brushstroke"
elif brushstroke_improved > 0:
    verdict = f"full_improves_{brushstroke_improved}_of_{brushstroke_total}_brushstroke"
elif full_wins_count > baseline_wins_count:
    verdict = "full_wins_overall"
elif baseline_wins_count > full_wins_count:
    verdict = "baseline_wins_overall"
else:
    verdict = "tie"

print(f"\n  Verdict: {verdict}")

# Save baseline_vs_full_comparison.json
bvf_json = {
    'timestamp': datetime.datetime.now().isoformat(),
    'baseline': {
        'checkpoint': str(BASELINE_OFFICIAL_CKPT),
        'evaluation_json': str(BASELINE_EVAL_JSON),
        'epoch': baseline_eval.get('epoch', 46),
        'model_type': 'baseline',
    },
    'full': {
        'checkpoint': str(frozen_path),
        'evaluation_json': str(EVAL_DIR_BEST / 'evaluation_results.json')
                           if selected == 'full_best'
                           else str(EVAL_DIR_EP50 / 'evaluation_results.json'),
        'epoch': selected_epoch,
        'model_type': 'full',
        'selected_from': selected,
    },
    'comparison': comparison_rows,
    'wins': {
        'baseline': baseline_wins_count,
        'full': full_wins_count,
    },
    'verdict': verdict,
    'brushstroke_improvement': {
        'improved': brushstroke_improved,
        'total': brushstroke_total,
    },
}

bvf_path = PROJECT_ROOT / 'results' / 'baseline_vs_full_comparison.json'
with open(str(bvf_path), 'w') as f:
    json.dump(bvf_json, f, indent=2)
print(f"\n  ✓ Comparison saved: {bvf_path}")


# ============================================================================
# Step 8: Final verification
# ============================================================================
print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

checks = [
    (EVAL_DIR_BEST / 'evaluation_results.json', "full_best evaluation JSON"),
    (EVAL_DIR_BEST / 'visual_comparison.png', "full_best visual comparison"),
    (EVAL_DIR_BEST / 'brushstroke_analysis.png', "full_best brushstroke analysis"),
    (EVAL_DIR_EP50 / 'evaluation_results.json', "epoch_50 evaluation JSON"),
    (EVAL_DIR_EP50 / 'visual_comparison.png', "epoch_50 visual comparison"),
    (EVAL_DIR_EP50 / 'brushstroke_analysis.png', "epoch_50 brushstroke analysis"),
    (frozen_path, "full_official_best.pth"),
    (record_path, "selection_record.json"),
    (bvf_path, "baseline_vs_full_comparison.json"),
]

all_ok = True
for path, desc in checks:
    exists = Path(path).exists()
    size = Path(path).stat().st_size if exists else 0
    status = f"✓ {size:,} bytes" if exists else "✗ MISSING"
    print(f"  {status}  {desc}")
    if not exists:
        all_ok = False

if all_ok:
    print("\n  ALL CHECKS PASSED ✓")
else:
    print("\n  SOME CHECKS FAILED ✗")
    sys.exit(1)


# ============================================================================
# Step 9: Summary
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATION COMPLETE — SUMMARY")
print("=" * 80)
print(f"\n  Selected full checkpoint: {selected} (epoch {selected_epoch})")
print(f"  Official frozen at:       {frozen_path}")
print(f"  Selection record:         {record_path}")
print(f"  Baseline vs Full JSON:    {bvf_path}")
print(f"  Verdict:                  {verdict}")
print(f"\n  full_best eval:  {EVAL_DIR_BEST / 'evaluation_results.json'}")
print(f"  epoch_50 eval:   {EVAL_DIR_EP50 / 'evaluation_results.json'}")
print(f"\n  Next step: Run ablation experiments (dir_only, edge_only, hist_only)")
print("=" * 80)
