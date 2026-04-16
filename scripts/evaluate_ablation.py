#!/usr/bin/env python3
"""
ARTIFEX — Ablation Evaluation Pipeline (Generalized)

Evaluates <ablation>_best.pth and checkpoint_epoch_50.pth, selects the
official checkpoint, and creates a comparison against baseline.

Usage:
    python3 scripts/evaluate_ablation.py --ablation dir_only --run-dir runs/dir_only_YYYYMMDD_HHMMSS
"""

import os, sys, json, math, shutil, datetime, warnings, argparse
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

# ---- Parse arguments ----
parser = argparse.ArgumentParser(description='ARTIFEX Ablation Evaluation')
parser.add_argument('--ablation', required=True, help='Ablation name (e.g., dir_only, edge_only, hist_only)')
parser.add_argument('--run-dir', required=True, help='Run directory (e.g., runs/dir_only_20260304_123456)')
parser.add_argument('--best-ckpt', default=None, help='Path to best checkpoint (auto-detected if not given)')
parser.add_argument('--final-ckpt', default=None, help='Path to final checkpoint (auto-detected if not given)')
parser.add_argument('--epochs', type=int, default=50, help='Expected final epoch for auto-detection')
parser.add_argument('--skip-freeze', action='store_true', help='Skip freezing official checkpoint')
parser.add_argument('--skip-comparison', action='store_true', help='Skip baseline comparison')
args = parser.parse_args()

ABLATION = args.ablation

# ---- Project root ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_PATH = PROJECT_ROOT / 'artifex_training.ipynb'

# ---- Paths ----
RUN_DIR = PROJECT_ROOT / args.run_dir
CKPT_DIR = RUN_DIR / 'checkpoints'

# Auto-detect checkpoints if not explicitly provided
if args.best_ckpt:
    BEST_CKPT = Path(args.best_ckpt)
else:
    BEST_CKPT = CKPT_DIR / f'{ABLATION}_best.pth'

if args.final_ckpt:
    FINAL_CKPT = Path(args.final_ckpt)
else:
    FINAL_CKPT = CKPT_DIR / f'checkpoint_epoch_{args.epochs}.pth'

BASELINE_EVAL_JSON = PROJECT_ROOT / 'results' / 'baseline_ep46' / 'evaluation_results.json'
BASELINE_OFFICIAL_CKPT = PROJECT_ROOT / 'models' / 'baseline_official' / 'baseline_official_best.pth'

# Output directories
EVAL_DIR = PROJECT_ROOT / 'results' / f'{ABLATION}_eval'
EVAL_DIR_BEST = EVAL_DIR / f'{ABLATION}_best'
EVAL_DIR_FINAL = EVAL_DIR / f'epoch_{args.epochs}'
OFFICIAL_DIR = PROJECT_ROOT / 'models' / f'{ABLATION}_official'

# Fixed visual sample indices (same deterministic seed as all evaluations)
_rng = np.random.RandomState(42)
VISUAL_INDICES = sorted(_rng.choice(305, size=8, replace=False).tolist())

print("=" * 80)
print(f"ARTIFEX — {ABLATION.upper()} Ablation Evaluation Pipeline")
print("=" * 80)
print(f"Project root:       {PROJECT_ROOT}")
print(f"Run directory:      {RUN_DIR}")
print(f"Best checkpoint:    {BEST_CKPT}")
print(f"Final checkpoint:   {FINAL_CKPT}")
print(f"Baseline eval JSON: {BASELINE_EVAL_JSON}")
print(f"Visual indices:     {VISUAL_INDICES}")
print()

# ---- Verify inputs ----
for p, desc in [
    (BEST_CKPT, f"{ABLATION}_best checkpoint"),
    (FINAL_CKPT, f"epoch_{args.epochs} checkpoint"),
    (BASELINE_EVAL_JSON, "baseline evaluation JSON"),
    (NOTEBOOK_PATH, "canonical notebook"),
]:
    if p.exists():
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc}: {p}")
        if p == BEST_CKPT or p == FINAL_CKPT:
            # Try to find what's actually available
            if CKPT_DIR.exists():
                avail = sorted(CKPT_DIR.glob('*.pth'))
                print(f"    Available checkpoints: {[a.name for a in avail]}")
            sys.exit(1)
        else:
            sys.exit(1)
print()

# ============================================================================
# Step 1: Setup environment from notebook cells
# ============================================================================
print("Setting up environment from notebook cells...")

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open(NOTEBOOK_PATH) as f:
    nb = json.load(f)

def get_cell_source(cell_idx_1based):
    cell = nb['cells'][cell_idx_1based - 1]
    assert cell['cell_type'] == 'code', f"Cell {cell_idx_1based} is not code"
    return ''.join(cell['source'])

ns = {'__builtins__': __builtins__, '__name__': '__main__'}

cells_to_exec = [2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]

for cell_num in cells_to_exec:
    src = get_cell_source(cell_num)
    src = '\n'.join(line for line in src.split('\n') if not line.strip().startswith('!pip'))
    src = src.replace('plt.show()', 'plt.close("all")')
    try:
        exec(compile(src, f'<cell_{cell_num}>', 'exec'), ns)
        print(f"  ✓ Cell {cell_num}")
    except Exception as e:
        print(f"  ✗ Cell {cell_num}: {e}")

# Cell 6 (data setup)
src6 = get_cell_source(6)
src6 = src6.replace('plt.show()', 'plt.close("all")')
try:
    exec(compile(src6, '<cell_6>', 'exec'), ns)
    print(f"  ✓ Cell 6 (data)")
except Exception as e:
    print(f"  ✗ Cell 6: {e}")

# Cell 8 (mask check)
try:
    exec(compile(get_cell_source(8), '<cell_8>', 'exec'), ns)
    print(f"  ✓ Cell 8 (mask)")
except Exception as e:
    print(f"  ! Cell 8 skipped: {e}")

# Cell 18 (metrics)
src18 = get_cell_source(18)
src18 = '\n'.join(line for line in src18.split('\n') if not line.strip().startswith('!pip'))
try:
    exec(compile(src18, '<cell_18>', 'exec'), ns)
    print(f"  ✓ Cell 18 (metrics)")
except Exception as e:
    print(f"  ✗ Cell 18: {e}")

# Extract key objects
device = ns['device']
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
CONFIG = ns['CONFIG']

print(f"\nDevice: {device}")
print(f"LPIPS available: {ns.get('LPIPS_AVAILABLE', False)}")

# ============================================================================
# Step 2: Test dataset
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


# ============================================================================
# Step 3: Evaluation function (same as full model eval)
# ============================================================================

def evaluate_checkpoint(label, ckpt_path, out_dir, test_dataset, test_loader,
                        visual_indices, save_restored=True):
    ckpt_path = str(ckpt_path)
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    restored_dir = os.path.join(out_dir, 'restored_images')
    if save_restored:
        os.makedirs(restored_dir, exist_ok=True)

    gen = SGRGANGenerator(in_channels=4, out_channels=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['generator_state_dict'])
    gen.eval()

    epoch_num = ckpt.get('epoch', -1)
    if epoch_num >= 0:
        epoch_num += 1
    val_loss = ckpt.get('val_loss', float('nan'))
    best_val_loss = ckpt.get('best_val_loss', val_loss)

    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {label}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Epoch: {epoch_num}, val_loss={val_loss:.6f}")
    print(f"  Output dir: {out_dir}")
    print(f"{'=' * 80}")

    all_metrics = {
        'psnr': [], 'ssim': [], 'lpips': [], 'l1': [], 'l2': [],
        'style': [], 'perceptual': [],
        'direction': [], 'edge_strength': [], 'histogram': [],
        'image_names': []
    }

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing [{label}]"):
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

            if save_restored:
                from torchvision.utils import save_image
                # img_name already has .png extension — don't double it
                save_image(comp[0], os.path.join(restored_dir, img_name))

    # Averages
    avg_metrics = {}
    for key in all_metrics:
        if key != 'image_names':
            vals = [v for v in all_metrics[key] if not (isinstance(v, float) and math.isnan(v))]
            avg_metrics[f'avg_{key}'] = float(np.mean(vals)) if vals else float('nan')
            avg_metrics[f'std_{key}'] = float(np.std(vals)) if vals else float('nan')

    print(f"\n  Results for {label}:")
    print(f"  PSNR:       {avg_metrics['avg_psnr']:.2f} ± {avg_metrics['std_psnr']:.2f} dB")
    print(f"  SSIM:       {avg_metrics['avg_ssim']:.4f} ± {avg_metrics['std_ssim']:.4f}")
    lp = avg_metrics.get('avg_lpips', float('nan'))
    print(f"  LPIPS:      {lp:.4f}" if not math.isnan(lp) else "  LPIPS:      N/A")
    print(f"  L1:         {avg_metrics['avg_l1']:.4f}")
    print(f"  L2:         {avg_metrics['avg_l2']:.6f}")
    print(f"  Perceptual: {avg_metrics['avg_perceptual']:.4f}")
    print(f"  Style:      {avg_metrics['avg_style']:.6f}")
    print(f"  Direction:  {avg_metrics['avg_direction']:.4f}")
    print(f"  Edge:       {avg_metrics['avg_edge_strength']:.4f}")
    print(f"  Histogram:  {avg_metrics['avg_histogram']:.4f}")

    # Save JSON
    results_json = {
        'label': label,
        'ablation': ABLATION,
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

    # ---- Visual comparison ----
    n = len(visual_indices)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(visual_indices):
            sample = test_dataset[idx]
            c = sample['corrupted'].unsqueeze(0).to(device)
            m = sample['mask'].unsqueeze(0).to(device)
            o = sample['original'].unsqueeze(0).to(device)
            nm = sample['image_name']
            r = gen(c, m)
            comp_s = c * (1 - m) + r * m
            corr_np = c[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            comp_np = comp_s[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            mask_np = m[0, 0].cpu().numpy()
            orig_np = o[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            pv = calculate_psnr(comp_s[0], o[0])
            sv = calculate_ssim(comp_s[0], o[0])
            axes[row, 0].imshow(corr_np); axes[row, 0].set_title(f'Corrupted\n{nm}', fontsize=9); axes[row, 0].axis('off')
            axes[row, 1].imshow(mask_np, cmap='gray'); axes[row, 1].set_title('Mask', fontsize=9); axes[row, 1].axis('off')
            axes[row, 2].imshow(comp_np); axes[row, 2].set_title(f'Restored ({label})\nPSNR={pv:.2f} SSIM={sv:.4f}', fontsize=9); axes[row, 2].axis('off')
            axes[row, 3].imshow(orig_np); axes[row, 3].set_title('Ground Truth', fontsize=9); axes[row, 3].axis('off')

    fig.suptitle(f'{ABLATION.upper()} — {label} Visual Results', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'visual_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Visual comparison saved")

    # ---- Brushstroke analysis ----
    bs_indices = visual_indices[:3]
    fig2, axes2 = plt.subplots(len(bs_indices), 5, figsize=(25, 5 * len(bs_indices)))
    if len(bs_indices) == 1:
        axes2 = axes2[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(bs_indices):
            sample = test_dataset[idx]
            c = sample['corrupted'].unsqueeze(0).to(device)
            m = sample['mask'].unsqueeze(0).to(device)
            o = sample['original'].unsqueeze(0).to(device)
            gt_or = sample['orientation_field'].unsqueeze(0).to(device)
            r = gen(c, m)
            comp_s = c * (1 - m) + r * m
            po = brushstroke_losses._extract_orientation(comp_s)
            pe = brushstroke_losses._sobel_magnitude(comp_s)
            comp_np = comp_s[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            orig_np = o[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            po_np = po[0, 0].cpu().numpy()
            go_np = gt_or[0, 0].cpu().numpy() if gt_or.shape[1] == 1 else gt_or[0].argmax(0).cpu().numpy() * 22.5
            pe_np = pe[0, 0].cpu().numpy()
            axes2[row, 0].imshow(comp_np); axes2[row, 0].set_title('Restored', fontsize=10); axes2[row, 0].axis('off')
            axes2[row, 1].imshow(orig_np); axes2[row, 1].set_title('GT', fontsize=10); axes2[row, 1].axis('off')
            axes2[row, 2].imshow(po_np, cmap='hsv', vmin=0, vmax=180); axes2[row, 2].set_title('Pred Orient', fontsize=10); axes2[row, 2].axis('off')
            axes2[row, 3].imshow(go_np, cmap='hsv', vmin=0, vmax=180); axes2[row, 3].set_title('GT Orient', fontsize=10); axes2[row, 3].axis('off')
            axes2[row, 4].imshow(pe_np, cmap='hot'); axes2[row, 4].set_title('Pred Edge', fontsize=10); axes2[row, 4].axis('off')

    fig2.suptitle(f'Brushstroke Analysis — {ABLATION} {label}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'brushstroke_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ Brushstroke analysis saved")

    del gen, ckpt
    import gc; gc.collect()
    if device.type == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return avg_metrics, results_json


# ============================================================================
# Step 4: Evaluate both checkpoints
# ============================================================================
print("\n" + "=" * 80)
print(f"EVALUATING {ABLATION.upper()} CHECKPOINTS")
print("=" * 80)

avg_best, results_best = evaluate_checkpoint(
    label=f'{ABLATION}_best', ckpt_path=BEST_CKPT, out_dir=EVAL_DIR_BEST,
    test_dataset=test_dataset, test_loader=test_loader,
    visual_indices=VISUAL_INDICES, save_restored=True
)

avg_final, results_final = evaluate_checkpoint(
    label=f'epoch_{args.epochs}', ckpt_path=FINAL_CKPT, out_dir=EVAL_DIR_FINAL,
    test_dataset=test_dataset, test_loader=test_loader,
    visual_indices=VISUAL_INDICES, save_restored=True
)


# ============================================================================
# Step 5: Head-to-head comparison & selection
# ============================================================================
print("\n" + "=" * 80)
print(f"HEAD-TO-HEAD: {ABLATION}_best vs epoch_{args.epochs}")
print("=" * 80)

comparison_keys = ['avg_psnr', 'avg_ssim', 'avg_lpips', 'avg_l1', 'avg_l2',
                   'avg_perceptual', 'avg_style',
                   'avg_direction', 'avg_edge_strength', 'avg_histogram']
higher_better = {'avg_psnr', 'avg_ssim'}

best_label = f'{ABLATION}_best'
final_label = f'epoch_{args.epochs}'

print(f"\n{'Metric':<22} {best_label:>14} {final_label:>14} {'Delta':>12} {'Winner':>10}")
print("-" * 74)

wins = {best_label: 0, final_label: 0, 'tie': 0}
rows_json = []

for key in comparison_keys:
    a = avg_best.get(key, float('nan'))
    b = avg_final.get(key, float('nan'))
    if math.isnan(a) and math.isnan(b):
        rows_json.append({'metric': key.replace('avg_', ''), best_label: None, final_label: None, 'delta': None, 'winner': 'N/A'})
        print(f"  {key.replace('avg_', ''):<20} {'N/A':>14} {'N/A':>14} {'N/A':>12} {'N/A':>10}")
        continue
    delta = b - a
    name = key.replace('avg_', '')
    if key in higher_better:
        winner = best_label if a >= b else final_label
    else:
        winner = best_label if a <= b else final_label
    wins[winner] += 1
    fmt = '.2f' if 'psnr' in key else ('.6f' if key in ['avg_l2', 'avg_style'] else '.4f')
    print(f"  {name:<20} {a:{fmt}:>14} {b:{fmt}:>14} {delta:+{fmt}:>12} {winner:>10}")
    rows_json.append({'metric': name, best_label: a, final_label: b, 'delta': delta, 'winner': winner})

print("-" * 74)
print(f"  Wins: {best_label}={wins[best_label]}  {final_label}={wins[final_label]}")

# Selection: prioritize brushstroke metrics for ablations
brushstroke_keys = ['avg_direction', 'avg_edge_strength', 'avg_histogram']
bs_wins = {best_label: 0, final_label: 0}
for key in brushstroke_keys:
    a = avg_best.get(key, float('nan'))
    b = avg_final.get(key, float('nan'))
    if not math.isnan(a) and not math.isnan(b):
        if a <= b:
            bs_wins[best_label] += 1
        else:
            bs_wins[final_label] += 1

if bs_wins[best_label] > bs_wins[final_label]:
    selected = best_label
    selected_ckpt = str(BEST_CKPT)
    selected_epoch = results_best['epoch']
    selected_metrics = avg_best
elif bs_wins[final_label] > bs_wins[best_label]:
    selected = final_label
    selected_ckpt = str(FINAL_CKPT)
    selected_epoch = results_final['epoch']
    selected_metrics = avg_final
else:
    if wins[best_label] >= wins[final_label]:
        selected = best_label
        selected_ckpt = str(BEST_CKPT)
        selected_epoch = results_best['epoch']
        selected_metrics = avg_best
    else:
        selected = final_label
        selected_ckpt = str(FINAL_CKPT)
        selected_epoch = results_final['epoch']
        selected_metrics = avg_final

reason = (
    f"Selected {selected} as official {ABLATION} model. "
    f"Brushstroke wins: {best_label}={bs_wins[best_label]}, {final_label}={bs_wins[final_label]}. "
    f"Overall wins: {best_label}={wins[best_label]}, {final_label}={wins[final_label]}. "
    f"PSNR: {avg_best.get('avg_psnr',0):.2f} vs {avg_final.get('avg_psnr',0):.2f}. "
    f"Direction: {avg_best.get('avg_direction',0):.4f} vs {avg_final.get('avg_direction',0):.4f}."
)
print(f"\n  >>> SELECTED: {selected} (epoch {selected_epoch})")
print(f"  Reason: {reason}")


# ============================================================================
# Step 6: Freeze official checkpoint
# ============================================================================
if not args.skip_freeze:
    print("\n" + "=" * 80)
    print(f"FREEZING OFFICIAL {ABLATION.upper()} CHECKPOINT")
    print("=" * 80)

    os.makedirs(str(OFFICIAL_DIR), exist_ok=True)
    frozen_path = OFFICIAL_DIR / f'{ABLATION}_official_best.pth'

    shutil.copy2(selected_ckpt, str(frozen_path))
    print(f"  ✓ Copied → {frozen_path}")

    sel_record = {
        'selected_checkpoint': selected_ckpt,
        'selected_label': selected,
        'frozen_copy': str(frozen_path),
        'epoch': selected_epoch,
        'ablation': ABLATION,
        'run_folder': str(RUN_DIR),
        'selection_timestamp': datetime.datetime.now().isoformat(),
        'selection_reason': reason,
        'candidates': {
            best_label: {k: avg_best.get(k) for k in ['avg_psnr', 'avg_ssim', 'avg_l1',
                         'avg_direction', 'avg_edge_strength', 'avg_histogram', 'avg_perceptual', 'avg_style']},
            final_label: {k: avg_final.get(k) for k in ['avg_psnr', 'avg_ssim', 'avg_l1',
                          'avg_direction', 'avg_edge_strength', 'avg_histogram', 'avg_perceptual', 'avg_style']},
        },
        'comparison': rows_json,
        'wins': wins,
        'brushstroke_wins': bs_wins,
    }
    record_path = OFFICIAL_DIR / 'selection_record.json'
    with open(str(record_path), 'w') as f:
        json.dump(sel_record, f, indent=2)
    print(f"  ✓ Selection record: {record_path}")
else:
    frozen_path = None
    record_path = None
    print("\n  Skipping freeze (--skip-freeze)")


# ============================================================================
# Step 7: Baseline comparison
# ============================================================================
if not args.skip_comparison:
    print("\n" + "=" * 80)
    print(f"BASELINE vs {ABLATION.upper()} COMPARISON")
    print("=" * 80)

    with open(str(BASELINE_EVAL_JSON)) as f:
        baseline_eval = json.load(f)

    metric_names = ['psnr', 'ssim', 'lpips', 'l1', 'l2', 'perceptual', 'style',
                    'direction', 'edge_strength', 'histogram']
    hi = {'psnr', 'ssim'}

    cmp_rows = []
    bw = 0; fw = 0
    print(f"\n{'Metric':<22} {'Baseline':>14} {ABLATION:>14} {'Delta':>12} {'Winner':>10}")
    print("-" * 74)

    for m in metric_names:
        bv = baseline_eval.get(f'avg_{m}', float('nan'))
        fv = selected_metrics.get(f'avg_{m}', float('nan'))
        if math.isnan(bv) or math.isnan(fv):
            w = 'N/A'; d = None
        else:
            d = fv - bv
            if m in hi:
                w = ABLATION if fv > bv else ('baseline' if bv > fv else 'tie')
            else:
                w = ABLATION if fv < bv else ('baseline' if bv < fv else 'tie')
        if w == 'baseline': bw += 1
        elif w == ABLATION: fw += 1
        fmt = '.2f' if m == 'psnr' else ('.6f' if m in ['l2', 'style'] else '.4f')
        bs = f"{bv:{fmt}}" if not math.isnan(bv) else 'N/A'
        fs = f"{fv:{fmt}}" if not math.isnan(fv) else 'N/A'
        ds = f"{d:+{fmt}}" if d is not None else 'N/A'
        print(f"  {m:<20} {bs:>14} {fs:>14} {ds:>12} {w:>10}")
        cmp_rows.append({'metric': m, 'baseline': float(bv) if not math.isnan(bv) else None,
                         ABLATION: float(fv) if not math.isnan(fv) else None,
                         'delta': float(d) if d is not None else None, 'winner': w})

    print("-" * 74)
    print(f"  Wins: baseline={bw}  {ABLATION}={fw}")

    # Verdict
    bs_improved = sum(1 for r in cmp_rows if r['metric'] in ['direction', 'edge_strength', 'histogram'] and r['winner'] == ABLATION)
    bs_total = sum(1 for r in cmp_rows if r['metric'] in ['direction', 'edge_strength', 'histogram'] and r['winner'] != 'N/A')

    if bs_improved == bs_total and bs_total > 0:
        verdict = f"{ABLATION}_improves_all_brushstroke"
    elif bs_improved > 0:
        verdict = f"{ABLATION}_improves_{bs_improved}_of_{bs_total}_brushstroke"
    elif fw > bw:
        verdict = f"{ABLATION}_wins_overall"
    elif bw > fw:
        verdict = "baseline_wins_overall"
    else:
        verdict = "tie"
    print(f"\n  Verdict: {verdict}")

    bvf_path = PROJECT_ROOT / 'results' / f'baseline_vs_{ABLATION}_comparison.json'
    bvf = {
        'timestamp': datetime.datetime.now().isoformat(),
        'baseline': {'checkpoint': str(BASELINE_OFFICIAL_CKPT), 'epoch': baseline_eval.get('epoch', 46)},
        ABLATION: {'checkpoint': str(frozen_path) if frozen_path else selected_ckpt, 'epoch': selected_epoch},
        'comparison': cmp_rows,
        'wins': {'baseline': bw, ABLATION: fw},
        'verdict': verdict,
        'brushstroke_improvement': {'improved': bs_improved, 'total': bs_total},
    }
    with open(str(bvf_path), 'w') as f:
        json.dump(bvf, f, indent=2)
    print(f"  ✓ Comparison saved: {bvf_path}")
else:
    print(f"\n  Skipping comparison (--skip-comparison)")


# ============================================================================
# Step 8: Final verification
# ============================================================================
print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

checks = [
    (EVAL_DIR_BEST / 'evaluation_results.json', f"{ABLATION}_best eval JSON"),
    (EVAL_DIR_BEST / 'visual_comparison.png', f"{ABLATION}_best visual"),
    (EVAL_DIR_BEST / 'brushstroke_analysis.png', f"{ABLATION}_best brushstroke"),
    (EVAL_DIR_FINAL / 'evaluation_results.json', f"epoch_{args.epochs} eval JSON"),
    (EVAL_DIR_FINAL / 'visual_comparison.png', f"epoch_{args.epochs} visual"),
    (EVAL_DIR_FINAL / 'brushstroke_analysis.png', f"epoch_{args.epochs} brushstroke"),
]
if not args.skip_freeze:
    checks.extend([
        (frozen_path, f"{ABLATION}_official_best.pth"),
        (record_path, "selection_record.json"),
    ])
if not args.skip_comparison:
    checks.append((bvf_path, "baseline comparison JSON"))

all_ok = True
for p, desc in checks:
    ex = Path(p).exists()
    sz = Path(p).stat().st_size if ex else 0
    print(f"  {'✓' if ex else '✗'} {sz:>12,} bytes  {desc}")
    if not ex: all_ok = False

print(f"\n  {'ALL CHECKS PASSED ✓' if all_ok else 'SOME CHECKS FAILED ✗'}")

print("\n" + "=" * 80)
print(f"{ABLATION.upper()} EVALUATION COMPLETE")
print("=" * 80)
print(f"  Selected: {selected} (epoch {selected_epoch})")
if frozen_path:
    print(f"  Frozen at: {frozen_path}")
print(f"  Next: Evaluate remaining ablations")
print("=" * 80)
