# PROJECT_PROGRESS.md

## Project

ARTIFEX — Brushstroke-Aware Deep Inpainting for Van Gogh Art Restoration

## Purpose

This file tracks meaningful project changes, experiment progress, baseline/full-model decisions, bugs, fixes, and thesis-relevant milestones.
It should be updated before or immediately after each important commit.

---

# Current Status Snapshot

## Current branch

`thesis-brushstroke-experiments`

## Current official baseline

- **Checkpoint:** `models/baseline_official/baseline_official_best.pth`
- **Selected from:** epoch 46
- **Why selected:** best overall checkpoint after comparison against epoch 60
- **Reference metadata:** `models/baseline_official/selection_record.json`

## Current project state

- Data pipeline complete
- HDF5 brushstroke priors complete
- Baseline selected and frozen
- Brushstroke-aware smoke test passed
- Generator-only warm-start helper implemented and verified
- **5-epoch warm-start verification run completed successfully**
- Ready for real 50-epoch full brushstroke-aware training

## Current risks

- Must keep run folders isolated
- Full 50-epoch run will take ~25 hours on M2 MPS
- Val loss (~0.91) is higher than baseline (~0.30) due to additional brushstroke loss terms — this is expected and not a concern

## Immediate next step

Restart kernel, re-run setup, set epochs=50, launch real full brushstroke-aware training run

---

# Progress Log

---

## Entry Template

### [YYYY-MM-DD] Short title of change

- **Branch:** `branch-name`
- **Commit:** `commit-hash-or-TBD`
- **Type:** code / experiment / evaluation / bugfix / refactor / thesis / config
- **Status:** planned / in progress / completed

#### What changed

-
-
-

#### Why this change was made

-
-

#### Files touched

-
-
-

#### Outputs / artifacts

-
-
-

#### Results / observations

-
-
-

#### Risks / issues found

-
-
-

#### Decision taken

-
-

#### Next step

- ***

## [2026-03-01] Baseline checkpoint comparison and freeze

- **Branch:** `thesis-brushstroke-experiments`
- **Commit:** `68c7e37`
- **Type:** evaluation / experiment-management
- **Status:** completed

#### What changed

- Added a clean comparison workflow for baseline candidate checkpoints
- Compared `baseline_best.pth` (epoch 46) vs `checkpoint_epoch_60.pth` (epoch 60)
- Froze the selected official baseline checkpoint
- Added metadata recording the baseline selection decision

#### Why this change was made

- The baseline had plateaued and should not be blindly trained further
- A single official baseline was required before full-model training and ablation comparisons

#### Files touched

- `artifex_FULLY_FIXED_M2_PATHS.ipynb`
- `results/baseline_comparison.json`
- `models/baseline_official/selection_record.json`

#### Outputs / artifacts

- `models/baseline_official/baseline_official_best.pth`
- `models/baseline_official/selection_record.json`
- `results/baseline_comparison.json`
- `results/baseline_ep46/`
- `results/baseline_ep60/`

#### Results / observations

- Epoch 46 won **7/9** comparison metrics over epoch 60
- PSNR: **25.61 vs 25.38 dB**
- SSIM: **0.8622 vs 0.8599**
- Epoch 46 also performed better on L1, L2, perceptual, direction, and edge metrics
- Epoch 60 only had tiny advantages on style and histogram

#### Risks / issues found

- Existing evaluation cells were too coupled to notebook config state
- Needed isolated comparison outputs to avoid mixing with old runs

#### Decision taken

- Official baseline fixed as **epoch 46**
- No more baseline continuation unless a future major problem is discovered

#### Next step

- Add safe generator-only initialization from the frozen baseline for the full brushstroke-aware run

---

## [2026-03-01] Full ablation smoke test passed

- **Branch:** `thesis-brushstroke-experiments`
- **Commit:** `TBD`
- **Type:** experiment / verification
- **Status:** completed

#### What changed

- Ran a 2-epoch smoke test for the `full` brushstroke-aware setup
- Verified all three brushstroke losses were active and decreasing
- Confirmed isolated full-run folder creation and checkpoint/log saving

#### Why this change was made

- Needed runtime proof that the full brushstroke-aware pipeline worked before launching long experiments

#### Files touched

- `artifex_FULLY_FIXED_M2_PATHS.ipynb`

#### Outputs / artifacts

- `runs/full_20260301_063323/`
- `full_best.pth`
- training history JSON/CSV
- live training curves PNG

#### Results / observations

- direction loss > 0
- edge-strength loss > 0
- histogram loss > 0
- validation loss improved from epoch 1 to epoch 2
- no NaNs, no shape issues, no path contamination

#### Risks / issues found

- Smoke test used temporary 2-epoch config and had to be reset after verification

#### Decision taken

- Full brushstroke-aware training pipeline is healthy
- Safe to move toward real full-model training once baseline is frozen

#### Next step

- Prepare full-model run initialized from frozen baseline generator weights

---

## [2026-03-01] Generator-only warm-start helpers added

- **Branch:** `thesis-brushstroke-experiments`
- **Commit:** `9335a8f`
- **Type:** code / experiment-management
- **Status:** completed

#### What changed

- Added `init_generator_from_baseline()` helper (cell 29) — loads only generator state_dict from frozen baseline
- Added `verify_full_run_setup()` sanity check (cell 30) — 10-point verification before training
- Added markdown header cell (cell 28) documenting the warm-start workflow

#### Why this change was made

- Neither `train_full_model()` nor `run_ablation()` supported generator-only weight initialization
- Needed a safe path to warm-start the full model from the frozen baseline without loading discriminators, optimizers, or epoch counters

#### Files touched

- `artifex_FULLY_FIXED_M2_PATHS.ipynb` (cells 28-30 inserted)

#### Decision taken

- This is a "warm-start" / "baseline-initialized" approach, NOT a "resume" — fresh optimizers, fresh discriminators, epoch starts at 1

#### Next step

- Run 5-epoch verification to confirm no crash, no NaN, brushstroke losses active

---

## [2026-03-01] 5-epoch warm-start verification run — PASSED

- **Branch:** `thesis-brushstroke-experiments`
- **Commit:** TBD
- **Type:** experiment / verification
- **Status:** completed

#### What changed

- Ran full brushstroke-aware model for 5 epochs using warm-started generator from frozen baseline (epoch 46)
- Fresh discriminators and optimizers; brushstroke lambdas: direction=2.0, edge=1.0, histogram=1.0
- Added two scratch cells (31-32) for warm-start prep and training execution

#### Why this change was made

- Needed to verify the warm-start pipeline works end-to-end before committing to a long 50-epoch training run
- Confirms: no crash, no NaN, brushstroke losses active and decreasing, run folder isolation

#### Files touched

- `artifex_FULLY_FIXED_M2_PATHS.ipynb` (cells 31-32 added)

#### Outputs / artifacts

- `runs/full_20260301_120318/` — isolated run folder
- `runs/full_20260301_120318/checkpoints/full_best.pth` (276MB)
- `runs/full_20260301_120318/checkpoints/checkpoint_epoch_5.pth` (276MB)
- `runs/full_20260301_120318/logs/training_history_live.csv`
- `runs/full_20260301_120318/logs/training_history_final.json`
- `runs/full_20260301_120318/outputs/training_curves_live.png`

#### Results / observations

- **No NaN, no crashes, all 5 epochs completed (~2.5 hr total)**
- Train G loss: 0.987 → 0.821 (steadily decreasing)
- Brushstroke losses (train, all decreasing):
  - direction: 0.158 → 0.093
  - edge_strength: 0.137 → 0.134
  - histogram: 0.041 → 0.028
- Val loss: 0.931 → 0.908 (stable, slightly improving)
- D losses stable at ~0.25 (healthy GAN dynamics)
- Val loss (~0.91) higher than baseline val loss (~0.30) — expected because brushstroke terms add extra loss components

#### Risks / issues found

- Config snapshot records `epochs: 100` (the original CONFIG default) even though runtime was overridden to 5 — cosmetic only
- Kernel reset between sessions requires re-running all setup cells

#### Decision taken

- Warm-start pipeline is verified and healthy
- Ready to proceed with real 50-epoch full training run
- Will treat this as "baseline-initialized full model" experiment (not "resume")

#### Next step

- Restart kernel, re-run setup cells, set CONFIG['epochs']=50, launch real full training
- After completion: evaluate on test set vs frozen baseline

---

# Milestone Tracker

## Milestone 1 — Data and Priors

- [x] Image preprocessing
- [x] Mask pipeline
- [x] Corrupted image creation
- [x] Brushstroke prior extraction
- [x] HDF5 storage and verification

## Milestone 2 — Baseline System

- [x] Baseline architecture working
- [x] Baseline training completed enough for checkpoint selection
- [x] Official baseline selected
- [x] Official baseline frozen

## Milestone 3 — Brushstroke-Aware Training

- [x] Brushstroke losses implemented
- [x] Full smoke test passed
- [x] Generator-only warm-start helpers added
- [x] 5-epoch warm-start verification passed
- [ ] Full 50-epoch training run
- [ ] Full model evaluated on test set

## Milestone 4 — Ablation Study

- [ ] dir_only trained
- [ ] edge_only trained
- [ ] hist_only trained
- [ ] all ablations evaluated
- [ ] final comparison table completed

## Milestone 5 — Thesis Artifacts

- [ ] quantitative result tables
- [ ] qualitative comparison figures
- [ ] zoom/brushstroke visuals
- [ ] method section finalized
- [ ] experiment section finalized
- [ ] ablation discussion finalized

---

# Final Model Decision Log

## Official baseline

- **Selected checkpoint:** epoch 46
- **Reason:** best checkpoint after direct comparison with epoch 60
- **Frozen on:** 2026-03-01

## Official full model

- **Selected checkpoint:** TBD
- **Reason:** TBD
- **Frozen on:** TBD

---

# Notes for Each Commit

Before each important commit:

1. Add a new dated log entry
2. Write what changed in plain language
3. Record any results or metric differences
4. Write the next immediate step
5. Then commit code + this file together

Recommended rule:

- If a commit changes the experiment state, this file must also be updated.
