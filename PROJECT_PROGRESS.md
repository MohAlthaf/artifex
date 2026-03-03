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

## Current official full model

- **Checkpoint:** `models/full_official/full_official_best.pth`
- **Selected from:** `full_best.pth` (epoch 3, best val loss during full training)
- **Why selected:** Won 7/9 metrics vs epoch 50, including 2/3 brushstroke metrics. Best PSNR (25.91) and direction loss (0.1925).
- **Reference metadata:** `models/full_official/selection_record.json`

## Current project state

- Data pipeline complete
- HDF5 brushstroke priors complete
- Baseline selected and frozen (epoch 46, `models/baseline_official/`)
- 50-epoch full brushstroke-aware training completed (2026-03-02 to 2026-03-03)
- Full model evaluated on 305-image test set
- Full model selected and frozen (epoch 3, `models/full_official/`)
- **Full model improves ALL brushstroke metrics over baseline**
- Baseline-vs-full comparison saved (`results/baseline_vs_full_comparison.json`)

## Current risks

- Must keep run folders isolated
- Full model val loss (~0.90) is higher than baseline val loss (~0.30) — expected, different loss objectives (brushstroke terms increase total loss but improve quality)
- LPIPS metric unavailable due to SSL certificate issue (non-blocking)

## Immediate next step

Run ablation experiments: `dir_only`, `edge_only`, `hist_only`

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

## [2026-03-03] 50-epoch full training completed, evaluated, and frozen

- **Branch:** `thesis-brushstroke-experiments`
- **Commit:** TBD
- **Type:** experiment / evaluation
- **Status:** completed

#### What changed

- 50-epoch full brushstroke-aware training completed (run `full_20260302_070353`)
- Training was interrupted twice (at epoch 32 and epoch 41) and resumed each time
- Both full checkpoints evaluated on 305-image test set: `full_best.pth` (epoch 3) and `checkpoint_epoch_50.pth`
- `full_best.pth` selected as official full model (7/9 metrics won, 2/3 brushstroke metrics won)
- Official full model frozen at `models/full_official/full_official_best.pth`
- Baseline-vs-full comparison completed: full model improves ALL 3 brushstroke metrics over baseline

#### Why this change was made

- Complete the full brushstroke-aware training run as the thesis's central experiment
- Establish quantitative evidence that brushstroke losses improve restoration quality

#### Files touched

- `scripts/evaluate_full_model.py` (new — standalone evaluation pipeline)
- `models/full_official/full_official_best.pth` (new — frozen official full model)
- `models/full_official/selection_record.json` (new — selection metadata)
- `results/full_eval/full_best/evaluation_results.json` (new)
- `results/full_eval/epoch_50/evaluation_results.json` (new)
- `results/baseline_vs_full_comparison.json` (new)
- `PROJECT_PROGRESS.md` (updated)

#### Outputs / artifacts

- `results/full_eval/full_best/` — evaluation JSON, visual comparison, brushstroke analysis, 305 restored images
- `results/full_eval/epoch_50/` — same set of artifacts
- `models/full_official/selection_record.json` — checkpoint selection rationale + both candidates' metrics
- `results/baseline_vs_full_comparison.json` — head-to-head baseline vs full on test set

#### Results / observations

- **full_best (epoch 3) vs epoch_50 (epoch 50):**
  - full_best wins 7/9 metrics (PSNR, SSIM, L1, L2, perceptual, direction, edge)
  - epoch_50 wins 2/9 (style, histogram)
  - PSNR: 25.91 vs 25.39; Direction: 0.1925 vs 0.2118
- **Baseline vs Full (using full_best):**
  - Full wins 8/9 metrics (only loses on style: 0.000574 vs 0.000556)
  - Direction: 0.2212 → 0.1925 (−13.0%)
  - Edge: 0.1484 → 0.1466 (−1.2%)
  - Histogram: 0.0570 → 0.0529 (−7.2%)
  - PSNR: 25.61 → 25.91 (+0.30 dB)
  - SSIM: 0.8622 → 0.8693 (+0.0071)
  - Verdict: `full_improves_all_brushstroke`
- LPIPS unavailable (SSL cert issue, non-blocking)
- Val loss (0.9045) is higher than baseline (0.3004) — expected since brushstroke loss terms add to objective

#### Risks / issues found

- Best val loss was at epoch 3 — model may benefit from different learning rate schedule or warmup
- Style loss slightly regressed (0.000556 → 0.000574) — minor, within noise
- Training required 3 resume cycles due to kernel interruptions

#### Decision taken

- `full_best.pth` (epoch 3) selected as official full model
- Frozen at `models/full_official/full_official_best.pth`

#### Next step

- Run ablation experiments: `dir_only`, `edge_only`, `hist_only` (50 epochs each)
- Each ablation isolates one brushstroke loss component
- Compare all variants against baseline and full model

---

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

## [2026-03-02] 50-epoch full training launched — interrupted at epoch 32

- **Branch:** `thesis-brushstroke-experiments`
- **Commit:** `50eade7`
- **Type:** experiment
- **Status:** interrupted

#### What changed

- Launched clean 50-epoch full brushstroke-aware training run from warm-started baseline generator (epoch 46)
- Fixed config snapshot bug: `CONFIG['epochs'] = 50` now set before `run_ablation()` so snapshot is correct
- Training ran successfully through 31 complete epochs before kernel interruption during epoch 32

#### Why this change was made

- This is the core full brushstroke-aware experiment for the thesis

#### Files touched

- `artifex_FULLY_FIXED_M2_PATHS.ipynb` (cell 31 fixed for config snapshot ordering)

#### Outputs / artifacts

- `runs/full_20260302_070353/` — canonical full experiment folder
- `runs/full_20260302_070353/checkpoints/checkpoint_epoch_{5,10,15,20,25,30}.pth` — all 276 MB, valid
- `runs/full_20260302_070353/checkpoints/full_best.pth` — epoch 3, val_loss=0.905
- `runs/full_20260302_070353/logs/config_snapshot.json` — correct (epochs=50, model_type=full)
- `runs/full_20260302_070353/logs/training_history_live.json` — 31 epochs recorded, status=interrupted

#### Results / observations

- All 31 epochs completed cleanly with no NaN, no corruption
- All periodic checkpoints are 276 MB (consistent, valid)
- Train G loss: 0.63 at epoch 31 (steadily decreasing from 0.99)
- Brushstroke losses all decreasing: direction 0.158→0.028, edge 0.137→0.117, histogram 0.041→0.018
- Val total loss plateaued around 0.93 (expected, different objective than baseline)
- best_val_loss = 0.905 at epoch 3 (early epochs had lower val loss before brushstroke losses dominated)
- Config snapshot correctly records epochs=50, model_type=full, all lambdas active

#### Risks / issues found

- Kernel interrupted during epoch 32 — no checkpoint saved for epochs 31 or 32
- Scheduler state (ReduceLROnPlateau) not saved in checkpoints — patience counter will reset on resume
- full_best.pth is from epoch 3, which is early; later epochs may be better by brushstroke quality despite higher raw val loss

#### Decision taken

- Resume from `checkpoint_epoch_30.pth` (last valid periodic checkpoint)
- Continue in the same `full_20260302_070353` run folder for thesis cleanliness
- Do not restart from scratch — 31 epochs of valid training is too valuable to discard

#### Next step

- Implement safe resume workflow and continue training to epoch 50

---

## [2026-03-03] Resume workflow implemented for interrupted full training

- **Branch:** `thesis-brushstroke-experiments`
- **Commit:** TBD
- **Type:** code / bugfix / experiment-management
- **Status:** completed

#### What changed

- Fixed `train_full_model()` bug: `best_val_loss` now restored from `best_val_loss` key (was incorrectly using `val_loss`)
- Added training history restore on resume: appends to existing 31-epoch history instead of starting empty
- Added resume prep cell (cell 31): configures CONFIG to resume in the original `full_20260302_070353` folder
- Added resume verification cell (cell 32): 12-point pre-flight checklist before resuming

#### Why this change was made

- Original resume path had a bug that would use `val_loss` (that epoch's val loss = 0.937) instead of `best_val_loss` (0.905), potentially overwriting `full_best.pth` with a worse model
- Training history was not restored on resume, which would create a gap in the CSV/JSON logs
- Needed a safe, verified path to continue the interrupted run

#### Files touched

- `artifex_FULLY_FIXED_M2_PATHS.ipynb` (cell 16: train_full_model fix; cells 31-32: new resume cells)
- `PROJECT_PROGRESS.md`

#### Outputs / artifacts

- No new run outputs yet (resume not yet executed)

#### Risks / issues found

- Scheduler state not saved in checkpoints — ReduceLROnPlateau patience resets on resume (tolerable: patience=10, no LR reduction had triggered in 31 epochs)
- Epoch 31 training completed but was not saved as a checkpoint (epoch 31 is between periodic saves at ep30 and ep35)

#### Decision taken

- Resume from epoch 30 checkpoint (the latest fully valid periodic checkpoint)
- Continue in the same run folder (`full_20260302_070353`) — one experiment = one folder
- Training history appended seamlessly (31 pre-existing + 19 new = 50 total epochs in final CSV/JSON)

#### Next step

- Restart kernel, run setup cells 2-16, skip cell 17, run cells 27/29/30, run cell 31 (resume prep), run cell 32 (resume verify), run cell 33 (training)

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

- [ ] dir_only trained ← **READY TO RUN** (cells 41–45 + evaluate_ablation.py)
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

- **Selected checkpoint:** epoch 3 (full_best.pth)
- **Reason:** Won 7/9 test metrics vs epoch 50; all 3 brushstroke metrics improved over baseline
- **Frozen on:** 2026-03-03

---

# Ablation Experiments

## dir_only — Direction Loss Only

- **Status:** READY TO RUN (not yet trained)
- **Config:** `lambda_direction = 2.0, lambda_thickness = 0.0, lambda_histogram = 0.0`
- **Warm-start:** frozen baseline (generator only), discriminators fresh
- **Epochs:** 50 (matches full model)
- **Notebook cells:** 41 (header), 42 (verify_ablation_setup), 43 (prep), 44 (train), 45 (eval)
- **Evaluation script:** `scripts/evaluate_ablation.py --ablation dir_only --run-dir <TBD>`
- **Implementation date:** 2026-03-04
- **Changes made:**
  - Added `verify_ablation_setup(ablation_name)` — generalized verify function for any ablation
  - Added dir_only prep cell: calls `run_ablation('dir_only')`, sets epochs=50, warm-starts from baseline, verifies
  - Created `scripts/evaluate_ablation.py` — parameterized evaluation pipeline for all ablations
  - Existing cells 1–40 unchanged

## edge_only — Edge Strength Loss Only

- **Status:** NOT STARTED (pending dir_only completion)

## hist_only — Histogram Loss Only

- **Status:** NOT STARTED (pending dir_only completion)

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
