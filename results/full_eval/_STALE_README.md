# STALE — DO NOT USE

This folder contains **v1 evaluation results** with LPIPS=NaN (AlexNet download failure).

**Superseded by:** `results/full_eval_v2/evaluation_results.json`

The per-image direction and edge_strength values in the `full_best/` subfolder are
still valid (they match v2 averages exactly) and are used by `scripts/statistical_analysis.py`
for brushstroke significance testing. However, LPIPS, perceptual, style, and
histogram values are from the v1 pipeline and MUST NOT be used as final metrics.
