"use client";

import BeforeAfterSlider from "./BeforeAfterSlider";
import MetricBadge from "./MetricBadge";

const OFFICIAL_METRIC_DEFS = [
  { key: "avg_psnr", label: "PSNR (dB)", decimals: 3 },
  { key: "avg_ssim", label: "SSIM", decimals: 4 },
  { key: "avg_l1", label: "L1", decimals: 4 },
  { key: "avg_direction", label: "Direction", decimals: 4 },
  { key: "avg_edge_strength", label: "Edge Str.", decimals: 4 },
  { key: "avg_histogram", label: "Histogram", decimals: 4 },
  { key: "avg_perceptual", label: "Perceptual", decimals: 4 },
];

const PER_UPLOAD_METRIC_DEFS = [
  { key: "psnr", label: "PSNR (dB)", decimals: 2 },
  { key: "ssim", label: "SSIM", decimals: 4 },
  { key: "l1", label: "L1", decimals: 4 },
  { key: "l2", label: "L2", decimals: 6 },
  { key: "perceptual", label: "Perceptual", decimals: 4 },
  { key: "style", label: "Style", decimals: 6 },
];

/**
 * ModelResultCard — shows restoration result + metrics for one model.
 * Distinguishes between:
 *   - Per-upload metrics (when GT provided)
 *   - Official test-set averages (always available for models with eval data)
 */
export default function ModelResultCard({
  result,
  originalSrc,
  hasGroundTruth,
}) {
  const {
    model_id,
    display_name,
    available,
    restored_image_b64,
    error,
    checkpoint_metadata,
    checkpoint_path,
    official_eval_summary,
    per_upload_metrics,
    inference_time_s,
    mask_coverage_pct,
  } = result;

  const epoch = checkpoint_metadata?.epoch;

  return (
    <div
      className={`glass rounded-2xl p-5 flex flex-col gap-3 min-w-[280px] flex-1 ${
        available ? "border-[var(--color-gold)]/20" : "opacity-60"
      }`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 flex-wrap">
        <h3 className="text-sm font-semibold text-[var(--color-gold-light)] flex-1 font-serif">
          {display_name}
        </h3>
        {available ? (
          <span className="bg-[var(--color-success)]/20 text-[var(--color-success)] border border-[var(--color-success)]/30 rounded-full px-2 py-0.5 text-[10px]">
            Available
          </span>
        ) : (
          <span className="bg-[var(--color-warning)]/20 text-[var(--color-warning)] border border-[var(--color-warning)]/30 rounded-full px-2 py-0.5 text-[10px]">
            Coming Soon
          </span>
        )}
      </div>

      {/* Model ID */}
      <code className="text-[10px] text-gray-400 bg-black/30 px-2 py-0.5 rounded self-start">
        {model_id}
      </code>

      {/* Image area */}
      {!available ? (
        <div className="aspect-square flex flex-col items-center justify-center bg-white/[0.03] rounded-lg gap-2 p-6">
          <span className="text-3xl">🔬</span>
          <span className="text-xs text-gray-400 text-center">
            Checkpoint not yet available.
            <br />
            Training in progress.
          </span>
        </div>
      ) : error ? (
        <div className="aspect-square flex flex-col items-center justify-center bg-[var(--color-error)]/5 border border-[var(--color-error)]/20 rounded-lg gap-2 p-4">
          <span className="text-2xl">⚠️</span>
          <span className="text-[11px] text-[var(--color-error)] text-center">
            {error}
          </span>
        </div>
      ) : restored_image_b64 && originalSrc ? (
        <BeforeAfterSlider
          originalSrc={originalSrc}
          restoredSrc={restored_image_b64}
        />
      ) : restored_image_b64 ? (
        <img
          src={restored_image_b64}
          alt={`Restored by ${display_name}`}
          className="w-full rounded-lg aspect-square object-cover"
        />
      ) : null}

      {/* Inference metadata */}
      {available && restored_image_b64 && (
        <div className="flex gap-3 flex-wrap text-[11px] text-gray-400 bg-black/20 px-3 py-2 rounded-lg">
          {inference_time_s != null && (
            <span>Inference: {inference_time_s}s</span>
          )}
          {mask_coverage_pct != null && (
            <span>Mask: {mask_coverage_pct.toFixed(1)}%</span>
          )}
          {epoch !== undefined && <span>Epoch {epoch}</span>}
        </div>
      )}

      {/* Checkpoint path */}
      {checkpoint_path && (
        <div
          className="text-[9px] text-gray-500 break-all bg-black/15 px-2 py-1 rounded"
          title={checkpoint_path}
        >
          {checkpoint_path.split("/").slice(-2).join("/")}
        </div>
      )}

      {/* ============ PER-UPLOAD METRICS (only when GT provided) ============ */}
      {hasGroundTruth &&
        per_upload_metrics &&
        !per_upload_metrics.error && (
          <div>
            <div className="text-[10px] text-[var(--color-success)] uppercase tracking-wider mb-2 flex items-center gap-1.5 font-semibold">
              <span>📐</span> Per-Upload Metrics
              <span className="ml-auto text-gray-500 normal-case tracking-normal font-normal">
                vs. your ground truth
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {PER_UPLOAD_METRIC_DEFS.map(({ key, label, decimals }) => (
                <MetricBadge
                  key={key}
                  value={per_upload_metrics[key]}
                  label={label}
                  decimals={decimals}
                />
              ))}
            </div>
          </div>
        )}

      {/* ============ OFFICIAL BENCHMARK METRICS (always shown if available) ============ */}
      {official_eval_summary && (
        <div>
          <div className="text-[10px] text-[var(--color-gold)] uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <span>📊</span> Official Test-Set Averages
            <span className="ml-auto text-gray-500 normal-case tracking-normal">
              n={official_eval_summary.num_images || "?"}
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {OFFICIAL_METRIC_DEFS.map(({ key, label, decimals }) => (
              <MetricBadge
                key={key}
                value={official_eval_summary[key]}
                label={label}
                decimals={decimals}
              />
            ))}
          </div>
          <p className="text-[9px] text-gray-500 mt-2 italic">
            Official benchmark (n=305 images). Not computed for your upload.
          </p>
        </div>
      )}

      {/* Download */}
      {restored_image_b64 && (
        <a
          href={restored_image_b64}
          download={`restored_${model_id}.png`}
          className="block text-center py-2 px-4 bg-[var(--color-gold)]/15 border border-[var(--color-gold)]/30 rounded-lg text-[var(--color-gold)] text-xs no-underline hover:bg-[var(--color-gold)]/25 transition-colors"
        >
          ⬇ Download
        </a>
      )}
    </div>
  );
}
