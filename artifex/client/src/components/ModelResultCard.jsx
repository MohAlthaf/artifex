/**
 * ModelResultCard.jsx
 * ====================
 * Displays the restoration result for a single model:
 *   - Model name + availability badge
 *   - Restored image (or unavailable/loading state)
 *   - Before/after slider (when result is present)
 *   - Checkpoint metadata
 *   - Official test-set metric summary  (clearly labelled as benchmark metrics)
 */

import { useState } from "react";

const METRIC_LABELS = {
  avg_psnr: { label: "PSNR (dB)", better: "higher", decimals: 3 },
  avg_ssim: { label: "SSIM", better: "higher", decimals: 4 },
  avg_l1: { label: "L1", better: "lower", decimals: 4 },
  avg_direction: { label: "Direction", better: "lower", decimals: 4 },
  avg_edge_strength: { label: "Edge Strength", better: "lower", decimals: 4 },
  avg_histogram: { label: "Histogram χ²", better: "lower", decimals: 4 },
  avg_perceptual: { label: "Perceptual", better: "lower", decimals: 4 },
};

function MetricBadge({ value, label, decimals }) {
  if (value === null || value === undefined) return null;
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        background: "rgba(255,255,255,0.05)",
        border: "1px solid rgba(212,175,55,0.15)",
        borderRadius: "var(--radius-md)",
        padding: "8px 12px",
        minWidth: "80px",
      }}
    >
      <span
        style={{
          fontSize: "0.95rem",
          fontWeight: "600",
          color: "var(--color-gold-light)",
        }}
      >
        {typeof value === "number" ? value.toFixed(decimals) : value}
      </span>
      <span
        style={{
          fontSize: "0.65rem",
          color: "var(--color-gray-400)",
          textAlign: "center",
          marginTop: "2px",
        }}
      >
        {label}
      </span>
    </div>
  );
}

function BeforeAfterSlider({ originalSrc, restoredSrc }) {
  const [pos, setPos] = useState(50);
  const handleMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x =
      (e.clientX || (e.touches && e.touches[0].clientX) || 0) - rect.left;
    setPos(Math.max(0, Math.min(100, (x / rect.width) * 100)));
  };
  return (
    <div
      onMouseMove={handleMove}
      onTouchMove={handleMove}
      style={{
        position: "relative",
        overflow: "hidden",
        borderRadius: "var(--radius-lg)",
        cursor: "ew-resize",
        userSelect: "none",
        background: "#000",
        aspectRatio: "1",
        width: "100%",
      }}
    >
      <img
        src={restoredSrc}
        alt="Restored"
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: `${pos}%`,
          height: "100%",
          overflow: "hidden",
        }}
      >
        <img
          src={originalSrc}
          alt="Original"
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            height: "100%",
            width: `${10000 / pos}%`,
            maxWidth: "none",
            objectFit: "cover",
          }}
        />
      </div>
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `${pos}%`,
          width: "2px",
          background: "var(--color-gold)",
          transform: "translateX(-50%)",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%,-50%)",
            background: "var(--color-gold)",
            borderRadius: "50%",
            width: "28px",
            height: "28px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "0.75rem",
            color: "#000",
            fontWeight: "bold",
          }}
        >
          ⟷
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          bottom: "8px",
          left: "8px",
          background: "rgba(0,0,0,0.7)",
          padding: "2px 8px",
          borderRadius: "4px",
          fontSize: "0.65rem",
          color: "#fff",
        }}
      >
        DAMAGED
      </div>
      <div
        style={{
          position: "absolute",
          bottom: "8px",
          right: "8px",
          background: "rgba(212,175,55,0.9)",
          padding: "2px 8px",
          borderRadius: "4px",
          fontSize: "0.65rem",
          color: "#000",
          fontWeight: "600",
        }}
      >
        RESTORED
      </div>
    </div>
  );
}

export default function ModelResultCard({ result, originalSrc, isLoading }) {
  const {
    model_id,
    display_name,
    available,
    restored_image_b64,
    error,
    checkpoint_metadata,
    checkpoint_path,
    official_eval_summary,
    inference_time_s,
    mask_coverage_pct,
  } = result;

  const epoch = checkpoint_metadata?.epoch;
  const valLoss = checkpoint_metadata?.val_loss;

  return (
    <div
      className="glass"
      style={{
        borderRadius: "var(--radius-xl)",
        padding: "var(--spacing-lg)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--spacing-md)",
        minWidth: "280px",
        flex: "1 1 280px",
        border: available
          ? "1px solid rgba(212,175,55,0.2)"
          : "1px solid rgba(255,255,255,0.05)",
        opacity: available ? 1 : 0.6,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
          flexWrap: "wrap",
        }}
      >
        <h3
          style={{
            fontSize: "0.9rem",
            color: "var(--color-gold-light)",
            flex: 1,
          }}
        >
          {display_name}
        </h3>
        {available ? (
          <span
            style={{
              background: "rgba(16,185,129,0.2)",
              color: "#10B981",
              border: "1px solid rgba(16,185,129,0.3)",
              borderRadius: "var(--radius-full)",
              padding: "2px 8px",
              fontSize: "0.65rem",
            }}
          >
            Available
          </span>
        ) : (
          <span
            style={{
              background: "rgba(245,158,11,0.2)",
              color: "#F59E0B",
              border: "1px solid rgba(245,158,11,0.3)",
              borderRadius: "var(--radius-full)",
              padding: "2px 8px",
              fontSize: "0.65rem",
            }}
          >
            Coming Soon
          </span>
        )}
      </div>

      {/* Model ID chip */}
      <code
        style={{
          fontSize: "0.65rem",
          color: "var(--color-gray-400)",
          background: "rgba(0,0,0,0.3)",
          padding: "2px 8px",
          borderRadius: "var(--radius-sm)",
          alignSelf: "flex-start",
        }}
      >
        {model_id}
      </code>

      {/* Image area */}
      {isLoading ? (
        <div
          style={{
            aspectRatio: "1",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(255,255,255,0.03)",
            borderRadius: "var(--radius-lg)",
            flexDirection: "column",
            gap: "8px",
          }}
        >
          <div
            className="spinner"
            style={{ width: "32px", height: "32px", borderWidth: "3px" }}
          />
          <span style={{ fontSize: "0.75rem", color: "var(--color-gray-400)" }}>
            Running inference…
          </span>
        </div>
      ) : !available ? (
        <div
          style={{
            aspectRatio: "1",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(255,255,255,0.03)",
            borderRadius: "var(--radius-lg)",
            flexDirection: "column",
            gap: "8px",
            padding: "24px",
          }}
        >
          <span style={{ fontSize: "2rem" }}>🔬</span>
          <span
            style={{
              fontSize: "0.75rem",
              color: "var(--color-gray-400)",
              textAlign: "center",
            }}
          >
            Checkpoint not yet available.
            <br />
            Training in progress.
          </span>
        </div>
      ) : error ? (
        <div
          style={{
            aspectRatio: "1",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(239,68,68,0.05)",
            borderRadius: "var(--radius-lg)",
            border: "1px solid rgba(239,68,68,0.2)",
            flexDirection: "column",
            gap: "8px",
            padding: "16px",
          }}
        >
          <span style={{ fontSize: "1.5rem" }}>⚠️</span>
          <span
            style={{
              fontSize: "0.7rem",
              color: "var(--color-error)",
              textAlign: "center",
            }}
          >
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
          style={{
            width: "100%",
            borderRadius: "var(--radius-lg)",
            aspectRatio: "1",
            objectFit: "cover",
          }}
        />
      ) : null}

      {/* Checkpoint metadata */}
      {available && (epoch !== undefined || valLoss !== undefined) && (
        <div
          style={{
            fontSize: "0.7rem",
            color: "var(--color-gray-400)",
            display: "flex",
            gap: "12px",
            flexWrap: "wrap",
          }}
        >
          {epoch !== undefined && <span>Epoch {epoch}</span>}
          {valLoss !== undefined && valLoss !== null && (
            <span>Val Loss {Number(valLoss).toFixed(4)}</span>
          )}
        </div>
      )}

      {/* Inference metadata */}
      {available && restored_image_b64 && (
        <div
          style={{
            fontSize: "0.7rem",
            color: "var(--color-gray-400)",
            display: "flex",
            gap: "12px",
            flexWrap: "wrap",
            background: "rgba(0,0,0,0.2)",
            padding: "6px 10px",
            borderRadius: "var(--radius-md)",
          }}
        >
          {inference_time_s != null && (
            <span>Inference: {inference_time_s}s</span>
          )}
          {mask_coverage_pct != null && (
            <span>Mask: {mask_coverage_pct.toFixed(1)}%</span>
          )}
        </div>
      )}
      {checkpoint_path && (
        <div
          style={{
            fontSize: "0.6rem",
            color: "var(--color-gray-500)",
            wordBreak: "break-all",
            background: "rgba(0,0,0,0.15)",
            padding: "4px 8px",
            borderRadius: "var(--radius-sm)",
          }}
          title={checkpoint_path}
        >
          Checkpoint: {checkpoint_path.split("/").slice(-2).join("/")}
        </div>
      )}

      {/* Official eval metrics */}
      {official_eval_summary && (
        <div>
          <div
            style={{
              fontSize: "0.65rem",
              color: "var(--color-gold)",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              marginBottom: "8px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span>📊</span> Official Test-Set Metrics
            <span
              style={{
                marginLeft: "auto",
                color: "var(--color-gray-500)",
                textTransform: "none",
                letterSpacing: 0,
              }}
            >
              n={official_eval_summary.num_images || "?"}
            </span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {Object.entries(METRIC_LABELS).map(([key, cfg]) => (
              <MetricBadge
                key={key}
                value={official_eval_summary[key]}
                label={cfg.label}
                decimals={cfg.decimals}
              />
            ))}
          </div>
          <p
            style={{
              fontSize: "0.6rem",
              color: "var(--color-gray-500)",
              marginTop: "8px",
              fontStyle: "italic",
            }}
          >
            These are official test-set averages (n=305). Not computed for your
            uploaded image.
          </p>
        </div>
      )}

      {/* Download */}
      {restored_image_b64 && (
        <a
          href={restored_image_b64}
          download={`restored_${model_id}.png`}
          style={{
            display: "block",
            textAlign: "center",
            padding: "8px 16px",
            background: "rgba(212,175,55,0.15)",
            border: "1px solid rgba(212,175,55,0.3)",
            borderRadius: "var(--radius-md)",
            color: "var(--color-gold)",
            fontSize: "0.75rem",
            textDecoration: "none",
            cursor: "pointer",
          }}
        >
          ⬇ Download
        </a>
      )}
    </div>
  );
}
