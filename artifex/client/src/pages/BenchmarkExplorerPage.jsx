/**
 * BenchmarkExplorerPage.jsx
 * ==========================
 * Benchmark Explorer — inspect official test-set results with ground truth.
 *
 * This page shows REAL evaluation evidence:
 *   - Ground truth images from the official test split
 *   - Pre-computed restored images (where available, e.g. full_official)
 *   - Per-image metrics from saved evaluation JSONs
 *
 * Metric honesty: all values come from saved benchmark evaluation files.
 * No on-the-fly metric computation is done here.
 */

import { useState, useEffect } from "react";

const API_URL = "http://localhost:3001/api";

const METRIC_DEFS = [
  { key: "psnr", label: "PSNR (dB)", better: "higher", decimals: 2 },
  { key: "ssim", label: "SSIM", better: "higher", decimals: 4 },
  { key: "l1", label: "L1", better: "lower", decimals: 4 },
  { key: "direction", label: "Direction", better: "lower", decimals: 4 },
  {
    key: "edge_strength",
    label: "Edge Strength",
    better: "lower",
    decimals: 4,
  },
  { key: "histogram", label: "Histogram χ²", better: "lower", decimals: 4 },
  { key: "perceptual", label: "Perceptual", better: "lower", decimals: 4 },
];

// ============================================================
// Sub-component: aggregate comparison summary
// ============================================================
function ComparisonSummary({ comparison }) {
  if (!comparison) return null;
  const { wins, verdict, comparison: rows } = comparison;
  if (!rows) return null;
  return (
    <div
      className="glass"
      style={{
        borderRadius: "var(--radius-xl)",
        padding: "var(--spacing-lg)",
        marginBottom: "var(--spacing-xl)",
      }}
    >
      <h3
        style={{
          color: "var(--color-gold)",
          marginBottom: "var(--spacing-md)",
          fontSize: "1rem",
        }}
      >
        📊 Official Model Comparison (Baseline vs Full)
      </h3>
      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "0.8rem",
          }}
        >
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(212,175,55,0.2)" }}>
              {["Metric", "Baseline", "Full", "Δ", "Winner"].map((h) => (
                <th
                  key={h}
                  style={{
                    padding: "8px 12px",
                    textAlign: "center",
                    color: "var(--color-gray-300)",
                    fontWeight: "600",
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.metric}
                style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}
              >
                <td
                  style={{
                    padding: "6px 12px",
                    color: "var(--color-gray-300)",
                  }}
                >
                  {row.metric}
                </td>
                <td
                  style={{
                    padding: "6px 12px",
                    textAlign: "center",
                    color: "var(--color-gray-200)",
                  }}
                >
                  {row.baseline !== null
                    ? Number(row.baseline).toFixed(4)
                    : "N/A"}
                </td>
                <td
                  style={{
                    padding: "6px 12px",
                    textAlign: "center",
                    color: "var(--color-gray-200)",
                  }}
                >
                  {row.full !== null ? Number(row.full).toFixed(4) : "N/A"}
                </td>
                <td
                  style={{
                    padding: "6px 12px",
                    textAlign: "center",
                    color:
                      row.delta < 0
                        ? "#10B981"
                        : row.delta > 0
                          ? "#EF4444"
                          : "var(--color-gray-400)",
                  }}
                >
                  {row.delta !== null
                    ? (row.delta >= 0 ? "+" : "") + Number(row.delta).toFixed(4)
                    : "N/A"}
                </td>
                <td style={{ padding: "6px 12px", textAlign: "center" }}>
                  <span
                    style={{
                      background:
                        row.winner === "full"
                          ? "rgba(16,185,129,0.15)"
                          : row.winner === "baseline"
                            ? "rgba(239,68,68,0.1)"
                            : "transparent",
                      color:
                        row.winner === "full"
                          ? "#10B981"
                          : row.winner === "baseline"
                            ? "#EF4444"
                            : "var(--color-gray-500)",
                      padding: "2px 8px",
                      borderRadius: "4px",
                      fontSize: "0.7rem",
                    }}
                  >
                    {row.winner || "N/A"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div
        style={{
          marginTop: "var(--spacing-md)",
          fontSize: "0.75rem",
          color: "var(--color-gray-400)",
          display: "flex",
          gap: "16px",
          flexWrap: "wrap",
        }}
      >
        <span>
          Baseline wins: <strong>{wins?.baseline ?? "?"}</strong>
        </span>
        <span>
          Full wins:{" "}
          <strong style={{ color: "#10B981" }}>{wins?.full ?? "?"}</strong>
        </span>
        <span>
          Verdict:{" "}
          <strong style={{ color: "var(--color-gold)" }}>
            {verdict?.replace(/_/g, " ")}
          </strong>
        </span>
      </div>
    </div>
  );
}

// ============================================================
// Sub-component: per-image sample viewer
// ============================================================
function SampleViewer({ sampleId, benchmarkModels }) {
  const [sampleData, setSampleData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!sampleId) return;
    setLoading(true);
    setError(null);
    fetch(`${API_URL}/benchmark/sample/${encodeURIComponent(sampleId)}`)
      .then((r) => r.json())
      .then((d) => {
        setSampleData(d);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [sampleId]);

  if (!sampleId) return null;
  if (loading)
    return (
      <div
        style={{
          textAlign: "center",
          padding: "48px",
          color: "var(--color-gray-400)",
        }}
      >
        <div className="spinner" style={{ margin: "0 auto 16px" }} />
        Loading sample…
      </div>
    );
  if (error)
    return (
      <p style={{ color: "var(--color-error)", padding: "16px" }}>
        Error: {error}
      </p>
    );
  if (!sampleData) return null;

  const {
    damaged_b64,
    mask_b64,
    ground_truth_b64,
    restored,
    per_image_metrics,
  } = sampleData;

  return (
    <div>
      {/* Ground images row */}
      <h4
        style={{
          color: "var(--color-gold-light)",
          marginBottom: "var(--spacing-md)",
          fontSize: "0.9rem",
        }}
      >
        {sampleId}
      </h4>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "var(--spacing-md)",
          marginBottom: "var(--spacing-lg)",
        }}
      >
        {[
          { src: damaged_b64, label: "Damaged Input" },
          { src: mask_b64, label: "Damage Mask" },
          { src: ground_truth_b64, label: "Ground Truth" },
        ].map(({ src, label }) => (
          <div key={label} style={{ flex: "1 1 160px", minWidth: "140px" }}>
            <p
              style={{
                fontSize: "0.7rem",
                color: "var(--color-gray-400)",
                marginBottom: "6px",
                textAlign: "center",
              }}
            >
              {label}
            </p>
            {src ? (
              <img
                src={src}
                alt={label}
                style={{
                  width: "100%",
                  borderRadius: "var(--radius-md)",
                  aspectRatio: "1",
                  objectFit: "cover",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              />
            ) : (
              <div
                style={{
                  width: "100%",
                  aspectRatio: "1",
                  background: "rgba(255,255,255,0.03)",
                  borderRadius: "var(--radius-md)",
                  border: "1px dashed rgba(255,255,255,0.1)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "0.7rem",
                  color: "var(--color-gray-500)",
                }}
              >
                N/A
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Model outputs + metrics */}
      <div
        style={{ display: "flex", flexWrap: "wrap", gap: "var(--spacing-md)" }}
      >
        {Object.entries(benchmarkModels || {}).map(([modelId, modelInfo]) => {
          const restoredSrc = restored?.[modelId] || null;
          const imgMetrics = per_image_metrics?.[modelId] || null;
          return (
            <div
              key={modelId}
              className="glass"
              style={{
                flex: "1 1 200px",
                minWidth: "180px",
                borderRadius: "var(--radius-lg)",
                padding: "var(--spacing-md)",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                border: modelInfo.available
                  ? "1px solid rgba(212,175,55,0.15)"
                  : "1px solid rgba(255,255,255,0.05)",
                opacity: modelInfo.available ? 1 : 0.5,
              }}
            >
              <p
                style={{
                  fontSize: "0.75rem",
                  color: "var(--color-gold-light)",
                  fontWeight: "600",
                }}
              >
                {modelInfo.display_name}
              </p>
              {restoredSrc ? (
                <img
                  src={restoredSrc}
                  alt={`Restored by ${modelId}`}
                  style={{
                    width: "100%",
                    aspectRatio: "1",
                    objectFit: "cover",
                    borderRadius: "var(--radius-md)",
                    border: "1px solid rgba(212,175,55,0.1)",
                  }}
                />
              ) : (
                <div
                  style={{
                    width: "100%",
                    aspectRatio: "1",
                    background: "rgba(255,255,255,0.03)",
                    borderRadius: "var(--radius-md)",
                    border: "1px dashed rgba(255,255,255,0.1)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexDirection: "column",
                    gap: "4px",
                    padding: "8px",
                  }}
                >
                  <span style={{ fontSize: "1.2rem" }}>
                    {modelInfo.available ? "🖼" : "🔬"}
                  </span>
                  <span
                    style={{
                      fontSize: "0.65rem",
                      color: "var(--color-gray-500)",
                      textAlign: "center",
                    }}
                  >
                    {modelInfo.available
                      ? "No pre-computed image"
                      : "Not yet available"}
                  </span>
                </div>
              )}
              {/* Per-image metrics */}
              {imgMetrics && (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "3px",
                  }}
                >
                  {METRIC_DEFS.filter(
                    (d) =>
                      imgMetrics[d.key] !== undefined &&
                      imgMetrics[d.key] !== null,
                  ).map((d) => (
                    <div
                      key={d.key}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "0.65rem",
                      }}
                    >
                      <span style={{ color: "var(--color-gray-400)" }}>
                        {d.label}
                      </span>
                      <span
                        style={{
                          color: "var(--color-gray-100)",
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        {Number(imgMetrics[d.key]).toFixed(d.decimals)}
                      </span>
                    </div>
                  ))}
                  <p
                    style={{
                      fontSize: "0.55rem",
                      color: "var(--color-gray-600)",
                      marginTop: "2px",
                      fontStyle: "italic",
                    }}
                  >
                    Per-image benchmark metrics (vs ground truth)
                  </p>
                </div>
              )}
              {!imgMetrics && modelInfo.available && (
                <p
                  style={{
                    fontSize: "0.65rem",
                    color: "var(--color-gray-500)",
                    fontStyle: "italic",
                  }}
                >
                  No per-image metrics for this model
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
// Main page
// ============================================================
export default function BenchmarkExplorerPage() {
  const [benchmarkModels, setBenchmarkModels] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [samples, setSamples] = useState([]);
  const [totalSamples, setTotalSamples] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [selectedSample, setSelectedSample] = useState(null);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const PER_PAGE = 20;

  // Load models + comparison once
  useEffect(() => {
    setLoadingMeta(true);
    Promise.all([
      fetch(`${API_URL}/benchmark/models`)
        .then((r) => r.json())
        .catch(() => null),
      fetch(`${API_URL}/benchmark/comparison`)
        .then((r) => r.json())
        .catch(() => null),
    ]).then(([modelsData, cmpData]) => {
      setBenchmarkModels(modelsData);
      setComparison(cmpData);
      setLoadingMeta(false);
    });
  }, []);

  // Load sample list
  useEffect(() => {
    fetch(`${API_URL}/benchmark/samples?page=${page}&per_page=${PER_PAGE}`)
      .then((r) => r.json())
      .then((d) => {
        setSamples(d.samples || []);
        setTotalSamples(d.total || 0);
        setTotalPages(d.pages || 1);
      })
      .catch(() => {});
  }, [page]);

  const filteredSamples = searchQuery
    ? samples.filter((s) => s.includes(searchQuery.replace(/[^0-9]/g, "")))
    : samples;

  if (loadingMeta)
    return (
      <div
        style={{
          textAlign: "center",
          padding: "80px",
          color: "var(--color-gray-400)",
        }}
      >
        <div className="spinner" style={{ margin: "0 auto 16px" }} />
        Loading benchmark data…
      </div>
    );

  return (
    <div>
      {/* Header */}
      <section
        style={{
          background: "rgba(0,0,0,0.2)",
          padding: "var(--spacing-2xl) 0",
          textAlign: "center",
          marginBottom: "var(--spacing-xl)",
        }}
      >
        <div className="container">
          <span className="hero-badge">📊 Official Evaluation Evidence</span>
          <h1
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: "2rem",
              marginTop: "12px",
              marginBottom: "12px",
              color: "var(--color-wheat)",
            }}
          >
            Benchmark Explorer
          </h1>
          <p
            style={{
              color: "var(--color-gray-300)",
              maxWidth: "600px",
              margin: "0 auto",
              fontSize: "0.95rem",
            }}
          >
            Browse the official 305-image Van Gogh test set. View damaged
            inputs, ground truth, model outputs, and real per-image evaluation
            metrics from saved benchmark runs.
          </p>
          {totalSamples > 0 && (
            <p
              style={{
                color: "var(--color-gold)",
                marginTop: "12px",
                fontSize: "0.85rem",
              }}
            >
              {totalSamples} test images ·{" "}
              {
                Object.values(benchmarkModels || {}).filter((m) => m.has_eval)
                  .length
              }{" "}
              models evaluated
            </p>
          )}
        </div>
      </section>

      <div className="container">
        {/* Comparison summary */}
        <ComparisonSummary comparison={comparison} />

        {/* Model availability cards */}
        {benchmarkModels && (
          <div style={{ marginBottom: "var(--spacing-xl)" }}>
            <h3
              style={{
                color: "var(--color-gold-light)",
                marginBottom: "var(--spacing-md)",
                fontSize: "0.9rem",
              }}
            >
              Model Status
            </h3>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "var(--spacing-sm)",
              }}
            >
              {Object.entries(benchmarkModels)
                .sort((a, b) => (a[1].order || 99) - (b[1].order || 99))
                .map(([modelId, info]) => (
                  <div
                    key={modelId}
                    style={{
                      background: info.has_eval
                        ? "rgba(16,185,129,0.1)"
                        : "rgba(255,255,255,0.03)",
                      border: `1px solid ${info.has_eval ? "rgba(16,185,129,0.3)" : "rgba(255,255,255,0.08)"}`,
                      borderRadius: "var(--radius-lg)",
                      padding: "10px 16px",
                      display: "flex",
                      flexDirection: "column",
                      gap: "4px",
                      minWidth: "160px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.8rem",
                        color: "var(--color-gray-200)",
                        fontWeight: "600",
                      }}
                    >
                      {info.display_name}
                    </span>
                    <div
                      style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}
                    >
                      {info.has_eval ? (
                        <span style={{ fontSize: "0.65rem", color: "#10B981" }}>
                          ✓ Eval available
                        </span>
                      ) : (
                        <span
                          style={{
                            fontSize: "0.65rem",
                            color: "var(--color-gray-500)",
                          }}
                        >
                          Training pending
                        </span>
                      )}
                      {info.has_restored_images && (
                        <span style={{ fontSize: "0.65rem", color: "#60A5FA" }}>
                          · Pre-computed images
                        </span>
                      )}
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Sample browser */}
        <div
          style={{
            display: "flex",
            gap: "var(--spacing-xl)",
            alignItems: "flex-start",
            flexWrap: "wrap",
          }}
        >
          {/* Sample list */}
          <div style={{ flex: "0 0 220px", minWidth: "200px" }}>
            <div style={{ marginBottom: "var(--spacing-md)" }}>
              <input
                type="text"
                placeholder="Filter by image #…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: "100%",
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "var(--radius-md)",
                  padding: "8px 12px",
                  color: "var(--color-gray-100)",
                  fontSize: "0.8rem",
                  outline: "none",
                }}
              />
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                maxHeight: "600px",
                overflowY: "auto",
              }}
            >
              {filteredSamples.map((s) => (
                <button
                  key={s}
                  onClick={() => setSelectedSample(s)}
                  style={{
                    background:
                      selectedSample === s
                        ? "rgba(212,175,55,0.2)"
                        : "rgba(255,255,255,0.03)",
                    border: `1px solid ${selectedSample === s ? "rgba(212,175,55,0.4)" : "rgba(255,255,255,0.08)"}`,
                    borderRadius: "var(--radius-md)",
                    padding: "8px 12px",
                    textAlign: "left",
                    cursor: "pointer",
                    color:
                      selectedSample === s
                        ? "var(--color-gold)"
                        : "var(--color-gray-300)",
                    fontSize: "0.72rem",
                    fontFamily: "monospace",
                    transition: "all 0.15s",
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
            {/* Pagination */}
            {totalPages > 1 && (
              <div
                style={{
                  display: "flex",
                  gap: "8px",
                  marginTop: "12px",
                  justifyContent: "center",
                }}
              >
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((p) => p - 1)}
                  className="btn btn-secondary"
                  style={{ padding: "6px 12px", fontSize: "0.75rem" }}
                >
                  ←
                </button>
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--color-gray-400)",
                    alignSelf: "center",
                  }}
                >
                  {page} / {totalPages}
                </span>
                <button
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => p + 1)}
                  className="btn btn-secondary"
                  style={{ padding: "6px 12px", fontSize: "0.75rem" }}
                >
                  →
                </button>
              </div>
            )}
          </div>

          {/* Sample viewer */}
          <div style={{ flex: "1 1 400px", minWidth: "300px" }}>
            {selectedSample ? (
              <div
                className="glass"
                style={{
                  borderRadius: "var(--radius-xl)",
                  padding: "var(--spacing-lg)",
                }}
              >
                <SampleViewer
                  sampleId={selectedSample}
                  benchmarkModels={benchmarkModels}
                />
              </div>
            ) : (
              <div
                className="glass"
                style={{
                  borderRadius: "var(--radius-xl)",
                  padding: "var(--spacing-3xl)",
                  textAlign: "center",
                  color: "var(--color-gray-500)",
                }}
              >
                <span
                  style={{
                    fontSize: "3rem",
                    display: "block",
                    marginBottom: "16px",
                  }}
                >
                  🖼
                </span>
                Select a test image from the list to view benchmark results.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
