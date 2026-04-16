"use client";

import { useEffect, useState } from "react";
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Search,
} from "lucide-react";

import {
  getBenchmarkComparison,
  getBenchmarkModels,
  getBenchmarkSample,
  getBenchmarkSamples,
} from "@/lib/api";

const METRIC_DEFS = [
  { key: "psnr", label: "PSNR (dB)", better: "higher", decimals: 2 },
  { key: "ssim", label: "SSIM", better: "higher", decimals: 4 },
  { key: "lpips", label: "LPIPS", better: "lower", decimals: 4 },
  { key: "l1", label: "L1", better: "lower", decimals: 4 },
  { key: "direction", label: "Direction", better: "lower", decimals: 4 },
  { key: "edge_strength", label: "Edge Strength", better: "lower", decimals: 4 },
  { key: "histogram", label: "Histogram", better: "lower", decimals: 4 },
  { key: "perceptual", label: "Perceptual", better: "lower", decimals: 4 },
];

export default function BenchmarkEvidence() {
  const [comparison, setComparison] = useState(null);
  const [benchmarkModels, setBenchmarkModels] = useState({});
  const [samples, setSamples] = useState([]);
  const [totalSamples, setTotalSamples] = useState(0);
  const [page, setPage] = useState(1);
  const [selectedSample, setSelectedSample] = useState(null);
  const [sampleData, setSampleData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sampleLoading, setSampleLoading] = useState(false);

  useEffect(() => {
    async function loadBenchmarkData() {
      try {
        const [comparisonData, modelsData, samplePage] = await Promise.all([
          getBenchmarkComparison(),
          getBenchmarkModels(),
          getBenchmarkSamples(1, 8),
        ]);

        setComparison(comparisonData);
        setBenchmarkModels(modelsData);
        setSamples(samplePage.samples || []);
        setTotalSamples(samplePage.total || 0);
      } catch (error) {
        console.error("ARTIFEX benchmark data could not be loaded:", error);
      } finally {
        setLoading(false);
      }
    }

    loadBenchmarkData();
  }, []);

  useEffect(() => {
    if (!selectedSample) {
      setSampleData(null);
      return;
    }

    setSampleLoading(true);

    getBenchmarkSample(selectedSample)
      .then(setSampleData)
      .catch((error) => {
        console.error("ARTIFEX sample data could not be loaded:", error);
      })
      .finally(() => {
        setSampleLoading(false);
      });
  }, [selectedSample]);

  const loadPage = async (nextPage) => {
    setPage(nextPage);

    try {
      const response = await getBenchmarkSamples(nextPage, 8);
      setSamples(response.samples || []);
    } catch (error) {
      console.error("ARTIFEX benchmark samples could not be loaded:", error);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12 text-gray-400">
        <div className="spinner mx-auto mb-4" />
        Loading benchmark data...
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {comparison?.comparison && (
        <div className="glass rounded-2xl p-6">
          <h3 className="text-[var(--color-gold)] font-serif text-lg mb-4 flex items-center gap-2">
            <BarChart3 size={18} aria-hidden="true" />
            Official Model Comparison | Baseline vs Full
          </h3>

          <div className="overflow-x-auto">
            <table className="comparison-table w-full text-xs">
              <thead>
                <tr>
                  {["Metric", "Baseline", "Full", "Delta", "Winner"].map(
                    (heading) => (
                      <th key={heading}>{heading}</th>
                    ),
                  )}
                </tr>
              </thead>

              <tbody>
                {comparison.comparison.map((row) => (
                  <tr key={row.metric}>
                    <td className="text-left text-gray-300">{row.metric}</td>
                    <td>
                      {row.baseline != null
                        ? Number(row.baseline).toFixed(4)
                        : "N/A"}
                    </td>
                    <td>
                      {row.full != null
                        ? Number(row.full).toFixed(4)
                        : "N/A"}
                    </td>
                    <td
                      className={
                        row.delta < 0
                          ? "text-[var(--color-success)]"
                          : row.delta > 0
                            ? "text-[var(--color-error)]"
                            : "text-gray-400"
                      }
                    >
                      {row.delta != null
                        ? `${row.delta >= 0 ? "+" : ""}${Number(row.delta).toFixed(4)}`
                        : "N/A"}
                    </td>
                    <td>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] ${
                          row.winner === "full"
                            ? "bg-[var(--color-success)]/15 text-[var(--color-success)]"
                            : row.winner === "baseline"
                              ? "bg-[var(--color-error)]/10 text-[var(--color-error)]"
                              : "text-gray-500"
                        }`}
                      >
                        {row.winner || "N/A"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {comparison.wins && (
            <div className="mt-4 flex gap-4 text-[11px] text-gray-400 flex-wrap">
              <span>
                Baseline wins:{" "}
                <strong>{comparison.wins.baseline ?? "?"}</strong>
              </span>
              <span>
                Full wins:{" "}
                <strong className="text-[var(--color-success)]">
                  {comparison.wins.full ?? "?"}
                </strong>
              </span>
              <span>
                Verdict:{" "}
                <strong className="text-[var(--color-gold)]">
                  {comparison.verdict?.replace(/_/g, " ")}
                </strong>
              </span>
            </div>
          )}
        </div>
      )}

      <div className="glass rounded-2xl p-6">
        <h3 className="text-[var(--color-gold)] font-serif text-lg mb-1 flex items-center gap-2">
          <Search size={18} aria-hidden="true" />
          Benchmark Sample Browser
        </h3>

        <p className="text-xs text-gray-400 mb-4">
          {totalSamples} test images with ground truth. Select a sample to
          inspect its outputs and per-image metrics.
        </p>

        <div className="grid grid-cols-4 md:grid-cols-8 gap-2 mb-4">
          {samples.map((sampleId) => (
            <button
              key={sampleId}
              onClick={() =>
                setSelectedSample(
                  selectedSample === sampleId ? null : sampleId,
                )
              }
              className={`text-[9px] py-2 px-1 rounded-lg text-center transition-all truncate ${
                selectedSample === sampleId
                  ? "bg-[var(--color-gold)]/20 text-[var(--color-gold)] border border-[var(--color-gold)]/40"
                  : "bg-white/5 text-gray-400 hover:bg-white/10 border border-transparent"
              }`}
            >
              {sampleId.replace(".png", "")}
            </button>
          ))}
        </div>

        {totalSamples > 8 && (
          <div className="flex justify-center gap-2 mb-6">
            <button
              disabled={page <= 1}
              onClick={() => loadPage(page - 1)}
              className="btn-secondary rounded px-3 py-1 text-xs disabled:opacity-40 inline-flex items-center gap-1"
            >
              <ChevronLeft size={14} aria-hidden="true" />
              Previous
            </button>

            <span className="text-xs text-gray-400 self-center">
              Page {page} of {Math.ceil(totalSamples / 8)}
            </span>

            <button
              disabled={page >= Math.ceil(totalSamples / 8)}
              onClick={() => loadPage(page + 1)}
              className="btn-secondary rounded px-3 py-1 text-xs disabled:opacity-40 inline-flex items-center gap-1"
            >
              Next
              <ChevronRight size={14} aria-hidden="true" />
            </button>
          </div>
        )}

        {selectedSample && sampleLoading && (
          <div className="text-center py-8 text-gray-400">
            <div className="spinner mx-auto mb-3 w-8 h-8" />
            Loading sample...
          </div>
        )}

        {sampleData && !sampleLoading && (
          <div className="glass-dark rounded-xl p-5 space-y-4">
            <h4 className="text-sm font-semibold text-[var(--color-gold-light)]">
              {sampleData.sample_id}
            </h4>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {sampleData.damaged_b64 && (
                <div>
                  <p className="text-[9px] text-gray-400 mb-1 text-center">
                    Damaged
                  </p>
                  <img
                    src={sampleData.damaged_b64}
                    alt="Damaged artwork sample"
                    className="w-full aspect-square object-cover rounded-lg"
                  />
                </div>
              )}

              {sampleData.mask_b64 && (
                <div>
                  <p className="text-[9px] text-gray-400 mb-1 text-center">
                    Mask
                  </p>
                  <img
                    src={sampleData.mask_b64}
                    alt="Damage mask sample"
                    className="w-full aspect-square object-cover rounded-lg"
                  />
                </div>
              )}

              {sampleData.ground_truth_b64 && (
                <div>
                  <p className="text-[9px] text-[var(--color-success)] mb-1 text-center">
                    Ground Truth
                  </p>
                  <img
                    src={sampleData.ground_truth_b64}
                    alt="Ground-truth artwork sample"
                    className="w-full aspect-square object-cover rounded-lg"
                  />
                </div>
              )}

              {Object.entries(sampleData.restored || {}).map(
                ([modelId, imageData]) => (
                  <div key={modelId}>
                    <p className="text-[9px] text-[var(--color-gold)] mb-1 text-center">
                      {modelId.replace(/_/g, " ")}
                    </p>
                    <img
                      src={imageData}
                      alt={`Restored output from ${modelId}`}
                      className="w-full aspect-square object-cover rounded-lg"
                    />
                  </div>
                ),
              )}
            </div>

            {Object.entries(sampleData.per_image_metrics || {}).length > 0 && (
              <div>
                <p className="text-[10px] text-[var(--color-gold)] mb-2 uppercase tracking-wider">
                  Per-Image Metrics
                </p>

                {Object.entries(sampleData.per_image_metrics).map(
                  ([modelId, metrics]) => (
                    <div key={modelId} className="mb-2">
                      <p className="text-[9px] text-gray-400 mb-1">
                        {modelId.replace(/_/g, " ")}:
                      </p>

                      <div className="flex flex-wrap gap-1">
                        {METRIC_DEFS.map(({ key, label, decimals }) =>
                          metrics[key] != null ? (
                            <span
                              key={key}
                              className="bg-white/5 px-2 py-0.5 rounded text-[9px] text-gray-300"
                            >
                              {label}: {Number(metrics[key]).toFixed(decimals)}
                            </span>
                          ) : null,
                        )}
                      </div>
                    </div>
                  ),
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}