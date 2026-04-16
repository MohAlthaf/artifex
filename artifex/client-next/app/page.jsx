"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, CheckCircle2, Info, Palette } from "lucide-react";

import BenchmarkEvidence from "@/components/BenchmarkEvidence";
import ModelResultCard from "@/components/ModelResultCard";
import UploadZone from "@/components/UploadZone";
import { healthCheck, restoreWithEval } from "@/lib/api";

export default function ThesisDemoPage() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [hasGroundTruth, setHasGroundTruth] = useState(false);
  const [metricPolicy, setMetricPolicy] = useState("");
  const [error, setError] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  useEffect(() => {
    healthCheck()
      .then((data) => {
        const hasAvailableModels = data.available_models?.length > 0;
        setBackendStatus(hasAvailableModels ? "ready" : "no-models");
      })
      .catch(() => {
        setBackendStatus("offline");
      });
  }, []);

  const handleSubmit = async ({
    image,
    mask,
    groundTruth,
    imagePreview: preview,
  }) => {
    setIsLoading(true);
    setError(null);
    setResults(null);
    setImagePreview(preview);

    try {
      const data = await restoreWithEval(image, mask, groundTruth);

      setResults(data.results);
      setHasGroundTruth(data.has_ground_truth);
      setMetricPolicy(data.metric_policy);
    } catch (err) {
      setError(err.message || "Restoration failed.");
      console.error("ARTIFEX restoration failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
    setImagePreview(null);
    setHasGroundTruth(false);
    setMetricPolicy("");
  };

  return (
    <div className="min-h-screen">
      <header className="fixed top-0 left-0 right-0 z-50 py-3 bg-[var(--color-deep-navy)]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 bg-gradient-to-br from-[var(--color-gold)] to-[var(--color-gold-dark)] rounded-lg flex items-center justify-center">
              <Palette size={20} className="text-white" aria-hidden="true" />
            </div>

            <span className="font-serif text-xl font-bold text-white">
              ARTIFEX
            </span>
          </div>

          <span
            className={`text-[10px] px-2 py-0.5 rounded-full border ${
              backendStatus === "ready"
                ? "bg-[var(--color-success)]/15 text-[var(--color-success)] border-[var(--color-success)]/30"
                : backendStatus === "checking"
                  ? "bg-[var(--color-warning)]/15 text-[var(--color-warning)] border-[var(--color-warning)]/30"
                  : "bg-[var(--color-error)]/15 text-[var(--color-error)] border-[var(--color-error)]/30"
            }`}
          >
            {backendStatus === "ready"
              ? "Backend Ready"
              : backendStatus === "checking"
                ? "Connecting..."
                : "Backend Offline"}
          </span>
        </div>
      </header>

      <section className="hero-bg relative min-h-[50vh] flex items-center justify-center pt-20 pb-10 overflow-hidden">
        <div className="relative text-center max-w-3xl px-6">
          <span className="inline-block px-4 py-1 bg-[var(--color-gold)]/20 text-[var(--color-gold-light)] rounded-full text-sm font-medium mb-6 border border-[var(--color-gold)]/30 animate-fade-in">
            Thesis Research Prototype
          </span>

          <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6 text-gradient-gold animate-fade-in">
            Van Gogh Art Restoration
          </h1>

          <p className="text-lg text-gray-300 max-w-xl mx-auto animate-fade-in">
            Upload a damaged painting, run it through the official ARTIFEX
            thesis models, and view upload-specific metrics when ground truth is
            provided.
          </p>
        </div>
      </section>

      <section className="max-w-5xl mx-auto px-6 py-12">
        {!results && !isLoading && (
          <>
            <div className="text-center mb-8">
              <h2 className="font-serif text-2xl mb-2">Upload Your Image</h2>
              <p className="text-gray-400 text-sm">
                Damaged image is required. A mask and clean ground-truth image
                can be added for evaluation.
              </p>
            </div>

            <UploadZone onSubmit={handleSubmit} isLoading={isLoading} />
          </>
        )}

        {isLoading && (
          <div className="text-center py-16 animate-fade-in">
            <div className="spinner mx-auto mb-6" />
            <p className="text-gray-300 text-lg">
              Running inference through all available models...
            </p>
            <p className="text-gray-500 text-sm mt-2">
              Results will appear once the restoration process is complete.
            </p>
          </div>
        )}

        {error && (
          <div className="text-center py-8 animate-fade-in">
            <p className="text-[var(--color-error)] mb-4">{error}</p>
            <button
              className="btn-secondary rounded-lg px-6 py-2"
              onClick={handleReset}
            >
              Try Again
            </button>
          </div>
        )}

        {results && !isLoading && (
          <div className="animate-fade-in">
            <div className="text-center mb-6">
              <h3 className="font-serif text-xl text-[var(--color-gold)] mb-1">
                Restoration Results
              </h3>
              <p className="text-sm text-gray-400">
                Use the slider to compare the damaged and restored images for
                each model.
              </p>
            </div>

            <div
              className={`glass rounded-xl p-4 mb-8 flex items-start gap-3 text-sm ${
                hasGroundTruth
                  ? "border border-[var(--color-success)]/20"
                  : "border border-[var(--color-gold)]/20"
              }`}
            >
              {hasGroundTruth ? (
                <CheckCircle2
                  size={20}
                  className="shrink-0 text-[var(--color-success)]"
                  aria-hidden="true"
                />
              ) : (
                <Info
                  size={20}
                  className="shrink-0 text-[var(--color-gold)]"
                  aria-hidden="true"
                />
              )}

              <span className="text-gray-300">{metricPolicy}</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
              {results.map((result) => (
                <ModelResultCard
                  key={result.model_id}
                  result={result}
                  originalSrc={imagePreview}
                  hasGroundTruth={hasGroundTruth}
                />
              ))}
            </div>

            <div className="flex justify-center">
              <button
                className="btn-secondary rounded-lg px-8 py-3 font-semibold inline-flex items-center gap-2"
                onClick={handleReset}
              >
                <ArrowLeft size={18} aria-hidden="true" />
                Try Another Image
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="max-w-5xl mx-auto px-6 py-12 border-t border-white/5">
        <div className="text-center mb-8">
          <h2 className="font-serif text-2xl mb-2">
            Official Benchmark Evidence
          </h2>
          <p className="text-gray-400 text-sm">
            Results from the 305-image Van Gogh test set with ground truth,
            using metrics from saved ARTIFEX evaluation JSON files.
          </p>
        </div>

        <BenchmarkEvidence />
      </section>

      <footer className="py-8 text-center border-t border-white/5">
        <p className="text-gray-500 text-xs">
          ARTIFEX | Van Gogh Art Restoration | Thesis Research Prototype | v3.0
        </p>
      </footer>
    </div>
  );
}