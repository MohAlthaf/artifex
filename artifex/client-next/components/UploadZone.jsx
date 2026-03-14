"use client";

import { useState, useCallback } from "react";

/**
 * UploadZone — drag-and-drop upload area for the thesis demo.
 * Supports: damaged image (required), mask (optional), ground truth (optional).
 */
export default function UploadZone({ onSubmit, isLoading }) {
  const [image, setImage] = useState(null);
  const [mask, setMask] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [maskPreview, setMaskPreview] = useState(null);
  const [gtPreview, setGtPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImage(file);
    setImagePreview(URL.createObjectURL(file));
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      handleFile(file);
    },
    [handleFile]
  );

  const handleReset = () => {
    setImage(null);
    setMask(null);
    setGroundTruth(null);
    setImagePreview(null);
    setMaskPreview(null);
    setGtPreview(null);
  };

  const handleSubmit = () => {
    if (!image || isLoading) return;
    onSubmit({ image, mask, groundTruth, imagePreview });
  };

  // No image uploaded yet — show drop zone
  if (!imagePreview) {
    return (
      <div
        className={`upload-zone glass rounded-2xl p-12 text-center cursor-pointer ${isDragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setIsDragging(false);
        }}
        onDrop={handleDrop}
        onClick={() => document.getElementById("file-input").click()}
      >
        <input
          id="file-input"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-[var(--color-gold)] to-[var(--color-gold-dark)] rounded-xl flex items-center justify-center text-3xl">
          📤
        </div>
        <p className="text-lg text-gray-200 mb-2">
          Drop your damaged painting here
        </p>
        <p className="text-sm text-gray-400">PNG, JPG, WebP</p>
      </div>
    );
  }

  // Image uploaded — show preview + optional uploads + submit
  return (
    <div className="animate-fade-in">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Damaged image */}
        <div className="glass rounded-xl p-4">
          <p className="text-xs font-semibold text-[var(--color-gold)] uppercase tracking-wider mb-3">
            Damaged Image *
          </p>
          <img
            src={imagePreview}
            alt="Damaged"
            className="w-full aspect-square object-cover rounded-lg"
          />
        </div>

        {/* Mask */}
        <div className="glass rounded-xl p-4">
          <p className="text-xs font-semibold text-[var(--color-gold)] uppercase tracking-wider mb-3">
            Damage Mask
            <span className="text-gray-500 normal-case ml-1">(optional)</span>
          </p>
          {maskPreview ? (
            <div className="relative">
              <img
                src={maskPreview}
                alt="Mask"
                className="w-full aspect-square object-cover rounded-lg"
              />
              <button
                onClick={() => {
                  setMask(null);
                  setMaskPreview(null);
                }}
                className="absolute top-2 right-2 bg-black/60 hover:bg-black/80 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs"
              >
                ✕
              </button>
            </div>
          ) : (
            <div
              onClick={() => document.getElementById("mask-input").click()}
              className="w-full aspect-square rounded-lg border-2 border-dashed border-[rgba(212,175,55,0.3)] flex flex-col items-center justify-center cursor-pointer hover:border-[var(--color-gold)] transition-colors"
            >
              <input
                id="mask-input"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files[0];
                  if (f) {
                    setMask(f);
                    setMaskPreview(URL.createObjectURL(f));
                  }
                }}
              />
              <span className="text-2xl mb-2">🎭</span>
              <span className="text-xs text-gray-400 text-center px-4">
                Upload mask
                <br />
                White = damaged
              </span>
            </div>
          )}
        </div>

        {/* Ground Truth */}
        <div className="glass rounded-xl p-4">
          <p className="text-xs font-semibold text-[var(--color-gold)] uppercase tracking-wider mb-3">
            Ground Truth
            <span className="text-gray-500 normal-case ml-1">(optional)</span>
          </p>
          {gtPreview ? (
            <div className="relative">
              <img
                src={gtPreview}
                alt="Ground truth"
                className="w-full aspect-square object-cover rounded-lg"
              />
              <button
                onClick={() => {
                  setGroundTruth(null);
                  setGtPreview(null);
                }}
                className="absolute top-2 right-2 bg-black/60 hover:bg-black/80 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs"
              >
                ✕
              </button>
              <div className="absolute bottom-2 left-2 bg-[var(--color-success)]/20 text-[var(--color-success)] border border-[var(--color-success)]/30 rounded-full px-2 py-0.5 text-[10px]">
                Metrics will be computed
              </div>
            </div>
          ) : (
            <div
              onClick={() => document.getElementById("gt-input").click()}
              className="w-full aspect-square rounded-lg border-2 border-dashed border-[rgba(212,175,55,0.3)] flex flex-col items-center justify-center cursor-pointer hover:border-[var(--color-gold)] transition-colors"
            >
              <input
                id="gt-input"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files[0];
                  if (f) {
                    setGroundTruth(f);
                    setGtPreview(URL.createObjectURL(f));
                  }
                }}
              />
              <span className="text-2xl mb-2">🖼️</span>
              <span className="text-xs text-gray-400 text-center px-4">
                Upload clean original
                <br />
                to enable real metrics
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Info banner about metrics */}
      {!groundTruth && (
        <div className="glass rounded-xl p-4 mb-6 flex items-start gap-3 text-sm">
          <span className="text-lg shrink-0">ℹ️</span>
          <span className="text-gray-300">
            <strong className="text-[var(--color-gold)]">
              No ground truth provided
            </strong>{" "}
            — restoration will run, but per-upload metrics (PSNR, SSIM, etc.)
            will <em>not</em> be computed. This is honest: metrics without
            ground truth would be meaningless.
          </span>
        </div>
      )}

      {groundTruth && (
        <div className="glass rounded-xl p-4 mb-6 flex items-start gap-3 text-sm border border-[var(--color-success)]/20">
          <span className="text-lg shrink-0">✓</span>
          <span className="text-gray-300">
            <strong className="text-[var(--color-success)]">
              Ground truth provided
            </strong>{" "}
            — real per-upload PSNR, SSIM, L1, L2, Perceptual, and Style
            metrics will be computed for each model.
          </span>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex justify-center gap-4">
        <button
          className="btn-secondary rounded-lg px-6 py-3 font-semibold"
          onClick={handleReset}
        >
          Reset
        </button>
        <button
          className="btn-primary rounded-lg px-8 py-3 font-bold text-lg"
          onClick={handleSubmit}
          disabled={isLoading}
        >
          {isLoading ? "Running Inference…" : "✨ Restore Across All Models"}
        </button>
      </div>
    </div>
  );
}
