"use client";

import { useCallback, useRef, useState } from "react";
import {
  CheckCircle2,
  ImageIcon,
  Info,
  Layers,
  UploadCloud,
  Wand2,
  X,
} from "lucide-react";

export default function UploadZone({ onSubmit, isLoading }) {
  const [image, setImage] = useState(null);
  const [mask, setMask] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [maskPreview, setMaskPreview] = useState(null);
  const [gtPreview, setGtPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const imageInputRef = useRef(null);
  const maskInputRef = useRef(null);
  const groundTruthInputRef = useRef(null);

  const createPreview = (file, setFile, setPreview) => {
    if (!file || !file.type.startsWith("image/")) {
      return;
    }

    setFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleFile = useCallback((file) => {
    createPreview(file, setImage, setImagePreview);
  }, []);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setIsDragging(false);

      const file = event.dataTransfer.files[0];
      handleFile(file);
    },
    [handleFile],
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
    if (!image || isLoading) {
      return;
    }

    onSubmit({ image, mask, groundTruth, imagePreview });
  };

  if (!imagePreview) {
    return (
      <div
        className={`upload-zone glass rounded-2xl p-12 text-center cursor-pointer ${
          isDragging ? "dragging" : ""
        }`}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={handleDrop}
        onClick={() => imageInputRef.current?.click()}
      >
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => handleFile(event.target.files[0])}
        />

        <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-[var(--color-gold)] to-[var(--color-gold-dark)] rounded-xl flex items-center justify-center">
          <UploadCloud size={34} className="text-white" aria-hidden="true" />
        </div>

        <p className="text-lg text-gray-200 mb-2">
          Drop your damaged painting here
        </p>
        <p className="text-sm text-gray-400">PNG, JPG, or WebP</p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass rounded-xl p-4">
          <p className="text-xs font-semibold text-[var(--color-gold)] uppercase tracking-wider mb-3">
            Damaged Image *
          </p>
          <img
            src={imagePreview}
            alt="Damaged artwork"
            className="w-full aspect-square object-cover rounded-lg"
          />
        </div>

        <div className="glass rounded-xl p-4">
          <p className="text-xs font-semibold text-[var(--color-gold)] uppercase tracking-wider mb-3">
            Damage Mask
            <span className="text-gray-500 normal-case ml-1">(optional)</span>
          </p>

          {maskPreview ? (
            <div className="relative">
              <img
                src={maskPreview}
                alt="Damage mask"
                className="w-full aspect-square object-cover rounded-lg"
              />
              <button
                onClick={() => {
                  setMask(null);
                  setMaskPreview(null);
                }}
                className="absolute top-2 right-2 bg-black/60 hover:bg-black/80 text-white rounded-full w-6 h-6 flex items-center justify-center"
                aria-label="Remove damage mask"
              >
                <X size={14} aria-hidden="true" />
              </button>
            </div>
          ) : (
            <div
              onClick={() => maskInputRef.current?.click()}
              className="w-full aspect-square rounded-lg border-2 border-dashed border-[rgba(212,175,55,0.3)] flex flex-col items-center justify-center cursor-pointer hover:border-[var(--color-gold)] transition-colors"
            >
              <input
                ref={maskInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) =>
                  createPreview(event.target.files[0], setMask, setMaskPreview)
                }
              />

              <Layers
                size={28}
                className="text-gray-400 mb-2"
                aria-hidden="true"
              />
              <span className="text-xs text-gray-400 text-center px-4">
                Upload mask
                <br />
                White = damaged
              </span>
            </div>
          )}
        </div>

        <div className="glass rounded-xl p-4">
          <p className="text-xs font-semibold text-[var(--color-gold)] uppercase tracking-wider mb-3">
            Ground Truth
            <span className="text-gray-500 normal-case ml-1">(optional)</span>
          </p>

          {gtPreview ? (
            <div className="relative">
              <img
                src={gtPreview}
                alt="Ground-truth artwork"
                className="w-full aspect-square object-cover rounded-lg"
              />
              <button
                onClick={() => {
                  setGroundTruth(null);
                  setGtPreview(null);
                }}
                className="absolute top-2 right-2 bg-black/60 hover:bg-black/80 text-white rounded-full w-6 h-6 flex items-center justify-center"
                aria-label="Remove ground truth"
              >
                <X size={14} aria-hidden="true" />
              </button>

              <div className="absolute bottom-2 left-2 bg-[var(--color-success)]/20 text-[var(--color-success)] border border-[var(--color-success)]/30 rounded-full px-2 py-0.5 text-[10px]">
                Metrics will be computed
              </div>
            </div>
          ) : (
            <div
              onClick={() => groundTruthInputRef.current?.click()}
              className="w-full aspect-square rounded-lg border-2 border-dashed border-[rgba(212,175,55,0.3)] flex flex-col items-center justify-center cursor-pointer hover:border-[var(--color-gold)] transition-colors"
            >
              <input
                ref={groundTruthInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) =>
                  createPreview(
                    event.target.files[0],
                    setGroundTruth,
                    setGtPreview,
                  )
                }
              />

              <ImageIcon
                size={28}
                className="text-gray-400 mb-2"
                aria-hidden="true"
              />
              <span className="text-xs text-gray-400 text-center px-4">
                Upload clean original
                <br />
                to enable real metrics
              </span>
            </div>
          )}
        </div>
      </div>

      {!groundTruth && (
        <div className="glass rounded-xl p-4 mb-6 flex items-start gap-3 text-sm">
          <Info
            size={20}
            className="shrink-0 text-[var(--color-gold)]"
            aria-hidden="true"
          />
          {/* <span className="text-gray-300">
            <strong className="text-[var(--color-gold)]">
              No ground truth provided
            </strong>{" "}
            — restoration will run, but per-upload metrics such as PSNR and
            SSIM will not be computed because they require a clean reference
            image.
          </span> */}
          <span className="text-gray-300">
            <strong className="text-[var(--color-success)]">
              Ground truth provided
            </strong>{" "}
            — real per-upload PSNR, SSIM, L1, L2, perceptual, and style metrics
            will be computed for each model. If a mask is also provided, live
            brushstroke metrics for direction, edge strength, and histogram will
            be computed as well.
          </span>
        </div>
      )}

      {groundTruth && (
        <div className="glass rounded-xl p-4 mb-6 flex items-start gap-3 text-sm border border-[var(--color-success)]/20">
          <CheckCircle2
            size={20}
            className="shrink-0 text-[var(--color-success)]"
            aria-hidden="true"
          />
          <span className="text-gray-300">
            <strong className="text-[var(--color-success)]">
              Ground truth provided
            </strong>{" "}
            — real per-upload PSNR, SSIM, L1, L2, perceptual, and style metrics
            will be computed for each model.
          </span>
        </div>
      )}

      <div className="flex justify-center gap-4">
        <button
          className="btn-secondary rounded-lg px-6 py-3 font-semibold"
          onClick={handleReset}
        >
          Reset
        </button>

        <button
          className="btn-primary rounded-lg px-8 py-3 font-bold text-lg inline-flex items-center gap-2"
          onClick={handleSubmit}
          disabled={isLoading}
        >
          <Wand2 size={18} aria-hidden="true" />
          {isLoading ? "Running Inference..." : "Restore Across All Models"}
        </button>
      </div>
    </div>
  );
}
