/**
 * LiveRestorePage.jsx
 * ====================
 * Live restoration demo page.
 *
 * Flow:
 *   1. User uploads a damaged image (drag-drop or browse)
 *   2. Optional: upload a damage mask
 *   3. Click "Restore" → POST /api/restore-all
 *   4. All available models run on the image
 *   5. Results shown as side-by-side model cards
 *
 * Metric honesty:
 *   - Each card shows the model's OFFICIAL test-set metric summary
 *   - Cards clearly label these as "Official Test-Set Metrics" with disclaimer
 *   - No per-upload PSNR/SSIM is computed or shown
 */

import { useState, useCallback } from "react";
import ModelResultCard from "../components/ModelResultCard";

const API_URL = "http://localhost:3001/api";

export default function LiveRestorePage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedMask, setSelectedMask] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [maskPreviewUrl, setMaskPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null); // array from /api/restore-all
  const [error, setError] = useState(null);

  // ---------- upload handlers ----------
  const handleFileSelect = (file) => {
    if (!file) return;
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleFileSelect(file);
  }, []);

  const handleMaskSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedMask(file);
      setMaskPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setSelectedMask(null);
    setPreviewUrl(null);
    setMaskPreviewUrl(null);
    setResults(null);
    setError(null);
  };

  // ---------- restore ----------
  const handleRestore = async () => {
    if (!selectedImage) return;
    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("image", selectedImage);
      if (selectedMask) formData.append("mask", selectedMask);

      const resp = await fetch(`${API_URL}/restore-all`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      const data = await resp.json();
      setResults(data.results);
    } catch (err) {
      setError("Restoration failed. Make sure both servers are running.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // ---------- render ----------
  return (
    <div>
      {/* Hero */}
      <section
        className="hero"
        style={{ paddingTop: "60px", paddingBottom: "40px" }}
      >
        <div className="hero-content">
          <span className="hero-badge fade-in">✨ Live Inference Demo</span>
          <h1 className="hero-title fade-in delay-1">
            Restore Damaged Paintings
          </h1>
          <p className="hero-subtitle fade-in delay-2">
            Upload a damaged image and run it through all available official
            thesis models simultaneously.
          </p>
        </div>
      </section>

      {/* Upload section */}
      <section className="upload-section" style={{ paddingTop: "0" }}>
        <div className="container">
          {/* Upload zone */}
          {!previewUrl ? (
            <>
              <div
                className="section-title"
                style={{ marginBottom: "var(--spacing-xl)" }}
              >
                <h2>Upload Your Image</h2>
                <p>Drag & drop a damaged painting, or click to browse</p>
              </div>
              <div
                className={`upload-zone glass ${isDragging ? "dragging" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                }}
                onDrop={handleDrop}
                onClick={() => document.getElementById("lr-file-input").click()}
              >
                <input
                  id="lr-file-input"
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                />
                <div className="upload-icon">📤</div>
                <p className="upload-text">Drop your damaged painting here</p>
                <p className="upload-hint">PNG, JPG, WebP · Max 10 MB</p>
              </div>
            </>
          ) : (
            <div className="fade-in">
              {/* Preview + mask upload */}
              {!results && (
                <div>
                  <div className="preview-container">
                    <div className="preview-card glass">
                      <p className="preview-label">Uploaded Image</p>
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="preview-image"
                      />
                    </div>
                  </div>

                  {/* Mask upload */}
                  <div
                    className="glass"
                    style={{
                      padding: "var(--spacing-lg)",
                      marginTop: "var(--spacing-lg)",
                      borderRadius: "var(--radius-xl)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        marginBottom: "var(--spacing-md)",
                        flexWrap: "wrap",
                        gap: "8px",
                      }}
                    >
                      <div>
                        <p
                          className="preview-label"
                          style={{ marginBottom: "4px" }}
                        >
                          Damage Mask (Optional)
                        </p>
                        <p
                          style={{
                            color: "var(--color-gray-400)",
                            fontSize: "0.75rem",
                          }}
                        >
                          White = damaged, Black = intact
                        </p>
                      </div>
                      {selectedMask && (
                        <button
                          className="btn btn-secondary"
                          style={{ padding: "8px 16px", fontSize: "0.875rem" }}
                          onClick={() => {
                            setSelectedMask(null);
                            setMaskPreviewUrl(null);
                          }}
                        >
                          Remove Mask
                        </button>
                      )}
                    </div>
                    {maskPreviewUrl ? (
                      <div
                        style={{
                          display: "flex",
                          gap: "var(--spacing-md)",
                          alignItems: "center",
                        }}
                      >
                        <img
                          src={maskPreviewUrl}
                          alt="Mask"
                          style={{
                            width: "100px",
                            height: "100px",
                            objectFit: "cover",
                            borderRadius: "var(--radius-md)",
                            border: "2px solid var(--color-gold)",
                          }}
                        />
                        <span
                          style={{
                            color: "var(--color-success)",
                            fontSize: "0.875rem",
                          }}
                        >
                          ✓ Mask uploaded
                        </span>
                      </div>
                    ) : (
                      <div
                        onClick={() =>
                          document.getElementById("lr-mask-input").click()
                        }
                        style={{
                          border: "2px dashed rgba(212,175,55,0.3)",
                          borderRadius: "var(--radius-lg)",
                          padding: "var(--spacing-lg)",
                          textAlign: "center",
                          cursor: "pointer",
                        }}
                      >
                        <input
                          id="lr-mask-input"
                          type="file"
                          accept="image/*"
                          style={{ display: "none" }}
                          onChange={handleMaskSelect}
                        />
                        <p
                          style={{
                            color: "var(--color-gray-300)",
                            marginBottom: "4px",
                          }}
                        >
                          🎭 Click to upload mask
                        </p>
                        <p
                          style={{
                            color: "var(--color-gray-500)",
                            fontSize: "0.75rem",
                          }}
                        >
                          If not provided, damaged areas are auto-detected from
                          black pixels
                        </p>
                      </div>
                    )}
                  </div>

                  {error && (
                    <p
                      style={{
                        color: "var(--color-error)",
                        textAlign: "center",
                        marginTop: "var(--spacing-lg)",
                      }}
                    >
                      {error}
                    </p>
                  )}

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      gap: "var(--spacing-md)",
                      marginTop: "var(--spacing-xl)",
                    }}
                  >
                    <button className="btn btn-secondary" onClick={handleReset}>
                      Cancel
                    </button>
                    <button
                      className="btn btn-primary btn-lg"
                      onClick={handleRestore}
                      disabled={isLoading}
                    >
                      {isLoading ? "Running…" : "✨ Restore Across All Models"}
                    </button>
                  </div>
                </div>
              )}

              {/* Loading shimmer */}
              {isLoading && (
                <div
                  style={{
                    textAlign: "center",
                    padding: "var(--spacing-3xl)",
                    color: "var(--color-gray-300)",
                  }}
                >
                  <div
                    className="spinner"
                    style={{ margin: "0 auto var(--spacing-lg)" }}
                  />
                  <p>Running inference through all available models…</p>
                  <p
                    style={{
                      fontSize: "0.875rem",
                      color: "var(--color-gray-500)",
                      marginTop: "8px",
                    }}
                  >
                    This may take 30–90 seconds per model on CPU
                  </p>
                </div>
              )}

              {/* Results */}
              {results && !isLoading && (
                <div>
                  <div
                    className="section-title"
                    style={{ marginBottom: "var(--spacing-xl)" }}
                  >
                    <h3 style={{ color: "var(--color-gold)" }}>
                      Restoration Results
                    </h3>
                    <p>Slide to compare damaged vs restored for each model</p>
                  </div>

                  {/* Metric disclaimer banner */}
                  <div
                    style={{
                      background: "rgba(212,175,55,0.08)",
                      border: "1px solid rgba(212,175,55,0.2)",
                      borderRadius: "var(--radius-lg)",
                      padding: "var(--spacing-md)",
                      marginBottom: "var(--spacing-xl)",
                      fontSize: "0.8rem",
                      color: "var(--color-gray-300)",
                      display: "flex",
                      gap: "12px",
                      alignItems: "flex-start",
                    }}
                  >
                    <span style={{ fontSize: "1.2rem", flexShrink: 0 }}>
                      ℹ️
                    </span>
                    <span>
                      <strong style={{ color: "var(--color-gold)" }}>
                        Metric note:
                      </strong>{" "}
                      Metrics shown below each card are
                      <strong> official test-set averages</strong> computed over
                      the Van Gogh benchmark (305 images with ground truth).
                      They are <em>not</em> computed for your current upload.
                      For per-image metrics with ground truth, use the{" "}
                      <strong>Benchmark Explorer</strong>.
                    </span>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "var(--spacing-lg)",
                    }}
                  >
                    {results.map((result) => (
                      <ModelResultCard
                        key={result.model_id}
                        result={result}
                        originalSrc={previewUrl}
                        isLoading={false}
                      />
                    ))}
                  </div>

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      marginTop: "var(--spacing-2xl)",
                    }}
                  >
                    <button className="btn btn-secondary" onClick={handleReset}>
                      Try Another Image
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
