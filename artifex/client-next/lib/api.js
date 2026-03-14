/**
 * API client for ARTIFEX Flask backend.
 * Direct connection — no Express proxy layer.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

/**
 * POST /api/restore-with-eval
 * Sends damaged image + mask + optional ground truth.
 * Returns restoration results + per-upload metrics (only when GT provided).
 */
export async function restoreWithEval(imageFile, maskFile, groundTruthFile) {
  const formData = new FormData();
  formData.append("image", imageFile);
  if (maskFile) formData.append("mask", maskFile);
  if (groundTruthFile) formData.append("ground_truth", groundTruthFile);

  const resp = await fetch(`${API_URL}/api/restore-with-eval`, {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }));
    throw new Error(err.error || `Server returned ${resp.status}`);
  }
  return resp.json();
}

/**
 * GET /api/models
 * Returns the full model registry with availability and eval summaries.
 */
export async function getModels() {
  const resp = await fetch(`${API_URL}/api/models`);
  if (!resp.ok) throw new Error("Failed to fetch models");
  return resp.json();
}

/**
 * GET /health
 * Health check.
 */
export async function healthCheck() {
  const resp = await fetch(`${API_URL}/health`);
  if (!resp.ok) throw new Error("Backend unavailable");
  return resp.json();
}

/**
 * GET /api/benchmark/comparison
 * Returns baseline vs full comparison data.
 */
export async function getBenchmarkComparison() {
  const resp = await fetch(`${API_URL}/api/benchmark/comparison`);
  if (!resp.ok) return null;
  return resp.json();
}

/**
 * GET /api/benchmark/models
 * Returns models with evaluation data.
 */
export async function getBenchmarkModels() {
  const resp = await fetch(`${API_URL}/api/benchmark/models`);
  if (!resp.ok) return {};
  return resp.json();
}

/**
 * GET /api/benchmark/samples?page=1&per_page=6
 * Paginated benchmark sample list.
 */
export async function getBenchmarkSamples(page = 1, perPage = 6) {
  const resp = await fetch(
    `${API_URL}/api/benchmark/samples?page=${page}&per_page=${perPage}`
  );
  if (!resp.ok) return { samples: [], total: 0, page: 1, pages: 0 };
  return resp.json();
}

/**
 * GET /api/benchmark/sample/:id
 * Full data for one benchmark sample (images + metrics).
 */
export async function getBenchmarkSample(sampleId) {
  const resp = await fetch(`${API_URL}/api/benchmark/sample/${sampleId}`);
  if (!resp.ok) throw new Error(`Failed to load sample ${sampleId}`);
  return resp.json();
}
