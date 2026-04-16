const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

async function readError(response, fallbackMessage) {
  const data = await response.json().catch(() => null);
  return data?.error || `${fallbackMessage} (${response.status})`;
}

export async function restoreWithEval(imageFile, maskFile, groundTruthFile) {
  const formData = new FormData();

  formData.append("image", imageFile);

  if (maskFile) {
    formData.append("mask", maskFile);
  }

  if (groundTruthFile) {
    formData.append("ground_truth", groundTruthFile);
  }

  const response = await fetch(`${API_URL}/api/restore-with-eval`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(
      await readError(response, "ARTIFEX restoration request failed"),
    );
  }

  return response.json();
}

export async function getModels() {
  const response = await fetch(`${API_URL}/api/models`);

  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch ARTIFEX models"));
  }

  return response.json();
}

export async function healthCheck() {
  const response = await fetch(`${API_URL}/health`);

  if (!response.ok) {
    throw new Error("ARTIFEX backend is unavailable");
  }

  return response.json();
}

export async function getBenchmarkComparison() {
  const response = await fetch(`${API_URL}/api/benchmark/comparison`);

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export async function getBenchmarkModels() {
  const response = await fetch(`${API_URL}/api/benchmark/models`);

  if (!response.ok) {
    return {};
  }

  return response.json();
}

export async function getBenchmarkSamples(page = 1, perPage = 6) {
  const response = await fetch(
    `${API_URL}/api/benchmark/samples?page=${page}&per_page=${perPage}`,
  );

  if (!response.ok) {
    return { samples: [], total: 0, page: 1, pages: 0 };
  }

  return response.json();
}

export async function getBenchmarkSample(sampleId) {
  const response = await fetch(`${API_URL}/api/benchmark/sample/${sampleId}`);

  if (!response.ok) {
    throw new Error(
      await readError(response, `Failed to load ARTIFEX sample ${sampleId}`),
    );
  }

  return response.json();
}