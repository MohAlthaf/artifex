/**
 * Express API Server for Artifex (v2)
 * =====================================
 * Proxies all /api/* requests to the Flask ML service on port 5001.
 * Handles multipart file uploads where needed (multer in-memory).
 *
 * Routes:
 *   GET  /api/health                          → Flask /health
 *   GET  /api/models                          → Flask /api/models
 *   POST /api/restore-all                     → Flask /api/restore-all  (multipart)
 *   POST /api/restore-one                     → Flask /api/restore-one  (multipart)
 *   POST /api/restore        (legacy)         → Flask /predict          (multipart)
 *   GET  /api/benchmark/models                → Flask /api/benchmark/models
 *   GET  /api/benchmark/samples               → Flask /api/benchmark/samples
 *   GET  /api/benchmark/sample/:id            → Flask /api/benchmark/sample/:id
 *   GET  /api/benchmark/metrics/:modelId      → Flask /api/benchmark/metrics/:modelId
 *   GET  /api/benchmark/comparison            → Flask /api/benchmark/comparison
 */

import express from "express";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;
const ML_SERVER_URL = process.env.ML_SERVER_URL || "http://localhost:5001";

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type. Only JPEG, PNG, and WebP are allowed."));
    }
  },
});

// Serve static files from samples directory
app.use("/samples", express.static(path.join(__dirname, "samples")));
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

// ---------------------------------------------------------------------------
// Helper: forward multipart upload to Flask
// ---------------------------------------------------------------------------
async function _forwardUpload(flaskPath, req, res, responseType = "json") {
  try {
    if (!req.files || !req.files.image) {
      return res.status(400).json({ error: "No image provided" });
    }
    const formData = new FormData();
    formData.append("image", req.files.image[0].buffer, {
      filename: req.files.image[0].originalname,
      contentType: req.files.image[0].mimetype,
    });
    if (req.files.mask) {
      formData.append("mask", req.files.mask[0].buffer, {
        filename: req.files.mask[0].originalname,
        contentType: req.files.mask[0].mimetype,
      });
    }
    // Forward any form fields (e.g. model_id)
    for (const [key, val] of Object.entries(req.body || {})) {
      formData.append(key, String(val));
    }
    const resp = await axios.post(`${ML_SERVER_URL}${flaskPath}`, formData, {
      headers: formData.getHeaders(),
      responseType,
      timeout: 120000,
    });
    if (responseType === "arraybuffer") {
      res.set("Content-Type", "image/png");
      res.send(Buffer.from(resp.data));
    } else {
      res.json(resp.data);
    }
  } catch (err) {
    console.error(`Upload proxy error (${flaskPath}):`, err.message);
    res.status(500).json({ error: err.message });
  }
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
app.get("/api/health", async (req, res) => {
  try {
    const mlHealth = await axios.get(`${ML_SERVER_URL}/health`);
    res.json({ status: "healthy", mlServer: mlHealth.data });
  } catch {
    res.json({ status: "healthy", mlServer: { status: "unavailable" } });
  }
});

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------
app.get("/api/models", async (req, res) => {
  try {
    const resp = await axios.get(`${ML_SERVER_URL}/api/models`);
    res.json(resp.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Restore endpoints
// ---------------------------------------------------------------------------
const uploadFields = upload.fields([
  { name: "image", maxCount: 1 },
  { name: "mask", maxCount: 1 },
]);

// Run ALL available models on the uploaded image → returns JSON with b64 results
app.post("/api/restore-all", uploadFields, (req, res) =>
  _forwardUpload("/api/restore-all", req, res, "json"),
);

// Run ONE specific model on the uploaded image → returns PNG binary
app.post("/api/restore-one", uploadFields, (req, res) =>
  _forwardUpload("/api/restore-one", req, res, "arraybuffer"),
);

// Legacy single-model restore (kept for backward compat) → PNG binary
app.post("/api/restore", uploadFields, (req, res) =>
  _forwardUpload("/predict", req, res, "arraybuffer"),
);

// ---------------------------------------------------------------------------
// Benchmark endpoints — simple JSON proxy (no file upload)
// ---------------------------------------------------------------------------
async function _proxyGet(flaskPath, req, res) {
  try {
    const url = `${ML_SERVER_URL}${flaskPath}`;
    const resp = await axios.get(url, {
      params: req.query,
      timeout: 30000,
    });
    res.json(resp.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.message });
  }
}

app.get("/api/benchmark/models", (req, res) =>
  _proxyGet("/api/benchmark/models", req, res),
);
app.get("/api/benchmark/samples", (req, res) =>
  _proxyGet("/api/benchmark/samples", req, res),
);
app.get("/api/benchmark/sample/:id", (req, res) =>
  _proxyGet(`/api/benchmark/sample/${req.params.id}`, req, res),
);
app.get("/api/benchmark/metrics/:modelId", (req, res) =>
  _proxyGet(`/api/benchmark/metrics/${req.params.modelId}`, req, res),
);
app.get("/api/benchmark/comparison", (req, res) =>
  _proxyGet("/api/benchmark/comparison", req, res),
);

// ---------------------------------------------------------------------------
// Legacy samples endpoint (static)
// ---------------------------------------------------------------------------
app.get("/api/samples", (req, res) => {
  const samplesDir = path.join(__dirname, "samples");
  if (!fs.existsSync(samplesDir)) return res.json({ samples: [] });
  const samples = fs
    .readdirSync(samplesDir)
    .filter((f) => /\.(png|jpg|jpeg)$/i.test(f))
    .slice(0, 10)
    .map((f) => ({
      name: f.replace(/\.[^/.]+$/, "").replace(/_/g, " "),
      damaged: `/samples/${f}`,
      restored: `/samples/restored_${f}`,
    }));
  res.json({ samples });
});

// Error handler
app.use((err, req, res, next) => {
  console.error("Server error:", err);
  res.status(500).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(` Artifex API server running on http://localhost:${PORT}`);
  console.log(` ML Server URL: ${ML_SERVER_URL}`);
});
