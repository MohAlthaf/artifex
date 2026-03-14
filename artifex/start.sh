#!/usr/bin/env bash
# =============================================================================
# start.sh — ARTIFEX Demo Startup Script (v3)
# =============================================================================
# Starts both servers:
#   1. Flask ML Service   (port 5001)  — loads official thesis checkpoints
#   2. Next.js Frontend   (port 3000)  — App Router, talks to Flask directly
#
# Express proxy is NO LONGER NEEDED — Next.js calls Flask via CORS.
#
# Requirements:
#   - Python 3.10+ with torch, flask, flask-cors, numpy, Pillow, torchvision
#     (activate your pyenv/conda environment before running, or set PYTHON= below)
#   - Node.js 18+ with npm
#   - Run `npm install` in client-next/ before first use
#
# Usage:
#   ./start.sh                          # uses python3 on PATH
#   PYTHON=python3.10 ./start.sh        # specify Python explicitly
# =============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Python selection: use PYTHON env var, or fall back to common defaults
# ---------------------------------------------------------------------------
if [ -n "$PYTHON" ]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
  PYTHON_BIN="python"
else
  echo "ERROR: python3 not found. Install Python 3.10+ or set: PYTHON=/path/to/python3 ./start.sh"
  exit 1
fi

echo -e "${GREEN}=========================================="
echo " ARTIFEX — Van Gogh Art Restoration Demo"
echo "=========================================="
echo -e "${NC}"
echo -e "${BLUE}Python: ${PYTHON_BIN}${NC}"
echo ""

# Start Flask ML Server
echo -e "${BLUE}[1/2] Starting Flask ML Service (port 5001)...${NC}"
echo "      Loading official thesis checkpoints — expect 30-60s on first start"
cd "$PROJECT_DIR/server/ml"
"$PYTHON_BIN" app.py &
ML_PID=$!
sleep 5   # give Flask time to initialise models

# Start Next.js Frontend
echo -e "${BLUE}[2/2] Starting Next.js Frontend (port 3000)...${NC}"
cd "$PROJECT_DIR/client-next"
npm run dev &
NEXT_PID=$!

echo ""
echo -e "${GREEN}=========================================="
echo " All servers started!"
echo ""
echo "  Flask ML Service : http://localhost:5001/health"
echo "  Next.js Frontend : http://localhost:3000"
echo ""
echo "  Open http://localhost:3000 in your browser."
echo ""
echo "  Single-page thesis demo:"
echo "    Upload damaged image + mask + optional clean ground truth"
echo "    → All-model inference with per-upload metrics (when GT provided)"
echo "    → Official benchmark evidence (n=305 test images)"
echo ""
echo "  Press Ctrl+C to stop all servers."
echo -e "==========================================${NC}"

# Cleanup on exit
trap "echo ''; echo 'Stopping all servers…'; kill \$ML_PID \$NEXT_PID 2>/dev/null; exit 0" INT TERM
wait
