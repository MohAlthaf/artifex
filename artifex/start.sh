#!/usr/bin/env bash
# =============================================================================
# start.sh — ARTIFEX Demo Startup Script (v2)
# =============================================================================
# Starts all three servers in the correct order:
#   1. Flask ML Service   (port 5001)  — loads official thesis checkpoints
#   2. Express API        (port 3001)  — proxies to Flask
#   3. React Frontend     (port 5173)  — Vite dev server
#
# Requirements:
#   - Python 3.10+ with torch, flask, flask-cors, numpy, Pillow installed
#     (activate your pyenv/conda environment before running, or set PYTHON= below)
#   - Node.js 18+ with npm
#   - Run `npm install` in server/ and client/ before first use
#
# Usage:
#   ./start.sh                          # uses python3 on PATH
#   PYTHON=python3.10 ./start.sh        # specify Python explicitly
#   PYTHON=~/.pyenv/versions/3.10.9/bin/python3 ./start.sh
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
echo -e "${BLUE}[1/3] Starting Flask ML Service (port 5001)...${NC}"
echo "      Loading official thesis checkpoints — expect 30-60s on first start"
cd "$PROJECT_DIR/server/ml"
"$PYTHON_BIN" app.py &
ML_PID=$!
sleep 5   # give Flask time to initialise models before Express connects

# Start Express API Server
echo -e "${BLUE}[2/3] Starting Express API Server (port 3001)...${NC}"
cd "$PROJECT_DIR/server"
npm run dev &
EXPRESS_PID=$!
sleep 2

# Start React Frontend
echo -e "${BLUE}[3/3] Starting React Frontend (port 5173)...${NC}"
cd "$PROJECT_DIR/client"
npm run dev &
REACT_PID=$!

echo ""
echo -e "${GREEN}=========================================="
echo " All servers started!"
echo ""
echo "  Flask ML Service : http://localhost:5001/health"
echo "  Express API      : http://localhost:3001/api/health"
echo "  React Frontend   : http://localhost:5173"
echo ""
echo "  Open http://localhost:5173 in your browser."
echo ""
echo "  Tabs:"
echo "    ✨ Live Restore      — upload an image → all-model inference"
echo "    📊 Benchmark Explorer — official test-set results (n=305)"
echo ""
echo "  Press Ctrl+C to stop all servers."
echo -e "==========================================${NC}"

# Cleanup on exit
trap "echo ''; echo 'Stopping all servers…'; kill \$ML_PID \$EXPRESS_PID \$REACT_PID 2>/dev/null; exit 0" INT TERM
wait
