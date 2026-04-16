#!/usr/bin/env bash

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$PYTHON" ]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python was not found."
  echo "Install Python 3.10 or newer, or run:"
  echo "PYTHON=/path/to/python3 ./start.sh"
  exit 1
fi

cleanup() {
  echo ""
  echo "Stopping ARTIFEX servers..."
  kill "$ML_PID" "$NEXT_PID" 2>/dev/null || true
  exit 0
}

trap cleanup INT TERM

echo -e "${GREEN}=========================================="
echo " ARTIFEX | Van Gogh Art Restoration Demo"
echo "==========================================${NC}"
echo -e "${BLUE}Python: ${PYTHON_BIN}${NC}"
echo ""

echo -e "${BLUE}[1/2] Starting Flask ML service on port 5001...${NC}"
cd "$PROJECT_DIR/server/ml"
"$PYTHON_BIN" app.py &
ML_PID=$!

sleep 5

echo -e "${BLUE}[2/2] Starting Next.js frontend on port 3000...${NC}"
cd "$PROJECT_DIR/client-next"
npm run dev &
NEXT_PID=$!

echo ""
echo -e "${GREEN}=========================================="
echo "ARTIFEX servers are running."
echo ""
echo "Flask ML service : http://localhost:5001/health"
echo "Next.js frontend : http://localhost:3000"
echo ""
echo "Open http://localhost:3000 in your browser."
echo ""
echo "Demo workflow:"
echo "  1. Upload a damaged image."
echo "  2. Add an optional mask and clean ground-truth image."
echo "  3. Run restoration across the available ARTIFEX models."
echo "  4. Review per-upload metrics when ground truth is provided."
echo "  5. Review official benchmark evidence from the 305-image test set."
echo ""
echo "Press Ctrl+C to stop all servers."
echo -e "==========================================${NC}"

wait