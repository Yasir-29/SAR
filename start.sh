#!/bin/bash
set -e

echo "========================================"
echo "Tamil Nadu Disaster Resilience System"
echo "========================================"
echo ""

# 1. Verify Python Virtual Environment
if [ ! -d "venv" ]; then
    echo "[ERROR] Python venv not found! Please create environment."
    exit 1
fi

PYTHON_BIN="./venv/bin/python"

# 2. Verify Final xBD-S12 Transfer Model Checkpoint
CKPT="data/tamil_nadu/final/checkpoints/tamil_nadu_xbd_s12_tn_final.pt"

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Final xBD-S12 transfer model production checkpoint missing at $CKPT"
    exit 1
fi

ACTUAL_SHA256=$($PYTHON_BIN -c "import hashlib; print(hashlib.sha256(open('$CKPT', 'rb').read()).hexdigest())")
EXPECTED_SHA256="232abe6fd662a91073d46dd32deec56fa79ccb02ab39815e59c0e6ed46df9c13"

if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "[WARNING] Checkpoint SHA256 mismatch! Expected: $EXPECTED_SHA256, Actual: $ACTUAL_SHA256"
fi

# Cleanup old background processes on port 5001 or 3000 if needed
lsof -ti:5001 | xargs kill -9 2>/dev/null || true

echo "Backend:"
echo "http://127.0.0.1:5001"
echo ""
echo "Frontend:"
echo "http://localhost:3000"
echo ""
echo "Production Model:"
echo "TamilNaduTransferNet (xBD-S12 Pretrained)"
echo ""
echo "Checkpoint SHA256:"
echo "$ACTUAL_SHA256"
echo ""
echo "Map:"
echo "Leaflet + OpenStreetMap (Token-Free)"
echo ""
echo "Status:"
echo "FINAL PRODUCTION RECOVERY BUILD"
echo "========================================"

trap 'kill 0' EXIT

# Start Flask Backend in background
$PYTHON_BIN app/backend/server.py &
BACKEND_PID=$!

# Start Next.js Frontend in background (if not already running)
if ! lsof -ti:3000 > /dev/null 2>&1; then
    cd frontend && npm run dev &
fi

wait
