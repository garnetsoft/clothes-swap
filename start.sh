#!/bin/bash
set -e
VENV=/mnt/ssd/envs/llm-orin-310
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Clothes Swap v2..."

# Backend
$VENV/bin/python -m uvicorn backend.api:app \
    --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend started (PID $BACKEND_PID)"

sleep 2

# Frontend
$VENV/bin/streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 &
FRONTEND_PID=$!
echo "Frontend started (PID $FRONTEND_PID)"

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "  App:     http://$IP:8501"
echo "  API:     http://$IP:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
