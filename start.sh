#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Stopping Nextis App..."
    kill $(jobs -p)
    exit
}

trap cleanup SIGINT SIGTERM

echo "Starting Nextis App..."

# Kill any leftover backend processes first
echo "Cleaning up any existing backend processes..."
pkill -f "uvicorn.*app.main" 2>/dev/null || true
pkill -f "run_backend.py" 2>/dev/null || true
sleep 1

# Start Backend (show output in terminal, also log to file)
echo "Starting Backend on http://localhost:8000 (using 'umbra' env)..."
export PYTHONPATH=$(pwd)/lerobot/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
/home/roberto/miniconda3/bin/python run_backend.py 2>&1 | tee app.log &
BACKEND_PID=$!

# Wait for backend to be ready (optional, but good practice)
sleep 2

# Start Frontend
echo "Starting Frontend on http://localhost:3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Wait for both
wait $BACKEND_PID $FRONTEND_PID
