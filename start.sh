#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Stopping Nextis App..."
    kill $(jobs -p)
    exit
}

trap cleanup SIGINT SIGTERM

echo "Starting Nextis App..."

# Start Backend
echo "Starting Backend on http://localhost:8000 (using 'umbra' env)..."
export PYTHONPATH=$(pwd)/lerobot/src:$PYTHONPATH
/home/roberto/miniconda3/bin/python run_backend.py > app.log 2>&1 &
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
