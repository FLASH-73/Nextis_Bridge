#!/bin/bash
echo "Stopping Nextis App processes..."

# Kill processes on specific ports
echo "Freeing port 8000 (Backend)..."
fuser -k 8000/tcp > /dev/null 2>&1

echo "Freeing port 3000 (Frontend)..."
fuser -k 3000/tcp > /dev/null 2>&1

# Kill by name just in case
echo "Killing python backend..."
pkill -f "run_backend.py"

echo "Killing next.js frontend..."
pkill -f "next-server"
pkill -f "next dev"

echo "Done."
