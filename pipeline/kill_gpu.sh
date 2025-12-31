#!/bin/bash
# Quick script to kill all GPU processes

echo "GPU processes before:"
nvidia-smi --query-compute-apps=pid,name --format=csv

echo ""
echo "Killing all GPU compute processes..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    if [ -n "$pid" ]; then
        echo "Killing PID $pid"
        kill -9 $pid 2>/dev/null
    fi
done

sleep 1

echo ""
echo "GPU processes after:"
nvidia-smi --query-compute-apps=pid,name --format=csv

