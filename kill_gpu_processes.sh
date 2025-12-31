#!/bin/bash
# Script to kill all GPU processes launched from this WSL machine

set -e

echo "Finding GPU processes..."

# Get all PIDs using the GPU
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | tr -d ' ' || true)

if [ -z "$PIDS" ]; then
    echo "No GPU processes found."
    exit 0
fi

echo "Found GPU processes: $PIDS"

# Count how many we'll try to kill
COUNT=0
KILLED=0
FAILED=0

# Try to kill each PID
for PID in $PIDS; do
    # Check if PID is a valid number
    if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
        continue
    fi
    
    COUNT=$((COUNT + 1))
    
    # Check if process exists and is accessible (WSL processes will be accessible)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Killing PID $PID..."
        if kill -9 "$PID" 2>/dev/null; then
            KILLED=$((KILLED + 1))
            echo "  ✓ Successfully killed PID $PID"
        else
            FAILED=$((FAILED + 1))
            echo "  ✗ Failed to kill PID $PID (may not be accessible from WSL)"
        fi
    else
        echo "  - PID $PID not accessible from WSL (likely a Windows process)"
    fi
done

echo ""
echo "Summary:"
echo "  Total GPU processes found: $COUNT"
echo "  Successfully killed: $KILLED"
echo "  Failed to kill: $FAILED"

if [ $KILLED -gt 0 ]; then
    echo ""
    echo "Waiting 2 seconds and checking for remaining processes..."
    sleep 2
    REMAINING=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l || echo "0")
    if [ "$REMAINING" -gt 0 ]; then
        echo "Warning: $REMAINING GPU process(es) still running (may be Windows processes)"
    else
        echo "All WSL GPU processes have been terminated."
    fi
fi

