#!/bin/bash

# Script to compile and run all NAS kernels
# Usage: ./test_all_nas_kernels.sh

DATA_DIR="/root/codex_baseline/serial_omp_nas_workdir/data/src"
GOLDEN_DIR="/root/codex_baseline/serial_omp_nas_workdir/golden_labels/src"

DATA_SUCCESS=()
DATA_FAILED_COMPILE=()
DATA_FAILED_RUN=()
GOLDEN_SUCCESS=()
GOLDEN_FAILED_COMPILE=()
GOLDEN_FAILED_RUN=()

TIMEOUT_SECONDS=240

echo "=========================================="
echo "Testing NAS Kernels in data/src"
echo "=========================================="
echo ""

cd "$DATA_DIR" || exit 1

for dir in */; do
    dir=${dir%/}
    if [ ! -f "$dir/Makefile" ]; then
        continue
    fi
    
    echo "----------------------------------------"
    echo "Data: $dir"
    echo "----------------------------------------"
    
    cd "$dir" || continue
    
    # Clean first
    make CC=nvc++ CLASS=S clean > /dev/null 2>&1
    
    # Test compilation
    echo "  Compiling..."
    if make CC=nvc++ CLASS=S > /tmp/data_compile_${dir}.log 2>&1; then
        echo "  ✓ Compilation successful"
        
        # Test running
        echo "  Running..."
        if timeout $TIMEOUT_SECONDS make CC=nvc++ CLASS=S run > /tmp/data_run_${dir}.log 2>&1; then
            echo "  ✓ Run successful"
            DATA_SUCCESS+=("$dir")
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "  ✗ Run timeout (>${TIMEOUT_SECONDS}s)"
            else
                echo "  ✗ Run failed"
                echo "    Last lines:"
                tail -3 /tmp/data_run_${dir}.log | sed 's/^/    /'
            fi
            DATA_FAILED_RUN+=("$dir")
        fi
    else
        echo "  ✗ Compilation failed"
        echo "    Last lines:"
        tail -5 /tmp/data_compile_${dir}.log | sed 's/^/    /'
        DATA_FAILED_COMPILE+=("$dir")
    fi
    
    cd "$DATA_DIR" || exit 1
    echo ""
done

echo ""
echo "=========================================="
echo "Testing NAS Kernels in golden_labels/src"
echo "=========================================="
echo ""

cd "$GOLDEN_DIR" || exit 1

for dir in */; do
    dir=${dir%/}
    if [ ! -f "$dir/Makefile" ]; then
        continue
    fi
    
    echo "----------------------------------------"
    echo "Golden: $dir"
    echo "----------------------------------------"
    
    cd "$dir" || continue
    
    # Clean first
    make CC=nvc++ CLASS=S clean > /dev/null 2>&1
    
    # Test compilation
    echo "  Compiling..."
    if make CC=nvc++ CLASS=S > /tmp/golden_compile_${dir}.log 2>&1; then
        echo "  ✓ Compilation successful"
        
        # Test running
        echo "  Running..."
        if timeout $TIMEOUT_SECONDS make CC=nvc++ CLASS=S run > /tmp/golden_run_${dir}.log 2>&1; then
            echo "  ✓ Run successful"
            GOLDEN_SUCCESS+=("$dir")
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "  ✗ Run timeout (>${TIMEOUT_SECONDS}s)"
            else
                echo "  ✗ Run failed"
                echo "    Last lines:"
                tail -3 /tmp/golden_run_${dir}.log | sed 's/^/    /'
            fi
            GOLDEN_FAILED_RUN+=("$dir")
        fi
    else
        echo "  ✗ Compilation failed"
        echo "    Last lines:"
        tail -5 /tmp/golden_compile_${dir}.log | sed 's/^/    /'
        GOLDEN_FAILED_COMPILE+=("$dir")
    fi
    
    cd "$GOLDEN_DIR" || exit 1
    echo ""
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Data Kernels:"
echo "  Successful: ${#DATA_SUCCESS[@]}"
echo "  Failed compilation: ${#DATA_FAILED_COMPILE[@]}"
echo "  Failed run: ${#DATA_FAILED_RUN[@]}"
echo ""
echo "Golden Kernels:"
echo "  Successful: ${#GOLDEN_SUCCESS[@]}"
echo "  Failed compilation: ${#GOLDEN_FAILED_COMPILE[@]}"
echo "  Failed run: ${#GOLDEN_FAILED_RUN[@]}"
echo ""

if [ ${#DATA_FAILED_COMPILE[@]} -gt 0 ]; then
    echo "Data - Failed to compile:"
    for dir in "${DATA_FAILED_COMPILE[@]}"; do
        echo "  - $dir"
    done
    echo ""
fi

if [ ${#DATA_FAILED_RUN[@]} -gt 0 ]; then
    echo "Data - Failed to run:"
    for dir in "${DATA_FAILED_RUN[@]}"; do
        echo "  - $dir"
    done
    echo ""
fi

if [ ${#GOLDEN_FAILED_COMPILE[@]} -gt 0 ]; then
    echo "Golden - Failed to compile:"
    for dir in "${GOLDEN_FAILED_COMPILE[@]}"; do
        echo "  - $dir"
    done
    echo ""
fi

if [ ${#GOLDEN_FAILED_RUN[@]} -gt 0 ]; then
    echo "Golden - Failed to run:"
    for dir in "${GOLDEN_FAILED_RUN[@]}"; do
        echo "  - $dir"
    done
    echo ""
fi

TOTAL_DATA=$((${#DATA_SUCCESS[@]} + ${#DATA_FAILED_COMPILE[@]} + ${#DATA_FAILED_RUN[@]}))
TOTAL_GOLDEN=$((${#GOLDEN_SUCCESS[@]} + ${#GOLDEN_FAILED_COMPILE[@]} + ${#GOLDEN_FAILED_RUN[@]}))

echo "=========================================="
echo "Overall Statistics"
echo "=========================================="
echo "Data: ${#DATA_SUCCESS[@]}/${TOTAL_DATA} kernels compile and run successfully"
echo "Golden: ${#GOLDEN_SUCCESS[@]}/${TOTAL_GOLDEN} kernels compile and run successfully"
echo "Total: $((${#DATA_SUCCESS[@]} + ${#GOLDEN_SUCCESS[@]}))/$(($TOTAL_DATA + $TOTAL_GOLDEN)) kernels successful"
echo ""

# Exit with error if there are any failures
if [ ${#DATA_FAILED_COMPILE[@]} -gt 0 ] || [ ${#DATA_FAILED_RUN[@]} -gt 0 ] || \
   [ ${#GOLDEN_FAILED_COMPILE[@]} -gt 0 ] || [ ${#GOLDEN_FAILED_RUN[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi

