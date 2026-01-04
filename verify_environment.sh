#!/bin/bash
echo "=== ParaCodex Environment Verification ==="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

echo "Node.js:"
node --version 2>&1 || echo "❌ Node.js not found"
echo ""

echo "npm:"
npm --version 2>&1 || echo "❌ npm not found"
echo ""

echo "Codex CLI:"
codex --version 2>&1 | head -1 || echo "❌ Codex CLI not found (install: npm install -g @openai/codex)"
echo ""

echo "Python:"
python3 --version || echo "❌ Python not found"
echo ""

echo "NVIDIA HPC SDK (nvc++):"
nvc++ --version 2>&1 | head -1 || echo "❌ nvc++ not found"
echo ""

echo "Nsight Systems:"
nsys --version 2>&1 | head -1 || echo "❌ nsys not found"
echo ""

echo "CUDA (optional):"
nvcc --version 2>&1 | grep "release" || echo "⚠️  CUDA not found (optional)"
echo ""

echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "❌ GPU not accessible"
echo ""

echo "OpenMP Support:"
if command_exists nvc++; then
    OMP_CHECK=$(nvc++ -mp -V 2>&1 | grep -i "openmp\|-mp" | head -1)
    if [ -n "$OMP_CHECK" ]; then
        echo "✅ OpenMP support available"
    else
        # Try alternative check
        nvc++ -mp --help 2>&1 | grep -qi "openmp" && echo "✅ OpenMP support available" || echo "⚠️  OpenMP support check inconclusive"
    fi
else
    echo "⚠️  nvc++ not found, cannot check OpenMP support"
fi
echo ""

echo "OpenAI API Access:"
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY is set"
elif command_exists codex; then
    CODEX_LOGIN_STATUS=$(codex login status 2>&1)
    if echo "$CODEX_LOGIN_STATUS" | grep -qi "logged in\|authenticated"; then
        echo "✅ Codex CLI is logged in"
    else
        echo "❌ OPENAI_API_KEY not set and Codex not logged in"
    fi
else
    echo "❌ OPENAI_API_KEY not set and Codex CLI not available"
fi
echo ""

echo "Python packages:"
echo "  openai: $(pip show openai 2>/dev/null | grep Version || echo 'not installed')"
echo "  numpy: $(pip show numpy 2>/dev/null | grep Version || echo 'not installed')"
echo "  matplotlib: $(pip show matplotlib 2>/dev/null | grep Version || echo 'not installed')"
echo ""

echo "=== Summary ==="
MISSING_COUNT=0
if ! command -v node &> /dev/null; then ((MISSING_COUNT++)); fi
if ! command -v npm &> /dev/null; then ((MISSING_COUNT++)); fi
if ! command -v codex &> /dev/null; then ((MISSING_COUNT++)); fi
if ! command -v nvc++ &> /dev/null; then ((MISSING_COUNT++)); fi
if ! command -v nsys &> /dev/null; then ((MISSING_COUNT++)); fi
if ! command -v python3 &> /dev/null; then ((MISSING_COUNT++)); fi

# Check OpenAI API access (either API key or Codex login)
if [ -z "$OPENAI_API_KEY" ]; then
    if command_exists codex; then
        CODEX_LOGIN_STATUS=$(codex login status 2>&1)
        if ! echo "$CODEX_LOGIN_STATUS" | grep -qi "logged in\|authenticated"; then
            ((MISSING_COUNT++))
        fi
    else
        ((MISSING_COUNT++))
    fi
fi

if [ $MISSING_COUNT -eq 0 ]; then
    echo "✅ All core dependencies are installed and configured"
else
    echo "❌ $MISSING_COUNT core dependencies are missing"
    echo "   Run './setup_environment.sh' or see ENVIRONMENT.md for setup instructions"
fi
