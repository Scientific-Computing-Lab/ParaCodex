#!/bin/bash
# ParaCodex Environment Setup Script
# This script helps set up the environment for running the ParaCodex pipeline

set -e

echo "=== ParaCodex Environment Setup ==="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check Node.js
echo "Checking Node.js..."
if command_exists node; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js $NODE_VERSION found"
else
    echo "❌ Node.js not found."
    echo "   Install with nvm (recommended):"
    echo "     curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "     source ~/.bashrc"
    echo "     nvm install 22"
    echo "   Or with apt:"
    echo "     curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -"
    echo "     sudo apt-get install -y nodejs"
fi

# Check npm
echo "Checking npm..."
if command_exists npm; then
    NPM_VERSION=$(npm --version)
    echo "✅ npm $NPM_VERSION found"
else
    echo "❌ npm not found. Install Node.js first (includes npm)."
fi

# Check and install Codex CLI
echo ""
echo "Checking Codex CLI..."
if command_exists codex; then
    CODEX_VERSION=$(codex --version 2>&1 | head -1)
    echo "✅ Codex CLI found: $CODEX_VERSION"
else
    echo "❌ Codex CLI not found."
    if command_exists npm; then
        echo "   Installing Codex CLI..."
        npm install -g @openai/codex
        if [ $? -eq 0 ]; then
            echo "✅ Codex CLI installed successfully"
        else
            echo "❌ Failed to install Codex CLI"
        fi
    else
        echo "   Install npm first, then run: npm install -g @openai/codex"
    fi
fi

# Check Python
echo ""
echo "Checking Python..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check pip
echo "Checking pip..."
if command_exists pip || command_exists pip3; then
    echo "✅ pip found"
else
    echo "❌ pip not found. Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️  requirements.txt not found in current directory"
fi

# Check NVIDIA HPC SDK
echo ""
echo "Checking NVIDIA HPC SDK..."
if command_exists nvc++; then
    NVC_VERSION=$(nvc++ --version 2>&1 | head -1)
    echo "✅ NVIDIA HPC SDK found: $NVC_VERSION"
else
    echo "❌ NVIDIA HPC SDK (nvc++) not found"
    echo "   Please install from: https://developer.nvidia.com/hpc-sdk"
    echo "   After installation, add to PATH:"
    echo "   export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:\$PATH"
fi

# Check Nsight Systems
echo ""
echo "Checking NVIDIA Nsight Systems..."
if command_exists nsys; then
    NSYS_VERSION=$(nsys --version 2>&1 | head -1)
    echo "✅ Nsight Systems found: $NSYS_VERSION"
else
    echo "❌ Nsight Systems (nsys) not found"
    echo "   Please install from: https://developer.nvidia.com/nsight-systems"
fi

# Check GPU
echo ""
echo "Checking GPU..."
if command_exists nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✅ GPU found: $GPU_INFO"
else
    echo "❌ nvidia-smi not found. GPU may not be accessible."
fi

# Check OpenAI API Key
echo ""
echo "Checking OpenAI API Key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY environment variable not set"
    echo "   Set it with: export OPENAI_API_KEY='your-api-key'"
    echo "   Add to ~/.bashrc for persistence"
else
    echo "✅ OPENAI_API_KEY is set"
fi

# Summary
echo ""
echo "=== Setup Summary ==="
MISSING=0

if ! command_exists node; then ((MISSING++)); fi
if ! command_exists npm; then ((MISSING++)); fi
if ! command_exists codex; then ((MISSING++)); fi
if ! command_exists python3; then ((MISSING++)); fi
if ! command_exists nvc++; then ((MISSING++)); fi
if ! command_exists nsys; then ((MISSING++)); fi
if [ -z "$OPENAI_API_KEY" ]; then ((MISSING++)); fi

if [ $MISSING -eq 0 ]; then
    echo "✅ All required dependencies are installed!"
    echo "   You can now run the ParaCodex pipeline."
else
    echo "⚠️  $MISSING required dependencies are missing."
    echo "   Please install the missing dependencies listed above."
fi

echo ""
echo "Run './verify_environment.sh' to verify your setup."
