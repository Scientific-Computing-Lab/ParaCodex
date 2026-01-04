# Environment Setup Guide

This document describes the complete environment setup required to run the ParaCodex pipeline.

## System Requirements
## Software Dependencies

### 1. NVIDIA HPC SDK (Required)
The NVIDIA HPC SDK provides the `nvc++` compiler with OpenMP GPU offload support.

**Version**: 25.7-0 or later

**Installation Option 1: Automated (Recommended)**:
```bash
# Run the automated installer
sudo ./install_nvidia_hpc_sdk.sh
```

The script will:
- Check system requirements
- Download NVIDIA HPC SDK (~3GB)
- Install to /opt/nvidia/hpc_sdk
- Configure environment variables
- Verify installation

**Installation Option 2: Manual**:
```bash
# Download from NVIDIA website
wget https://developer.download.nvidia.com/hpc-sdk/25.7/nvhpc_2025_257_Linux_x86_64_cuda_12.6.tar.gz

# Extract and install
tar xpzf nvhpc_2025_257_Linux_x86_64_cuda_12.6.tar.gz
cd nvhpc_2025_257_Linux_x86_64_cuda_12.6
sudo ./install

# Add to PATH (add to ~/.bashrc)
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH
```

**Verify installation**:
```bash
nvc++ --version
# Should output: nvc++ 25.7-0 64-bit target on x86-64 Linux
```
**Verify installation**:
```bash
nsys --version
# Should output: NVIDIA Nsight Systems version 2025.3.1.90-253135822126v0
```

### 2. CUDA Toolkit (Optional, for CUDA translation)
Required if translating to/from CUDA.

**Version**: 11.8 or later (tested with 12.6)

**Installation**:
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

### 3. Node.js and npm (Required for Codex CLI)
**Version**: Node.js 18+ (tested with v22.19.0), npm 9+ (tested with 11.6.2)

**Installation**:
```bash
# Option 1: Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 22
nvm use 22

# Option 2: Using apt (Ubuntu/Debian)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version  # Should show v22.x.x or later
npm --version   # Should show 9.x.x or later
```

### 4. Codex CLI (Required for AI Agent)
The Codex CLI is the interface for interacting with OpenAI's Codex API.

**Version**: @openai/codex 0.77.0 or later

**Installation**:
```bash
# Install globally via npm
npm install -g @openai/codex

# Verify installation
codex --version
# Should output: codex-cli 0.77.0 or later
```

**Authentication**:

**Option 1: Login with OpenAI Plus/Pro Account**
If you have an OpenAI Pro account, you can log in directly to Codex:
```bash
# Login interactively (will open browser for authentication)
codex login

# Or login with API key from environment variable
echo $OPENAI_API_KEY | codex login --with-api-key

# Check login status
codex login status
```

**Option 2: Use API Key Environment Variable**
Alternatively, you can set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
# Add to ~/.bashrc for persistence
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
```

**Note**: Pro OpenAI users are recommended to use `codex login` for seamless authentication.

### 5. Python Environment
**Version**: Python 3.10.12 (3.8+ should work)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
```

### 6. Python Dependencies
All Python packages are listed in `requirements.txt`. Key dependencies include:

- **openai**: For AI agent interactions (Codex API)
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **torch**: PyTorch (for CUDA support)
- **httpx**: HTTP client for API calls

**Installation**:
```bash
# Install all Python dependencies
pip install -r requirements.txt

# Or create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 7. OpenAI API Access
The pipeline requires access to OpenAI's Codex API.

**Setup Options**:

**Option 1: Login with OpenAI Pro Account (Recommended)**
If you have an OpenAI Pro account, you can authenticate directly:
```bash
# Interactive login (opens browser)
codex login

# Or login with existing API key
echo $OPENAI_API_KEY | codex login --with-api-key

# Verify login
codex login status
```

**Option 2: Use API Key Environment Variable**
1. Obtain an API key from OpenAI (https://platform.openai.com/api-keys)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. Add to `~/.bashrc` for persistence:
   ```bash
   echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
   ```

**Note**: Pro OpenAI users can use `codex login` for easier authentication without managing environment variables.

### 9. OpenMP Libraries
OpenMP runtime libraries are included with the NVIDIA HPC SDK.

**Verification**:
```bash
# OpenMP libraries should be in the HPC SDK lib directory
ls /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib/libomp*

# Test OpenMP compilation
echo '#include <omp.h>
#include <stdio.h>
int main() { 
  #pragma omp parallel
  printf("OpenMP thread %d\\n", omp_get_thread_num());
  return 0;
}' > test_omp.c

nvc++ -mp test_omp.c -o test_omp
./test_omp
rm test_omp test_omp.c
```

**GPU Offload Test**:
```bash
# Test OpenMP GPU offload
echo '#include <omp.h>
#include <stdio.h>
int main() {
  int n = 100;
  int sum = 0;
  #pragma omp target teams distribute parallel for reduction(+:sum)
  for(int i = 0; i < n; i++) sum += i;
  printf("Sum: %d\\n", sum);
  return 0;
}' > test_gpu.c

nvc++ -mp=gpu test_gpu.c -o test_gpu
./test_gpu
rm test_gpu test_gpu.c
```

## Environment Variables

Add these to your `~/.bashrc` or `~/.bash_profile`:

```bash
# Node.js (if using nvm)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# NVIDIA HPC SDK
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH

# CUDA (if installed)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# OpenAI API
export OPENAI_API_KEY="your-api-key-here"

# Optional: Set default GPU
export CUDA_VISIBLE_DEVICES=0
```

## Verification

After setup, verify your environment:

```bash
# Check Node.js and npm
node --version
npm --version

# Check Codex CLI
codex --version

# Check Python
python3 --version

# Check NVIDIA HPC SDK
nvc++ --version

# Check Nsight Systems
nsys --version

# Check CUDA (if installed)
nvcc --version

# Check GPU
nvidia-smi

# Check Python packages
pip list | grep -E "openai|numpy|torch|matplotlib"

# Check OpenMP support
nvc++ -mp -V 2>&1 | grep -i openmp
```

## Quick Setup Script

For convenience, here's a script to verify all dependencies:

```bash
#!/bin/bash
echo "=== Environment Verification ==="
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

echo "OpenAI API Key:"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set"
else
    echo "✅ OPENAI_API_KEY is set"
fi
echo ""

echo "Python packages:"
pip list | grep -E "openai|numpy|torch" || echo "❌ Some Python packages missing"
```

Save this as `verify_environment.sh` and run with `bash verify_environment.sh`.

## Troubleshooting

### Common Issues

1. **`nvc++: command not found`**
   - Ensure NVIDIA HPC SDK is installed and PATH is set correctly
   - Run: `export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH`

2. **`nsys: command not found`**
   - Install NVIDIA Nsight Systems
   - Check if `/usr/local/bin/nsys` or `/opt/nvidia/nsight-systems/bin/nsys` exists

3. **OpenMP offload failures**
   - Verify GPU is accessible: `nvidia-smi`
   - Check CUDA runtime: `nvc++ -gpu=managed -mp=gpu test.c`

4. **Python package conflicts**
   - Use a virtual environment: `python3 -m venv venv && source venv/bin/activate`
   - Reinstall: `pip install -r requirements.txt --force-reinstall`

## Docker Alternative (Coming Soon)

For easier setup, a Docker container with all dependencies pre-installed will be provided.

## Contact

For environment setup issues, please open an issue on GitHub.
