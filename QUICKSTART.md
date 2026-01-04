# ParaCodex Quick Start Guide

This is a condensed guide to get you up and running quickly. For detailed information, see [README.md](README.md).

## 🎯 Prerequisites Checklist

Before running ParaCodex, ensure you have:

- [ ] NVIDIA GPU with CUDA support
- [ ] Node.js 22+ and npm 9+ installed
- [ ] Codex CLI installed (`npm install -g @openai/codex`)
- [ ] NVIDIA HPC SDK 25.7+ installed (`nvc++` compiler)
- [ ] NVIDIA Nsight Systems 2025.3+ installed (`nsys` profiler)
- [ ] Python 3.8+ installed
- [ ] OpenAI API key

## ⚡ Quick Setup (5 minutes)

### 1. Clone and Navigate
```bash
git clone https://github.com/Scientific-Computing-Lab/ParaCodex.git
cd ParaCodex
```

### 2. Verify Environment
```bash
./verify_environment.sh
```

If you see ❌ errors, you can auto-install missing dependencies:

```bash
# Auto-install NVIDIA HPC SDK (if missing)
sudo ./install_nvidia_hpc_sdk.sh

# Note: Nsight Systems (nsys) is bundled with HPC SDK - no separate install needed!
# If nsys is not found after HPC SDK installation, ensure HPC SDK is in your PATH.
```

### 3. Install Node.js Dependencies
```bash
# Install Codex CLI
npm install -g @openai/codex

# Verify
codex --version
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up OpenAI API Access

**Option 1: Login with OpenAI Pro Account (Recommended)**
```bash
# Interactive login (opens browser)
codex login

# Or login with API key
echo $OPENAI_API_KEY | codex login --with-api-key

# Verify login
codex login status
```

**Option 2: Use API Key Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 6. Test Your Setup
```bash
# Check Node.js and Codex CLI
node --version
codex --version

# Check compiler
nvc++ --version

# Check profiler
nsys --version

# Check GPU
nvidia-smi

# Check Python packages
python3 -c "import openai; print('OpenAI package OK')"
```

## 🚀 Running Your First Translation

### Example 1: Serial to OpenMP (Single Kernel)
```bash
python pipeline/initial_translation_codex.py \
    --source-api serial \
    --target-api omp \
    --kernels jacobi \
    --results-dir ./results
```

### Example 2: Serial to OpenMP with Optimization
```bash
python pipeline/initial_translation_codex.py \
    --source-api serial \
    --target-api omp \
    --kernels jacobi \
    --results-dir ./results \
    --optimize
```

### Example 3: Full Pipeline (Translation + Optimization + Correctness Checking)
```bash
python pipeline/initial_translation_codex.py \
    --source-api serial \
    --target-api omp \
    --results-dir ./results \
    --optimize \
    --supervise \
    --opt-supervisor-steps 2
```

### Example 4: Translate All Kernels
```bash
# Remove --kernels flag to process all kernels in the jsonl file
python pipeline/initial_translation_codex.py \
    --source-api serial \
    --target-api omp \
    --results-dir ./results \
    --optimize
```

## 📊 Checking Results

After translation, check the results directory:

```bash
ls -la results/

# View the translated code
cat results/jacobi-omp/initial/main.c

# View optimization steps
cat results/jacobi-omp/step1/main.c
cat results/jacobi-omp/step2/main.c

# View profiling data
cat results/jacobi-omp/step1/nsys_relevant.txt

# View AI agent reasoning
cat results/jacobi-omp/analysis.md
cat results/jacobi-omp/data_plan.md
cat results/jacobi-omp/optimization_plan.md
```

## 🐳 Docker Alternative (Even Faster!)

If you have Docker and NVIDIA Container Toolkit:

```bash
# Clone
git clone https://github.com/Scientific-Computing-Lab/ParaCodex.git
cd paracodex

# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Build and run
docker build -t paracodex:latest .
docker run --gpus all -it -e OPENAI_API_KEY="$OPENAI_API_KEY" -v $(pwd):/workspace paracodex:latest

# Inside container, run translations
python pipeline/initial_translation_codex.py --source-api serial --target-api omp --optimize
```

## 🔧 Common Issues

### Issue: `codex: command not found`
**Solution**: Install Codex CLI via npm:
```bash
npm install -g @openai/codex
# If npm is not found, install Node.js first
```

### Issue: `node: command not found`
**Solution**: Install Node.js:
```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 22

# Or using apt (Ubuntu)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Issue: `nvc++: command not found`
**Solution**: Add NVIDIA HPC SDK to PATH:
```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH
```

### Issue: `nsys: command not found`
**Solution**: Install NVIDIA Nsight Systems from [NVIDIA website](https://developer.nvidia.com/nsight-systems).

### Issue: `OPENAI_API_KEY not set` or Codex authentication failed
**Solution**: 
```bash
# Option 1: Login with OpenAI Pro account (recommended)
codex login

# Option 2: Set API key environment variable
export OPENAI_API_KEY="your-api-key-here"

# Verify authentication
codex login status
```

### Issue: GPU not accessible
**Solution**: Check GPU status:
```bash
nvidia-smi
# If this fails, check NVIDIA driver installation
```

## 📚 Next Steps

- Read [README.md](README.md) for detailed usage examples and setup instructions
- Check `pipeline/prompts.md` for AI prompt documentation (if available)
- Explore example results in the `results/` directory

## 🆘 Getting Help

- Run `./verify_environment.sh` to diagnose setup issues
- Check [README.md](README.md) for troubleshooting
- Open an issue on GitHub for support

---

**Ready to parallelize some code? Start with a simple kernel and work your way up!** 🚀

