# ParaCodex Quick Start Guide

Get up and running in 5 minutes. For full documentation, see [README.md](README.md).

## 🎯 Prerequisites Checklist

- [ ] NVIDIA GPU with CUDA support
- [ ] Node.js 22+ and npm 9+
- [ ] NVIDIA HPC SDK 25.7+ (`nvc++` compiler)
- [ ] Python 3.8+
- [ ] OpenAI API key or OpenAI Pro account

## ⚡ Quick Setup

### 1. Clone and navigate
```bash
git clone https://github.com/Scientific-Computing-Lab/ParaCodex.git
cd paracodex
chmod +x setup_environment.sh install_nvidia_hpc_sdk.sh verify_environment.sh
```

### 2. Install NVIDIA HPC SDK (if not already installed)
```bash
sudo ./install_nvidia_hpc_sdk.sh
source ~/.bashrc
```

### 3. Run the environment setup
```bash
./setup_environment.sh
```

This installs all dependencies in one step: Codex CLI, Python packages, webapp dependencies, and copies the ParaCodex Codex skills to `~/.codex/skills/paracodex/`.

### 4. Set up OpenAI access

**Option 1: OpenAI Pro account (recommended)**
```bash
codex login
```

**Option 2: API key**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 5. Verify your environment
```bash
./verify_environment.sh
```

## 🚀 Running ParaCodex

```bash
cd pipeline/webapp
bash start.sh
```

Open **http://localhost:5000** in your browser.

### Using the web interface

1. **Source Directory** — Browse to the folder with the code you want to translate
2. **Source API** — The parallel API of your code (e.g. `serial`, `cuda`)
3. **Target API** — The API to translate to (e.g. `omp`, `hip`)
4. Click **Start Pipeline** — live logs stream in as the agent works
5. Inspect translated code and profiling artifacts in the job view

### Benchmark workdirs

| Suite | Source directory |
|-------|-----------------|
| Rodinia | `workdirs/serial_omp_rodinia_workdir/data/src/` |
| NAS | `workdirs/serial_omp_nas_workdir/data/src/` |
| HeCBench | `workdirs/serial_omp_hecbench_workdir/data/src/` |
| ParEval | `workdirs/cuda_omp_pareval_workdir/data/src/` |

## 🔧 Common Issues

**`codex: command not found`**
```bash
npm install -g @openai/codex
```

**`node: command not found`**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 22
```

**`nvc++: command not found`**
```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH
```

**Port 5000 already in use**
```bash
FLASK_RUN_PORT=8080 python pipeline/webapp/app.py
```

**GPU not accessible**
```bash
nvidia-smi
# If this fails, check NVIDIA driver installation
```

## 📚 Next Steps

- See [README.md](README.md) for detailed usage, troubleshooting, and benchmark workflows
- See [pipeline/AGENTS.md](pipeline/AGENTS.md) for agent documentation
- See [pipeline/prompts/](pipeline/prompts/) for AI prompt documentation
