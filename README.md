# 🚀 ParaCodex: A Profiling-Guided Autonomous Coding Agent for Reliable Parallel Code Generation and Translation

![ParaCodex Demo](paracodex_demo.gif)

A framework for translating code between serial and parallel programming models (OpenMP, CUDA, HIP, SYCL, OpenCL) using AI agents, with automated performance optimization and correctness verification. Supports Rodinia, NAS, HeCBench, ParEval, and custom benchmarks.

## 📋 Overview

ParaCodex implements a complete pipeline for:
- 🔄 **Code Translation**: Converting between serial C/C++ code and parallel implementations (OpenMP, CUDA, HIP, SYCL, OpenCL) using AI agents
- 🔀 **Cross-Parallel Translation**: Translating between different parallel programming models (e.g., CUDA → OpenMP)
- ⚡ **Performance Optimization**: Multi-stage optimization with GPU offloading and profiling
- ✅ **Correctness Verification**: Automated testing to ensure numerical equivalence

## 🏗️ Project Structure

```
paracodex/
├── pipeline/                            # Core translation and optimization pipeline
│   ├── agents/                          # AI agent implementations
│   ├── prompts/                         # AI prompts for each translation path
│   ├── utils/                           # Pipeline utilities (config, logging, codex runner)
│   ├── gate_sdk/                        # GATE SDK for correctness verification
│   ├── scripts/                         # Helper scripts
│   ├── webapp/                          # Web interface (primary way to run ParaCodex)
│   │   ├── app.py                       # Flask server
│   │   ├── start.sh                     # Convenience startup script
│   │   └── static/                      # HTML/CSS/JS frontend
│   ├── package.json                     # Node.js dependencies (@openai/codex-sdk)
│   └── AGENTS.md                        # Agent role definitions
├── paracodex_skills/                    # Codex CLI skills for each translation path
│   └── paracodex/                       # Skills installed to ~/.codex/skills/ on setup
├── workdirs/                            # Working directories for each benchmark suite
│   ├── serial_omp_rodinia_workdir/      # Rodinia benchmark workspace
│   ├── serial_omp_nas_workdir/          # NAS benchmark workspace
│   ├── serial_omp_hecbench_workdir/     # HeCBench workspace
│   └── cuda_omp_pareval_workdir/        # ParEval CUDA/OpenMP workspace
├── results/                             # Results and performance data
├── setup_environment.sh                 # Main environment setup script
├── install_nvidia_hpc_sdk.sh            # Automated NVIDIA HPC SDK installer
├── verify_environment.sh                # Environment verification tool
├── kill_gpu_processes.py/sh             # GPU process management utilities
└── requirements.txt                     # Python dependencies
```

## ✨ Key Features

### 🤖 AI-Powered Translation
- **Multi-Agent Pipeline**: Specialized AI agents for translation, optimization, and verification
- **Serial-to-Parallel Translation**: Converting serial code to OpenMP and CUDA implementations
- **Cross-Parallel Translation**: Translating between different parallel programming models
- **GPU Offloading**: Automatic translation to OpenMP with GPU acceleration
- **CUDA Implementation**: Direct CUDA kernel generation and optimization

### 🔧 Multi-Stage Optimization
- **2-Stage Process**: Systematic optimization from correctness to performance
- **GPU Profiling**: Integration with NVIDIA Nsight Systems (nsys) for detailed analysis
- **Performance Tracking**: Continuous monitoring of optimization progress

### ✅ Correctness Verification
- **GATE SDK Integration**: Automated numerical correctness checking
- **Supervisor Agent**: AI-powered code repair and correctness enforcement
- **Numerical Equivalence**: Ensures translated code produces identical results

### 🌐 Web Interface
- **Browser-Based UI**: No CLI required — select source directory, pick API pair, click Start
- **Live Log Streaming**: Watch the agent work in real time
- **Job Management**: Run multiple translations, inspect artifacts per job

## 🚀 Quick Start

### 📋 Prerequisites

#### Core Requirements
- **Node.js 22+** and **npm**: Required for the Codex agent engine
- **Python 3.8+**: For the pipeline and web server
- **OpenAI Codex CLI**: Installed automatically by `setup_environment.sh`
- **OpenAI API Access**:
  - **Pro/Plus Users**: `codex login` (recommended)
  - **API Key Users**: Set `OPENAI_API_KEY` environment variable

#### Compilation and GPU Tools
- **NVIDIA HPC SDK**: For OpenMP GPU offloading (`nvc++` compiler)
- **NVIDIA Nsight Systems (nsys)**: For GPU profiling (bundled with HPC SDK)
- **NVIDIA GPU**: With OpenMP offloading support

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Scientific-Computing-Lab/ParaCodex.git
cd paracodex

# Make scripts executable
chmod +x setup_environment.sh install_nvidia_hpc_sdk.sh verify_environment.sh

# Step 1: Install NVIDIA HPC SDK (if not already installed)
sudo ./install_nvidia_hpc_sdk.sh
source ~/.bashrc

# Step 2: Run the main environment setup
# This installs all dependencies AND copies the ParaCodex Codex skills
./setup_environment.sh

# Step 3: Verify your installation
./verify_environment.sh
```

`setup_environment.sh` handles:
- ✅ Node.js / npm check
- ✅ Codex CLI install (`@openai/codex`)
- ✅ Python dependencies (`requirements.txt`)
- ✅ Pipeline Node.js dependencies (`pipeline/package.json`)
- ✅ Webapp Python dependencies (`pipeline/webapp/requirements.txt`)
- ✅ ParaCodex Codex skills → `~/.codex/skills/paracodex/`
- ✅ NVIDIA HPC SDK / nsys / GPU checks
- ✅ OpenAI API key / Codex login check

#### Manual NVIDIA HPC SDK install

```bash
sudo ./install_nvidia_hpc_sdk.sh
source ~/.bashrc
```

- Downloads NVIDIA HPC SDK v25.7 (~3GB)
- Installs to `/opt/nvidia/hpc_sdk`
- Configures `PATH` and environment variables in `~/.bashrc`
- Tests OpenMP CPU and GPU offload support

#### Set Up OpenAI Access

**Option 1: Login with OpenAI Pro Account (Recommended)**
```bash
codex login
# or with API key:
echo $OPENAI_API_KEY | codex login --with-api-key
```

**Option 2: Environment Variable**
```bash
export OPENAI_API_KEY='your-api-key-here'
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.bashrc
```

### 💻 Running ParaCodex

ParaCodex is run through its **web interface**:

```bash
cd pipeline/webapp
bash start.sh
```

Then open **http://localhost:5000** in your browser.

#### Using the Web Interface

1. **Source Directory** — Browse to the folder containing the code you want to translate
2. **Source API** — The parallel API currently in your code (e.g. `serial`, `cuda`)
3. **Target API** — The API to translate to (e.g. `omp`, `hip`)
4. Click **Start Pipeline** and watch live logs stream in
5. Inspect translated code and profiling artifacts per job

#### Supported Translation Paths

| From | To |
|------|----|
| serial | omp, cuda, ocl, hip, sycl |
| cuda | omp, hip, sycl, ocl |
| omp | cuda, serial |
| … | … |

#### Engine Support

| Engine | Status |
|--------|--------|
| Codex (OpenAI) | ✅ Supported |
| OpenCode | 🔜 Coming soon |

### 🎯 Benchmark-Specific Workflows

Each benchmark suite has a pre-configured `workdir` containing the source kernels, GATE SDK, and golden reference implementations:

| Suite | Workdir | Kernels |
|-------|---------|---------|
| Rodinia | `workdirs/serial_omp_rodinia_workdir/` | nw, srad, lud, b+tree, backprop, bfs, hotspot |
| NAS | `workdirs/serial_omp_nas_workdir/` | cg, ep, ft, mg |
| HeCBench | `workdirs/serial_omp_hecbench_workdir/` | 26 kernels |
| ParEval | `workdirs/cuda_omp_pareval_workdir/` | 4 kernels (CUDA→OMP) |

Point the webapp's **Source Directory** to `workdirs/<suite>/data/src/` for each benchmark.

## 🛠️ Setup Scripts

### `setup_environment.sh`
Main dependency installer and environment checker. Run this after cloning.

### `install_nvidia_hpc_sdk.sh`
Automated installer for NVIDIA HPC SDK v25.7 with OpenMP GPU offload support.

### `verify_environment.sh`
Verifies all dependencies are installed and shows version info.

```bash
./verify_environment.sh
```

### `kill_gpu_processes.sh` / `kill_gpu_processes.py`
Terminates stale GPU processes.

```bash
./kill_gpu_processes.sh
```

## 🆘 Troubleshooting

### NVIDIA HPC SDK not found
```bash
sudo ./install_nvidia_hpc_sdk.sh
source ~/.bashrc
```
If installation fails: check disk space (need 10GB+), sudo privileges, and internet connection (~3GB download).

### Codex CLI not found
```bash
npm install -g @openai/codex
```

### OpenAI API key / auth issues
```bash
codex login           # Pro users
# or
export OPENAI_API_KEY='your-api-key'
```

### `nvc++` installed but not in PATH
```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH
echo 'export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Port 5000 already in use
```bash
FLASK_RUN_PORT=8080 python pipeline/webapp/app.py
```

### Python package errors
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r pipeline/webapp/requirements.txt
```

---

**Note**: ParaCodex is designed for research and development with multiple benchmark suites. Ensure you have an NVIDIA GPU with OpenMP offloading support and NVIDIA HPC SDK before use.

**Supported Benchmarks**: Rodinia (nw, srad, lud, b+tree, backprop, bfs, hotspot), NAS (cg, ep, ft, mg), HeCBench (26 kernels), ParEval (4 CUDA→OMP kernels).
