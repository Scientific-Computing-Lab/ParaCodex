# 🚀 ParaCodex: A Profiling-Guided Autonomous Coding Agent for Reliable Parallel Code Generation and Translation

A comprehensive framework for translating benchmark code between serial and parallel implementations (OpenMP, CUDA) and between different parallel programming models using AI agents, with automated performance testing and correctness verification. Supports Rodinia, NAS, HeCBench, and custom benchmarks.

## 📋 Overview

This repository implements a complete pipeline for:
- 🔄 **Code Translation**: Converting between serial C/C++ code and parallel implementations (OpenMP, CUDA) using AI agents
- 🔀 **Cross-Parallel Translation**: Translating between different parallel programming models (e.g., CUDA to OpenMP)
- ⚡ **Performance Optimization**: Multi-stage optimization with GPU offloading and profiling
- ✅ **Correctness Verification**: Automated testing to ensure numerical equivalence

## 🏗️ Project Structure

```
paracodex/
├── pipeline/                                    # Core translation and optimization pipeline
│   ├── initial_translation_codex.py             # Initial code translation using AI
│   ├── optimize_codex.py                        # Multi-stage optimization pipeline
│   ├── supervisor_codex.py                      # Correctness verification agent
│   ├── path_config.py                           # Path configuration and Codex CLI helpers
│   ├── SERIAL_OMP_PROMPTS.md                    # AI prompts for serial-to-OpenMP translation
│   ├── CUDA_PROMPTS.md                          # AI prompts for CUDA translation
│   ├── combined_serial_filenames.jsonl          # Serial kernel listings
│   ├── combined_omp_filenames.jsonl             # OpenMP kernel listings
│   ├── combined_cuda_filenames.jsonl            # CUDA kernel listings
│   └── combined_omp_pareval_filenames.jsonl     # ParEval benchmark listings
├── performance_testers/                         # Performance testing and benchmarking tools
│   └── performance_comparison.py                # Performance comparison utilities
├── utils/                                       # Utility scripts
│   └── clean_kernel_dirs.py                     # Cleanup utilities
├── workdirs/                                    # Working directories for different benchmarks
│   ├── serial_omp_rodinia_workdir/              # Rodinia benchmark workspace
│   │   ├── data/                                # Source code and benchmarks (parallel versions)
│   │   │   └── src/                             # Kernel directories (e.g., nw-omp, lud-omp)
│   │   ├── gate_sdk/                            # GATE SDK for correctness verification
│   │   ├── golden_labels/                       # Reference serial implementations
│   │   │   └── src/                             # Serial kernel directories
│   │   └── serial_kernels_changedVars/          # Transformed serial kernels
│   │       └── src/                             # Modified serial kernels for translation
│   ├── serial_omp_nas_workdir/                  # NAS benchmark workspace
│   ├── serial_omp_hecbench_workdir/             # HeCBench workspace
│   └── cuda_omp_pareval_workdir/                # ParEval CUDA/OpenMP workspace
├── results/                                     # Results and performance data
├── setup_environment.sh                         # Main environment setup script
├── install_nvidia_hpc_sdk.sh                    # Automated NVIDIA HPC SDK installer
├── verify_environment.sh                        # Environment verification tool
├── kill_gpu_processes.py/sh                     # GPU process management utilities
└── requirements.txt                             # Python dependencies
```

## ✨ Key Features

### 🤖 AI-Powered Translation
- **Multi-Agent Pipeline**: Specialized AI agents for translation, optimization, and verification
- **Serial-to-Parallel Translation**: Converting serial code to OpenMP and CUDA implementations
- **Cross-Parallel Translation**: Translating between different parallel programming models
- **Intelligent Analysis**: Automatic hotspot identification and offload target selection
- **GPU Offloading**: Automatic translation to OpenMP with GPU acceleration
- **CUDA Implementation**: Direct CUDA kernel generation and optimization

### 🔧 Multi-Stage Optimization
- **2-Stage Process**: Systematic optimization from correctness to performance (GPU offload + performance tuning)
- **GPU Profiling**: Integration with NVIDIA Nsight Systems (nsys) for detailed analysis
- **Retry Mechanisms**: Robust error handling with automatic retry logic
- **Performance Tracking**: Continuous monitoring of optimization progress
- **Cyclic Optimization**: Iterative refinement until target performance is achieved

### ✅ Correctness Verification
- **GATE SDK Integration**: Automated numerical correctness checking
- **Reference Comparison**: Validation against golden reference implementations
- **Supervisor Agent**: AI-powered code repair and correctness enforcement
- **Numerical Equivalence**: Ensures translated code produces identical results

### 📊 Performance Evaluation
- **Comprehensive Benchmarking**: CPU and GPU performance testing
- **Performance Comparison**: Side-by-side analysis of different implementations
- **Results Visualization**: JSON output for easy integration with analysis tools
- **Automated Testing**: Batch processing of multiple kernels and configurations

## 🚀 Quick Start

### 📋 Prerequisites

#### Core Requirements
- **Node.js 22+** and **npm**: For Codex CLI
- **Python 3.8+**: For the pipeline scripts
- **OpenAI Codex CLI**: For AI agent interactions (`npm install -g @openai/codex`)
- **OpenAI API Key**: Set as `OPENAI_API_KEY` environment variable

#### Compilation and GPU Tools
- **NVIDIA HPC SDK**: For OpenMP GPU offloading (nvc++ compiler)
- **CUDA Toolkit**: For CUDA development
- **NVIDIA Nsight Systems (nsys)**: For GPU profiling
- **NVIDIA GPU**: With OpenMP offloading support

### ⚙️ Installation

#### Quick Setup (Automated)

We provide automated setup scripts to streamline the installation process:

```bash
# Clone the repository
git clone <repository-url>
cd paracodex

# Make scripts executable
chmod +x setup_environment.sh install_nvidia_hpc_sdk.sh verify_environment.sh

# Step 1: Install NVIDIA HPC SDK (if not already installed)
sudo ./install_nvidia_hpc_sdk.sh

# Step 2: Run the main environment setup
./setup_environment.sh

# Step 3: Verify your installation
./verify_environment.sh
```

#### 🔧 Setup Scripts Overview

**1. `install_nvidia_hpc_sdk.sh`** - NVIDIA HPC SDK Automated Installer
   
   This script automates the download and installation of NVIDIA HPC SDK v25.7:
   - ✅ Downloads NVIDIA HPC SDK (~3GB)
   - ✅ Verifies system requirements (10GB disk space, x86_64 architecture)
   - ✅ Installs compilers (nvc++, nvfortran) with OpenMP GPU offload support
   - ✅ Configures PATH and environment variables in `~/.bashrc`
   - ✅ Tests OpenMP CPU and GPU offload support
   - ⚠️ Requires sudo privileges
   
   ```bash
   sudo ./install_nvidia_hpc_sdk.sh
   source ~/.bashrc  # Activate the new environment
   ```

**2. `setup_environment.sh`** - Main Environment Setup Script
   
   Checks and installs all ParaCodex dependencies:
   - ✅ Check for Node.js v22+ and npm
   - ✅ Install Codex CLI (`@openai/codex`) if missing
   - ✅ Verify Python 3.8+ installation
   - ✅ Install Python dependencies from `requirements.txt`
   - ✅ Check for NVIDIA HPC SDK (nvc++)
   - ✅ Check for Nsight Systems (nsys)
   - ✅ Verify GPU accessibility
   - ✅ Confirm OpenAI API key is set
   - 📊 Provides summary of missing dependencies
   
   ```bash
   ./setup_environment.sh
   ```

**3. `verify_environment.sh`** - Environment Verification Tool
   
   Comprehensive check of your ParaCodex environment:
   - ✅ Verifies all required tools are installed and accessible
   - ✅ Shows version information for each component
   - ✅ Tests OpenMP support
   - ✅ Checks GPU accessibility with detailed info
   - ✅ Validates Python package installations
   - 📊 Provides a summary of any missing dependencies
   
   ```bash
   ./verify_environment.sh
   ```

#### Manual Setup

If you prefer manual installation or need to install specific components:

##### 1. Install NVIDIA HPC SDK

**Option A: Automated (Recommended)**
```bash
sudo ./install_nvidia_hpc_sdk.sh
source ~/.bashrc
```

**Option B: Manual**
- Download from: https://developer.nvidia.com/hpc-sdk
- Install version 25.7 for CUDA 12.6 support
- Add to PATH:
  ```bash
  export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
  export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH
  export MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/man:$MANPATH
  ```
- Add to `~/.bashrc` for persistence

##### 2. Install Node.js and npm

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 22

# Or using apt
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
```

##### 3. Install Codex CLI

```bash
npm install -g @openai/codex
```

##### 4. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
# Add to ~/.bashrc for persistence
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.bashrc
```

##### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

##### 6. Install NVIDIA Nsight Systems

- Download from: https://developer.nvidia.com/nsight-systems
- Or use package manager if available

#### Verify Installation

After setup, verify your environment is correctly configured:

```bash
./verify_environment.sh
```

This will check and display:
- ✅ Node.js and npm versions
- ✅ Codex CLI installation
- ✅ Python version and key packages (openai, numpy, torch, matplotlib)
- ✅ NVIDIA HPC SDK (nvc++) version
- ✅ Nsight Systems (nsys) installation
- ✅ CUDA toolkit (optional)
- ✅ GPU information (name, driver, memory)
- ✅ OpenMP support
- ✅ OpenAI API key configuration

**Expected output for a properly configured system:**
```
✅ All core dependencies are installed and configured
```

**If you see missing dependencies:**
```
❌ X core dependencies are missing
   Run './setup_environment.sh' or see installation instructions
```

Manual verification commands:
```bash
codex --version
python3 --version
nvc++ --version
nsys --version
nvidia-smi
echo $OPENAI_API_KEY
```

#### Python Dependencies

Key Python packages (see `requirements.txt` for full list):
- `openai>=2.8.1` - OpenAI API client
- `pandas>=2.3.3` - Data processing
- `numpy>=2.1.2` - Numerical computing
- `matplotlib>=3.10.6` - Visualization
- `seaborn>=0.13.2` - Statistical visualization
- `tqdm>=4.67.1` - Progress bars
- NVIDIA CUDA libraries for GPU support

### 💻 Basic Usage

#### Initial Setup

After installation, ensure your working directory is properly configured:

```bash
# The workdirs contain benchmark-specific source code and configurations
# For Rodinia benchmarks, use workdirs/serial_omp_rodinia_workdir/
# Ensure proper directory structure:
#   - data/src/ containing parallel kernel directories (e.g., nw-omp, lud-omp)
#   - golden_labels/src/ containing serial reference implementations
#   - serial_kernels_changedVars/src/ containing transformed serial kernels (optional)
# Ensure jsonl file with kernel names present in pipeline/combined_*_filenames.jsonl
# Ensure `golden_labels/src` exists if you want to run the supervisor agent
```

#### 🔄 1. Initial Translation

Translate Rodinia benchmarks from serial to OpenMP:

```bash
# To translate all the kernels in the jsonl file run without --kernels
# Serial to OpenMP for Rodinia benchmarks
python pipeline/initial_translation_codex.py \
    --codex-workdir /path/to/paracodex/workdirs/serial_omp_rodinia_workdir/ \
    --source-api serial \
    --target-api omp \
    --kernels nw,srad,lud
```

**Note**: The `--codex-workdir` flag specifies the working directory containing your benchmark kernels. The output will be saved to `pipeline/rodinia_outputs/` (or appropriate benchmark output directory) by default.

#### 2. Translation with optimization

```bash
# Serial to OpenMP with optimization
python pipeline/initial_translation_codex.py \
    --codex-workdir /path/to/paracodex/workdirs/serial_omp_rodinia_workdir/ \
    --source-api serial \
    --target-api omp \
    --optimize
```

#### ⚡ 3. Translation with supervision (correctness gate) after initial translation

```bash
# Serial to OpenMP with supervision for correctness verification
python pipeline/initial_translation_codex.py \
    --codex-workdir /path/to/paracodex/workdirs/serial_omp_rodinia_workdir/ \
    --source-api serial \
    --target-api omp \
    --supervise
```

This will create `initial_supervised_*` files in the output directory, including:
- `initial_supervised_ncu_output.txt` - Full nsys profiling output
- `initial_supervised_ncu_relevant.txt` - Extracted GPU performance metrics
- `initial_supervised_compilation.txt` - Compilation logs
- `initial_supervised_output.txt` - Execution output

#### 🔧 4. Translation with optimization and supervision after optimization steps

```bash
# Serial to OpenMP with optimization and supervision after optimization steps
python pipeline/initial_translation_codex.py \
    --codex-workdir /path/to/paracodex/workdirs/serial_omp_rodinia_workdir/ \
    --source-api serial \
    --target-api omp \
    --optimize \
    --supervise \
    --opt-supervisor-steps 2
```

This will:
1. Perform initial translation
2. Run supervision (correctness verification)
3. Run optimization steps (step1, step2)
4. Run supervision after specified optimization steps (creates `step2_supervised/` directory)
#### 📁 Running all the steps will result in a folder with the following structure:

```
pipeline/rodinia_outputs/
├── {kernel_name}-{target_api}/                # Per-kernel results (e.g., nw-omp, lud-omp)
│   ├── compilation_result.txt                 # Initial compilation result
│   ├── initial_compilation.txt                # Initial compilation result
│   ├── initial_transcript.txt                 # Initial translation transcript
│   ├── initial_transcript_summary.txt         # Summary of initial translation
│   ├── {file}_initial.c                       # Initial translated code (root level)
│   ├── initial/                               # Initial translation directory
│   │   └── {file}.c                           # Initial translated code
│   ├── initial_supervised_ncu_output.txt       # Full nsys profiling output (if --supervise)
│   ├── initial_supervised_ncu_relevant.txt    # Extracted GPU metrics (if --supervise)
│   ├── initial_supervised_compilation.txt     # Compilation logs (if --supervise)
│   ├── initial_supervised_output.txt          # Execution output (if --supervise)
│   ├── initial_correct/                       # After supervisor correction
│   │   └── {file}.c                           # Supervised initial code
│   ├── step1/                                 # Optimization step 1
│   │   ├── {file}.c                           # Code after step 1
│   │   ├── transcript.txt                     # AI agent transcript
│   │   ├── transcript_summary.txt             # Transcript summary
│   │   ├── nsys_output.txt                    # Full nsys profiling output
│   │   └── nsys_relevant.txt                  # Extracted relevant nsys metrics
│   ├── step2/                                 # Optimization step 2
│   │   ├── {file}.c                           # Code after step 2
│   │   ├── transcript.txt                     # AI agent transcript
│   │   ├── transcript_summary.txt             # Transcript summary
│   │   ├── nsys_output.txt                    # Full nsys profiling output
│   │   └── nsys_relevant.txt                  # Extracted relevant nsys metrics
│   ├── step2_supervised/                      # Supervision after step 2 (if --opt-supervisor-steps 2)
│   │   ├── {file}.c                           # Supervised code after step 2
│   │   ├── supervised_nsys_output.txt         # Full nsys profiling output
│   │   └── supervised_nsys_relevant.txt       # Extracted GPU metrics
│   └── optimized/                             # Final optimized code
│       └── {file}.c                           # Final optimized code
├── {kernel2_name}-{target_api}/               # Results for second kernel
│   └── [same structure as above]
└── [additional kernels...]                     # Results for other kernels
```

**Key Artifacts Explained:**
- **Source Code Snapshots**: Versioned code at each optimization stage (in `initial/`, `step1/`, `step2/`, `optimized/` directories)
- **Transcripts**: AI agent conversations and decision logs (`initial_transcript.txt`, `step*/transcript.txt`)
- **Nsys Outputs**: GPU profiling data (`step*/nsys_output.txt`, `initial_supervised_ncu_output.txt`) and extracted metrics (`step*/nsys_relevant.txt`, `initial_supervised_ncu_relevant.txt`)
- **Supervised Files**: Correctness-verified code and performance metrics from supervision phase

#### 📊 5. Running performance test (against golden label parallel code)
```bash
python performance_testers/performance_comparison.py \
    --candidate_dir <your path to the parent directory of translated code> \
    --reference_dir <parent directory of golden label parallel code> \
    --output <output directory for generated artifacts>
```
#### 💡 Example
```bash
python performance_testers/performance_comparison.py \
    --candidate_dir /path/to/paracodex/pipeline/rodinia_outputs \
    --reference_dir /path/to/paracodex/workdirs/serial_omp_rodinia_workdir/data/src \
    --output /path/to/paracodex/results/perf_rodinia_nsys
```
### 🛤️ Supported Translation Paths

The framework supports the following translation paths across multiple benchmark suites:
- **Serial → OpenMP**: Converting serial code to OpenMP with GPU offloading (primary use case for Rodinia, NAS, HeCBench)
- **Serial → CUDA**: Converting serial code to CUDA kernels
- **OpenMP → CUDA**: Translating OpenMP code to CUDA implementations
- **CUDA → OpenMP**: Converting CUDA kernels to OpenMP with GPU offloading (ParEval benchmarks)

### 🎯 Benchmark-Specific Workflows

#### Rodinia Benchmarks
For Rodinia benchmarks, the typical workflow is:

1. **Prepare serial kernels**: Place serial reference implementations in `workdirs/serial_omp_rodinia_workdir/golden_labels/src/{kernel}-serial/`
2. **Optional transformations**: Apply variable renaming, comment stripping, and reorderings to create `serial_kernels_changedVars/src/{kernel}-serial/`
3. **Run translation**: Use `initial_translation_codex.py` with `--codex-workdir` pointing to `workdirs/serial_omp_rodinia_workdir/`
4. **Verify correctness**: Use `--supervise` flag to run GATE SDK correctness checks
5. **Optimize performance**: Use `--optimize` flag for multi-stage GPU optimization
6. **Review metrics**: Check `*_ncu_relevant.txt` or `*_nsys_relevant.txt` files for GPU performance metrics

#### NAS Benchmarks
For NAS benchmarks, use `workdirs/serial_omp_nas_workdir/` as the working directory with similar workflow steps.

#### HeCBench Benchmarks
For HeCBench benchmarks, use `workdirs/serial_omp_hecbench_workdir/` as the working directory.

#### ParEval Benchmarks
For ParEval CUDA/OpenMP translation, use `workdirs/cuda_omp_pareval_workdir/` as the working directory.

## 🛠️ Utility Scripts

ParaCodex includes several utility scripts to simplify environment management and troubleshooting:

### Setup and Verification Scripts

#### `setup_environment.sh`
Main environment setup and dependency checker. Checks for all required tools and installs missing Python dependencies.

**Usage:**
```bash
./setup_environment.sh
```

**What it does:**
- Checks Node.js, npm, and Codex CLI
- Installs Codex CLI if npm is available
- Verifies Python and pip
- Installs Python packages from `requirements.txt`
- Checks NVIDIA HPC SDK, Nsight Systems, and GPU
- Validates OpenAI API key
- Provides summary of missing dependencies

#### `install_nvidia_hpc_sdk.sh`
Automated installer for NVIDIA HPC SDK v25.7 with OpenMP GPU offload support.

**Usage:**
```bash
sudo ./install_nvidia_hpc_sdk.sh
source ~/.bashrc
```

**Features:**
- Downloads NVIDIA HPC SDK v25.7 (~3GB)
- Checks system requirements (10GB disk, x86_64 arch)
- Installs to `/opt/nvidia/hpc_sdk`
- Configures environment variables automatically
- Tests OpenMP CPU and GPU offload support
- Colorful progress indicators

**Requirements:** sudo access, 10GB free disk space

#### `verify_environment.sh`
Comprehensive environment verification tool that checks all dependencies and displays detailed version information.

**Usage:**
```bash
./verify_environment.sh
```

**Checks:**
- Node.js, npm, Codex CLI versions
- Python version and key packages
- NVIDIA HPC SDK (nvc++) and OpenMP support
- Nsight Systems and CUDA toolkit
- GPU info (name, driver, memory)
- OpenAI API key configuration

### GPU Management Scripts

#### `kill_gpu_processes.sh` / `kill_gpu_processes.py`
Utilities to terminate GPU processes when needed (e.g., clearing hung processes during development).

**Usage:**
```bash
./kill_gpu_processes.sh
# or
python3 kill_gpu_processes.py
```

### Cleanup Utilities

#### `utils/clean_kernel_dirs.py`
Cleans up generated files in kernel directories during development.

**Usage:**
```bash
python3 utils/clean_kernel_dirs.py --help
```



## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the prompts documentation:
  - `pipeline/SERIAL_OMP_PROMPTS.md` for serial-to-OpenMP translation
  - `pipeline/CUDA_PROMPTS.md` for CUDA-related translations

### Troubleshooting

#### Environment Issues

**Using the automated scripts:**

If `setup_environment.sh` reports missing dependencies, install them as indicated. Common issues:

1. **NVIDIA HPC SDK not found**: 
   ```bash
   sudo ./install_nvidia_hpc_sdk.sh
   source ~/.bashrc
   ```
   
   If installation fails:
   - Check disk space (need 10GB+)
   - Verify you have sudo privileges
   - Check internet connection (downloads ~3GB)
   - Manually download from https://developer.nvidia.com/hpc-sdk

2. **Codex CLI not found**: 
   ```bash
   npm install -g @openai/codex
   ```
   If this fails, ensure Node.js and npm are installed first.

3. **OpenAI API Key not set**:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   echo "export OPENAI_API_KEY='your-api-key'" >> ~/.bashrc
   source ~/.bashrc
   ```

4. **GPU not accessible**:
   - Check with `nvidia-smi`
   - Install NVIDIA drivers if needed: `sudo apt install nvidia-driver-XXX`
   - Verify CUDA compatibility
   - Reboot after driver installation

5. **nvc++ installed but not in PATH**:
   ```bash
   export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
   export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH
   echo 'export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

6. **Python package installation fails**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   If specific packages fail, install them individually:
   ```bash
   pip install openai pandas numpy matplotlib seaborn tqdm
   ```

#### Common Runtime Issues

- **Compilation errors**: Ensure NVIDIA HPC SDK is properly installed and in PATH
- **Permission denied on scripts**: Run `chmod +x setup_environment.sh`
- **Python import errors**: Run `pip install -r requirements.txt`

---

**Note**: This framework is designed for research and development purposes with multiple benchmark suites. Ensure you have appropriate hardware (NVIDIA GPU with OpenMP offloading support) and software (NVIDIA HPC SDK, CUDA Toolkit) before use. The framework supports translation between serial, OpenMP, and CUDA programming models with automated correctness verification and performance optimization.

**Supported Benchmarks**: The framework is configured for multiple benchmark suites:
- **Rodinia**: nw (Needleman-Wunsch), srad (Speckle Reducing Anisotropic Diffusion), lud (LU Decomposition), b+tree, backprop, bfs, hotspot, and others
- **NAS Parallel Benchmarks**: Scientific computing kernels
- **HeCBench**: Heterogeneous computing benchmarks
- **ParEval**: Parallel evaluation benchmarks for CUDA/OpenMP translation
