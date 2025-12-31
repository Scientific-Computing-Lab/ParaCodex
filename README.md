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
└── kill_gpu_processes.py/sh                     # GPU process management utilities
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

- **NVIDIA HPC SDK**: For OpenMP GPU offloading (nvc++ compiler)
- **CUDA Toolkit**: For CUDA development
- **Python 3.8+**: For the pipeline scripts
- **Codex CLI**: For AI agent interactions
- **NVIDIA Nsight Compute**: For GPU profiling

### ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd paracodex
   ```

2. **Set up the working directory**:
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

3. **Install dependencies**:
   ```bash
   # Ensure NVIDIA HPC SDK is installed and nvc++ is in PATH
   # Install Python dependencies (if any requirements.txt exists)
   ```

### 💻 Basic Usage

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



## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the prompts documentation:
  - `pipeline/SERIAL_OMP_PROMPTS.md` for serial-to-OpenMP translation
  - `pipeline/CUDA_PROMPTS.md` for CUDA-related translations

---

**Note**: This framework is designed for research and development purposes with multiple benchmark suites. Ensure you have appropriate hardware (NVIDIA GPU with OpenMP offloading support) and software (NVIDIA HPC SDK, CUDA Toolkit) before use. The framework supports translation between serial, OpenMP, and CUDA programming models with automated correctness verification and performance optimization.

**Supported Benchmarks**: The framework is configured for multiple benchmark suites:
- **Rodinia**: nw (Needleman-Wunsch), srad (Speckle Reducing Anisotropic Diffusion), lud (LU Decomposition), b+tree, backprop, bfs, hotspot, and others
- **NAS Parallel Benchmarks**: Scientific computing kernels
- **HeCBench**: Heterogeneous computing benchmarks
- **ParEval**: Parallel evaluation benchmarks for CUDA/OpenMP translation
