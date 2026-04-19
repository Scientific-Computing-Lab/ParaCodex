---
name: paracodex-cuda-sycl-analysis
description: "ParaCodex prompt for cuda to sycl analysis step."
---

# CUDA to SYCL Migration - Analysis Phase
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Task
Analyze CUDA code in `{source_dir}/` for migration to SYCL. produce `{kernel_dir}/analysis.md`.
**Copy source files** to `{kernel_dir}/`.

## Process

### 0. Setup
- Copy `{file_listing}` from `{source_dir}` to `{kernel_dir}`.

### 1. API & Pattern Scan
**Reference:** `references/examples.md`.
- **Textures:** Check for texture memory (High complexity).
- **Dynamic Parallelism:** Check for `cudaLaunchKernel` in device code (Not supported).
- **Libraries:** Check usage of `cu*` libs vs `oneMKL`.
- **Structure:** Decide whether the CUDA kernel/file split should be preserved, fused, or flattened in SYCL.
- Check whether the default run size is too small to expose submission, wait, or transfer overhead.
- Mark kernels as small-input sensitive only when tiny runs are likely to bias the SYCL decomposition or memory-model choice.
- Recommend a larger profiling size that materially exercises the GPU on the available hardware; do not stop at an input that still produces only short, launch-dominated kernels unless memory or timeout limits force it.
- Read `system_info_summary.txt` and choose that size against the actual device memory and compute capability; prefer the largest short-run, memory-safe input that should load the GPU meaningfully.
- If the chosen larger size still leaves the GPU weakly utilized or mostly launch/setup bound, mark it insufficient and recommend pushing higher within the practical budget.

### 2. Output Report
**Reference:** `references/output.md` (Use template).

- Fill "Migration Complexity".
- Fill "API Inventory".
- Fill "Kernel Details".
- Fill "Structural Recommendations".
- Fill "Scalability Check".
- **Flag Issues:** Textures, Dynamic Parallelism, Proprietary libs.

## Constraints
- No git commands.
