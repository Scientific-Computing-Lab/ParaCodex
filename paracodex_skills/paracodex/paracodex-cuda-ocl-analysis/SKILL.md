---
name: paracodex-cuda-ocl-analysis
description: "ParaCodex prompt for cuda ocl analysis step."
---

# CUDA to OpenCL Migration - Analysis Phase
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Task
Analyze CUDA kernels in `{source_dir}/` and produce `{kernel_dir}/analysis.md`.
**Copy source files** to `{kernel_dir}/` with suffix conversion (.cu → .cl for kernels, .c/.cpp for host).
**Do not modify code in this step (logic-wise).**

## Process

### 0. Copy Source Files (Suffix Conversion)
- Copy `{file_listing}` from `{source_dir}/`.
- **Convert suffixes**: `.cu` → `.cl` / `.cpp`.
- Update includes (`.cuh` → `.h`).
- Get baseline output.
- **Reference:** `references/output.md` (File Mapping).

### 1. Create Environment (Makefile)
- Create `Makefile` if needed or check existing.

### 2. Find Kernels and Loops
```bash
grep -n "__global__\|__device__" *.cu 2>/dev/null
grep -n "<<<.*>>>" *.cu 2>/dev/null
grep -n "__syncthreads\|cudaDeviceSynchronize" *.cu 2>/dev/null
```

### 3. Classify Priority
- **CRITICAL:** >50% runtime OR called every iteration.

### 4. Determine Type & Data Analysis
**Reference:** `references/examples.md` (Decision Tree & Patterns).
- Check `__syncthreads` (maps to barrier), `atomicAdd`, `__shared__` (maps to local).
- Identify warp-level primitives (NO OpenCL equivalent).
- Determine the natural kernel/file split for the migration and flag whether preserving CUDA kernel boundaries is good or harmful.
- Check whether the default run size is too small to expose real enqueue, transfer, or compilation overhead.
- Mark kernels as small-input sensitive only when tiny runs are likely to bias the OpenCL split or data strategy.
- Recommend a larger profiling size that materially exercises the GPU on the available hardware; do not stop at an input that still produces only short, launch-dominated kernels unless memory or timeout limits force it.
- Read `system_info_summary.txt` and choose that size against the actual device memory and compute capability; prefer the largest short-run, memory-safe input that should load the GPU meaningfully.
- If the chosen larger size still leaves the GPU weakly utilized or mostly launch/setup bound, mark it insufficient and recommend pushing higher within the practical budget.

### 5. Create Analysis Report
**Reference:** `references/output.md` (Use this template).

- Fill "File Conversion Mapping".
- Fill "Kernel/Loop Nesting".
- Fill "Kernel Details".
- Fill "OpenCL Mapping Table".
- Fill "Structural Recommendations".
- Fill "Scalability Check".
- **Flag Issues:** Warp primitives, dynamic parallelism, texture memory.

## Constraints
- **Critical:** Identify warp-level primitives (major block).
- Copy files with suffix conversion.
- No code modifications.
