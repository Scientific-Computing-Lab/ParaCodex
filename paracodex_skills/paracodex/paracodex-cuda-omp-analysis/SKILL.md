---
name: paracodex-cuda-omp-analysis
description: "ParaCodex prompt for cuda omp analysis step."
---

# Loop Classification for OMP Migration - Analysis Phase
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Task
Analyze CUDA kernels in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. 
**Copy source files** to `{kernel_dir}/` with suffix conversion (.cu → .c or .cpp).
**Do not modify code in this step.**

## Process

### 0. Copy Source Files (Suffix Conversion)
- Copy `{file_listing}` from `{source_dir}/` to `{kernel_dir}/`.
- **Convert suffixes**: `.cu` → `.c` (C) or `.cpp` (C++).
- Get baseline output (run in source dir, copy to kernel dir).
- **No Code Mod:** Preserve content exactly.
- **Reference:** `references/output.md` (File Conversion Mapping).

### 1. Create Environment (Makefile.nvc)
- Create `Makefile.nvc` using `nvc++`.
- Create any needed headers/utils.

### 2. Find Kernels and Loops
```bash
grep -n "__global__\|__device__" *.cu 2>/dev/null
grep -n "<<<.*>>>" *.cu 2>/dev/null
grep -n "for\s*(" *.cu 2>/dev/null | head -100
grep -n "for.*iter\|for.*it\|while" *.cu 2>/dev/null | head -50
```

### 3. Classify Priority
- **CRITICAL:** >50% runtime OR called every iteration.
- **IMPORTANT:** 5-50% runtime.
- **SECONDARY:** Setup only.
- **AVOID:** <10K threads.

### 4. Determine Type & Data Analysis
**Reference:** `references/examples.md` (Decision Tree for CUDA Kernels).
- Check `__syncthreads`, `atomicAdd`, `__shared__`.
- Analyze each array (Memory type, Transfer pattern).
- Identify whether the CUDA code's natural unit of work is a single kernel, a fused multi-stage pipeline, or a sequence of distinct kernels.
- Flag any CUDA kernel structure that should be preserved or deliberately fused during OMP migration.
- Check whether the default run size is too small to expose real launch, transfer, or occupancy behavior after migration.
- Mark kernels as small-input sensitive only when tiny runs are likely to bias the OpenMP design choice.
- Recommend a larger profiling size that materially exercises the GPU on the available hardware; do not stop at an input that still produces only short, launch-dominated kernels unless memory or timeout limits force it.
- Read `system_info_summary.txt` and choose that size against the actual device memory and compute capability; prefer the largest short-run, memory-safe input that should load the GPU meaningfully.
- If the chosen larger size still leaves the GPU weakly utilized or mostly launch/setup bound, mark it insufficient and recommend pushing higher within the practical budget.

### 5. Create Analysis Report
**Reference:** `references/output.md` (Use this template).

- Fill "File Conversion Mapping".
- Fill "Kernel/Loop Nesting".
- Fill "Kernel Details".
- Fill "Summary Table".
- Fill "CUDA-Specific Details".
- Fill "Structural Recommendations".
- Fill "Scalability Check".
- **Flag Issues:** `__syncthreads`, shared memory, atomics.

## Constraints
- **Critical:** Identify `__syncthreads()` patterns (no direct OMP equivalent).
- Copy files with suffix conversion.
- No code modifications.
