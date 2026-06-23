---
name: paracodex-ocl-cuda-analysis
description: "ParaCodex prompt for ocl cuda analysis step."
---

# OpenCL to CUDA Migration - Analysis Phase
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Task
Analyze OpenCL kernels in `{source_dir}/` and produce `{kernel_dir}/analysis.md`.
**Copy source files** to `{kernel_dir}/` with suffix conversion (.cl → .cu for combined source).
**Do not modify code in this step (logic-wise).**

## Process

### 0. Copy Source Files (Suffix Conversion)
- Copy `{file_listing}` from `{source_dir}/`.
- **Convert suffixes**: `.cl` / `.cpp` → `.cu` (Unified Source).
- Update includes (`.h` → `.cuh`).
- Get baseline output.
- **Reference:** `references/output.md` (File Mapping).

### 1. Create Environment (Makefile)
- Create `Makefile` if needed or check existing.

### 2. Find Kernels and Loops
```bash
grep -n "__kernel\|kernel void" *.cl 2>/dev/null
grep -n "barrier\|clFinish" *.cl *.c *.cpp 2>/dev/null
```

### 3. Classify Priority
- **CRITICAL:** >50% runtime OR called every iteration.

### 4. Determine Type & Data Analysis
**Reference:** `references/examples.md` (Decision Tree & Patterns).
- Check `barrier` (maps to syncthreads), `atomic_add`, `__local` (maps to shared).
- Identify sub-groups (maps to warps).
- Identify whether the OpenCL kernel/file split should be preserved, fused, or simplified in CUDA.
- Check whether the default run size is too small to expose real launch, transfer, or occupancy behavior after migration.
- Mark kernels as small-input sensitive only when tiny runs are likely to bias the CUDA structure choice.
- Recommend a larger profiling size that materially exercises the GPU on the available hardware; do not stop at an input that still produces only short, launch-dominated kernels unless memory or timeout limits force it.
- Read `system_info_summary.txt` and choose that size against the actual device memory and compute capability; prefer the largest short-run, memory-safe input that should load the GPU meaningfully.
- If the chosen larger size still leaves the GPU weakly utilized or mostly launch/setup bound, mark it insufficient and recommend pushing higher within the practical budget.

### 5. Create Analysis Report
**Reference:** `references/output.md` (Use this template).

- Fill "File Conversion Mapping".
- Fill "Kernel/Loop Nesting".
- Fill "Kernel Details".
- Fill "CUDA Mapping Table".
- Fill "Structural Recommendations".
- Fill "Scalability Check".
- **Flag Issues:** Device Enqueue, Pipes, Image Objects.

## Constraints
- **Critical:** Identify Device Enqueue (major block).
- Copy files with suffix conversion.
- No code modifications.
