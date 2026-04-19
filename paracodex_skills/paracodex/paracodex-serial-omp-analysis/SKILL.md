---
name: paracodex-serial-omp-analysis
description: "ParaCodex prompt for serial omp analysis step."
---

# Loop Classification for GPU Offload - Analysis Phase
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Task
Analyze loops in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. 
Copy source files unmodified to `{kernel_dir}/`.

The analysis must identify the natural GPU offload unit, helper-fragmentation risk, and whether the default input is too small to guide optimization reliably.

## Process

### 0. COPY SOURCE FILES
Copy {file_listing} to {kernel_dir} unmodified.

### 1. Find All Loops
```bash
grep -n "for.*iter\|for.*it\|while\|main(" *.c *.cpp 2>/dev/null | head -50
grep -n "for\s*(" *.c *.cpp 2>/dev/null | head -100
```
Prioritize functions called in main compute loop:
- Every iteration → CRITICAL/IMPORTANT
- Once at setup → SECONDARY/AVOID

### 2. Classify Priority
For each loop: `iterations × ops/iter = total work`
- **CRITICAL:** >50% runtime OR called every iteration with O(N) work
- **IMPORTANT:** 5-50% runtime OR called every iteration with small work
- **SECONDARY:** Called once at setup
- **AVOID:** Setup/IO/RNG OR <10K iterations

### 3. Determine Loop Type & Data Analysis
**Reference:** `references/examples.md` (See Decision Tree and Type Reference).

**Check for Type H first:** If an outer loop iterates over independent work items and contains an inner loop whose body is guarded by `if (inner_var == 0)`, classify the inner loop as **Type H (CUDA-serial port)** and flag it as **RESTRUCTURE NEEDED** in analysis.md.

Analyze each array:
- Flat vs pointer-to-pointer?
- Static vs dynamic?
- Structs/Globals?

Structural questions to answer during analysis:
- Is the timed region already a single GPU-shaped loop nest?
- Is the hot path split across helper functions with tiny outer loops or host-visible stage boundaries?
- Would preserving the current helper boundaries likely create multiple tiny kernels, extra host/device sync, or poor scaling?
- If yes, recommend a fused offload unit in `analysis.md` instead of helper-by-helper offload.

Scalability questions to answer during analysis:
- Is the default input likely too small to expose real kernel, transfer, launch, or host-wait behavior?
- Is this benchmark likely small-input sensitive, or is the default size already representative?
- Which quantities should scale with problem size: kernel count, transfer count, transfer volume, host waits, memory footprint?
- What larger practical profile size should later stages use on the available hardware?
- Choose a larger profiling size that is large enough to expose sustained GPU behavior on the available device, not just millisecond-scale launch noise; it should materially exercise the GPU while still remaining comfortably within memory and time limits.
- Read `system_info_summary.txt` and choose that size against the actual device memory and compute capability; prefer the largest short-run, memory-safe input that should load the GPU meaningfully.
- If the chosen larger size still leaves the GPU weakly utilized or mostly launch/setup bound, mark it insufficient and recommend pushing higher within the practical budget.

### 4. Create Analysis Report
**Reference:** `references/output.md` (Use this template).

- Fill "Loop Nesting Structure".
- Fill "Loop Details" for CRITICAL/IMPORTANT/SECONDARY loops.
- Fill "Summary Table".
- Fill "Structural Recommendations".
- Fill "Scalability Check".
- Fill "Data Details".
- **Flag Issues:** Variable bounds, reductions, atomics, stage dependency, RNG.

## Constraints
- Find all loops in functions called from main compute loop.
- Document only - no pragmas or code modifications.
- When uncertain between B and C, choose C.
- Copy all source files unmodified to `{kernel_dir}/`.
