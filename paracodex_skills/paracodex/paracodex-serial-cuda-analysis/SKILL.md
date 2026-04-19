---
name: paracodex-serial-cuda-analysis
description: "ParaCodex prompt for serial to cuda analysis step."
---

# Serial to CUDA Migration - Analysis Phase
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/`

## Task
Analyze Serial code in `{source_dir}/` for migration to CUDA. produce `{kernel_dir}/analysis.md`.
**Copy source files** to `{kernel_dir}/`.

## Process

### 0. Setup
- Copy `{file_listing}` from `{source_dir}` to `{kernel_dir}`.

### 1. Identify Hotspots
- Run the code, time it.
- Identify Loop/Function taking most time.
- Distinguish hotspot kernel time from end-to-end runtime, including setup and data movement.
- If several helper functions form one timed pipeline, treat that pipeline as the optimization target.

### 2. Check Parallelizability
**Reference:** `references/examples.md`.
- **Dependencies:** Loop carried? (Hard).
- **Independent:** (Easy).
- **Reduction:** (Medium).
- **Structural fit:** Does the hotspot need a fused CUDA kernel rather than a direct helper-by-helper translation?
- Check whether the default run size is too small to expose real transfer, launch, or occupancy behavior.
- Mark the hotspot as small-input sensitive only when tiny-run conclusions are likely to mislead the migration.
- Read `system_info_summary.txt` and recommend a larger profiling size using the actual device memory and compute capability; prefer the largest short-run, memory-safe input that should load the GPU meaningfully.
- If that larger size still leaves the GPU weakly utilized or mostly launch/setup bound, mark it insufficient and recommend pushing higher within the practical budget.

### 3. Output Report
**Reference:** `references/output.md` (Use template).

- Fill "Hotspot Analysis".
- Fill "Migration Strategy".
- Fill "Structural Risk".
- Fill "Scalability Check".
- **Estimate Speedup:** (Potential).

## Constraints
- No git commands.
