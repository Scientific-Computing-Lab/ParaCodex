---
name: paracodex-cuda-hip-analysis
description: "ParaCodex prompt for cuda to hip analysis step."
---

# CUDA to HIP Migration - Analysis Phase
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Task
Analyze CUDA code in `{source_dir}/` for migration to AMD HIP. produce `{kernel_dir}/analysis.md`.
**Copy source files** to `{kernel_dir}/`.

## Process

### 0. Setup
- Copy `{file_listing}` from `{source_dir}` to `{kernel_dir}`.

### 1. API & Pattern Scan
**Reference:** `references/examples.md`.
- **Warp Size:** Check for `32`, `0x1f`. (AMD is 64).
- **Inline PTX:** Check for `asm`. (Needs rewrite).
- **Libraries:** Check usage of `cu*` libs.

### 2. Output Report
**Reference:** `references/output.md` (Use template).

- Fill "Migration Complexity".
- Fill "API Inventory".
- Fill "Kernel Details".
- **Flag Issues:** Warp assumptions, PTX, Proprietary libs.

## Constraints
- No git commands.
