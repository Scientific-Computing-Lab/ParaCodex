---
name: paracodex-cuda-hip-step2
description: "ParaCodex prompt for cuda to hip step2 (optimization) step."
---

# Performance Tuning - CUDA to HIP Migration
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Reference:** `{kernel_dir}/hip_migration_plan.md`

## Context
Migrated from CUDA → HIP. Target hardware is AMD GPU (CDNA/RDNA).

## Workflow

### 1. Verify Baseline Correctness
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt
```

### 2. Analyze Performance Profile
1. Read profile (`rocprof` output).
2. Compare with CUDA/Baseline.

### 3. Create Optimization Plan
**MANDATORY:** Create `optimization_plan.md` in `{kernel_dir}` using template in `references/output.md`.

Fill Analysis:
- **Block Size:** Multiple of 64? (AMD Wavefront)
- **Registers:** VGPR usage high?
- **LDS:** Bank conflicts?

### 4. Execute Optimization Plan
**Reference:** `references/examples.md`.

- **4A. Tune Block Size:** Try 256, 128, 64. Avoid 32 (too small on CDNA where wavefront=64). 256 is safe for both CDNA (wavefront=64) and RDNA (wavefront=32). Always use `warpSize` — not `__AMDGCN_WAVEFRONT_SIZE` (removed in ROCm 7.x).
- **4B. Registers:** Move uniform data to SGPRs.
- **4C. Memory:** Coalescing.

### 5. Verify & Profile (Iterative)
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > optimized_output.txt 2>&1
diff baseline_output.txt optimized_output.txt
{profile_cmd_str} > {profile_log_path}_optimized 2>&1
```

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- **PRESERVE CORRECTNESS** - diff against baseline.
- **VERIFY PERFORMANCE GAIN** - revert if slower.
