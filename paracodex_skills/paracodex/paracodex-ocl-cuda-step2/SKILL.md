---
name: paracodex-ocl-cuda-step2
description: "ParaCodex prompt for ocl cuda step2 step."
---

# Performance Tuning - OpenCL to CUDA Migration
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Reference:** `{kernel_dir}/cuda_migration_plan.md`

## Context
Migrated from OpenCL → CUDA. Target is >1.0x speedup (match or beat OpenCL).

## Workflow

### 1. Verify Baseline Correctness
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt
```

### 2. Analyze Performance Profile
1. Read profile (`cuda_gpu_kern_sum`).
2. Compare with OpenCL baseline.
3. If slower: Major optimization needed.
4. Compare combined GPU kernel + memcpy + sync time to end-to-end runtime and host overhead.
5. If kernel time improves but combined cost regresses, structural work is still needed.
6. Recheck with at least one larger practical input when hardware and timeout budget allow; choose a size that materially exercises the GPU on the available hardware rather than one that remains launch-dominated, and reject changes that only win on the smallest/default run.
7. Use `system_info_summary.txt` to size that run against the actual device. Prefer the largest short-run, memory-safe input that should load the GPU meaningfully, not merely a modest increase over the default.

### 3. Create Optimization Plan
**MANDATORY:** Create `optimization_plan.md` in `{kernel_dir}` using template in `references/output.md`.

Fill Analysis:
- **Block Size:** Suboptimal?
- **Transfer Overhead:** >50%?
- **Slow Math:** Using `sinf` vs `__sinf`?
- **Warp Divergence?**
- **Wrong kernel decomposition?**

### 4. Execute Optimization Plan
**Reference:** `references/examples.md` (Optimization Examples).

- **4A. Block Size:** Tune for occupancy (multiple of 32).
- **4S. Structural Rewrite:** Fuse kernels, flatten data, or replace a weak step1 split before micro-tuning.
- **4B. Fast Math:** Use intrinsics `__sinf`, `__cosf`.
- **4C. Warp Primitives:** Use `__shfl_down_sync` for reductions.
- **4D. Transfers:** Async, Pinned, or Unified Memory.
- **4E. Shared Mem:** Pad to avoid bank conflicts.
- Keep combined GPU kernel + memcpy + sync cost improving; do not accept a kernel-only win that increases transfer cost.
- Reject any change that improves only the tiny/default run but scales worse at the larger practical input.
- Do not accept a "larger" input that still leaves the GPU mostly idle. If utilization is still weak, push higher within the short-run budget and record the hardware/memory reason if you stop.

### 5. Verify & Profile (Iterative)
```bash
{clean_cmd_str}
{build_cmd_str}
# Check correctness
timeout 300 {run_cmd_str} > optimized_output.txt 2>&1
diff baseline_output.txt optimized_output.txt
# Profile
{profile_cmd_str} > {profile_log_path}_optimized 2>&1
```

### 6. Final Summary
Update `optimization_plan.md` with final metrics and checklist.

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- **PRESERVE CORRECTNESS** - diff against baseline.
- **VERIFY PERFORMANCE GAIN** - revert if slower.
