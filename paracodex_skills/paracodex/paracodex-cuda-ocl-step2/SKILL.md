---
name: paracodex-cuda-ocl-step2
description: "ParaCodex prompt for cuda ocl step2 step."
---

# Performance Tuning - CUDA to OpenCL Migration
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Reference:** `{kernel_dir}/opencl_migration_plan.md`

## Context
Migrated from CUDA → OpenCL. Differences in runtime, compiler, and memory handling often cause performance gaps.

## Workflow

### 1. Verify Baseline Correctness
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt
```
If errors, fix correctness issues before optimization.

### 2. Analyze Performance Profile
1. Read profile data (`cat {profile_log_path} | grep ...`).
2. Compare with CUDA baseline.
3. If >1.5x slowdown: Significant optimization needed.
4. Compare kernel time to end-to-end runtime; if they diverge, check structural issues before micro-tuning.
5. Recheck with at least one larger practical input when hardware and timeout budget allow; choose a size that materially exercises the GPU on the available hardware rather than one that remains launch-dominated, and reject changes that only win on the smallest/default run.
6. Use `system_info_summary.txt` to size that run against the actual device. Prefer the largest short-run, memory-safe input that should load the GPU meaningfully, not merely a modest increase over the default.

### 3. Create Optimization Plan
**MANDATORY:** Create `optimization_plan.md` in `{kernel_dir}` using template in `references/output.md`.
**Reference:** `references/examples.md` (Bottleneck Analysis).

Analyses to check:
- Work-Group Size Suboptimal?
- Memory Transfer Overhead?
- Math Function Precision?
- Local Memory Bank Conflicts?
- Kernel Launch Overhead?
- Wrong kernel/program split?

### 4. Execute Optimizations
**Reference:** `references/examples.md` (Optimization Examples).

- **4A. Optimize Work-Group Size:** Query `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` and set `local_work_size` to a multiple of it.
- **4S. Structural Rewrite:** Fuse kernels, move build/setup out of the hot path, or replace a weak step1 split before micro-optimizing.
- **4B. Native Math:** Use `native_sin`, `native_cos`, `native_sqrt` where precision allows (avoids IEEE-754 overhead).
- **4C. Transfers:** Use `CL_MEM_ALLOC_HOST_PTR` (pinned) buffers; use `clEnqueueWriteBuffer` with `CL_FALSE` (non-blocking) + events for async.
- **4D. Barriers:** Use `CLK_LOCAL_MEM_FENCE` over `CLK_GLOBAL_MEM_FENCE` when only local memory requires synchronization.
- **4E. Vector Loads:** Use `float4`/`int4` types for memory reads/writes to improve coalescing and instruction throughput.
- **Micro-Opts:** `restrict`, private variable caching of repeated global reads, `#pragma unroll`.
- Reject any change that improves only the tiny/default run but scales worse at the larger practical input.
- Do not accept a "larger" input that still leaves the GPU mostly idle. If utilization is still weak, push higher within the short-run budget and record the hardware/memory reason if you stop.

### 5. Verify & Profile (Iterative)
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > optimized_output.txt 2>&1
diff baseline_output.txt optimized_output.txt
{profile_cmd_str} > {profile_log_path}_optimized 2>&1
```

### 6. Final Summary
Update `optimization_plan.md` using "Final Performance Summary" template, including whether larger-input profiling changed the decision.

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- **PRESERVE CORRECTNESS** - diff against baseline after each change.
- **VERIFY PERFORMANCE GAIN** - revert if slower.
