---
name: paracodex-cuda-omp-step2
description: "ParaCodex prompt for cuda omp step2 step."
---

# Performance Tuning - CUDA to OMP Migration
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**You may replace the step1 offload/data strategy if the profile shows the current structure is wrong**

## EARLY EXIT CHECK
If current combined GPU kernel + memcpy + sync cost is within 5% of expected optimal and kernel count / transfer cost / host overhead are already reasonable:
- Document current metrics in `optimization_plan.md` (See `references/output.md`).
- Skip optimization - code is already well-tuned.
- Focus only on micro-optimizations (const, restrict, cache locals).
- Do not early-exit on a tiny/default input alone when the design may be small-input sensitive.

## Context
Migrated from CUDA → OpenMP. See `references/examples.md` for Context and Common Bottlenecks.

## Workflow

### 1. Verify Baseline
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt | grep -E "Verification|SUCCESSFUL|FAILED"
```
If errors, fix them before continuing.

### 2. Analyze Profile and Create Plan
1. Read profile (`cuda_gpu_kern_sum`, `cuda_mem_time`, `gpu` activity).
2. Check GPU capabilities (`nvidia-smi`).
3. Compare with original CUDA metrics (if available).
4. Compare combined GPU kernel + memcpy + sync time to baseline; if kernel time improves but combined cost regresses, do not accept the result.
5. Recheck with at least one larger practical input when hardware and timeout budget allow; choose a size that materially exercises the GPU on the available hardware rather than one that remains launch-dominated, and reject changes that only win on the smallest/default run.
6. Use `system_info_summary.txt` to size that run against the actual device. Prefer the largest short-run, memory-safe input that should load the GPU meaningfully, not merely a modest increase over the default.
6. **Create `optimization_plan.md`** using template in `references/output.md`.
   - Fill "Current Metrics".
   - Fill "Structural Check".
   - Select "Bottleneck Hypothesis".
   - Define "Actions".

**Reference:** See `references/examples.md` for "Fusion Rules" and "Optimization Checklist".

**MANDATORY structural check before tuning**
- Classify the result before changing code:
  - **Wrong residency/data strategy:** transfers are clearly higher than the recovered CUDA structure requires.
  - **Execution-structure limited:** residency is reasonable, but helper stages, many short kernels, or host waits dominate.
- If residency is already reasonable, do not replace it blindly; first reduce launch/sync overhead and recover the useful CUDA execution structure in OpenMP form.
- If the current decomposition preserves a weak step1 structure, step2 may fuse helpers, flatten buffers, or replace the offload unit entirely.

### 3. Execute Optimization Plan
- Treat the larger profiling size as valid only if it materially exercises the GPU; if the run is still dominated by launch/setup noise, increase the size again within practical limits before making the optimization decision.
- Do not accept a "larger" input that still leaves the GPU mostly idle. If utilization is still weak, push higher within the short-run budget and record the hardware/memory reason if you stop.
- Apply changes and document in `optimization_plan.md`.
- If kernel-time improvements do not improve end-to-end runtime, keep tuning or revert.

### 4. Optimization Actions
**Reference:** `references/examples.md` (Checklist & Syntax).
- **Transfers High:** Hoist data, use `omp_target_alloc`.
- **Wrong offload unit:** Fuse helpers, flatten buffers, or replace a weak step1 decomposition.
- **Too many regions:** Fuse loops, inline helpers (Match CUDA structure).
- **Grid shape mismatch:** Add `collapse(N)`.
- **Hot Kernel:** Micro-opts.
- If residency is correct but execution is launch/sync limited, prefer launch reduction and CUDA-structure recovery before changing the data strategy.
- If an async/dependency-based rewrite is attempted and is unstable on this compiler/runtime, revert it and record that the current result is structurally correct but runtime-limited.
- Reject any change that improves kernel time but increases combined GPU kernel + memcpy + sync cost.
- Reject any change that improves only the tiny/default run but scales worse at the larger practical input.

### 5. Final Summary
Update `optimization_plan.md` with:
- Baseline vs Final metrics.
- Slowdown vs CUDA (Target <1.5x).
- Optimizations applied.
- State explicitly whether larger-input profiling changed the optimization decision.
- State explicitly whether the final result is:
  - wrong-data-strategy fixed
  - execution-structure limited but stable
  - blocked by compiler/runtime instability
(Use "Final Performance Summary" from `references/output.md`).

## Profiling
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found"
```

# Rules
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- ALWAYS CLEAN BEFORE BUILD.
- PRESERVE CORRECTNESS.
- **Syntax Reminder:** Extract struct members to local variables (See `references/examples.md`).
