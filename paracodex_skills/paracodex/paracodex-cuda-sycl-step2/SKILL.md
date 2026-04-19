---
name: paracodex-cuda-sycl-step2
description: "ParaCodex prompt for cuda to sycl step2 (optimization) step."
---

# Performance Tuning - CUDA to SYCL Migration
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Reference:** `{kernel_dir}/sycl_migration_plan.md`

## Context
Migrated from CUDA → SYCL. Target hardware can varies (Intel GPU, CPU, NVIDIA via PTX).

## Workflow

### 1. Verify Baseline Correctness
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt
```

### 2. Analyze Performance Profile
1. Read profile (e.g., VTune or simple timers).
2. Compare with CUDA/Baseline.
3. Compare combined GPU kernel + memcpy + sync time to end-to-end runtime and host wait time.
4. If kernel/submission time improves but combined cost regresses, the structure is still wrong.
5. Recheck with at least one larger practical input when hardware and timeout budget allow; choose a size that materially exercises the GPU on the available hardware rather than one that remains launch-dominated, and reject changes that only win on the smallest/default run.
6. Use `system_info_summary.txt` to size that run against the actual device. Prefer the largest short-run, memory-safe input that should load the GPU meaningfully, not merely a modest increase over the default.

### 3. Create Optimization Plan
**MANDATORY:** Create `optimization_plan.md` in `{kernel_dir}` using template in `references/output.md`.

Fill Analysis:
- **USM Usage:** Are we using explicit USM? (Prefer over buffers/accessors for lower overhead.)
- **Sub-groups:** Can we use sub-group shuffle for reductions?
- **Work-group Size:** Tuned for target? Query at runtime:
  ```cpp
  auto wg_size = kernel_bundle.get_kernel(kernel_id)
      .get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
  ```
- **Submission structure:** Too many tiny submissions or `.wait()` calls?

### 4. Execute Optimization Plan
**Reference:** `references/examples.md`.

- **4A. Switch to USM:** Replace Accessors/Buffers with `sycl::malloc_device`; use `q.memcpy` for explicit transfers.
- **4S. Structural Rewrite:** Fuse submissions, remove needless waits, or replace a weak step1 decomposition before micro-tuning.
- **4B. Sub-groups:** Replace manual shared-memory reductions with `sycl::reduce_over_group` or `sub_group::shuffle_down`.
- **4C. Async Chaining:** Chain kernels with `.depends_on(event)` instead of `.wait()` after every submission to overlap transfers with compute.
- **4D. Loop Unrolling:** Use `[[sycl::reqd_work_group_size(N)]]` attribute + `#pragma unroll` for inner loops.
- **4E. Sub-group Size:** Pin sub-group width with `[[sycl::reqd_sub_group_size(N)]]` (e.g., `N=32`) to ensure warp-width optimizations apply and avoid fallback to sub-group size 1.
- Reject any change that improves kernel time but increases combined GPU kernel + memcpy + sync cost.
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

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- **PRESERVE CORRECTNESS** - diff against baseline.
- **VERIFY PERFORMANCE GAIN** - revert if slower.
