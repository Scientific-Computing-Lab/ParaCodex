---
name: paracodex-serial-cuda-step2
description: "ParaCodex prompt for serial to cuda step2 (optimization) step."
---

# Performance Tuning - Serial to CUDA Migration
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Reference:** `{kernel_dir}/cuda_migration_plan.md`

## Context
Migrated from Serial → CUDA. Correct but likely slow (naive port).

## Workflow

### 1. Verify Baseline Correctness
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt
```

### 2. Analyze Performance Profile
1. Read profile (e.g., Nsight Systems or `nvprof`).
2. Identify bottlenecks (Global Mem throughput, Occupancy).
3. Compare combined GPU kernel + memcpy + sync time to end-to-end runtime; if kernel time improves but combined cost regresses, look for structural issues.
4. Recheck with at least one larger practical input when hardware and timeout budget allow; choose a size that materially exercises the GPU on the available hardware rather than one that remains launch-dominated, and reject changes that only win on the smallest/default run.
5. Use `system_info_summary.txt` to size that run against the actual device. Prefer the largest short-run, memory-safe input that should load the GPU meaningfully, not merely a modest increase over the default.

### 3. Create Optimization Plan
**MANDATORY:** Create `optimization_plan.md` in `{kernel_dir}` using template in `references/output.md`.

Fill Analysis:
- **Global Load Efficiency:** Low? Use Coalescing.
- **Latency:** High? Use Shared Memory.
- **Structural fit:** Is the current kernel decomposition wrong for the workload?

### 4. Execute Optimization Plan
**Reference:** `references/examples.md`.

- **4A. Shared Memory:** Tile data access; pad shared arrays by 1 to avoid bank conflicts.
- **4S. Structural Rewrite:** Fuse kernels, flatten data, or replace a weak step1 decomposition if that is the real bottleneck.
- **4B. Threads:** Use `cudaOccupancyMaxPotentialBlockSize` to find optimal block size; annotate kernels with `__launch_bounds__(BLOCK_SIZE)`.
- **4C. Memory:** Use `cudaMemcpyAsync` + streams to overlap transfers with compute.
- **4D. Warp Shuffle:** Replace shared-memory reductions with `__shfl_down_sync(0xffffffff, val, offset)` for warp-level reductions. Preferred modern alternative: `cg::reduce(warp, val, cg::plus<float>())` from `cooperative_groups.h`.
- **4E. Fast Math:** Add `--use_fast_math` (or `__sinf`, `__cosf`, `__expf` intrinsics) where precision allows.
- **4F. CUDA Graphs:** For repeat-launch kernels, capture into a CUDA Graph with `cudaStreamBeginCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch` to eliminate per-launch overhead.
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
- **VERIFY SPEEDUP** - revert if slower.
- Optimize for end-to-end runtime, not only kernel-time wins.
