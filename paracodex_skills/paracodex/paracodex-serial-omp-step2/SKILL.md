---
name: paracodex-serial-omp-step2
description: "ParaCodex prompt for serial omp step2 step."
---

# Performance Tuning
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**You may replace the step1 data/offload strategy if the profile shows the current structure is fundamentally wrong**

## EARLY EXIT CHECK
If current combined GPU kernel + memcpy + sync cost is within 5% of expected optimal AND kernel count / transfer count / host overhead are already reasonable:
- Document current metrics in `optimization_plan.md` (See `references/output.md`).
- Skip optimization - code is already well-tuned.
- Focus only on micro-optimizations (const, restrict, cache locals).
- Never early-exit based on GPU kernel time alone.
- Do not early-exit on a tiny/default input alone when the design may be small-input sensitive.

## Workflow

### 1. Verify Baseline 
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt | grep -E "Verification|SUCCESSFUL|FAILED"
```
If results differ, fix Step 2 first. If errors, fix them before continuing.

### 2. Analyze Profile and Create Plan
1. Read profile data (`cuda_gpu_kern_sum`, `cuda_api_sum`, `cuda_gpu_mem_time_sum`).
2. Run `nvidia-smi` to estimate GPU saturation.
3. Compare combined GPU kernel + memcpy + sync time to baseline; if kernel time improves but combined cost regresses, do not accept the result.
4. Recheck with at least one larger practical input when hardware and timeout budget allow; choose a size that materially exercises the GPU on the available hardware rather than one that remains launch-dominated, and reject changes that only win on the smallest/default run.
5. Use `system_info_summary.txt` to size that run against the actual device. Prefer the largest short-run, memory-safe input that should load the GPU meaningfully, not merely a modest increase over the default.
5. **Create `optimization_plan.md` in `{kernel_dir}`.**
   - **Reference:** `references/output.md` (Use this template).
   - Fill "Current Metrics", "Structural Check", "Fusion Opportunities", "Iteration Loop", "SpMV Inner Loop Decision".
   - Consult "Bottleneck Checklist" in `references/output.md` to identify issues.
   - List actions in "Strategy".

**MANDATORY structural check before tuning**
- Count hot kernels and helper stages in the timed path.
- Compare GPU kernel time to end-to-end runtime.
- Classify the result before changing code:
  - **Wrong residency/data strategy:** transfer volume or transfer count is clearly excessive for the algorithm.
  - **Execution-structure limited:** data residency is already reasonable, but many short kernels, helper staging, or host waits dominate.
- If the current code has fragmented tiny kernels, helper-call launch overhead, repeated host/device sync, or a preserved serial decomposition that limits parallelism, the FIRST action in the plan must be a structural rewrite.
- If data residency is already reasonable, do not replace it blindly; first reduce launch/sync overhead, fuse stages where legal, and remove unnecessary host waits.
- Step2 is allowed to inline helpers, fuse stages, flatten buffers, and replace the step1 offload unit entirely when this is needed for performance.

**Fusion Rules Reference:** `references/examples.md`.

### 3. Execute Optimization Plan
- Apply changes and document in `optimization_plan.md`.
- Measure end-to-end runtime again after each structural optimization. If GPU kernel time improves but total runtime regresses, revert or continue until total runtime also improves.
- Re-profile at the larger practical input before accepting the final result when the kernel was marked small-input sensitive or the default run is tiny. If the larger input still yields only short, weakly utilized GPU work, increase it again until the profile reflects sustained GPU behavior or documented memory/time limits stop you.
- Do not accept a "larger" input that still leaves the GPU mostly idle. If utilization is still weak, push higher within the short-run budget and record the hardware/memory reason if you stop.

### 4. Optimization Actions
**Reference:** `references/examples.md` (Consult for specific patterns).
- **Structural Rewrite (4S):** If step1 chose the wrong offload unit, replace it with a fused GPU-oriented routine before micro-tuning.
- **Fix Data Movement (4A):** Hoist data, use `omp_target_alloc`. Use `omp_target_memset` (OpenMP 6.0) for device-side zero-initialization without host round-trip.
- **Optimize Hot Kernel (4B):** Combine loops, collapse, SIMD, cache locals.
- **Launch Overhead (4C):** Inline helper functions.
- **Fix Type C1 (4D):** Remove inner pragmas from stage loops.
- **Increase Parallelism (4E):** Collapse depth, tile sizes.
- **Performance Portability (4F):** Use `metadirective` (OpenMP 5.0+) to select different parallelization strategies at compile time for different targets (e.g., GPU `teams loop` vs CPU `parallel for`).
- If residency is correct but execution is launch/sync limited, prefer low-risk launch reduction before replacing the data strategy.
- If a more aggressive async/dependency rewrite is attempted and proves unstable on this compiler/runtime, revert it, keep the stable version, and record that the code is structurally correct but runtime-limited.
- Reject any change that improves kernel time but increases combined GPU kernel + memcpy + sync cost.

### 5. Final Summary
Update `optimization_plan.md` with:
- Final vs Baseline metrics.
- Speedup.
- Optimizations applied vs reverted.
- Key Insights.
- State explicitly whether the final code changed the step1 offload/data strategy and why.
- State explicitly whether larger-input profiling changed the optimization decision.
- State explicitly whether the final result is:
  - wrong-data-strategy fixed
  - execution-structure limited but stable
  - blocked by compiler/runtime instability
(Use the "Optimization Plan - Final Summary" section from `references/output.md`).

## Profiling Command
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
- Never run git commands.
