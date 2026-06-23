# Performance Analysis

## Current Metrics
- Runtime: 0.496s profiled run on 2048x2048, 12 iterations
- End-to-end runtime source: profiled run
- Main kernel: `nvkernel__Z16single_iterationPfS_S_iiffff_F1L54_2`, 100% GPU, 12 instances
- Memory transfer: 68.1% of transfer time, 50.331 MB total
- Kernel launches: 12
- Host overhead / waits: `wait` and `poll` dominate the profiled trace, but launch count matches iteration count
- Combined GPU+mem time: 7,258,514 ns

## Scalability Check
- Default correctness size: 1024x1024, 2 iterations
- Larger practical profiling size: 2048x2048, 12 iterations
- Why this size materially exercises the GPU: it keeps the kernel launch count modest while raising per-launch work to millions of cells
- Small-input sensitive? NO
- Larger-input profiling changed the decision? NO

## Structural Check
- Step1 offload unit: fused routine
- Current structural risk: execution-structure limited
- Residency strategy correct? YES
- Execution-structure limited? YES
- Can step2 keep the current structure? YES
- If NO, required rewrite: none

## Fusion Opportunities:

### Identified Fusions:
- None required. The timestep recurrence stays on the host, and each timestep is already fused into one GPU kernel.

## Iteration Loop (if present):
- Main: line 172, 12 iters
- `single_iteration` line 53/54: 12 kernel launches total
- Total: 12 ops

## SpMV Inner Loop Decision
- N/A. This kernel is a stencil update, not SpMV.

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong offload unit | Not observed | No structural rewrite needed |
| Data transfers | Low transfer count and fixed residency | Keep `omp_target_alloc` / `is_device_ptr` path |
| Launch overhead | 12 kernels for 12 iterations | Acceptable |
| Over-parallelization | Not observed | No change |
| Hot kernel | Kernel time is already small | No change |
| Stage parallelization | Verification harness unavailable in this workspace | Keep stable path |

## Strategy (priority)
1. [KEEP]: Retain the current fused per-timestep GPU kernel and persistent device allocations - the profile already shows minimal launch and transfer count.
2. [MICRO]: Only apply low-risk local cleanups if needed later - avoid structural changes unless a new profile shows regression.

## Micro-opts
[x] const locals, firstprivate values, cached dimensions

## Target
- Runtime: unchanged until a larger profile shows a real bottleneck
- Kernels: ~12 for 12 iterations
- Memory: low transfer volume, single HtoD setup plus one DtoH teardown
- Structural target: fused hot path

# Optimization Plan - Final Summary (Append this after execution)

## Final Performance Summary

### Baseline (Step 2)
- Runtime: 0.496s
- Main kernel: 12 instances, 2.789810ms total

### Final (Step 3)
- Runtime: 0.496s
- Speedup: 1.00x
- Main kernel: 12 instances, 2.789810ms total
- Combined GPU+mem cost improved: NO

### Optimizations Applied
1. [ ] ACTION 4S: none; retained the existing fused GPU timestep kernel
2. [ ] ACTION: none; no structural rewrite was justified by the profile

### Micro-optimizations Applied
1. [x] MICRO-OPT: kept the current const/cached-locals style

### Key Insights
- The step1 offload/data strategy was not changed because residency and transfer volume were already reasonable.
- Larger-input profiling did not change the decision.
- The final result is execution-structure limited but stable.
- End-to-end runtime did not justify a more aggressive async/dependency rewrite.
- Combined GPU+mem cost stayed low; host wait time in the profile is not caused by excess kernel count or transfer churn.
