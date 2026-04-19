# Performance Analysis

## Current Metrics
- Runtime: `0.214766s` plain run at `2048`; `0.611678s` profiled run at `2048`; `0.785766s` profiled run at `4096`
- End-to-end runtime source: plain run + profiled run
- Main kernel: `nvkernel_lud_omp_F1L128_4` / `nvkernel_lud_omp_F1L133_6`, `58.4%` / `30.4%` GPU at `2048`, `127` instances each
- Memory transfer: `~3.235ms` total GPU memcpy time at `2048`; `~13.892ms` at `4096`
- Kernel launches: `382` at `2048`; `766` at `4096`
- Host overhead / waits: `cuStreamSynchronize` dominates (`72.7%` of CUDA API time at `2048`, `86.8%` at `4096`)
- Combined GPU+mem time: `107.138ms` at `2048`; `523.531ms` at `4096`

## Scalability Check
- Default correctness size: `32`
- Larger practical profiling size: `4096`
- Why this size materially exercises the GPU: it increases panel count and trailing-block work enough to expose launch/sync behavior beyond the small default run
- Small-input sensitive? `YES`
- Larger-input profiling changed the decision? `NO`

## Structural Check
- Step1 offload unit: fused routine
- Current structural risk: helper launch overhead and host sync
- Residency strategy correct? `YES`
- Execution-structure limited? `YES`
- Can step2 keep the current structure? `YES`
- If NO, required rewrite: `N/A`

## Fusion Opportunities:

### Identified Fusions:
- Diagonal, perimeter, and interior stages cannot be legally fused across the panel boundary without breaking the algorithmic dependency chain.
- The safe gain here is helper inlining and keeping the matrix resident, not collapsing the LU stages into one serial kernel.

## Iteration Loop (if present):
- Main offset loop: `127` iterations at `2048`, `255` iterations at `4096`
- Per iteration: one diagonal block, one perimeter chunk loop, one interior chunk loop
- Total: `3` device stages per panel

## SpMV Inner Loop Decision
- N/A

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong offload unit | tiny kernels, poor end-to-end runtime, helper staging | Structural rewrite (4S) |
| Data transfers | >30% transfer time | Move to Strategy C, use `is_device_ptr` |
| Launch overhead | instances >> iterations | Inline helper functions |
| Over-parallelization | Type C slow, outer saturated | Remove inner pragmas |
| Hot kernel | One kernel >50% time | collapse, simd, cache locals |
| Stage parallelization | FAIL verification | Remove pragma from stage loops |

## Strategy (priority)
1. Keep the matrix resident in one `target data` region and preserve the staged LU ordering on device, because transfer volume is already reasonable and the bottleneck is launch/sync overhead.
2. Inline the device helpers and keep the chunk loops on GPU, because that reduces launch fragmentation without changing correctness.
3. Reject async/depend rewrites unless they are stable on this runtime, because the tested dependency-token version segfaulted.

## Micro-opts
[x] `static inline` helpers
[ ] `const`, `restrict`, `firstprivate`, cache locals

## Target
- Runtime: `<0.25s` at `2048`
- Kernels: `382` launches at `2048`, but without extra helper staging
- Memory: `<5%` of runtime at `2048`
- Structural target: reduced helper stages, stable GPU execution

---
# Optimization Plan - Final Summary (Append this after execution)

## Final Performance Summary

### Baseline (Step 2)
- Runtime: `0.490157s` plain run at `2048`
- Main kernel: `127` instances each for the two hot chunk kernels; `47.141ms` combined GPU kernel time at `2048`

### Final (Step 3)
- Runtime: `0.214766s` plain run at `2048`
- Speedup: `2.28x`
- Main kernel: `127` instances each for the two hot chunk kernels; `46.568ms` combined GPU kernel time at `2048`
- Combined GPU+mem cost improved: `YES`

### Optimizations Applied
1. `[x] 4C / 4B`: Marked the device helpers `static inline` to remove avoidable call overhead.
2. `[x] 4A`: Kept the whole matrix inside one `target data` region with device-resident updates.
3. `[ ] 4S`: Async dependency rewrite was attempted and reverted because it segfaulted on this runtime.

### Micro-optimizations Applied
1. `[x] MICRO-OPT`: `static inline` on the three device helpers.
2. `[ ] MICRO-OPT`: `restrict` / `const` caching was not needed after the stable speedup landed.

### Key Insights
- The code was execution-structure limited, not data-movement limited.
- The stable win came from removing helper call overhead and keeping the original staged GPU structure intact.
- The larger 4096 run confirmed the same decision: the safe structure remained stable and the async rewrite was not worth keeping.
- End-to-end runtime improved substantially, while the combined GPU+mem+sync profile improved modestly.
- The final issue was launch/sync overhead, not wrong residency.
- The more aggressive async/dependency rewrite was reverted for compiler/runtime instability.
