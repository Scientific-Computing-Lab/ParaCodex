# Performance Analysis

## Current Metrics
- Runtime: 0.530s
- End-to-end runtime source: profiled run
- Main kernel: `nvkernel__Z12nw_optimizedPiS_S_iii_F1L102_2` and `nvkernel__Z12nw_optimizedPiS_S_iii_F1L153_4`, 255 instances total
- Memory transfer: 13.6% of CUDA API time, 33.587 MB HtoD + 16.794 MB DtoH
- Kernel launches: 255
- Host overhead / waits: 97.1% in `wait`; 16.39 ms `cuStreamSynchronize`
- Combined GPU+mem time: 36,147,651 ns

## Scalability Check
- Default correctness size: 2048
- Larger practical profiling size: 4096
- Small-input sensitive? YES
- Larger-input profiling changed the decision? NO

## Structural Check
- Step1 offload unit: fused routine
- Current structural risk: launch / host-sync overhead
- Residency strategy correct? YES
- Execution-structure limited? YES
- Can step2 keep the current structure? YES
- If NO, required rewrite: none; the more aggressive rewrite was rejected and reverted

## Fusion Opportunities:

### Identified Fusions:
- None that are legal across the wavefront dependence chain

## Iteration Loop (if present):
- Main: 128 + 127 diagonal iterations
- Per diagonal: up to 128 tiles
- Total: 255 tile-launch iterations in the timed path

## SpMV Inner Loop Decision
- Avg nonzeros per row (NONZER): N/A
- If NONZER < 50: Keep inner loop SERIAL
- If NONZER > 100: Add `#pragma omp loop reduction`

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong offload unit | not present | keep step1 data strategy |
| Data transfers | 0.1 to 0.2 of CUDA API time | no change |
| Launch overhead | 255 kernels and 512 stream syncs | keep fused helper path, avoid unstable rewrites |
| Over-parallelization | not present | no change |
| Hot kernel | two small kernels, each < 6 ms total | micro-tune only |
| Stage parallelization | pass | no change |

## Strategy (priority)
1. [REVERTED]: single-device-region launch reduction - produced incorrect traceback and worse combined CUDA cost
2. [KEEP]: retain the step1 fused wavefront layout and only apply low-risk micro-opts - preserves correctness and avoids extra syncs

## Micro-opts
[x] static inline `maximum`
[ ] cache locals / `restrict` / `firstprivate` beyond the step1 baseline

## Target
- Runtime: stable, no regression from step1
- Kernels: 255 for 2048 input
- Memory: about 0.1 of CUDA API time
- Structural target: stable fused wavefront path

---
# Optimization Plan - Final Summary

## Final Performance Summary

### Baseline (Step 2)
- Runtime: 0.489s
- Main kernel: 128 + 127 instances, 9.80 ms total

### Final (Step 3)
- Runtime: 0.530s
- Speedup: 0.92x
- Main kernel: 128 + 127 instances, 9.80 ms total
- Combined GPU+mem cost improved: NO

### Optimizations Applied
1. [x] [MICRO-OPT]: `maximum` kept device-callable and inline-friendly
2. [ ] [ACTION 4S]: attempted phase-level launch reduction - REVERTED (incorrect traceback / slower combined cost)
3. [ ] [ACTION]: no additional accepted structural change

### Micro-optimizations Applied
1. [x] [MICRO-OPT]: retain the step1 fused wavefront data residency and device mapping
2. [ ] [MICRO-OPT]: cache tile-local indices - REVERTED with the structural experiment

### Key Insights
- The step1 offload/data strategy was already the correct one; the problem was launch and sync overhead, not residency.
- A more aggressive single-device-region rewrite was unstable for this kernel and was reverted.
- End-to-end runtime did not improve materially on the final accepted code.
- Combined GPU+mem cost did not improve materially; the final accepted code is stable rather than faster.
- The remaining bottleneck is launch / sync overhead from the unavoidable 255-diagonal wavefront schedule.
