# Performance Analysis

## Current Metrics
- Runtime: [X]s
- End-to-end runtime source: [plain run / profiled run / both]
- Main kernel: [name], [Y]% GPU, [Z] instances
- Memory transfer: [%] time, [MB] total
- Kernel launches: [count]
- Host overhead / waits: [X]% or [description]
- Combined GPU+mem time: [X]ns

## Scalability Check
- Default correctness size: [value]
- Larger practical profiling size: [value or rule]
- Why this size materially exercises the GPU: [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- Hardware basis from `system_info_summary.txt`: [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]
- Small-input sensitive? [YES/NO]
- Larger-input profiling changed the decision? [YES/NO]

## Structural Check
- Step1 offload unit: [single loop nest / fused routine / helper-by-helper]
- Current structural risk: [none / fragmented tiny kernels / helper launch overhead / host sync / pointer-heavy layout]
- Residency strategy correct? [YES/NO]
- Execution-structure limited? [YES/NO]
- Can step2 keep the current structure? [YES/NO]
- If NO, required rewrite: [fuse helpers / flatten buffers / replace data strategy / inline hot path]

## Fusion Opportunities:

### Identified Fusions:
- Lines X-Y: init → FUSE (same bounds)
- Lines A-B: compute+reduce → FUSE (register value)

## Iteration Loop (if present):
- Main: lines [X-Y], [N] iters
- SpMV line Z: [N] times
- Update line W: [N] times
- Total: [N×M] ops

## SpMV Inner Loop Decision
- Avg nonzeros per row (NONZER): [value from code/headers]
- If NONZER < 50: Keep inner loop SERIAL
- If NONZER > 100: Add `#pragma omp loop reduction`

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong offload unit | tiny kernels, poor end-to-end runtime, helper staging | Structural rewrite (4S) |
| Data transfers | >30% transfer time | Move to Strategy C, use is_device_ptr |
| Launch overhead | instances >> iterations | Inline helper functions |
| Over-parallelization | Type C slow, outer saturated | Remove inner pragmas |
| Hot kernel | One kernel >50% time | collapse, simd, cache locals |
| Stage parallelization | FAIL verification | Remove pragma from stage loops |

## Strategy (priority)
1. [ACTION]: [what] - [why] - expect [gain]
2. [ACTION]: [what] - [why] - expect [gain]

## Micro-opts
[ ] const, restrict, firstprivate, cache locals

## Target
- Runtime: [X]s
- Kernels: ~[N] for [M] iters
- Memory: <[X]%
- Structural target: [fused hot path / reduced helper stages / no timed-path host sync]

---
# Optimization Plan - Final Summary (Append this after execution)

## Final Performance Summary

### Baseline (Step 2)
- Runtime: [X]s
- Main kernel: [Y] instances, [Z]ms total

### Final (Step 3)
- Runtime: [X]s
- Speedup: [X]x
- Main kernel: [Y] instances, [Z]ms total
- Combined GPU+mem cost improved: [YES/NO]

### Optimizations Applied
1. [] [ACTION 4S]: [description] → [±X%]
2. [] [ACTION]: [description] → [±X%]
3. [] [ACTION]: REVERTED (slower)

### Micro-optimizations Applied
1. [] [MICRO-OPT]: [description] → [±X%]
2. [] [MICRO-OPT]: REVERTED (slower)

### Key Insights
- [Most impactful optimization]
- [Remaining bottlenecks]
- [Whether end-to-end runtime and not just GPU kernel time improved]
- [Whether combined GPU+mem cost improved]
- [Whether the final issue was wrong residency or launch/sync overhead]
- [Whether a more aggressive async/dependency rewrite was reverted for compiler/runtime instability]

---

# Bottleneck Checklist & Analysis (Use this for planning)

## [ ] 1. Data Management Issue (CRITICAL - fix first!)
- Transfer ratio: [actual/expected] = [X]x
- If >2.5x: Data management wrong
- Root cause: [from data_plan.md verification]
- Fix: [specific action - e.g., offload missing functions, move scratch to device]
- Expected gain: [X]x speedup

## [ ] 2. Wrong Offload Unit / Structural Mismatch
- End-to-end runtime vs GPU kernel time: [compare]
- If GPU time is low but runtime is still poor, the structure is wrong
- Root cause: [helper staging / tiny kernels / host sync / pointer layout]
- Fix: Structural rewrite (ACTION 4S)
- Expected gain: [X]x

## [ ] 3. Kernel Launch Overhead
- Kernel instances: [count]
- Expected: ~[N] for [N] iterations
- If instances >> N: Helper functions called in loop
- Root cause: [which functions - e.g., device_spmv, device_axpy]
- Fix: Inline operations in loop (ACTION 4C)
- Expected gain: [X]x (reduce [Y] launches to [Z])

## [ ] 4. Memory Transfer Bottleneck
- Transfer time: [X]% of total time
- If >50% AND ratio <2x: Transfers correct but dominant
- Fix: Optimize data movement (ACTION 4A)
- Expected gain: [X]%

## [ ] 5. Hot Kernel Performance
- Kernel: [name] takes [X]% GPU time, [Y]ms avg
- Root cause: [inefficient algorithm/missing optimization]
- Fix: [collapse/simd/cache/etc.] (ACTION 4B)
- Expected gain: [X]% faster kernel

## [ ] 6. Type C Parallelization Error
- Verification: [PASS/FAIL]
- If FAIL: Wrong stage loop parallelization
- Fix: Remove inner pragmas (ACTION 4D)

## [ ] 7. Over-Parallelization (saturated outer loops)
- Outer parallelized iterations: [K × J = ?]
- Saturation threshold: [Saturation threshold]
- IF saturated AND inner has pragma → REMOVE inner pragmas
- Symptoms: Type C kernel slower after (or before) "optimization", GPU over-saturated
- Fix: Remove collapse/omp loop from inner/stage/writeback loops
- Expected gain: [X]%
