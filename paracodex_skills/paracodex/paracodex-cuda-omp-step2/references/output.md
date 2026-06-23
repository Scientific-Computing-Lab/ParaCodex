# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: [X]s
- Main kernel: [name], [Y]% GPU, [Z] instances
- Memory transfer: [%] time, [MB] total
- Combined GPU+mem time: [X]ns
- Kernel launches: [count]
- Host overhead / waits: [description]

## Scalability Check
- Default correctness size: [value]
- Larger practical profiling size: [value or rule]
- Why this size materially exercises the GPU: [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- Hardware basis from `system_info_summary.txt`: [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]
- Small-input sensitive? [YES/NO]
- Larger-input profiling changed the decision? [YES/NO]

## Structural Check
- Step1 offload unit: [single region / fused routine / helper-by-helper]
- Current structural risk: [none / fragmented regions / helper launch overhead / bad CUDA recovery]
- Residency strategy correct? [YES/NO]
- Execution-structure limited? [YES/NO]
- Can step2 keep the current structure? [YES/NO]
- If NO, required rewrite: [fuse / flatten / replace data strategy]
- Combined GPU+mem regression vs baseline? [YES/NO]

## Bottleneck Hypothesis (pick 1–2)
- [ ] Transfers too high (CUDA avoided transfers in loop)
- [ ] Too many kernels / target regions (launch overhead)
- [ ] Missing collapse vs CUDA grid dimensionality
- [ ] Hot kernel needs micro-opts

## Actions (1–3 max)
1. [ACTION]: [what] - [why] - expected [gain]
2. [ACTION]: ...

---

# Final Performance Summary - CUDA to OMP Migration

## Baseline (from CUDA)
- CUDA Runtime: [X]s (if available)
- CUDA Main kernel: [Y] launches, [Z]ms total

## OMP Before Optimization
- Runtime: [X]s
- Slowdown vs CUDA: [X]x
- Main kernel: [Y] instances, [Z]ms total

## OMP After Optimization
- Runtime: [X]s
- Slowdown vs CUDA: [X]x (target <1.5x)
- Speedup vs initial OMP: [X]x
- Main kernel: [Y] instances, [Z]ms total
- Combined GPU+mem cost improved: [YES/NO]

## Optimizations Applied
1. [X] [ACTION]: [description] → [±X%] [recovered CUDA pattern Y]
2. [X] [ACTION]: REVERTED (slower)

## CUDA→OMP Recovery Status
- [X] Restored 2D/3D grid mapping with collapse
- [X] Matched CUDA kernel fusion structure
- [X] Eliminated excessive transfers (matched CUDA pattern)
- [ ] Still missing: [any CUDA optimizations that couldn't be recovered]

## Micro-optimizations Applied
1. [X] [MICRO-OPT]: [description] → [±X%]
2. [X] [MICRO-OPT]: REVERTED (slower)

## Key Insights
- [Most impactful optimization - relate to CUDA pattern]
- [Remaining bottlenecks vs CUDA]
- [OMP limitations compared to CUDA]
- [Whether end-to-end runtime and not just GPU metrics improved]
- [Whether the final issue was wrong residency or launch/sync overhead]
- [Whether a more aggressive async/dependency rewrite was reverted for compiler/runtime instability]
