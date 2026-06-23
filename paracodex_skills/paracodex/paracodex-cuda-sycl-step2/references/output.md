# SYCL Optimization Plan

## Current Metrics
- Runtime: [X]s
- Main kernel: [Y]ms
- Host wait / submission overhead: [description]
- Combined GPU+mem time: [X]ns

## Scalability Check
- Default correctness size: [value]
- Larger practical profiling size: [value or rule]
- Why this size materially exercises the GPU: [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- Hardware basis from `system_info_summary.txt`: [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]
- Small-input sensitive? [YES/NO]
- Larger-input profiling changed the decision? [YES/NO]

## Bottlenecks
- [ ] **USM vs Buffers:** Is data movement implicit (Buffers)? Explicit USM is often faster.
- [ ] **Sub-groups:** Are we using `sub_group` primitives (warp equivalents)?
- [ ] **Work-group Size:** Is it tuned for the target device (e.g., 256 for PVC)?
- [ ] **JIT Overhead:** Initial compilation time?
- [ ] **Wrong submission structure:** Too many tiny submissions or `.wait()` calls?

## Optimization Actions
1. [ACTION]: Convert Buffers/Accessors to Unified Shared Memory (USM).
2. [ACTION]: Use `sub_group` shuffle instead of local memory reductions.
3. [ACTION]: Tune work-group size (multiple of sub-group size).

## Final Summary
- Baseline Runtime: [X]s
- Optimized Runtime: [Y]s
- Speedup: [X/Y]x
- Combined GPU+mem cost improved: [YES/NO]
- End-to-end runtime improved, not just kernel timing: [YES/NO]
- Final structure still plausible at larger size: [YES/NO]
