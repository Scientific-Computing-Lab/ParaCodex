# CUDA Optimization Plan

## Current Metrics
- Runtime: [X]s
- Hotspot kernel: [name], [Y]ms
- Combined GPU+mem time: [X]ns
- End-to-end runtime source: [plain run / profiled run / both]
- Host overhead / waits: [description]

## Scalability Check
- Default correctness size: [value]
- Larger practical profiling size: [value or rule]
- Why this size materially exercises the GPU: [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- Hardware basis from `system_info_summary.txt`: [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]
- Small-input sensitive? [YES/NO]
- Larger-input profiling changed the decision? [YES/NO]

## Bottlenecks
- [ ] **Wrong Kernel Decomposition:** Too many tiny kernels or avoidable staging?
- [ ] **Memory Bandwidth:** Is the code bound by global memory access?
- [ ] **Compute Bound:** Is it instruction heavy?
- [ ] **Latency:** Are threads waiting on dependencies?

## Optimization Actions
1. [ACTION]: Structural rewrite if needed.
2. [ACTION]: Use Shared Memory (`__shared__`) to reduce global access.
3. [ACTION]: Coalesce Global Memory Accesses.
4. [ACTION]: Use Fast Math Intrinsics.
5. [ACTION]: Minimize Host-Device Transfers (Unified Memory or Async).

## Final Summary
- Baseline Serial Runtime: [X]s
- Initial CUDA Runtime: [Y]s
- Optimized CUDA Runtime: [Z]s
- Speedup: [X/Z]x
- Combined GPU+mem cost improved: [YES/NO]
- End-to-end runtime improved, not just kernel time: [YES/NO]
- Final structure still plausible at larger size: [YES/NO]
