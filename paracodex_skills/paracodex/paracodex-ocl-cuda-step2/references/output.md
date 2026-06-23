# Performance Analysis - OpenCL to CUDA

## Current Metrics
- Total runtime: [X]s
- Main kernel: [name], [Y]ms total, [Z] calls
- Memory transfer: [W]ms, [V]MB total
- Combined GPU+mem time: [X]ns
- Host overhead / waits: [description]

## Scalability Check
- Default correctness size: [value]
- Larger practical profiling size: [value or rule]
- Why this size materially exercises the GPU: [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- Hardware basis from `system_info_summary.txt`: [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]
- Small-input sensitive? [YES/NO]
- Larger-input profiling changed the decision? [YES/NO]

## OpenCL vs CUDA Comparison
- OpenCL runtime: [X]s (baseline)
- CUDA runtime: [Y]s (current)
- Speedup: [X/Y]x
- Target: >1.0x (CUDA should be faster or equal)

## Bottleneck Analysis

### [ ] 1. Block Size Suboptimal
- Current: [X] threads per block
- Recommended: Multiple of 32 (warp size), typically 128-512
- Check occupancy: Use CUDA Occupancy Calculator or nvprof
- **Fix:** Tune block dimensions to maximize occupancy
- Expected gain: [X]%

### [ ] 2. Memory Transfer Overhead
- Transfer time: [X]% of total
- If >50%: Consider:
  - Pinned memory (cudaHostAlloc)
  - Async transfers (cudaMemcpyAsync + streams)
  - Unified memory (cudaMallocManaged)
- Expected gain: [X]%

### [ ] 3. Slow Math Functions
- Using sinf/cosf vs __sinf/__cosf
- **Fix:** Use intrinsics if precision allows
- Expected gain: [X]%

### [ ] 4. Shared Memory Bank Conflicts
- Check __shared__ memory access patterns
- Stride should avoid 32-bank conflicts
- Expected gain: [X]%

### [ ] 5. Warp Divergence
- Check if/else branches
- **Fix:** Restructure to minimize divergence
- Expected gain: [X]%

### [ ] 6. Memory Coalescing
- Sequential threads should access sequential memory
- Expected gain: [X]%

### [ ] 7. Wrong Kernel Structure
- If kernel time is good but total runtime is poor, check tiny kernels, host synchronization, or weak decomposition.
- **Fix:** Structural rewrite before micro-tuning.
- Expected gain: [X]%

## Optimization Strategy (priority order)
1. [ACTION]: [description] - expected [X]% gain
2. [ACTION]: [description] - expected [X]% gain

---

# Final Performance Summary

## Baseline (OpenCL)
- Runtime: [X]s
- Main kernel: [Y]ms

## Initial CUDA
- Runtime: [A]s
- Speedup vs OpenCL: [X/A]x
- Main kernel: [B]ms

## Optimized CUDA
- Runtime: [C]s
- Speedup vs OpenCL: [X/C]x (target: >1.0x)
- Speedup vs initial CUDA: [A/C]x
- Main kernel: [D]ms

## Optimizations Applied
1. [X] [ACTION]: [description] → [±X%]
2. [X] [ACTION]: [description] → [±X%]

## CUDA-Specific Optimizations
1. [X] Fast math intrinsics (__sinf, __cosf) → [±X%]
2. [X] Warp-level primitives (shuffle) → [±X%]
3. [X] Block size tuned (occupancy: [X]%) → [±X%]
4. [X] Unified memory (simplified code) → [±X%]

## Combined Cost
- Combined GPU+mem cost improved: [YES/NO]

## Device-Specific Metrics
- Device: [name]
- Optimal block size: [Z]
- Achieved occupancy: [X]%
- Branch efficiency: [Y]%
- Memory coalescing: [Z]%
- End-to-end runtime improved, not just kernel metrics: [YES/NO]
- Final structure still plausible at larger size: [YES/NO]
