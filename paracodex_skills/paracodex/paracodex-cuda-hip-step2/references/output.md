# HIP Optimization Plan

## Current Metrics
- Runtime: [X]s
- Main kernel: [Y]ms

## Bottlenecks
- [ ] **LDS Bank Conflicts:** AMD GCN has 32 banks (vs CUDA 32, but mapping differs).
- [ ] **Occupancy:** Limited by VGPRs/SGPRs (Vector/Scalar General Purpose Registers).
- [ ] **Divergence:** Wavefront size (64 on CDNA/GCN, 32 on RDNA).
- [ ] **Memory:** Global memory access coalescing.

## Optimization Actions
1. [ACTION]: Tune block size (Workgroup size) for AMD.
   - Recommended: Multiple of 64 (usually 64, 128, 256).
2. [ACTION]: Use ROCm-specific intrinsics/libraries.
3. [ACTION]: Optimize LDS usage (padding).

## Final Summary
- Baseline Runtime: [X]s
- Optimized Runtime: [Y]s
- Speedup: [X/Y]x
