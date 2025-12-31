# Performance Analysis

## Current Metrics
- Runtime: 0.57s (CLASS A)
- Main kernel: nvkernel__ZN25_INTERNAL_4_ft_c_edb84aa710cffts1_negEiiii_F1L682_12, 34.1% GPU, 20 instances
- Memory transfer: ~6% time, 1613.76 MB total (H2D)
- Kernel launches: 127 (cuLaunchKernel)

## Fusion Opportunities:

### Identified Fusions:
- None – FFT stages (cffts1/2/3) are separated by data dependencies; init/index map use different data sources.

## Iteration Loop (if present):
- Main: lines 281-295, 6 iters
- FFT pipeline per iter: fft(-1) -> checksum once per iter
- Total ops: dominated by 3D FFT butterflies over 256x256x128 grid

## SpMV Inner Loop Decision
- No SpMV present in this benchmark (FFT only)

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Data transfers | ~0.22s H2D (1.6 GB) upfront | Generate twiddle/IC directly on device to drop H2D volume |
| Launch overhead | cuLaunchKernel=127 vs 6 iterations | Inline/collapse simple loops to reduce helper launches |
| Over-parallelization | Not observed | N/A |
| Hot kernel | cffts1_neg 34% time | Cache strides, use thread_limit to improve occupancy on Ada (cc 8.9) |
| Stage parallelization | Verified OK | Keep stage ordering intact |

## Strategy (priority)
1. Compute twiddle and initial conditions directly on device (keep persistent device arrays) to eliminate ~0.22s H2D and 1.6 GB transfer; expect ~10-15% runtime drop.
2. Tune hot FFT kernels (cffts1/2/3) with cached strides and `thread_limit(256)` to reduce index arithmetic and improve SM occupancy on RTX 4060 (cc 8.9); expect a few percent gain.
3. Micro-opt: add const locals for offsets and reuse loaded twiddle values in evolve/init to reduce redundant math.

## Micro-opts
[ ] const, restrict, firstprivate, cache locals

## Target
- Runtime: ≤0.50s (CLASS A)
- Kernels: ~20 cffts1/2/3 invocations for 6 iters
- Memory: <3% time in H2D after device-side init

# Final Performance Summary

### Baseline (Step 2)
- Runtime: 0.57s (CLASS A)
- Main kernel: cffts1_neg, 20 instances, 1.26s total (profile log provided)

### Final (Step 3)
- Runtime: 0.57s (CLASS A)
- Speedup: ~1.0x (runtime unchanged; transfer volume reduced)
- Main kernel: cffts1_neg, 20 instances, 1.29s total (CLASS B profile)
- Memory transfers: 155.9ms H2D over 6 copies (1.07 GB total)

### Optimizations Applied
1. [x] Device twiddle computation: move compute_indexmap to GPU to drop large H2D copy (now ~0.4% kernel time).
2. [x] FFT kernels: added `thread_limit(256)` and cached plane/base offsets to trim index arithmetic and improve occupancy.
3. [ ] Device RNG path: reverted to host RNG to preserve checksums; data path unchanged for correctness.

### Micro-optimizations Applied
1. [x] Cached locals: plane/base offsets reused in init/evolve/FFT kernels to cut repeated multiplies.
2. [ ] restrict/const qualifiers: not applied (interfaces unchanged).

### Key Insights
- Biggest gain came from eliminating the host-to-device twiddle copy; H2D volume dropped from 1.61 GB (8 copies) to 1.07 GB (6 copies).
- FFT stage kernels remain dominant (~86% GPU time); further speedups will require algorithmic restructuring or fusing stage work.
- RNG path is sensitive—device-side generation altered checksums; keep host RNG unless a verified device implementation is available.
