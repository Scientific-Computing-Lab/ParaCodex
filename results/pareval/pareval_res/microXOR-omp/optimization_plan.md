# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: 0.24 s (measured by `/usr/bin/time` with `OMP_TARGET_OFFLOAD=MANDATORY`; device work is ~2.25 ms)
- Main kernel: `nvkernel__Z8cellsXORPKiPim_F1L3_2`, 100% GPU share, 1 instance, avg 32 543 ns
- Memory transfer: 2.22 ms total (82.6% DtoH, 17.4% HtoD) for 4.194 MB per direction
- Kernel launches: 1
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9)

## Bottleneck Hypothesis (pick 1–2)
- [ ] Transfers too high (CUDA avoided transfers in loop)
- [ ] Too many kernels / target regions (launch overhead)
- [ ] Missing collapse vs CUDA grid dimensionality
- [x] Hot kernel needs micro-opts

## Actions (1–3 max)
1. Cache the flattened `idx = i * N + j` inside `cellsXOR` so each neighbor check reuses one computed address rather than repeating 4 multiplications; this trims address-math overhead on the hot stencil and keeps the index in registers, so the kernel spends fewer cycles on integer arithmetic (expected sub-percent gain but still recovers some of the CUDA arithmetic density).

# Final Performance Summary - CUDA to OMP Migration

### Baseline (from CUDA)
- CUDA Runtime: N/A (not profiled in this workspace)
- CUDA Main kernel: `cellsXOR<<<...>>>` (dense 2D stencil that originally covered the entire grid)
- CUDA Main kernel time: (not captured; assume kernels dominated GPU time as before)

### OMP Before Optimization
- Runtime: ~0.24 s measured via `/usr/bin/time` with `OMP_TARGET_OFFLOAD=MANDATORY`; runtime was dominated by host RNG/validation since the GPU portion was only ~2.25 ms.
- Slowdown vs CUDA: unknown (CUDA measurements absent); the kernel mirrors the CUDA layout so the device portion is expected to be close.
- Main kernel: `nvkernel__Z8cellsXORPKiPim_F1L3_2`, 1 launch, avg 32 543 ns.

### OMP After Optimization
- Runtime: 0.24 s; micro-optimizations did not change the host-dominated wall time but slightly reduced the kernel’s index arithmetic pressure.
- Slowdown vs CUDA: unknown (probe limited by host overhead and missing CUDA baseline).
- Speedup vs initial OMP: ~1× (within measurement noise).
- Main kernel: unchanged from before, still the globally offloaded cellsXOR loop.

### Optimizations Applied
1. Cached the flattened index inside `cellsXOR` so each neighbor check reuses a single `i * N + j` instead of recomputing it four times, reducing redundant multiplications and keeping the index register-resident.

### CUDA→OMP Recovery Status
- [x] Restored 2D grid mapping with `collapse(2)` inside the offloaded `cellsXOR`.
- [x] Matched CUDA kernel fusion structure (single target region covers the whole stencil pass).
- [ ] Still missing: host RNG/validation parallelization to offset the 0.24 s wall time that is mostly spent off-GPU.

### Micro-optimizations Applied
1. [x] [MICRO-OPT]: Cached flattened index arithmetic in `cellsXOR` → lowered redundant address computations; device runtime impact is within noise but is now conceptually tighter.

### Key Insights
- Most impactful optimization remains to move more host-side work (RNG seeding/validation) off the critical path; the GPU kernel is already tiny.
- Remaining bottleneck: host RNG/state setup and verification dominate, so GPU-side gains must be large enough to matter amid the 0.24 s total.
- OMP limitation: `#pragma omp target teams loop` already mirrors CUDA’s grid, so further gains require host work parallelism or overlapping data transfers.
