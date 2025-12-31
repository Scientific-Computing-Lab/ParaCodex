# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: 0.276s (aggregated osrt wait ~276,000,000 ns after the collapse rewrite, down from 0.606s before optimizing).
- Main kernel: `cellsXOR_device`, 100% GPU time, 1 instance (45,184 ns average).
- Memory transfer: 83% of CUDA memory time is Device→Host (~1.8 ms) and 17% Host→Device (~0.37 ms), 4.194 MB moved each direction.
- Kernel launches: 1 (`cuLaunchKernel`).

## Bottleneck Hypothesis (pick 1–2)
- [ ] Transfers too high (CUDA avoided transfers in loop)
- [x] Missing collapse vs CUDA grid dimensionality (current kernel is flattened, CUDA version used 2D grid/block coverage)
- [x] Hot kernel needs micro-opts (pointer aliasing, repeated arithmetic, thread mapping hints)

## Actions (1–3 max)
1. [X] Rebuilt the target region as `#pragma omp target teams distribute parallel for collapse(2)` with canonical signed loops and `thread_limit(256)` so the compiler sees a 2D grid; the new mapping is numerically correct and reduced host wait time (0.276s runtime).
2. [X] Annotated `cellsXOR_device` arguments with `__restrict__` and kept locally scoped indices to help the optimizer trust the load/store pattern without changing the data-transport strategy.
