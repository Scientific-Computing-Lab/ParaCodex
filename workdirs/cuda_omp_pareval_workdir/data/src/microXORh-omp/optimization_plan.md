# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: 0.01 s (measured with `OMP_TARGET_OFFLOAD=MANDATORY /usr/bin/time -f "%e" ./microXORh.exe 1024 32` after the recent build).
- Main kernel: single `#pragma omp target teams thread_limit(...)` region with an inner `#pragma omp distribute parallel for collapse(2)` covering the full 1024×1024 grid once; no additional CUDA-style kernels exist and the compiled binary offloads to the NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9). The profile log reported only compiler warnings, so no official kernel timing percentages are available.
- Memory transfer: `#pragma omp target data map(to: input[0:N*N]) map(from: output[0:N*N])` moves ≈4 MB in each direction once; there are no per-iteration copies inside the loop.
- Kernel launches: 1 OpenMP offload teams/distribute region executes the stencil exactly once per run.

## Bottleneck Hypothesis (pick 1–2)
- [ ] Transfers too high (CUDA avoided transfers in loop)
- [ ] Too many kernels / target regions (launch overhead)
- [ ] Missing collapse vs CUDA grid dimensionality
- [x] Hot kernel needs micro-opts

## Actions
1. Cache the current row and its neighbor row pointers so each neighbor access reuses the same base address instead of recomputing `i * N + j`; this is a micro-optimization that reduces redundant index math inside the hot loop.
2. Add `__restrict__` aliases for `input`/`output` and simplify the boundary logic to a few boolean guards so the compiler can more easily hoist pointer arithmetic and flatten the `count == 1` store.

# Final Performance Summary - CUDA to OMP Migration

### Baseline (from CUDA)
- CUDA Runtime: not provided in the artifacts (only the `Validation passed.` message was recorded).
- CUDA Main kernel: `cellsXOR`, single launch; timing was not captured.

### OMP Before Optimization
- Runtime: approximately 0.01 s (same as after optimization; the hotspot is extremely light after earlier porting).
- Slowdown vs CUDA: unknown (CUDA timings absent).
- Main kernel: collapsed 2D `target teams` + `distribute parallel for` loop covering the N×N stencil.

### OMP After Optimization
- Runtime: 0.01 s (wall-clock unchanged but now with tighter pointer usage and local row caching).
- Slowdown vs CUDA: unknown.
- Speedup vs initial OMP: ≈1.0× (the code already lived in a good spot; micro-opts keep the runtime minimal while leaving the memory-bound behavior intact).
- Main kernel: same single offload region with pointer aliases/row caching.

### Optimizations Applied
1. Cached row pointers (`row`, `row_above`, `row_below`) so that each neighbor check reuses locally stored addresses rather than recomputing `i * N + j`; `collapse(2)` still spans the full grid.
2. Added `__restrict__` qualifiers on `input_ptr`/`output_ptr`, simplified the `count` branch, and removed the ternary store so the compiler easily recognizes the memory accesses and dependence patterns.

### CUDA→OMP Recovery Status
- [x] Restored 2D/3D grid mapping with collapse
- [x] Matched CUDA kernel fusion structure (single offload + validation loop separation)
- [x] Eliminated excessive transfers (single `target data` region)
- [ ] Still missing: explicit CUDA-style block-level scheduling knobs beyond `thread_limit`

### Micro-optimizations Applied
1. [x] Row pointer caching + neighbor aliases → micro gain (kept redundant index math out of the hotspot).
2. [x] `__restrict__` pointer hints + simplified output store → micro gain (compiler can keep the small working set in registers).

### Key Insights
- The stencil remains memory bound—each of the 1024² iterations still reads four neighbors before writing one value—so the runtime plateaus near 0.01 s even after the micro-optimizations.
- The profile log did not emit CUDA kernel timings; the only GPU-related information was the compiler warning about `-gpu`, so we rely on the observed wall-clock and the verified single-target region for correctness.
- For this simple stencil, OpenMP offload already mirrors the CUDA work distribution, so any further speedup would require hardware-specific scheduling or vectorization hints beyond the current micro-optimizations.
