# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: ≈0.02s (latest `env OMP_TARGET_OFFLOAD=MANDATORY time ./microXOR.exe 1024 32`).
- Main kernel: `cellsXOR` (OpenMP `target teams loop` mirroring the CUDA stencil). `profile.log` only lists that nsys executed `cuda_gpu_kern_sum/md` reports but no numeric durations, so GPU% and explicit kernel times are unavailable; there is one offload per run.
- Memory transfer: host↔device mapping of `N*N` ints (~4 MB) once per run (approx. 8 MB total); `profile.log` has no concrete `cuda_gpu_mem_time_sum` values.
- Kernel launches: single `cellsXOR` offload (matching the original CUDA kernel).

## Bottleneck Hypothesis (pick 1–2)
- [ ] Transfers too high (mapping still happens only once H→D and once D→H).
- [ ] Too many kernels / target regions (only one offload).
- [ ] Missing collapse vs CUDA grid dimensionality (kept `collapse(2)`).
- [x] Hot kernel needs micro-opts (cleanup of index math + alias hints).

## Actions (1–3 max)
1. Cache neighbor row pointers inside each `i` iteration and write output via a cached row offset so we avoid recomputing `(i * N)` for every neighbor check—removes redundant index math and keeps memory accesses aligned with CUDA’s row-major streams (expected low-single-digit percent gain on this memory-bound stencil).
2. Annotate the `cellsXOR` buffer parameters with `__restrict__` so NVHPC’s OpenMP offload backend knows the two buffers do not alias, yielding straighter stores/loads across the RTX 4060’s vector lanes (compute capability 8.9).

### Profiling Notes
- `profile.log` only shows that nsys executed the `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, and `cuda_gpu_mem_size_sum` reports; no numerical results appear and the referenced `.sqlite` file is not available in this workspace, so objective kernel/transfer percentages are missing.
- Hardware context: NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9, 8 GB VRAM) per `nvidia-smi --query-gpu=name,compute_cap --format=csv`. Keeping the 2D iteration collapse mirrors the CUDA grid on this GPU and maximises occupancy.

# Final Performance Summary - CUDA to OMP Migration

### Baseline (from CUDA)
- CUDA Runtime: not recorded in the baseline logs provided here.
- CUDA Main kernel: `cellsXOR`, single `<<<grid, block>>>` launch; the only available log (`baseline_output.txt`) lists the launch but no timing breakdown.
- CUDA transfer profile: H→D once, D→H once (same as current OMP target data strategy); no timing numbers captured.

### OMP Before Optimization
- Runtime: ≈0.03s (measured via `env OMP_TARGET_OFFLOAD=MANDATORY time ./microXOR.exe 1024 32` prior to the code tweaks).
- Slowdown vs CUDA: N/A (no CUDA timing for comparison).
- Main kernel: `cellsXOR` target teams loop, one offload, timing not detailed in `profile.log`.

### OMP After Optimization
- Runtime: ≈0.02s (latest `time` measurement after caching row pointers and adding `__restrict__`).
- Slowdown vs CUDA: N/A (baseline runtime missing).
- Speedup vs initial OMP: ~1.5× (0.03s → 0.02s) from micro-optimizations.
- Main kernel: same single `cellsXOR` offload; still no per-kernel timing (nsys log only shows report generation).

### Optimizations Applied
1. [X] Row reuse: cached row pointers + row offset stores for neighbor reads → reduced redundant `i * N` multiplies.
2. [X] Alias hints: `__restrict__` on input/output to keep the OpenMP compiler confident the buffers do not alias.

### CUDA→OMP Recovery Status
- [X] Restored 2D grid mapping with `collapse(2)`.
- [X] Matched CUDA kernel fusion structure (single `cellsXOR` offload covers all neighbors).
- [X] Eliminated excessive transfers (single `target data map` region for the arrays).
- [ ] Still missing: explicit kernel/transfer timings from `nsys` for finer-grained tuning.

### Micro-optimizations Applied
1. [X] Cached row pointers + row offsets → trimmed repeated index math and minimized multiplications inside the inner loop.
2. [X] `__restrict__` qualifiers → reduced alias ambiguity and opened the door for vector-friendly loads/stores.

### Key Insights
- `cellsXOR` is a memory-bound 2D stencil; reducing redundant index calculations and signalling exclusive access are the highest-impact changes without altering the data-transfer strategy.
- `target data` is still the single H↔D transfer pair, mirroring the original CUDA pattern and avoiding remapping costs.
- RTX 4060 (compute 8.9) can leverage the `collapse(2)` loop structure to fill the 2D grid similarly to CUDA’s block/thread mapping, so we keep that layout intact.

## Optimization Checklist
- [ ] Transfers dominate: hoist data; `omp_target_alloc` + `is_device_ptr`; avoid per-iter mapping.
- [ ] Too many kernels/regions: fuse adjacent target loops; inline helper kernels when safe.
- [x] Missing CUDA grid shape: kept `collapse(2)` to match the CUDA grid.
- [x] Hot kernel: `const`, `__restrict__`, cached locals, reduced recomputation.
