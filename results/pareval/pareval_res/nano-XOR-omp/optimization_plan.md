# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: 0.25s (single run of `./nanoXOR.exe 1024 32` with `OMP_TARGET_OFFLOAD=MANDATORY` after the micro-optimizations).
- Main kernel: `cellsXOR` (`nvkernel__Z8cellsXORPKiPim_F1L14_2`), still 100.0% of CUDA GPU kernel time with 1 launch.
- Memory transfer: 78.9% (1.998ms) is device→host, 21.1% (0.534ms) host→device; both still pull ~4.194MB each (the implicit OpenMP map still matches the CUDA data strategy).
- Kernel launches: 1 (single target kernel sweep).
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9).

## Bottleneck Hypothesis (pick 1–2)
- [x] Transfers too high (full-grid maps still require two 4.2MB round trips and already dominate runtime; can only be micro-optimized).
- [x] Hot kernel needs micro-opts (neighbor checks in `cellsXOR` are compute-light but repeated arithmetic and hash indexing could benefit from local accumulation).
- [ ] Too many kernels / target regions (single target loop per run).
- [ ] Missing collapse vs CUDA grid dimensionality (collapse already mirrors CUDA grid). 

## Actions (1–3 max)
1. Cache `i*N` outside the inner loop so each thread reuses the row base when scanning its four neighbors, reducing index multiplications inside the target kernel - expected ~1–2% at most.
2. Annotate the kernel pointers as `__restrict__` and use updating `const` locals to help LLVM/Clang vectorizers better understand the lack of aliasing and keep the simple pattern inline on the GPU - expected micro-gains.
3. [optional] Adjust collapse or loop scheduling if warranted after micro-ops (no change planned yet since the loop already matches the CUDA grid). 
