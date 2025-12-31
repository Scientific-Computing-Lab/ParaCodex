# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: 0.371 seconds (event-based baseline run with 1 warmup + timed iteration recorded in `current_output.txt` and `profile.log`).
- Main kernel: `xs_lookup_kernel_baseline`, with OpenMP timers reporting 111.55 ms of compute; two kernel launches (warmup + timed) still dominate the profile.
- Memory transfer: host→device 134.39 ms (~36% of runtime) for the 241 MB `SimulationData` copy; device→host 2.05 ms for the verification reduction.
- Kernel launches: 2 (warmup + timed); `profile.log` now mirrors the latest OpenMP run so the previous compilation failure is no longer present.
- Target GPU: NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, compute capability 8.9 from `nvidia-smi --query-gpu=name,compute_cap`).

## Bottleneck Hypothesis (pick 1–2)
- [x] Transfers too high (the ~134 ms host→device copy is ~36% of the run even though it is done only once).
- [ ] Too many kernels / target regions.
- [ ] Missing collapse vs CUDA grid dimensionality.
- [x] Hot kernel needs micro-opts (`xs_lookup_kernel_baseline` still takes 162 ms versus the ~1.7 ms CUDA kernel, so there is room for lowering per-lookup work).

## Actions (1–3 max)
1. Micro-optimize the RNG-based material sampling: replace the nested recompute of the `dist` array inside `pick_mat` with an explicit chain of precomputed thresholds so each thread tests against hardcoded cutoffs instead of rebuilding the prefix sum. This reduces instruction count/register usage inside `xs_lookup_kernel_baseline` and should recover a few percent of kernel time without touching data movement.
2. Assess transfer persistence: if host→device remains >25% of runtime in subsequent iterations, keep exploring `omp_target_alloc` lifetimes or overlapping copies in future experiments, but leave the existing one-time data strategy untouched for now.

# Final Performance Summary - CUDA to OMP Migration

### Baseline (from CUDA)
- CUDA Runtime: 0.147 seconds (CUDA baseline run that produced `baseline_output.txt`).
- CUDA Main kernel: `xs_lookup_kernel_baseline`, kernel_ms 1.662 and kernel launches 2 (1 warmup, 1 timed) with the same data/validation sequence.
- CUDA transfers: host→device 139.497 ms, device→host 0.450 ms for the verification buffer.

### OMP Before Optimization
- Runtime: 0.472 seconds (pre-optimization OpenMP run recorded before the pick_mat change).
- Slowdown vs CUDA: ~3.2×.
- Main kernel: `xs_lookup_kernel_baseline`, kernel_ms 162.58, 2 launches.
- Host→device copy: 140.27 ms (around 30% of total) with the same 241 MB data footprint.

### OMP After Optimization
- Runtime: 0.371 seconds with `xs_lookup_kernel_baseline`.
- Slowdown vs CUDA: ~2.52× (target <1.5× remains unmet).
- Speedup vs initial OMP: ~1.27×.
- Kernel_ms: 111.55, host→device_ms: 134.39, device→host_ms: 2.05 (event-based baseline run now matches `profile.log`).
- Kernel launches: 2 (warmup + timed); verification checksum remains 299541 (Valid).

### Optimizations Applied
1. [X] Micro-optimization: replaced `pick_mat`’s nested running-sum loops with an explicit chain of thresholds, which slashes the per-lookup material selection overhead and dropped kernel_ms from ~162.6 to 111.6 (≈-31%) while leaving the data movement strategy unchanged.
2. [ ] Transfer persistency / overlap: deferred (still a single copy via `move_simulation_data_to_device`) because the current copy happens once at startup and transfers still track the CUDA behavior.

### CUDA→OMP Recovery Status
- [X] Preserved the original lookup kernel structure so each OpenMP thread does the same RNG + `calculate_macro_xs` work as CUDA.
- [X] Kept the same data staging strategy (one-time host→device copy + final verification memcpy) to mirror the CUDA launch pattern.
- [ ] Still missing: matching CUDA-level kernel throughput (~1.7 ms per launch) without restructuring the algorithm (e.g., splitting RNG/lookup or introducing work sorting).

### Micro-optimizations Applied
1. [X] [MICRO-OPT]: Simplified `pick_mat` so each thread compares `roll` against hardwired thresholds rather than recomputing the `dist` prefix sum; kernel time dropped ~31% with zero checksum impact.

### Key Insights
- Material selection dominated the OpenMP kernel because the old version recomputed sums per lookup, so the threshold chain significantly reduces the inner instruction count and improves register availability on the Ada GPU.
- Host→device transfers still consume ~134 ms of the run because the 241 MB dataset must be staged once; overlapping/streaming could be explored later but would diverge from the established data strategy.
- Even after the micro-optimization, the OpenMP port remains ~2.5× slower than the CUDA baseline, so further wins need larger structural changes (e.g., splitting RNG from lookup or better organizing per-material work).
