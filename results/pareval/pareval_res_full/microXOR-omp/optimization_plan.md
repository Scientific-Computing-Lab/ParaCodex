# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: ~0.03s for `sqrt`-sized run (`env OMP_TARGET_OFFLOAD=MANDATORY ./microXOR.exe 1024 32`); GPU kernel stats not emitted in `profile.log`.
- Main kernel: `cellsXOR` implemented as an OpenMP `target teams loop` mapping the full `N×N` grid, 1 launch per run (kernel aliases are not listed in the profile output).
- Memory transfer: `input` mapped `to`/`output` mapped `tofrom` once; profiling log only lists report generation so precise transfer % is unavailable (effectively two implicit copies of `N×N` ints).
- Kernel launches: 1 explicit `target teams loop`.

## Bottleneck Hypothesis (pick 1–2)
- [ ] Transfers too high (CUDA avoided transfers in loop)
- [ ] Too many kernels / target regions (launch overhead)
- [x] Missing collapse vs CUDA grid dimensionality
- [x] Hot kernel needs micro-opts

## Actions (1–3 max)
1. Reintroduce the 2D loop structure with `collapse(2)` so the OpenMP teams/threads mirror the original CUDA grid and avoid per-iteration division/mod operations; expected to improve scheduling and cache locality.
2. Qualify `input`/`output` with `__restrict__` and hoist boundary constants (`N-1`) to locally cached variables so the compiler can better optimize the neighbor checks; expected to recover GPU-like throughput with minimal code change.

## Optimization Checklist (short)
- [ ] Transfers dominate: hoist data; `omp_target_alloc` + `is_device_ptr`; avoid per-iter mapping
- [ ] Too many kernels/regions: fuse adjacent target loops; inline helper kernels when safe
- [x] Missing CUDA grid shape: add `collapse(N)`
- [x] Hot kernel: `const`, `restrict`, cache locals, reduce recomputation

# Final Performance Summary - CUDA to OMP Migration

### Baseline (from CUDA)
- CUDA Runtime: not provided in this repo/profile logs, so the exact time is unknown.
- CUDA Main kernel: `cellsXOR` launched once with a `(ceil(N/blockEdge), ceil(N/blockEdge))` grid and `(blockEdge, blockEdge)` blocks; no kernel duration data captured.

### OMP Before Optimization
- Runtime: ~0.03s (measured with `env OMP_TARGET_OFFLOAD=MANDATORY ./microXOR.exe 1024 32` before applying loop reorganizations).
- Slowdown vs CUDA: unknown (lack of CUDA timing data).
- Main kernel: `cellsXOR` OpenMP target teams loop, one offload per run; kernel profiling output did not list GPU durations.

### OMP After Optimization
- Runtime: ~0.02s (measured after introducing collapse+restrict hints).
- Slowdown vs CUDA: still unknown (CUDA baselines missing).
- Speedup vs initial OMP: ~1.5× faster using the same input parameters.
- Main kernel: still a single `cellsXOR` target teams loop; kernel-level times remain unreported in the current profile.

### Optimizations Applied
1. Restored the CUDA-style 2D launch by using nested loops with `collapse(2)`, eliminating per-iteration division/mod arithmetic and aligning data accesses with the original grid.
2. Added pointer-level qualifiers (`__restrict__`) plus a cached `N-1` boundary to reduce redundant computations and expose locality to the compiler.

### CUDA→OMP Recovery Status
- [x] Restored 2D/3D grid mapping with collapse
- [ ] Matched CUDA kernel fusion structure (single kernel already in place)
- [ ] Eliminated excessive transfers (data mapping was already minimal)
- [ ] Still missing: richer kernel timing instrumentation to compare directly against CUDA

### Micro-optimizations Applied
1. [X] MICRO-OPT: Applied `collapse(2)` and nested loops so the OpenMP teams/threads iterate directly over `(i,j)` and avoid the `flat` index math from CUDA paradigms, which also reduces the amount of integer math per iteration.
2. [X] MICRO-OPT: Declared `input`/`output` as `__restrict__` and hoisted the `N-1` constant to a local variable, helping the compiler optimize neighbor loads and stores.

### Key Insights
- The kernel is already a compact stencil, so recovering the CUDA grid mapping and tightening pointer assumptions yielded the most measurable gain (~1.5× faster runtime).
- The current `nsys` log shows only the report generation steps, so future profiling should target the actual GPU kernel tables to capture `cellsXOR` durations and transfer percentages.
- Data movement remains the single map region per run, matching the CUDA strategy and minimizing transfer overhead without additional code changes.
