# Performance Analysis

## Current Metrics
- Runtime: 0.4355s (host timer, CLASS=S run).
- Main kernel: `nvkernel_main_F1L127_2`, 100.0% of reported GPU time, 1 instance (nsys `cuda_gpu_kern_sum` shows ~1.743s).
- Memory transfer: DtoH 41.2% + HtoD 33.2% = 74.4% of GPU time, total data <0.00001 MB (nsys `cuda_gpu_mem_size_sum`).
- Kernel launches: 1 (single offloaded sample loop in `main`).

## Fusion Opportunities:

### Identified Fusions:
- No safe fusion yet: lines 129-131 (histogram zeroing) and lines 221-225 (global histogram reduction) share the same iteration bounds but execute in distinct phases of the sampling loop, so fusing them would conflate initialization with accumulation and break correctness.

## Iteration Loop (if present):
- Main: lines 126-229, `v3` iterations (256 samples for CLASS=S) offloaded; each sample runs ~65K Gaussian pairs.
- SpMV line Z: N/A (EP has no CSR SpMV phase).
- Update line W: lines 221-225, one visit per bin (10 bins) per sample to atomically merge `local_hist` into `v59`.
- Total: 256 samples × 65,536 pair iterations ≈ 16.7M inner RNG/pair operations.

## SpMV Inner Loop Decision
- Avg nonzeros per row (NONZER): N/A for EP.
- Decision: keep loops serial; no reductions/inner parallelism beyond current scheme.

## Bottleneck Checklist (priority order)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Data transfers | 74.4% of GPU time spent in cudaMemcpy (DtoH + HtoD) even though the total payload is micro-byte sized | Maintain the single `#pragma omp target data` region that already maps `v59` for all samples, avoid additional host/device copies, and rely on the map clause inside the offload instead of pulling data out of the loop.
| Launch overhead | One kernel instance (expected 1) so launch overhead is negligible now | Keep the offload inlined exactly as in lines 122-229; avoid spawning helper kernels inside the sample loop.
| Memory transfer bottleneck | `cudaMemcpy` and small `cudaMemset` invocations collectively consume >99% of profiled GPU time (4 DtoH, 8 HtoD, 6 memset operations) | Keep arrays resident and limit future launches to the existing `target data` scope so we do not trigger additional map/unmap syncs.
| Hot kernel | `nvkernel_main_F1L127_2` currently dominates runtime (1.743s) due to heavy RNG math and repeated memory accesses | Cache loop-invariant constants, avoid rereading `local_hist` entries, and reduce register pressure inside the sampled pair loop to shrink kernel time slightly.
| Type C parallelization | Not present (EP is fully contained inside a single target loop) | N/A |

## Strategy (priority)
1. Cache repeated values inside the sample loop (`const int bins = v8`, capture `local_hist[bin]` before the atomic) so the GPU can keep frequently-read scalars in registers rather than spilling to shared/global memory—expect a small (<5%) reduction in kernel time.
2. Preserve the existing `target data` map strategy (per instructions) while leaning on `firstprivate` values for constants; keep per-sample scratch arrays private to avoid additional data movements and maintain correctness.

## Micro-opts
- [x] `const` aliases and loop-invariant locals to encourage register caching
- [ ] `restrict` (not currently applicable)
- [ ] `firstprivate` (already in use)
- [x] cache locals before atomic updates to reduce redundant loads

## Target
- Runtime: ≤0.42s (micro-opt goal); drop kernel time a few percent while retaining the 1.743s GPU kernel signature.
- Kernels: 1 dominant kernel (no new launches).
- Memory: keep transfer share below 70% of GPU time by avoiding extra host-device handshakes.

## Bottlenecks (mark applicable)
### [x] 1. Data Management Issue (CRITICAL - fix first!)
- Transfer ratio: 74.4% of GPU time yet only ~0.00001 MB moved (nsys `cuda_gpu_mem_size_sum`).
- Root cause: Half a dozen small memcpys/memsets happen as soon as the offload finishes (target data region closes). Nothing in the code actually remaps `v59` mid-loop, so the ratio is high simply because the kernel is compute-heavy and the copies are small.
- Fix: keep the single target data region open around the sample loop (already done) and avoid introducing extra transfers; rely on `map(tofrom:v59[0:v8])` and the existing `firstprivate` parameters.
- Expected gain: small improvement to data-driven overhead, likely within the noise band.

### [ ] 2. Kernel Launch Overhead
- Kernel instances: 1, which matches the number of sample groups.
- Expected: ~1 per CLASS S execution, so no extra launch tuning required.
- Root cause/Fix: N/A.

### [x] 3. Memory Transfer Bottleneck
- Transfer time 74.4% and `cudaMemset` adds another 25.5% of GPU time (total ~99.9%), showing that host-device synchronizations dominate relative to the computational kernel.
- Fix: keep data resident in device memory, do not remap `v59`, and keep updates aggregated on the device before copying results back.
- Expected gain: reduces overhead by a few percent if we avoid extra memcopies in future iterations.

### [x] 4. Hot Kernel Performance
- Kernel `nvkernel_main_F1L127_2` holds 100% of GPU time (1.743s) and executes ~16.7M RNG pairs per run.
- Root cause: repeated RNG loops with `LOG`, `SQRT`, and `MAX` plus frequent atomic updates on `v59`.
- Fix: reorganize the sample loop so invariant values become `const` locals, cache `local_hist` entries before atomics, and keep per-bin counts in registers before the final merge.
- Expected gain: ~1-5% kernel time reduction, depending on register reuse.

### [ ] 5. Type C Parallelization Error
- Verification: PASS.
- No inner stage loops are parallelized; therefore, no Type C issue.
- Fix: not applicable.

### [ ] 6. Over-Parallelization (saturated outer loops)
- Outer iterations: 256 samples × 65K pair loops; saturation threshold for RTX 4060 is far above this workload.
- Symptoms: GPU utilization currently low (<10%), so no inner-parallelism removal required.
- Fix: not applicable.
# Final Performance Summary

### Baseline (Step 2)
- Runtime: 0.4355s (CLASS=S host timer from initial run).
- Main kernel: `nvkernel_main_F1L127_2`, 1 instance, ~1743.2ms total (same name/time reported by the existing `profile.log`).

### Final (Step 3)
- Runtime: 0.4257s (CLASS=S host timer after micro-optimizations).
- Speedup: 1.024× (≈2.3% faster than baseline host time).
- Main kernel: `nvkernel_main_F1L127_2`, 1 instance, ~1743.2ms total (no new nsys profile, so assumed unchanged).

### Optimizations Applied
1. [x] Micro restructure: cache `v8` in `bin_count` and reuse it to avoid fetching the firstprivate in every bin loop → slight register reduction.
2. [x] Micro restructure: grab `local_hist[bin]` into `bin_value` before the atomic update to reduce repeated loads in the hot histogram path.

### Micro-optimizations Applied
1. [x] `const` loop alias for `bin_count` keeps the bin limit in registers.
2. [x] `cache locals` by storing `local_hist[bin]` before the atomic update and reusing the cached value.

### Key Insights
- The GPU run remains dominated by the single offloaded `nvkernel_main_F1L127_2` kernel; even a few register-friendly tweaks only shave a couple percent off the host timer.
- Host-device transfers still consume the majority of profiled GPU time because they are tiny but frequent relative to the compute kernel, so the current data strategy (single target data region) is left untouched.
