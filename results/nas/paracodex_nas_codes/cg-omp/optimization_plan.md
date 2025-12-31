# Performance Analysis

## Current Metrics
- Runtime: 0.11s (Class S run recorded in `current_output.txt`)
- Main kernel: nvkernel__ZN19_INTERNAL_4_cg_c_nz9conj_gradEPiS0_PdS1_S1_S1_S1_S1_S1__F1L355_20, 94.3% GPU, 1900 instances (Class C `profile.log`)
- Memory transfer: ~47.9ms (≈1.1% of GPU time), 461.5MB total (H2D 461.4MB, D2H 0.033MB, memset 0.033MB) – (Class C profile)
- Kernel launches: 9883 `cuLaunchKernel` calls (plus 23564 `cuStreamSynchronize`/overhead, Class C profile)

## Fusion Opportunities:

### Identified Fusions:
- Lines 187-198: initialization of `x` and zeroing of `q/z/r/p` all iterate over `NA` → can share a single `target teams` kernel and avoid the extra launch that was zeroing four vectors.
- Lines 233-254: `norm_temp1` and `norm_temp2` share the same trip count and can be reduced in one launch (already combined) and the subsequent `x` update can reuse the cached normalization scalar to stay fused as much as possible.
- Lines 398-417: final SpMV (looping over `rowstr`) and the residual norm loop share the same bounds; fusing them lets the device reuse `rowstr/colidx` loads and only launch one kernel per `conj_grad` exit.

## Iteration Loop (if present):
- Main: lines 230-255, NITER=15 iterations driving `conj_grad` (plus one warm-up call) while holding data on device via `target data`.
- SpMV line 356-365 is executed `cgitmax=25` times per `conj_grad`, so 16 conj_grad invocations (1 init + 15 benchmark) yield 400 SpMV passes.
- Update line 378-395 (z/r update and rho reduction) also executes 25×16=400 loops; the trailing residual/norm is executed once per conj_grad (16 times) but was fused so only one launch.
- Total: 400 cgit loops produce ∼400×(rows) operations (~400×1400 = 560k row-wise SpMV passes) and final residual norm, so ~NITER×cgitmax×NA arithmetic passes dominate.

## SpMV Inner Loop Decision
- Avg nonzeros per row (NONZER) = 7 (Class S) → <50, so inner `for (k=tmp1; k<tmp2; k++)` remains serial and fully unrolled by the compiler; rely on the outer `target teams loop` for parallelism.

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Data transfers | Single H2D map of 461MB dominates `cuda_gpu_mem_size_sum` but only 3 transfers (~47.9ms total); GPU kernels still >4.2s | Keep `target data` hoisted outside iterations; avoid extra host-device copies (already mapped) and reuse data.
| Launch overhead | 9883 `cuLaunchKernel` calls from repeated `target teams loop` directives and 23564 synchronizations inflate `cuda_api_sum` time to 91.7% so kernel overhead is visible | Merge reductions/updates (norm + residual) into fewer loops and keep vector updates paired inside the same kernel to shrink launches per `conj_grad`.
| Memory transfer | D2H 0.033MB per iteration, memsets tiny, transfers only 1.1% of GPU time → not a limiter | Keep existing offload mapping strategy; no extra device-host moves.
| Hot kernel | `conj_grad` routine (SpMV + the z/r loop) uses 94.3% of GPU time and each kernel is ~2.2ms avg | Cache row pointers/reg values, collapse repeated loops, and fuse the final SpMV+residual to amortize the `target` launch cost.
| Stage parallelization | Verification passes (VERIFICATION SUCCESSFUL) but inner stage loops still sequential | No change needed; the current stage loops are already serial for correctness.

## Strategy (priority)
1. Fuse the two dot-product kernels in the benchmark loop (lines 233-242) to accumulate `norm_temp1` and `norm_temp2` in one launch; this keeps the scalar `norm_temp2` ready for the subsequent `x` update and removes one kernel per iteration (expect ~5% wall-time shrinkage).
2. Combine the final SpMV and residual norm loops (lines 399-417) so a single kernel computes `r[j]` and the residual square sum; this reuses `rowstr/colidx` lookups and cuts one launch per `conj_grad` call (target ~10% GPU kernel speedup).
3. Keep vector updates (z/r) in the same kernel when possible and hoard normalization scalars locally to avoid implicit device-host copies; this reduces control overhead and avoids re-reading values from global memory.

## Micro-opts
[ ] const
[ ] restrict
[ ] firstprivate
[x] cache locals (introduced `diff` to reuse the computed residual inside the fused loop)

## Target
- Runtime: ≤0.08s (aim for ~25% improvement by trimming repeated kernels)
- Kernels: O(cgitmax×`conj_grad` calls) ≈400+ essential launches instead of the current ∼9883 `cuLaunchKernel` calls
- Memory: <5% of GPU time spent on transfers, keeping the 461MB H2D cost amortized over all iterations
