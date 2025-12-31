# Performance Analysis

## Current Metrics
- Runtime: 0.53s (Class B, RTX 4060 Laptop, nvc++)
- Main kernel: `resid` fused update (~50.9% GPU time, 170 instances, 1.27 ms avg)
- Memory transfer: 85.5% of mem time Device-to-Host (178 ms total, 10 copies), Host-to-Device 14.5% (30.4 ms, 2 copies)
- Kernel launches: 2983 `cuLaunchKernel` calls, 3281 `cuStreamSynchronize`

## Fusion Opportunities:

### Identified Fusions:
- Lines 464-485: `resid` partial-sum build + final update share bounds → fuse and compute in-register
- Lines 398-423: `psinv` partial neighbor sums + final smoothing share bounds → fuse into single teams loop

## Iteration Loop (if present):
- Main solve: `nit=20`, traverses multigrid V-cycle per iter
- Hot stages per V-cycle: `resid` and `psinv` each invoked across levels, ~170 kernels per phase
- Norm/reduction: `norm2u3` called for verification, small share of time

## Bottlenecks (mark applicable)
### [ ] 1. Data Management Issue (CRITICAL - fix first!)
- Transfer ratio: actual/expected ≈ 5x (10 D2H vs ~2 expected final copies)
- Root cause: Extra device-to-host copies observed; keep data strategy intact but avoid added transfers
- Fix: Reuse device temporaries and avoid implicit host-visible buffers
- Expected gain: ~15-25% if copies reduced

### [x] 2. Kernel Launch Overhead
- Kernel instances: 3321 launches for 20 iterations
- Expected: ~half of current; dual-kernel patterns inside `resid/psinv` drive counts
- Root cause: Helper kernels for temporary buffers per stencil pass
- Fix: Inline partial sums and final updates in one kernel (ACTION 4C)
- Expected gain: ~1.3-1.5x from fewer launches/syncs

### [x] 3. Memory Transfer Bottleneck
- Transfer time: 210 ms (~37% of runtime); D2H dominant
- Fix: Remove temporary buffer maps that trigger copies; keep persistent maps only (ACTION 4A-lite)
- Expected gain: 10-20%

### [x] 4. Hot Kernel Performance
- Kernel: `resid` (~31% GPU time, 0.80 ms avg)
- Root cause: Extra global memory traffic to temporary arrays; two passes over grid
- Fix: Single pass with cached neighbor sums; `__restrict__` and combined teams loop (ACTION 4B)
- Expected gain: 15-25% faster kernel

### [ ] 5. Type C Parallelization Error
- Verification: PASS
- If FAIL: N/A
- Fix: N/A


## Strategy (priority)
1. Fuse `resid` temp + update into one `target teams loop` with register temps and `__restrict__` pointers to cut kernel count and global traffic.
2. Fuse `psinv` neighbor sums + smoothing into one kernel, removing device allocations and extra synchronization while preserving existing data mapping.

## Micro-opts
[x] const, restrict, firstprivate, cache locals

## Target
- Runtime: ≤0.45s (Class B)
- Kernels: ~1700-1800 launches for 20 iterations
- Memory: D2H <25% of total time

# Final Performance Summary

### Baseline (Step 2)
- Runtime: 0.57s
- Main kernel: `resid` two-pass, 170+170 instances (~215 ms total)

### Final (Step 3)
- Runtime: 0.53s
- Speedup: 1.08x
- Main kernel: `resid` fused pass, 170 instances (215 ms total, fewer launches)

### Optimizations Applied
1. [x] Fused `resid` temp+update into single kernel with register neighbor sums; removed device temp allocations and one launch per call.
2. [x] Fused `psinv` temp+update into single kernel; eliminated temp buffers and extra launch while keeping persistent maps intact.

### Key Insights
- Kernel launch count dropped (~10%) and CPU-side malloc traffic removed without altering data mapping; modest end-to-end gain.
- Device-to-host copies still dominate mem time (10 transfers); further reduction would yield larger gains but data strategy is fixed.
