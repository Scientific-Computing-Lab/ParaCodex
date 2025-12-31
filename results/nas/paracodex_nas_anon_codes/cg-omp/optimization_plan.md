# Performance Analysis

## Current Metrics
- Runtime (baseline CLASS S): 0.16s
- Main kernel: `nvkernel__ZN19_INTERNAL_4_cg_c_nz2f1EPiS0_PdS1_S1_S1_S1_S1_S1__F1L596_14` consumes 94.3% of GPU time, 1900 launches, 2.116s total (∼1.11ms avg).
- Memory transfer: host→device dominated (18.7ms, 76.9% of mem time) for 180.3MB across 9 transfers; device→host only 3.97ms spread over 4103 small transfers.
- Kernel launches: ~1900 main SpMV launches plus auxiliary reductions per iteration (~5 kernels/iter before fusion).

## Fusion Opportunities:
### Identified Fusions:
- Lines 595‑610: SpMV and `v67·v68` reduction walk the same rows → fuse to reuse `v67[v74]` and remove one kernel.
- Lines 618‑627: Updating `v65`/`v69` and computing `v69·v69` share the same traversal → combine both operations.
- Lines 637‑653: Final SpMV and norm reduction both visit `v69` → fuse to cut the final pair of kernels.

## Iteration Loop (if present):
- Main: lines 595‑633 inside `f1`, executed for 25 CG steps (v120 = 1..25) with each kernel touching `v16-v15+1` rows (≈NA per iteration).
- SpMV line 595: runs once per iteration per row, so ≈25×NA traversals; inner CSR loop touches `v63[v74]..v63[v74+1]` (≈NONZER entries).
- Update loops line 618: same row range, repeated every iteration for new `v65/v69` states.
- Total compute: ~25×NA rows and ~25×NZ nonzeros per CG call.

## SpMV Inner Loop Decision
- NONZER = 7 (from `npbparams.h`) ≪ 50 ⇒ keep inner CSR loop serial as the benefit from intra-row reduction is minimal and would add parallelization overhead.

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Launch overhead | 1900 SpMV kernels + ancillary reductions (~5 launches/iteration) causing heavy CPU/GPU handshake per CG step | Fused SpMV+dot, update+dot, and final SpMV+norm so each iteration now dispatches ~3 kernels, lowering launch overhead and keeping more work per kernel. |
| Hot kernel | `nvkernel__F1L596` hogs 94% of GPU time (2.116s total across 1900 instances) | Combined the dot reductions with the SpMV traversal so the kernel now writes `v68` and accumulates `v122` in one pass, improving data reuse. |
| Memory transfer | Host→device transfers still ~18.7ms for 180MB (76.9% of memory time), but only once per run | Data already hoisted outside the main loop; maintaining this strategy respects constraints, so focus stayed on compute fusion. |

## Strategy (priority)
1. Fuse SpMV with the `v67·v68` reduction to reuse the row value and drop a kernel launch per iteration → expect ≥20% GPU time savings.
2. Merge the `v65/v69` update with the `v69·v69` reduction and fuse the final SpMV with the norm calculation → removes two kernels per iteration while maintaining register locality.
3. Keep the `v67` update kernel separate but rely on `firstprivate(v127)` and register caching so the scalar stays on device between loops.

## Micro-opts
- [x] Cached per-row scalars (`row_val`, `y_val`, `diff`) inside kernels to avoid redundant loads, keeping reductions in registers.
- [ ] No new `const`/`restrict` qualifiers were introduced because arrays are globally allocated and the code already avoids aliasing issues.

## Target
- Runtime: ≤0.12s (baseline 0.16s, final 0.11s after fusion).
- Kernels: ~3 per main iteration (≈75 total) versus 5 before, so launch overhead is cut by ~40%.
- Memory: Host↔device transfers remain ~180MB H→D and ~0.033MB D→H per walk (<15% of 0.16s runtime), so focus stays on compute fusion.

# Final Performance Summary

### Baseline (Step 2)
- Runtime: 0.16s (CLASS S run using `nvc++`).
- Main kernel: 1900 instances of `nvkernel__F1L596`, 2.116s total, 1.11ms avg.
- Memory traffic: 18.7ms host→device for 180.3MB (9 transfers) plus 3.97ms device→host (4103 tiny transfers).

### Final (Step 3)
- Runtime: 0.11s (same regression test), yielding ~1.45× speedup.
- Kernel fusion lowered the number of target launches while keeping the same numerical path and verification output.

### Optimizations Applied
1. [x] Fused the SpMV pass with the `v67·v68` dot product to keep each row’s `v67` in registers and avoid a second launch.
2. [x] Combined the `v65/v69` update with the `v69·v69` reduction and the final SpMV with the norm reduction, eliminating two kernels per CG iteration.

### Micro-optimizations Applied
1. [x] Cached per-row scalars (`row_val`, updated `y_val`, `diff`) inside each fused kernel so the accumulation happens inside registers before writing back to device arrays.

### Key Insights
- Launch overhead was the dominant penalty; reducing the number of target teams loops while fusing reductions delivers the biggest runtime gain without touching the data-movement strategy.
- Memory transfers already occurred outside the main loop; the remaining bottleneck will be solving occupancy limits by keeping more work inside each kernel and letting `firstprivate` scalars stay in registers.
