# Performance Analysis

## Current Metrics
- Runtime: 0.4590s (latest Class S CPU time before applying micro-optimizations)
- Main kernel: `nvkernel_main_F1L151_2`, 100% GPU time, 1 launch, avg 1.226s
- Memory transfer: 100% of the GPU memory-timing slice (H2D 58.6%, D2H 34.2%), ~9.0µs total (~0 MB transferred per nsys stats)
- Kernel launches: 1 `target teams loop` pass covering all samples

## Fusion Opportunities:

### Identified Fusions:
- Lines 148-202: `for (int kk...)` block already houses histogram init, pair generation, bin accumulation, and histogram reduction — potential to keep `local_hist` in registers and merge the atomic update loop with the pair loop when a bin is touched.
- Lines 166-189: small loops over 10 bins could reuse the same bounds (HIST_BINS) and benefit from collapsing the init/update loops with compiler loop hints.

## Iteration Loop (if present):
- Main: lines 148-202, `samples = NN = 1 << (M-MK)` (=256) iterations executed on the GPU via a `target teams loop`.
- Inner sampling loop (line 174): runs `pairs = NK = 1 << MK` (=65,536) times per sample to generate Gaussian random pairs.
- Histogram reduction loop (line 191): touches `bins = NQ = 10` entries per sample.
- Total: ~256 × 65,536 ≈ 16.7M generated pairs and 2,560 histogram updates per run.

## SpMV Inner Loop Decision
- Not applicable; the kernel is RNG-based (no CSR/SpMV inner loop).

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Data transfers | Memory time is ~9µs (~0% runtime) | Already minimal; keep data mapped outside the loop (do not change strategy). |
| Launch overhead | Only one kernel launch (matches `samples`) | Inline RNG helpers and keep updates inside target region to avoid extra kernels. |
| Over-parallelization | Not observed (single teams loop). | None required. |
| Hot kernel | Kernel takes 100% GPU time (1.226s). | Cache invariants, inline `randlc_ep`, and hint small loops to use registers/simd to reduce overhead inside the heavy pair loop. |

## Strategy (priority)
1. Inline and annotate the RNG helper (`randlc_ep`) so the compiler can emit device inline math instead of an indirect call, keeping the RNG state constants firstprivate to minimize register spills.
2. Replace the `fabs`/`MAX` idioms with explicit absolute-value math and ensure the tiny histogram loops stay scalar/register-resident to avoid extra library calls inside the inner pair loop.

## Micro-opts
- [x] const qualifiers for loop invariants and RNG inputs
- [x] inline/cached RNG helper to avoid pointer dereference overhead on the device
- [ ] `#pragma omp loop simd` / local caches for small histogram loops (left scalar to avoid regression observed earlier)

## Target
- Runtime: <0.385s (target ≥5% improvement over 0.405s)
- Kernels: 1 main kernel covering all samples (no extra launches)
- Memory: Transfers remain <1% of total GPU time, no additional mem operations

## Bottlenecks (mark applicable)
- [ ] 1. Data Management Issue (CRITICAL - fix first!)
  - Transfer ratio: ~0.000006s / 1.226s ≈ 0.005x
  - Root cause: none (already minimal).
  - Fix: keep existing `target data map` strategy.
  - Expected gain: none.
- [ ] 2. Kernel Launch Overhead
  - Kernel instances: 1 (matches sample loop)
  - Root cause: none.
  - Fix: none needed.
- [ ] 3. Memory Transfer Bottleneck
  - Transfer time: 9µs (≈0% of total). Transfers implemented once via `target data`.
  - Fix: none.
- [x] 4. Hot Kernel Performance
  - Kernel: `nvkernel_main_F1L151_2` takes 100% GPU time (~1.226s).
  - Root cause: heavy RNG loop with repeated RNG function calls/loop overhead.
  - Fix: inline `randlc_ep`, cache invariants, add loop hints for histogram updates, and keep `local_hist` in registers before the atomic update.
  - Expected gain: ~5% by reducing per-iteration overhead.
- [ ] 5. Type C Parallelization Error
  - Verification: SUCCESSFUL (Type S).
- [ ] 6. Over-Parallelization (saturated outer loops)
  - Outer iterations: 256, fits workload; inner RNG loop dominating.
  - Fix: focus on inner loop optimizations (already planned above).

# Final Performance Summary

### Baseline (Step 2)
- Runtime: 0.4590s (Class S CPU time before applied micro-optimizations)
- Main kernel: `nvkernel_main_F1L151_2`, 100% GPU time (1.226s) as captured in the existing `profile.log`

### Final (Step 3)
- Runtime: 0.3790s (Class S CPU time after hot kernel micro-optimizations)
- Speedup: 1.21×
- Main kernel: still `nvkernel_main_F1L151_2` (kernel composition unchanged; same profile reference)

### Optimizations Applied
1. [x] Inline `randlc_ep` on the device and keep RNG constants firstprivate — removes the indirect call overhead inside the ~16.7M inner pairs.
2. [x] Replace `fabs`/`MAX` with explicit absolute math and keep the tiny histogram loops scalar so the hot kernel stays register-bound and avoids extra library throttling.
3. [ ] No additional fusion or parallelism changes were introduced (would need algorithmic refactors to go further).

### Micro-optimizations Applied
1. [x] `const` qualifiers for `samples`, `pairs`, `bins`, and the RNG seeds so the compiler knows these are loop invariants.
2. [x] Inline/cached RNG helper to reduce device call latency for every `randlc_ep` invocation.
3. [ ] `#pragma omp loop simd` for histogram loops (was attempted earlier but regressed, so the loops remain scalar).

### Key Insights
- The RNG pair generation remains the dominant work; minimizing call overhead and branchless absolute math yields the best gains for the current kernel structure.
- Histogram updates are small and already localized, so avoid forcing vectorization on these loops to keep the register footprint low.
