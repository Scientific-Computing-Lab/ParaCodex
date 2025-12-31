# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `q` | `NQ = 10` doubles (80 bytes) | working | host (zeroed) | R/W via atomic merges during kernel, final read on host |
| `sx`/`sy`/`gc` | scalars (double) | working | host (zero) | reduction updates inside kernel |

**Note:** The original scratch buffers `xx` and `qq` are eliminated by folding the RNG + counting logic into each sample thread; per-thread locals now capture transient random samples and bin counts, removing the huge per-block allocations.

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| `main` timed sample loop | `q`, `sx`, `sy`, `gc` | once per `NN` sample, invoked from `target teams distribute parallel for` | device |

The timed region now consists of a single `for (kk = 0; kk < NN; ++kk)` loop that performs RNG initialization, generates paired random numbers, accumulates gaussian sums, and merges a per-thread local histogram into the global `q` array.

## Data Movement Strategy

**Chosen Strategy:** A

**Device Allocations (once):**
- `q` (10 doubles) enters the target data region with `alloc`/`tofrom` semantics so we can accumulate across samples without repeated transfers
- Scalars `sx`, `sy`, `gc`, RNG constants (`r23`, `r46`, `t23`, `t46`, `A`, `S`) remain mapped via the target data region and/or reduction clauses

**Host→Device Transfers:**
- When: once before the timed loop begins (during `#pragma omp target data map(tofrom: q[:NQ], sx, sy, gc)`) after zeroing `q`/scalars
- Arrays: `q` (80 bytes) and scalars → negligible (~0.01 MB)
- Total H→D: ~0.01 MB

**Device→Host Transfers:**
- When: once after the timed loop ends; target data region ensures final `q`, `sx`, `sy`, `gc` are copied back
- Total D→H: ~0.01 MB

**Transfers During Iterations:** NO – data stays on the device for the duration of the timed loop

## Critical Checks (for chosen strategy)

**Strategy A:**
- [x] The timed kernel is wrapped inside `#pragma omp target data map(tofrom: q[:NQ], sx, sy, gc)` so `q`/scalars are resident
- [x] There are no additional scratch arrays requiring enter/exit data or `omp_target_alloc`; per-thread locals replace the original `xx` and `qq`

**Common Mistakes:**
- Some helper RNG loops remain host-only (they are now inlined inside the device loop so no mismatched execution)
- Scratch buffers were host allocated before; ensure the new implementation uses local arrays (stack space) per sample instead
- Keep the entire sample loop on the device so we never copy large buffers back and forth

## Expected Transfer Volume
- Total: ~0.02 MB for the entire execution (q plus scalars)
- **Red flag:** not expected because data stays resident and we do not transfer per sample

## Additional Parallelization Notes
- **RNG Replicable?** YES – each `(kk)` sample re-derives the same seed sequence from `kk`, enabling deterministic execution on the device
- **Outer Saturation?** All `NN` samples (`≈ 2^{M−MK}`) run in one parallel loop intended to saturate the GPU
- **Sparse Matrix NONZER?** N/A
- **Histogram Strategy?** There are only `NQ = 10` bins; we use a per-thread local histogram (`double local_hist[NQ]`) and atomically merge into the global `q` after processing each sample; this avoids large shared scratch arrays

**Summary:** 1 working array (`q`), 3 scalar accumulators, Strategy A. Expected: ~0.02 MB H→D + ~0.02 MB D→H.
