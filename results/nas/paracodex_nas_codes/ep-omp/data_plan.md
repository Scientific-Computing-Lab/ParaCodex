# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `q` | `NQ` doubles (10 * 8 B ≈ 80 B) | working | zeroed on host before timed region | host–device `target data` region; atomic updates inside kernel; read after run |
| `xx` | `2 * NK * blksize` doubles (≈ 2.15 GB when `NK=2^16`, `blksize=2048`) | scratch | malloc host before timed region | per-block RNG storage on host (to be removed in GPU plan) |
| `qq` | `blksize * NQ` doubles (≈ 163 KB for `blksize=2048`, `NQ=10`) | scratch | cleared per block | per-block histogram to avoid atomics (currently host-only) |

`xx` and `qq` exist today purely to stage RNG outputs and histogram bins for each block on the host; the GPU plan eliminates both by generating values on the fly and using per-thread histograms that merge via atomics.

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| main processing loop | `q` (read/write), RNG state variables (`t1`, `t2`, `an`, `S`, `A`) | O(NN) iterations | device (target teams loop) |
| `randlc_ep` | internal RNG globals (`r23`, `t23`, `r46`, `t46`) | 2 * NK calls per sample + exponentiation helper calls | device |

## Data Movement Strategy

**Chosen Strategy:** A (target data region)

**Device Allocations (once):**
- `q[0:NQ]` – small, so it stays mapped with the target data region (`map(tofrom:q[0:NQ])`).
- `sx`, `sy` – scalars participate in reductions, so map them `tofrom`. They stay in the target data region without explicit `omp_target_alloc`.

**Host→Device Transfers:**
- When: once before the timed kernel runs while entering the `target data` region.
- Arrays: `q[0:NQ]` (≈ 80 B), `sx`, `sy` (16 B).
- Total H→D: ≈ 0.0001 MB. (All other persistent data—`A`, `S`, RNG constants—are globals declared target.)

**Device→Host Transfers:**
- When: upon exiting the `target data` region after the kernel completes.
- Arrays: same `q[0:NQ]`, `sx`, `sy`, so ≈ 0.0001 MB.

**Transfers During Iterations:** NO – all data stays on device for the duration of the kernel; histogram accumulation uses per-thread locals and atomic merges.

## Critical Checks (for Strategy A)
- [x] `randlc_ep` and RNG globals marked `declare target` so the kernel can call it without extra maps.
- [x] `q`/`sx`/`sy` remain inside a single `target data` block with `map(tofrom)` so the reduction results are synchronized back afterward.
- [x] Scratch arrays (`xx`, `qq`) are removed from the hotspot; per-thread locals and atomic updates replace them, eliminating extra `enter/exit data` or `target update` logic.

## Expected Transfer Volume
- Total: ~0.0002 MB both ways for the entire run.
- **Red flag:** Actual transfer > 0.5 MB would indicate we accidentally remapped large buffers.

## Additional Parallelization Notes
- **RNG Replicable?** YES – the `randlc_ep` state is reset per sample; the helper already implements the power-of-two jump.
- **Outer Saturation?:** The main loop simply runs from `kk = 0` to `NN-1`, so we parallelize across `NN` samples with a single target teams loop.
- **Histogram Strategy?:** `NQ=10` bins (<100), so each thread keeps a `double local_q[10]`, accumulates per sample, then merges via `#pragma omp atomic update`.

**Summary:** 3 arrays (1 working, 2 scratch), 1 critical helper function, Strategy A with minimal transfers (~0.0002 MB). Scratch arrays vanish in the GPU plan, leaving only the small working histogram plus scalar reductions inside the `target teams loop`.
