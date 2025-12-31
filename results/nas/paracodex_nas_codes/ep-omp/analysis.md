# Loop Classification for `ep.c`

### Loop Nesting Structure
- `q_init` (`data/src/ep-omp/ep.c`:134) Type A – SECONDARY (setup before timers).
- `seed_loop` (`data/src/ep-omp/ep.c`:146) Type E – SECONDARY; sequential `randlc_ep` to compute `an`.
- `blk_loop` (`data/src/ep-omp/ep.c`:157) Type A – CRITICAL; iterates over `numblks` sized by `NN`/`blksize`.
  ├ `qq_zero_k` (`data/src/ep-omp/ep.c`:165) Type A – IMPORTANT; resets each column of `qq`.
  │  └ `qq_zero_bin` (`data/src/ep-omp/ep.c`:167) Type A – IMPORTANT; inner `NQ` loop.
  └ `sample_loop` (`data/src/ep-omp/ep.c`:171) Type A – CRITICAL; each `k` emits one sample and drives RNG/type-specific inner loops.
     ├ `rng_recur` (`data/src/ep-omp/ep.c`:176) Type E – IMPORTANT; bit-folding recurrence to seed per-sample RNG.
     ├ `rand_fill` (`data/src/ep-omp/ep.c`:217) Type E – CRITICAL; fills `xx` with `2*NK` random numbers with loop-carried state.
     └ `gauss_bin` (`data/src/ep-omp/ep.c`:234) Type D – CRITICAL; converts pairs into `qq` buckets and accumulates `sx/sy`.
- `qq_reduce_i` (`data/src/ep-omp/ep.c`:255) Type A – SECONDARY; collapses per-bin `qq`.
  └ `qq_reduce_k` (`data/src/ep-omp/ep.c`:258) Type A – SECONDARY; inner summation over `blksize`.

### Loop Details

#### Loop: `q_init` in `main` (`data/src/ep-omp/ep.c`:134)
- **Iterations:** `NQ` (10); constant and <10K so purely setup.
- **Type:** A – dense, constant bound.
- **Parent loop:** none (prepares statistics before timed region).
- **Contains:** none.
- **Dependencies:** none.
- **Nested bounds:** constant.
- **Private vars:** `i`.
- **Arrays:** `q` (W).
- **Issues:** setup-only; no parallel dependencies.

#### Loop: `seed_loop` in `main` (`data/src/ep-omp/ep.c`:146)
- **Iterations:** `MK + 1` (≈17); used to advance the random number generator to `an`.
- **Type:** E – recurrence (updates `t1` sequentially via `randlc_ep`).
- **Parent loop:** none (still part of timer enclosure but only once).
- **Contains:** none.
- **Dependencies:** stage dependency on `t1`.
- **Nested bounds:** constant.
- **Private vars:** `i`.
- **Arrays:** none.
- **Issues:** loop-carried RNG/state, not parallelizable without altering the algorithm.

#### Loop: `blk_loop` in `main` (`data/src/ep-omp/ep.c`:157)
- **Iterations:** `numblks ≈ ceil(NN/blksize)`; drives the main timed chunk. For M=24 this is 1, but grows as `NN` grows.
- **Type:** A – dense outer driver.
- **Parent loop:** timed block started at `data/src/ep-omp/ep.c`:130.
- **Contains:** `qq_zero_k`, `qq_zero_bin`, `sample_loop`.
- **Dependencies:** reduction on `sx`, `sy` across `k` loops; per-block `qq` is local column-major (no cross-`blk` sharing).
- **Nested bounds:** variable (depends on `blksize`, which is trimmed when the final block is shorter).
- **Private vars:** `blk`, `koff`.
- **Arrays:** `qq` (RW via nested loops), `xx` (W in `sample_loop`), `q`/`sx`/`sy` (modified in contained loops).
- **Issues:** lifts RNG/sequencing per sample; `blksize` is mutated inside the block, so nested loops must treat the tail carefully.

#### Loop: `qq_zero_k` in `main` (`data/src/ep-omp/ep.c`:165)
- **Iterations:** `blksize` (per-block, ≤ 2048 but can shrink for the last block).
- **Type:** A – dense zeroing loop.
- **Parent loop:** `blk_loop`.
- **Contains:** `qq_zero_bin`.
- **Dependencies:** none (each column is reset before use).
- **Nested bounds:** variable (changes with the block size).
- **Private vars:** `k`.
- **Arrays:** `qq` (W, column-major).
- **Issues:** repeated per block but contributes <10% runtime; still linear in `blksize`.

#### Loop: `qq_zero_bin` in `main` (`data/src/ep-omp/ep.c`:167)
- **Iterations:** `NQ` (10) per column.
- **Type:** A – dense inner zeroing.
- **Parent loop:** `qq_zero_k`.
- **Contains:** none.
- **Dependencies:** none.
- **Nested bounds:** constant.
- **Private vars:** `i`.
- **Arrays:** `qq` (W).
- **Issues:** tiny; <10K iterations; no data hazards.

#### Loop: `sample_loop` in `main` (`data/src/ep-omp/ep.c`:171)
- **Iterations:** `blksize` per block → total ≈ `NN` samples.
- **Type:** A – driver for each sample.
- **Parent loop:** `blk_loop`.
- **Contains:** `rng_recur`, `rand_fill`, `gauss_bin`.
- **Dependencies:** reduction to `sx`/`sy` (scalar) and updates to `qq` via bucketed columns; uses per-sample `kk`, so RNG seeds are independent across `k`.
- **Nested bounds:** variable (controlled by `blksize` and number of elements remaining).
- **Private vars:** `k`, `kk`, `t1`, `t2`, `tmp_sx`, `tmp_sy`.
- **Arrays:** `xx` (W in `rand_fill`), `qq` (W, column `k-1`), `sx`/`sy` (RW with reductions).
- **Issues:** Outer Type A with inner RNG (special case); `sx`/`sy` reductions need scalar accumulation; `qq` updates depend on computed `l` but are isolated per column (no atomic required).

#### Loop: `rng_recur` in `main` (`data/src/ep-omp/ep.c`:176)
- **Iterations:** up to 100; typically runs `log2(kk)` iterations because of the `if (ik == 0) break`.
- **Type:** E – recurrence (bit folding to advance random state per sample).
- **Parent loop:** `sample_loop`.
- **Contains:** sequential `randlc` style updates on `t1`/`t2`.
- **Dependencies:** stage dependency on `t1`/`t2` and `kk`; each iteration uses the previous value.
- **Nested bounds:** constant (≤ 100).
- **Private vars:** `i`, `ik`, `t3`, `in_*`.
- **Arrays:** none.
- **Issues:** sequential RNG path; only replicable per sample (RNG replicable: YES if each thread can reconstruct `kk`).

#### Loop: `rand_fill` in `main` (`data/src/ep-omp/ep.c`:217)
- **Iterations:** `2 * NK` (≈131072) for each `k`.
- **Type:** E – loop-carried recurrence (updates `t1` to create the next Gaussian sample).
- **Parent loop:** `sample_loop`.
- **Contains:** none.
- **Dependencies:** each iteration depends on the previous `t1` value (no parallel speedup without altering RNG).
- **Nested bounds:** constant (depends on `NK`, itself `1 << MK`).
- **Private vars:** `i`, `in_t1`, `in_x1`, `in_x2`, `in_t2`, `in_z`, `in_t3`, `in_t4`.
- **Arrays:** `xx` (W, storing a scratch column for sample `k`).
- **Issues:** recurrence prevents vectorization; dominates runtime inside each sample.

#### Loop: `gauss_bin` in `main` (`data/src/ep-omp/ep.c`:234)
- **Iterations:** `NK` (≈65536) per sample.
- **Type:** D – histogram/indirect binning with computed `l`.
- **Parent loop:** `sample_loop`.
- **Contains:** conditional bucket update and reduction to `tmp_sx`/`tmp_sy`.
- **Dependencies:** histogram write to `qq[l*blksize + (k-1)]` (variable `l`); `tmp_sx`/`tmp_sy` reductions to be collapsed into `sx`/`sy`.
- **Nested bounds:** constant.
- **Private vars:** `i`, `x1`, `x2`, `t1`, `t2`, `t3`, `t4`, `l`.
- **Arrays:** `xx` (R), `qq` (W), `sx`/`sy` (RW through reduction).
- **Issues:** histogram indices vary; outer column (`k-1`) keeps the inner loop race-free but requires reduction logic in `sx`/`sy`.

#### Loop: `qq_reduce_i` in `main` (`data/src/ep-omp/ep.c`:255)
- **Iterations:** `NQ` (10); sums each bucket once after all blocks.
- **Type:** A – global reduction over histogram bins.
- **Parent loop:** timed block (lines 130–266).
- **Contains:** `qq_reduce_k`.
- **Dependencies:** reduction of `q[i]` and scalar `gc`.
- **Nested bounds:** constant (depends only on `NQ`).
- **Private vars:** `i`, `sum_qi`.
- **Arrays:** `qq` (R), `q` (RW), `gc` (RW).
- **Issues:** reduction over `gc`; guard for final `blksize` ensures only processed entries contribute.

#### Loop: `qq_reduce_k` in `main` (`data/src/ep-omp/ep.c`:258)
- **Iterations:** `blksize` (equal to final block size, potentially <2048).
- **Type:** A – inner reduction.
- **Parent loop:** `qq_reduce_i`.
- **Contains:** none.
- **Dependencies:** reads `qq[i*blksize + k]` to build each bucket’s sum.
- **Nested bounds:** variable (matching whichever `blksize` left after block processing).
- **Private vars:** `k`.
- **Arrays:** `qq` (R).
- **Issues:** if the last block was trimmed, `blksize` shrinks; the loop is executed only once after computation so it does not dominate runtime.

### Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| `main (q_init)` | A | SECONDARY | none | `NQ` | none | setup-only, <10K iters |
| `main (seed_loop)` | E | SECONDARY | none | `MK + 1` | recurrence on `t1` | sequential RNG amplification |
| `main (blk_loop)` | A | CRITICAL | `timed block` | `numblks ≈ ceil(NN/blksize)` | `sx`/`sy` reduction | RNG sequencing per sample |
| `main (qq_zero_k)` | A | IMPORTANT | `blk_loop` | `blksize` | none | repeated zeroing |
| `main (qq_zero_bin)` | A | IMPORTANT | `qq_zero_k` | `NQ` | none | <10K iterations |
| `main (sample_loop)` | A | CRITICAL | `blk_loop` | `≈ NN` | reduction to `sx`/`sy`, histogram writes | RNG replicable, histogram |
| `main (rng_recur)` | E | IMPORTANT | `sample_loop` | ≤100 | stage dependency on `t1`/`t2` | sequential RNG path |
| `main (rand_fill)` | E | CRITICAL | `sample_loop` | `2*NK` | recurrence on `t1` | sequential RNG core |
| `main (gauss_bin)` | D | CRITICAL | `sample_loop` | `NK` | histogram writes, scalar reduction | indirect binning |
| `main (qq_reduce_i)` | A | SECONDARY | timed block | `NQ` | `q`, `gc` reduction | global sum step |
| `main (qq_reduce_k)` | A | SECONDARY | `qq_reduce_i` | `blksize` | reads `qq` | depends on trimmed final block |

### Data Details
- **Dominant compute loop:** `for (k = 1; k <= blksize; k++)` inside `for (blk...)` (`data/src/ep-omp/ep.c`:171) – every iteration emits one sample, executes ~`2*NK` RNG steps and `NK` binning calculations.
- **Arrays swapped between functions?:** NO – `xx`, `qq`, `q`, etc., live entirely inside `main`.
- **Scratch arrays?:** YES – `xx` and `qq` are malloc’d per block (`xx`: `blksize * 2 * NK` doubles; `qq`: `blksize * NQ` doubles) and reused to hold per-`k` samples and buckets.
- **Mid-computation sync?:** NO – no explicit barriers or mutexes; dependencies are enforced by sequential loop order and reductions.
- **RNG in timed loop?:** YES – the `rng_recur` (`data/src/ep-omp/ep.c`:176) and `rand_fill` (`data/src/ep-omp/ep.c`:217) loops are inside the timed `sample_loop`. RNG replicable: YES, each sample recomputes its seed from `kk = k_offset + k + koff`.

#### Array Overview
- `double x[2*(1<<16)]`: static array on the stack (scalar, defined before `main`). Not used in the timed path beyond being declared for alignment/stubbed legacy.
- `double q[10]`: static stack array, final bin counts for the histogram. Accessed as RW across the aggregate reduction at the end, and reset before timing (`q_init`).
- `double *xx`: dynamically allocated scratch (`malloc(blksize * 2 * NK * sizeof(double))` after trimming `blksize`). Writes happen in `rand_fill` (each sample column is stored as `xx[i*blksize + (k-1)]`), reads happen in `gauss_bin`. The layout is flat, not pointer-to-pointer.
- `double *qq`: dynamically allocated histogram scratch (`malloc(blksize * NQ * sizeof(double))`). Zeroed per block and indexed as `qq[bin * blksize + k]` so that each sample writes to a unique column, enabling the per-`k` bin update without atomics. Later reduced into `q`.
- `double dum[3]` and other small temporaries: static stack arrays used for RNG seeding outside the timed loop.

#### Global state used by loops
- `MK`, `MM`, `NN`, `NK`, `NQ`, `BLKSIZE` drive loop bounds and the number of random samples per block.
- `A`, `S`, `EPSILON`, `r23`, `r46`, `t23`, `t46` control the RNG state machine inside `randlc_ep` and the RNG recurrence loops.
- `sx`, `sy`, `gc`, `q` are the reduction targets updated inside the critical loops.

