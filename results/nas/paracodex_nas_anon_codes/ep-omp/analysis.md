# EP Loop Classification

### Loop Nesting Structure
- `for (v38 = 0; v38 < v1 + 1; v38++)` (`data/src/ep-omp/ep.c`:123) — Type E (seed recurrence)
- `for (v76 = 0; v76 < v78; ++v76)` (`data/src/ep-omp/ep.c`:134) — Type A (chunk iteration)
  ├── `for (v42 = 0; v42 < v75; v42++)` (`data/src/ep-omp/ep.c`:142) — Type A (zero histograms)
  │   └── `for (v38 = 0; v38 < v8; v38++)` (`data/src/ep-omp/ep.c`:144) — Type A (per-bin clear)
  └── `for (v42 = 1; v42 <= v75; v42++)` (`data/src/ep-omp/ep.c`:148) — Type A (per-sample RNG + stats)
      ├── `for (v38 = 1; v38 <= 100; v38++)` (`data/src/ep-omp/ep.c`:153) — Type E (RNG recursion; break)
      ├── `for (v38 = 0; v38 < 2*v7; v38++)` (`data/src/ep-omp/ep.c`:194) — Type E (RNG stream stored in `v60`)
      └── `for (v38 = 0; v38 < v7; v38++)` (`data/src/ep-omp/ep.c`:211) — Type D (histogram binning into `v61`)
- `for (v38 = 0; v38 < v8; v38++)` (`data/src/ep-omp/ep.c`:232) — Type F (reduction over per-bin counts)
  └── `for (v42 = 0; v42 < v75; v42++)` (`data/src/ep-omp/ep.c`:235) — Type F (reduce across samples into `v80`)
- `for (v38 = 0; v38 < v8; v38++)` (`data/src/ep-omp/ep.c`:293) — Type A (post-validate printing)

### Loop Details

## Loop: timer warm-up RNG (`data/src/ep-omp/ep.c`:123)
- **Iterations:** `v1 + 1 = 17`
- **Type:** E — recursive RNG state (`f1`) depends on the previous iteration's `v16`
- **Parent loop:** none (runs immediately after `timer_start`)
- **Contains:** none
- **Dependencies:** strict loop-carried dependency on `v16`
- **Nested bounds:** constant upper bound `v1 + 1`
- **Private vars:** `v16`, `v17`
- **Arrays:** none (uses scalars)
- **Issues:** sequential warm-up; parallelization would require batching RNG reseeding and is unnecessary for this tiny loop

## Loop: block chunk loop (`data/src/ep-omp/ep.c`:134)
- **Iterations:** `v78 = ceil(v3 / v75)` (1 for default `M=24`, scalable to more chunks when `M` grows)
- **Type:** A — iterates over independent chunks of `v75` samples
- **Parent loop:** none (main timed region)
- **Contains:** histogram zeroing (`lines 142-145`), sample generation (`line 148`)
- **Dependencies:** only the `if (v77 + v75 > v37)` adjustment on the last chunk
- **Nested bounds:** `v75` constant within a chunk, `v78` derived from `v3`
- **Private vars:** `v77`, `v75` (mutated for final chunk)
- **Arrays:** `v61` (RW via the contained loops), `v60` (RW inside sample)
- **Issues:** needs to keep `v75`/`v37` pair in sync when splitting work; chunk-splitting decisions must happen before offload

## Loop: zero per-sample histograms (`data/src/ep-omp/ep.c`:142)
- **Iterations:** `v75` (maximum 256 samples per chunk)
- **Type:** A — dense initialization per sample
- **Parent loop:** chunk loop (`line 134`)
- **Contains:** single inner loop at `line 144`
- **Dependencies:** none
- **Nested bounds:** `v8` constant inside
- **Private vars:** `v42`
- **Arrays:** `v61` (W) — clears the `[sample][bin]` view
- **Issues:** repeated O(v75×v8) writes inside the timed region; must keep zeroing before the histogram loop

## Loop: zero bins for one sample (`data/src/ep-omp/ep.c`:144)
- **Iterations:** `v8 = 10`
- **Type:** A — dense constant-bounds loop inside the sample zeroing
- **Parent loop:** sample-zeroing aloft (`line 142`)
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `v38`
- **Arrays:** `v61` (W) — each bin reset independently
- **Issues:** trivial but executes every sample; no hazards

## Loop: per-sample RNG + stats (`data/src/ep-omp/ep.c`:148)
- **Iterations:** `v75` independent samples (256)
- **Type:** A — each sample uses its own RNG seed and stats accumulators
- **Parent loop:** chunk loop (`134`)
- **Contains:** RNG seed refinement (`line 153`), RNG stream generation (`line 194`), histogram accumulation (`line 211`)
- **Dependencies:** none across samples; per-sample RNG tasks are independent (`RNG replicable: YES`)
- **Nested bounds:** constants (`100`, `2*v7`, `v7`) for the contained loops
- **Private vars:** `v40`, `v71`, `v72`, `v16`, `v17`, `v18`, `v19`, `v41`
- **Arrays:** `v60` (W/R storing random values), `v61` (RW histogram), `v27`, `v28` (accumulate sums)
- **Issues:** orchestrates the dominant workload; any offload must ensure each thread maintains its own RNG state and local reductions before updating shared accumulators

## Loop: RNG seed refinement (`data/src/ep-omp/ep.c`:153)
- **Iterations:** up to 100 (breaks when `v39 == 0`)
- **Type:** E — loop-carried dependency through `v40` and `v17`
- **Parent loop:** per-sample RNG (`line 148`)
- **Contains:** `if`/`break` logic, repeated `f1` style recurrences
- **Dependencies:** sequential state updates are required for the RNG and cannot be parallelized within a sample
- **Nested bounds:** constant (1..100)
- **Private vars:** `v39`, `v62`, `v63`, `v66`-`v70`
- **Arrays:** none
- **Issues:** RNG replicable: YES (each sample drives this loop with its own `(v40, v16)` seed); keep the loop on a single thread or warp

## Loop: RNG stream to `v60` (`data/src/ep-omp/ep.c`:194)
- **Iterations:** `2*v7 = 131072`
- **Type:** E — each iteration updates `v16`, which feeds the next random number
- **Parent loop:** per-sample RNG (`line 148`)
- **Contains:** stores `v11 * v16` into the flattened `v60` buffer
- **Dependencies:** sequential RNG state; cannot be parallelized within a sample
- **Nested bounds:** constant
- **Private vars:** `v62`, `v63`, `v65`, `v68`, `v69`, `v70`
- **Arrays:** `v60` (W) — scratch RNG sequence for the following gaussian loop
- **Issues:** high iteration count with loop-carried dependency; preserve per-thread RNG state when offloading

## Loop: histogramming gaussian pairs (`data/src/ep-omp/ep.c`:211)
- **Iterations:** `v7 = 65536`
- **Type:** D — writes to `v61[v41*v75 + (v42-1)]` via indirect bin index `v41`
- **Parent loop:** per-sample RNG (`line 148`)
- **Contains:** pairwise transformation, modulus to bins, scalar accumulators `v71`, `v72`
- **Dependencies:** none between iterations, but many updates contend on the same `v61` bin
- **Nested bounds:** `v7` constant
- **Private vars:** `v22`, `v23`, `v41`
- **Arrays:** `v60` (R), `v61` (RW) — bin counts, `v27`, `v28` (reduce `v71`, `v72`)
- **Issues:** requires atomic/bin-local buffers when parallelizing; per-bin counters have to be merged carefully

## Loop: bin reduction (`data/src/ep-omp/ep.c`:232)
- **Iterations:** `v8 = 10`
- **Type:** F — aggregates bins across `v75` samples into a scalar
- **Parent loop:** timed block after chunk loop
- **Contains:** inner reduction (`line 235`)
- **Dependencies:** reduction into `v80`, then `v59[v38]` and `v32`
- **Nested bounds:** `v75` across samples
- **Private vars:** `v80`
- **Arrays:** `v61` (R), `v59` (RW)
- **Issues:** small but must be the final combine; if parallelized across `v38`, reductions need atomics or per-bin partials

## Loop: reduce across samples (`data/src/ep-omp/ep.c`:235)
- **Iterations:** `v75 = 256`
- **Type:** F — simple reduction to scalar `v80`
- **Parent loop:** bin reduction (`line 232`)
- **Contains:** none
- **Dependencies:** sequential reduction into `v80`
- **Nested bounds:** constant per chunk
- **Private vars:** `v80`
- **Arrays:** `v61` (R)
- **Issues:** no additional hazards beyond the reduction

## Loop: print counts (`data/src/ep-omp/ep.c`:293)
- **Iterations:** `v8 = 10`
- **Type:** A — dense output loop
- **Parent loop:** none (post-timer)
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `v38`
- **Arrays:** `v59` (R)
- **Issues:** I/O-only; negligible compute

### Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| `main` (`line 123`) | E | SECONDARY | none | 17 | RNG recurrence (`v16`) | sequential warm-up |
| `main` (`line 134`) | A | CRITICAL | none | `ceil(v3 / v75)` | chunk-size adjustment | adapt `v75` for last block |
| `main` (`line 142`) | A | IMPORTANT | `line 134` | `v75` | none | repeated zeroing in timed loop |
| `main` (`line 144`) | A | IMPORTANT | `line 142` | `v8` | none | none |
| `main` (`line 148`) | A | CRITICAL | `line 134` | `v75` | none (samples independent) | RNG/per-sample hist depend on inner loops |
| `main` (`line 153`) | E | IMPORTANT | `line 148` | 100 | loop-carried RNG (`v40`, `v17`) | sequential RNG; `RNG replicable: YES` |
| `main` (`line 194`) | E | CRITICAL | `line 148` | `2*v7` | sequential RNG state `v16` | sequential RNG stream |
| `main` (`line 211`) | D | CRITICAL | `line 148` | `v7` | histogram bin updates | needs atomic/privates on `v61[v41]` |
| `main` (`line 232`) | F | SECONDARY | none | `v8` | reduction to `v59`, `v32` | reduction needs careful merging |
| `main` (`line 235`) | F | SECONDARY | `line 232` | `v75` | reduction to `v80` | sequential reduction |
| `main` (`line 293`) | A | SECONDARY | none | `v8` | none | I/O-only |

### Data Details
- **Dominant compute loop:** the per-chunk/per-sample loop (`lines 134 & 148`) dominates the timed region; 256 samples each generate `2*v7 = 131072` RNG outputs and `v7 = 65536` gaussian pairs before accumulating into `v61`.
- **Arrays swapped between functions?:** NO — only scalars are passed to the helper `f1`.
- **Scratch arrays?:** YES — `v60` is a dynamic `v75 × (2*v7)` flat buffer for raw RNG values and `v61` is a dynamic `v75 × v8` histogram that is reset per chunk before accumulation.
- **Global variables driving loops:** `v7 = 2^16`, `v8 = 10`, `v75 = min(v3, 2048)` (256 by default), `v78 = ceil(v3 / v75)` and `v3 = 2^{M - v1}` (with `M = 24`, `v1 = 16`).
- **Mid-computation sync?:** NO — the code relies on sequential RNG and per-sample work without explicit barriers.
- **RNG in timed loop?:** YES — RNG state refinement, stream generation, and gaussian pairing all happen inside the timed region; the RNG is per-sample replicable.
- **Scratch usage:** `v60` stores the `(2*v7)` random values that feed the gaussian loop; `v61` stores histograms for each sample and is reduced into `v59`/`v32` after the chunk.
