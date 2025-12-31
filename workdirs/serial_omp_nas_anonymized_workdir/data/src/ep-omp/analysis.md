# Loop Classification for GPU Offload

## Loop Nesting Structure
- `main` → `for (blk=0; blk < numblks; ++blk)` (`ep.c:155`) Type A
  ├── `for (k=0; k<blksize; k++)` (`ep.c:163`) Type A
  │   └── `for (i=0; i<NQ; i++)` (`ep.c:165`) Type A
  ├── `for (k = 1; k <= blksize; k++)` (`ep.c:169`) Type A (outer sample loop)
  │   ├── `for (i = 1; i <= 100; i++)` (`ep.c:174`) Type E
  │   ├── `for(i=0; i<2*NK; i++)` (`ep.c:215`) Type E
  │   └── `for (i = 0; i < NK; i++)` (`ep.c:232`) Type A
  └── `for(i=0; i<NQ; i++)` (`ep.c:252`) Type F
- `main` → `for (i = 0; i < MK + 1; i++)` (`ep.c:144`) Type E

## Loop Details
### Loop: `main` at `ep.c:144`
- **Iterations:** `MK+1` (≈17 for MK=16) executed once before block processing
- **Type:** E – sequential `randlc_ep` updates the exponent `t1`, so each iteration depends on the prior seed
- **Parent loop:** none (setup)
- **Contains:** none
- **Dependencies:** recurrence through `t1`/`t2`; rng stream advances per iteration
- **Nested bounds:** constant
- **Private vars:** `i`, `t2`
- **Arrays:** none (scalar state only)
- **Issues:** RNG in loop, sequential setup

### Loop: `main` at `ep.c:155`
- **Iterations:** `numblks≈ceil(NN/blksize)` where `NN=2^{M−MK}`; dominates timed region
- **Type:** A – each block processes independent samples, the inner RNG (line 169) is sequential but per-sample
- **Parent loop:** none (outer timed loop)
- **Contains:** reset `qq` (lines 163-166), per-sample `k` loop (line 169), reduction to `q` (line 252)
- **Dependencies:** independent blocks except tail block shrinks `blksize`
- **Nested bounds:** variable (`blksize` is adjusted for the last block)
- **Private vars:** `blk`, `koff`, `blksize`, `k_offset`
- **Arrays:** `qq` (RW) scratch per block, `xx` (W) regenerated, `q` (RW) accumulates final counts
- **Issues:** Dominant timed loop, RNG replicable (unique `(blk,k)` seed), ensure tail `blksize` matches `xx`/`qq` allocations

### Loop: `main` at `ep.c:163`
- **Iterations:** `blksize` (2048 default) per block, `qq` cleared for each sample
- **Type:** A – dense zeroing of the per-block scratch buffer
- **Parent loop:** block loop (`ep.c:155`)
- **Contains:** inner `for (i=0; i<NQ; i++)` (line 165)
- **Dependencies:** none
- **Nested bounds:** outer bound (`blksize`) variable for tail block, inner bound constant (`NQ=10`)
- **Private vars:** `k`, `i`
- **Arrays:** `qq` (W)
- **Issues:** `<10K iterations` per block and small increments; can be fused into later kernels if needed

### Loop: `main` at `ep.c:169`
- **Iterations:** up to `blksize` samples per block (≈2048) executed in the timed loop
- **Type:** A – independent samples, each spawns sequential RNG loops; outer `k` loop fits the “outer A + inner E” pattern
- **Parent loop:** block loop (`ep.c:155`)
- **Contains:** RNG recurrence (line 174), random number fill (line 215), gaussian transform (line 232)
- **Dependencies:** none between samples; uses `kk` derived from `koff`, so seeds stay reproducible
- **Nested bounds:** `blksize` (variable on last block)
- **Private vars:** `k`, `kk`, `t1`, `t2`, `tmp_sx`, `tmp_sy`
- **Arrays:** `xx` (W) reinitialized per sample, `qq` (W) updated through the sample, `q` (RW) aggregated outside
- **Issues:** RNG replicable: YES (each `(blk,k)` recomputes its seed); inner RNG loops are sequential (Type E)

### Loop: `main` at `ep.c:174`
- **Iterations:** up to 100 per sample, executed for each `k`
- **Type:** E – recurrence from halving `kk` and updating `t1`/`t2`; sequential rng stream
- **Parent loop:** sample loop (`ep.c:169`)
- **Contains:** none
- **Dependencies:** `kk` and `t1` carry data between iterations; `if (ik==0) break` stops early
- **Nested bounds:** constant (≤100)
- **Private vars:** `i`, `ik`, `in_t1`, `in_a1`, `in_a2`, `in_x1`, `in_x2`, `in_z`, `in_t2`, `in_t3`, `in_t4`, `t3`
- **Arrays:** none
- **Issues:** RNG in loop, stage dependency (recurrence), sequential logic prevents parallelization

### Loop: `main` at `ep.c:215`
- **Iterations:** `2*NK` (≈131072) per sample; fills `xx` with random candidates
- **Type:** E – each iteration calls `randlc` via inline math that updates `t1`, so iteration `i` depends on `i-1`
- **Parent loop:** sample loop (`ep.c:169`)
- **Contains:** none
- **Dependencies:** recurrence in `t1`/`t2`; deterministic RNG stream
- **Nested bounds:** constant (`2*NK`)
- **Private vars:** `i`, `in_t1`, `in_x1`, `in_x2`, `in_a1`, `in_a2`, `in_t2`, `in_z`, `in_t3`, `in_t4`
- **Arrays:** `xx` (W)
- **Issues:** RNG in loop, sequential recurrence (Type E)

### Loop: `main` at `ep.c:232`
- **Iterations:** `NK` (≈65536) per sample, computes gaussian pairs and updates counters
- **Type:** A – dense traversal of the generated random pairs, writes to scratch counts
- **Parent loop:** sample loop (`ep.c:169`)
- **Contains:** conditional update, no further loops
- **Dependencies:** none between `i` iterations; `qq` updates are scatter to unique `l*blksize + (k-1)` slots
- **Nested bounds:** constant (`NK`)
- **Private vars:** `i`, `x1`, `x2`, `l`, `t1`, `t2`, `t3`, `t4`
- **Arrays:** `xx` (R), `qq` (RW)
- **Issues:** None besides scalar operations; no reduction/atomic needed because indexes are unique per sample

### Loop: `main` at `ep.c:252`
- **Iterations:** `NQ` (10) outer × `blksize` inner; aggregates the per-sample histogram into `q`
- **Type:** F – reduction of the `qq` scratch array into scalars `q[i]` and `gc`
- **Parent loop:** block loop (`ep.c:155`)
- **Contains:** inner reduction over `k`
- **Dependencies:** reduction to `sum_qi`; each `k` contributes a scalar per `i`
- **Nested bounds:** outer constant (`NQ=10`), inner `blksize` (variable for tail block)
- **Private vars:** `i`, `k`, `sum_qi`
- **Arrays:** `qq` (R), `q` (RW), `gc` (RW)
- **Issues:** Reduction patterns require accumulation or parallel reduction primitives when offloading

### Loop: `main` at `ep.c:132`
- **Iterations:** `NQ=10` executed once before timing; initializes global histogram
- **Type:** A – dense initialization
- **Parent loop:** none (setup)
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `i`
- **Arrays:** `q` (W)
- **Issues:** `<10K iterations` and setup-only; not part of the timed computation

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| `main` | E | Secondary | none | `MK+1` | recurrence on `t1` | RNG in setup, sequential rng stream |
| `main` | A | Critical | none | `numblks≈ceil(NN/blksize)` | block independence, tail `blksize` | Dominant timed loop, RNG replicable |
| `main` | A | Important | `blk` | `blksize` | none | small loops (<10K) |
| `main` | A | Important | `blk` → `k` | `blksize` | none (outer) | RNG replicable, inner E loops |
| `main` | E | Important | `k` | ≤100 per sample | recurrence on `kk`/`t1` | RNG in loop, stage dependency |
| `main` | E | Important | `k` | `2*NK` | recurrence on `t1` | RNG in loop, sequential filling |
| `main` | A | Important | `k` | `NK` | none | scatter writes to `qq` |
| `main` | F | Important | `blk` | `NQ × blksize` | reduction to `q`/`gc` | accumulation needed |
| `main` | A | Secondary | none | `NQ` | none | setup-only initialization |

## Data Details
- **Dominant compute loop:** `for (blk=0; blk < numblks; ++blk)` (`ep.c:155`) over independent space slices
- **Arrays swapped between functions?:** NO – `xx`, `qq`, `q`/`gc` remain in `main`
- **Scratch arrays?:** YES – `xx` (`malloc(blksize*2*NK)`) and `qq` (`malloc(blksize*NQ)`) are scratch buffers reused per block
- **Mid-computation sync?:** NO explicit synchronization inside the timed loop
- **RNG in timed loop?:** YES – `randlc` streams appear in the setup loop, the per-sample recurrence (`ep.c:174`), and the random-number fill (`ep.c:215`)
