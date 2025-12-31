# Loop Classification for GPU Offload - CG

## Loop Nesting Structure
- `for (it = 1; it <= NITER; it++)` (`data/src/cg-omp/cg.c:226`) Type E
  - `conj_grad` entry resets & pre-reduction (`data/src/cg-omp/cg.c:330-338`) Type A/F
  - `conj_grad` `cgit` iteration (`data/src/cg-omp/cg.c:341`) Type E
    - SpMV (`data/src/cg-omp/cg.c:345`) Type B (contains inner nonzero accumulation, `data/src/cg-omp/cg.c:350`)
    - dot product (`data/src/cg-omp/cg.c:360`) Type F
    - vector updates (`data/src/cg-omp/cg.c:369-385`) Type A/F
  - `conj_grad` post-iteration SpMV (`data/src/cg-omp/cg.c:390`) Type B
  - `conj_grad` final norm (`data/src/cg-omp/cg.c:401`) Type F
- `makea` matrix generator (`data/src/cg-omp/cg.c:434`) Type A (Secondary)
  - `sprnvc` RNG driver (`data/src/cg-omp/cg.c:585`) AVOID/Type E
- `sparse` assembly loops (`data/src/cg-omp/cg.c:474-572`) Type B (Secondary)

## Loop Details

### Loop: main `for (it = 1; it <= NITER; it++)` at data/src/cg-omp/cg.c:226
- **Iterations:** `NITER` (class-defined, e.g., 15/75/100)
- **Type:** E - stage dependency in the benchmark driver, since each iteration depends on the previous `x/z` state
- **Parent loop:** none
- **Contains:** `conj_grad` invocation, norm/reduction loops (`data/src/cg-omp/cg.c:330-245`)
- **Dependencies:** stage (the `x`/`z` glyph returned by `conj_grad` feeds the next iteration)
- **Nested bounds:** constant (`NITER`, fixed by `npbparams`)
- **Private vars:** `it`
- **Arrays:** `x`(RW), `z`(RW), `p`(RW), `q`(RW), `r`(RW)
- **Issues:** sequential outer loop; cannot reorder iterations without invalidating the convergence history or the printed `zeta` values

### Loop: main norm accumulation at data/src/cg-omp/cg.c:231
- **Iterations:** `end = lastcol - firstcol + 1` (~`NA`)
- **Type:** F - reduction for `norm_temp1` and `norm_temp2`
- **Parent loop:** `for (it = 1; it <= NITER; it++)` (`data/src/cg-omp/cg.c:226`)
- **Contains:** none
- **Dependencies:** reduction on `norm_temp1` and `norm_temp2`
- **Nested bounds:** constant (`end`, derived from `NA`)
- **Private vars:** `j`
- **Arrays:** `x`(R), `z`(R)
- **Issues:** parallel reductions must be managed (two scalar sums) when offloading

### Loop: main vector rescale at data/src/cg-omp/cg.c:243
- **Iterations:** `end = lastcol - firstcol + 1` (~`NA`)
- **Type:** A - dense elementwise scaling
- **Parent loop:** `for (it = 1; it <= NITER; it++)` (`data/src/cg-omp/cg.c:226`)
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `x`(W), `z`(R)
- **Issues:** memory-bound write to `x`; independent iterations parallelize easily once `z` is computed

### Loop: `conj_grad` initialization at data/src/cg-omp/cg.c:330
- **Iterations:** `naa = NA`
- **Type:** A - dense zeroing/assignment of vectors
- **Parent loop:** `conj_grad` entry scope
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant (`NA`)
- **Private vars:** `j`
- **Arrays:** `q`(W), `z`(W), `r`(W), `p`(W), `x`(R)
- **Issues:** simple parallel fill; no reductions

### Loop: initial rho reduction at data/src/cg-omp/cg.c:337
- **Iterations:** `lastcol - firstcol + 1` (~`NA`)
- **Type:** F - reduction into `rho`
- **Parent loop:** `conj_grad`
- **Contains:** none
- **Dependencies:** reduction on `rho`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `r`(R)
- **Issues:** reduction requires synchronization if parallelized; `rho` reused by `cgit`

### Loop: `cgit` iteration at data/src/cg-omp/cg.c:341
- **Iterations:** `cgitmax = 25`
- **Type:** E - loop-carried recurrence (`rho`, `p`, `r`, `beta` update sequentially)
- **Parent loop:** `conj_grad`
- **Contains:** SpMV (line 345), dot product (line 360), vector updates (`z`/`r`/`p` lines 369-385)
- **Dependencies:** stage/recurrence on `rho`/`beta`; each iteration uses the latest `p`/`r`
- **Nested bounds:** constant for `j` (sets `end = NA` each iteration); inner nonzero loop variable (per row)
- **Private vars:** `cgit`, `j`, `tmp1`, `tmp2`, `sum`, `tmp3`, `d`, `alpha`, `beta`, `rho0`
- **Arrays:** `rowstr`(R), `colidx`(R), `a`(R), `p`(R), `q`(W), `z`(RW), `r`(RW)
- **Issues:** limited parallelism across `cgit` iterations; careful reduction handling inside nested loops (see below)

### Loop: SpMV inside `cgit` at data/src/cg-omp/cg.c:345
- **Iterations:** `end = lastrow - firstrow + 1` (~`NA`); inner loop runs over `rowstr[j+1]-rowstr[j]` (≈ `row nnz`)
- **Type:** B - sparse matrix-vector multiply with variable inner bounds
- **Parent loop:** `for (cgit = 1; cgit <= cgitmax; cgit++)` (`data/src/cg-omp/cg.c:341`)
- **Contains:** inner reduction loop `for (k = tmp1; k < tmp2; k++)` (`data/src/cg-omp/cg.c:350`)
- **Dependencies:** none across `j`; each row is independent
- **Nested bounds:** outer constant (`NA`), inner variable (`rowstr` span per row)
- **Private vars:** `j`, `tmp1`, `tmp2`, `sum`, `tmp3`, `k`
- **Arrays:** `rowstr`(R), `colidx`(R), `a`(R), `p`(R), `q`(W)
- **Issues:** indirect accesses via `colidx`/`p` create irregular gathers; consider atomic free updates because each `q[j]` is distinct

### Loop: dot product inside `cgit` at data/src/cg-omp/cg.c:360
- **Iterations:** `end = lastcol - firstcol + 1` (~`NA`)
- **Type:** F - global reduction for `d`
- **Parent loop:** `cgit`
- **Contains:** none
- **Dependencies:** reduction on `d`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `p`(R), `q`(R)
- **Issues:** reduction requires synchronization (tree reduction or atomic) when parallelizing

### Loop: `z/r` update inside `cgit` at data/src/cg-omp/cg.c:369
- **Iterations:** `end = lastcol - firstcol + 1` (~`NA`)
- **Type:** A - dense SAXPY-like updates
- **Parent loop:** `cgit`
- **Contains:** none
- **Dependencies:** none per iteration
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `z`(RW), `r`(RW), `p`(R), `q`(R)
- **Issues:** memory bandwidth heavy, but embarrassingly parallel per element

### Loop: rho recompute inside `cgit` at data/src/cg-omp/cg.c:376
- **Iterations:** `end = lastrow - firstrow + 1` (~`NA`)
- **Type:** F - reduction
- **Parent loop:** `cgit`
- **Contains:** none
- **Dependencies:** reduction on `rho`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `r`(R)
- **Issues:** another reduction; needs careful accumulation to avoid divergence

### Loop: `p` refresh inside `cgit` at data/src/cg-omp/cg.c:384
- **Iterations:** `end = lastcol - firstcol + 1` (~`NA`)
- **Type:** A - dense PX+Y update
- **Parent loop:** `cgit`
- **Contains:** none
- **Dependencies:** none per iteration
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `p`(RW), `r`(R)
- **Issues:** memory-bound, but parallelizable with a simple loop

### Loop: post-`cgit` SpMV at data/src/cg-omp/cg.c:390
- **Iterations:** `end = lastrow - firstrow + 1` (~`NA`); inner loop `k` from `rowstr[j]` to `rowstr[j+1]`
- **Type:** B - sparse matrix-vector multiply computing `r = A * z`
- **Parent loop:** `conj_grad`
- **Contains:** inner reduction loop identical to the earlier SpMV
- **Dependencies:** none across `j`
- **Nested bounds:** outer constant, inner variable (row lengths)
- **Private vars:** `j`, `tmp1`, `tmp2`, `d`, `tmp3`, `k`
- **Arrays:** `rowstr`(R), `colidx`(R), `a`(R), `z`(R), `r`(W)
- **Issues:** irregular access pattern; each row writes to distinct `r[j]`, so no atomics needed

### Loop: final norm at data/src/cg-omp/cg.c:401
- **Iterations:** `lastcol - firstcol + 1` (~`NA`)
- **Type:** F - reduction for `sum`
- **Parent loop:** `conj_grad`
- **Contains:** none
- **Dependencies:** reduction on `sum`
- **Nested bounds:** constant
- **Private vars:** `j`, `d`
- **Arrays:** `x`(R), `r`(R)
- **Issues:** reduction needs explicit accumulation; used to seed `*rnorm`

### Loop: colidx shift at data/src/cg-omp/cg.c:175
- **Iterations:** `lastrow - firstrow + 1` (~`NA`); each row loops over stored nonzeros
- **Type:** B - sparse traversal adjusting column indices
- **Parent loop:** `main` initialization (before `timer_stop(T_init)`)
- **Contains:** inner `for (k = rowstr[j]; k < rowstr[j+1]; k++)`
- **Dependencies:** none
- **Nested bounds:** outer constant, inner variable (row lengths)
- **Private vars:** `j`, `k`
- **Arrays:** `rowstr`(R), `colidx`(RW)
- **Issues:** Secondary (setup) work; not part of timed region

### Loop: `makea` row generator at data/src/cg-omp/cg.c:434
- **Iterations:** `n` (=`NA`) once during setup
- **Type:** A - populates sparse CSR metadata per row
- **Parent loop:** `main` initialization (through `makea`)
- **Contains:** inner `for (ivelt = 0; ivelt < nzv; ivelt++)` storing column/value pairs
- **Dependencies:** none, but calls `sprnvc`/`vecset`
- **Nested bounds:** outer constant (`n`), inner variable (`nzv` up to `NONZER`)
- **Private vars:** `iouter`, `ivelt`, `nzv`, `vc`, `ivc`
- **Arrays:** `arow`(W), `acol`(W), `aelt`(W)
- **Issues:** Secondary work; relies on RNG/helper calls to build the matrix

### Loop: sparse assembly at data/src/cg-omp/cg.c:508
- **Iterations:** `i` over `n` rows; inner loops over `arow[i]` entries and secondary `nzrow` loops (≈ `NONZER`)
- **Type:** B - sparse assembly with variable inner bounds (`arow`)
- **Parent loop:** `sparse`, called from `makea`
- **Contains:** embedded search loop over `rowstr` slots (lines 522-539) and compaction loops (lines 558-566)
- **Dependencies:** none beyond keeping CSR invariants
- **Nested bounds:** multiple levels variable (depends on `arow[i]`, `rowstr` spans)
- **Private vars:** `i`, `j`, `nza`, `k`, `kk`, `nzrow`, `jcol`, `va`, `scale`, `cont40`
- **Arrays:** `arow`(R), `acol`(R), `aelt`(R), `rowstr`(RW), `colidx`(RW), `a`(RW), `nzloc`(RW)
- **Issues:** setup-only but irregular; relies on detects for row insert position and keeps counts

### Loop: `sprnvc` RNG loop at data/src/cg-omp/cg.c:585
- **Iterations:** `nz` (≈ `NONZER`, typically < 30)
- **Type:** E (RNG with loop-carried random state)
- **Parent loop:** `makea` via `sprnvc`
- **Contains:** inner `for (ii = 0; ii < nzv; ii++)` checking duplicates
- **Dependencies:** sequential RNG (`randlc`) using global `tran`/`amult`
- **Nested bounds:** while loop controlled by `nzv`; inner loop `ii` runs over current entries (< nz)
- **Private vars:** `nzv`, `ii`, `i`, `vecelt`, `vecloc`
- **Arrays:** `v`(W), `iv`(W)
- **Issues:** AVOID – runs outside timed region, few iterations, uses RNG state (deterministic but serial)

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| `main::for (it = 1; it <= NITER; it++)` | E | CRITICAL | none | `NITER` | stage | sequential driver, depends on previous `x/z` |
| `main::norm accumulation` | F | CRITICAL | `it` | ~`NA` | reduction | two scalar reductions per iteration |
| `conj_grad::cgit` | E | CRITICAL | `conj_grad` | 25 | stage | loop-carried `beta`/`rho` dependency |
| `conj_grad::SpMV` | B | CRITICAL | `cgit` | ~`NA * avg_nonzeros` | none | irregular gather via `colidx` |
| `conj_grad::dot product` | F | CRITICAL | `cgit` | ~`NA` | reduction | global reduction on `d` |
| `conj_grad::post SpMV` | B | CRITICAL | `conj_grad` | ~`NA * avg_nonzeros` | none | final `A*z` build, irregular indexing |
| `main::colidx shift` | B | SECONDARY | `main` | ~`NA * avg_nonzeros` | none | setup-only index fix |
| `makea::row generator` | A | SECONDARY | `main` | `NA` | none | RNG-backed build of CSR metadata |
| `sprnvc` RNG loop | E | AVOID | `makea` | `NONZER` | sequential RNG | <10K iterations, uses `randlc` |

## Data Details
- **Dominant compute loop:** `conj_grad`'s `cgit` iteration loop (`data/src/cg-omp/cg.c:341`) drives runtime, since it repeats SpMV/dot/vector updates `cgitmax` times per benchmark iteration.
- **Arrays:** CSR data lives in flat static arrays (`a[NZ]`, `colidx[NZ]`, `rowstr[NA+1]`). Vectors `x`, `z`, `p`, `q`, `r` are static length `NA+2`; temporary metadata like `arow`, `acol`, `aelt`, `iv`, `sum_array` also static and global.
- **Allocation:** Everything is statically allocated; there is no heap allocation or pointer-to-pointer indirection inside the timed phase.
- **Struct members:** none used; all data is in plain arrays.
- **Global vars:** `naa`, `nzz`, `firstrow`, `lastrow`, `firstcol`, `lastcol`, `amult`, `tran`, `timeron`, `loop_iter`, `conj_calls` influence loop bounds or RNG.
- **Arrays swapped between functions?:** NO — `makea` populates the CSR structures once, and `conj_grad` reuses the same global arrays without swapping buffers.
- **Scratch arrays?:** YES — `makea`/`sprnvc` rely on small scratch buffers (`vc`, `ivc`, `nzloc`), and `conj_grad` declares `sum_array[NA+2]` plus per-loop scalars.
- **Mid-computation sync?:** NO — the benchmark is serial, so there is no explicit synchronization barrier in the timed region.
- **RNG in timed loop?:** NO — `randlc` appears only during matrix construction (`makea`/`sprnvc`) before `timer_start(T_bench)`.
