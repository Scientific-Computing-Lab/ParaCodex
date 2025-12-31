# CG Loop Classification for GPU Offload Analysis

## Loop Nesting Structure
- `for (it = 1; it <= NITER; it++)` (`data/src/cg-omp/cg.c`:222) Type A
  ├─ Norm accumulation loop (`data/src/cg-omp/cg.c`:227) Type F
  ├─ Vector rescale loop (`data/src/cg-omp/cg.c`:239) Type A
  └─ `conj_grad(...)` (`data/src/cg-omp/cg.c`:325-398)
    ├─ Initialization sweep (`data/src/cg-omp/cg.c`:325) Type A
    ├─ Initial residual reduction (`data/src/cg-omp/cg.c`:332) Type F
    ├─ Stage loop `for (cgit = 1; cgit <= cgitmax; cgit++)` (`data/src/cg-omp/cg.c`:336) Type E
    │   ├─ Row-wise sparse mat-vec (`data/src/cg-omp/cg.c`:340) Type B → inner loop Type A
    │   ├─ Scalar dot reduction (`data/src/cg-omp/cg.c`:351) Type F
    │   ├─ Vector updates `z`/`r` (`data/src/cg-omp/cg.c`:361) Type A
    │   ├─ Residual norm reduction (`data/src/cg-omp/cg.c`:367) Type F
    │   └─ Direction update (`data/src/cg-omp/cg.c`:374) Type A
    ├─ Final sparse mat-vec (`data/src/cg-omp/cg.c`:379) Type B → inner loop Type A
    └─ Final residual reduction (`data/src/cg-omp/cg.c`:392) Type F

## Loop Details

### Loop: main benchmark driver at `data/src/cg-omp/cg.c`:222
- **Iterations:** `NITER` (>10K for standard classes)
- **Type:** A – dense timed driver with constant bounds and no indirect addressing
- **Parent loop:** none
- **Contains:** `conj_grad(...)` call plus two tight vector loops for norm/rescale
- **Dependencies:** stage dependency on `conj_grad` output, no reductions inside this loop body
- **Nested bounds:** constant (`NITER`)
- **Private vars:** `it`
- **Arrays:** `x(RW)`, `z(R)`, `r(R)`, `p(R)` (used inside child loops)
- **Issues:** none beyond reliance on `conj_grad` for residual convergence

### Loop: norm accumulations at `data/src/cg-omp/cg.c`:227
- **Iterations:** `lastcol - firstcol + 1` (=`NA`)
- **Type:** F – dual scalar reductions (`norm_temp1`, `norm_temp2`)
- **Parent loop:** main benchmark loop (line 222)
- **Contains:** none
- **Dependencies:** reductions on `norm_temp1`, `norm_temp2`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `x(R)`, `z(R)`
- **Issues:** reduction requires accumulation across iterations; parallel reduction or tiling needed

### Loop: vector rescale at `data/src/cg-omp/cg.c`:239
- **Iterations:** `lastcol - firstcol + 1`
- **Type:** A – dense scaling of `x`
- **Parent loop:** main benchmark loop (line 222)
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `j`, `norm_temp2`
- **Arrays:** `x(RW)`, `z(R)`
- **Issues:** independent updates, parallelizable with per-element work

### Loop: conj_grad initialization sweep at `data/src/cg-omp/cg.c`:325
- **Iterations:** `naa` (≈`NA`)
- **Type:** A – traverses contiguous vectors to zero/assign
- **Parent loop:** `conj_grad`
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `q(W)`, `z(W)`, `r(RW)`, `p(W)`, `x(R)`
- **Issues:** straightforward initialization; no reductions or atomic needs

### Loop: initial residual reduction at `data/src/cg-omp/cg.c`:332
- **Iterations:** `lastcol - firstcol + 1`
- **Type:** F – computes `rho = Σ r[j]^2`
- **Parent loop:** `conj_grad`
- **Contains:** none
- **Dependencies:** reduction on `rho`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `r(R)`
- **Issues:** requires a reduction; reduction tree or atomic updates needed if parallelized

### Loop: CG stage iteration at `data/src/cg-omp/cg.c`:336
- **Iterations:** `cgitmax` (25)
- **Type:** E – recurrence/stage loop (next iteration depends on previous `rho`, `beta`, `p`)
- **Parent loop:** `conj_grad`
- **Contains:** sparse mat-vec, reductions, vector updates (child loops listed below)
- **Dependencies:** stage dependency via `rho`, `rho0`, `beta`, `p`, `z`, `r`
- **Nested bounds:** constant outer loop but contains variable inner structures
- **Private vars:** `cgit`, `end`
- **Arrays:** `a(R)`, `colidx(R)`, `rowstr(R)`, `p(R)`, `q(W)`, `z(RW)`, `r(RW)`, `x(R)`, `rho(RW)`
- **Issues:** sequential stage barrier; parallelization must keep iterations ordered or rely on pipelined variants

### Loop: sparse mat-vec inside stage at `data/src/cg-omp/cg.c`:340
- **Iterations:** `end` rows (`lastrow - firstrow + 1`)
- **Type:** B – outer loop over rows with varying inner counts due to CSR structure
- **Parent loop:** stage loop (line 336)
- **Contains:** inner dot-product loop over `k` with variable bounds
- **Dependencies:** none across rows
- **Nested bounds:** outer constant, inner variable (`rowstr[j]`/`rowstr[j+1]`)
- **Private vars:** `j`, `tmp1`, `tmp2`, `sum`, `k`, `tmp3`
- **Arrays:** `rowstr(R)`, `colidx(R)`, `a(R)`, `p(R)`, `q(W)`
- **Issues:** indirect reads of `colidx`; the inner accumulation can be parallelized per row, but sparse indexing limits coalescing

### Loop: scalar dot reduction at `data/src/cg-omp/cg.c`:351
- **Iterations:** `lastcol - firstcol + 1`
- **Type:** F – computes `d = Σ p[j]*q[j]`
- **Parent loop:** stage loop (line 336)
- **Contains:** none
- **Dependencies:** reduction for `d`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `p(R)`, `q(R)`
- **Issues:** reduction requires tree or atomic; independent per iteration but needs accumulation

### Loop: vector update z/r at `data/src/cg-omp/cg.c`:361
- **Iterations:** `lastcol - firstcol + 1`
- **Type:** A – two independent vector updates per iteration
- **Parent loop:** stage loop (line 336)
- **Contains:** none
- **Dependencies:** none beyond using scalars `alpha`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `z(RW)`, `r(RW)`, `p(R)`, `q(R)`
- **Issues:** per-element independent; parallel-friendly

### Loop: residual norm reduction at `data/src/cg-omp/cg.c`:367
- **Iterations:** `lastcol - firstcol + 1`
- **Type:** F – recomputes `rho = Σ r[j]^2`
- **Parent loop:** stage loop (line 336)
- **Contains:** none
- **Dependencies:** reduction on `rho`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `r(R)`
- **Issues:** reduction needs synchronization if parallelized; reused by next stage iteration

### Loop: direction update at `data/src/cg-omp/cg.c`:374
- **Iterations:** `lastcol - firstcol + 1`
- **Type:** A – updates search direction `p`
- **Parent loop:** stage loop (line 336)
- **Contains:** none
- **Dependencies:** none besides scalar `beta`
- **Nested bounds:** constant
- **Private vars:** `j`
- **Arrays:** `p(RW)`, `r(R)`
- **Issues:** independent; vectorizable

### Loop: final sparse mat-vec at `data/src/cg-omp/cg.c`:379
- **Iterations:** `end` rows
- **Type:** B – same CSR access pattern as the stage mat-vec
- **Parent loop:** `conj_grad`
- **Contains:** inner dot product loop with variable bounds
- **Dependencies:** none across rows
- **Nested bounds:** variable inner bounds from `rowstr`
- **Private vars:** `j`, `tmp1`, `tmp2`, `d`, `k`, `tmp3`
- **Arrays:** `rowstr(R)`, `colidx(R)`, `a(R)`, `z(R)`, `r(W)`
- **Issues:** indirect access; parallelization needs atomic-free CSR scatter per row

### Loop: final residual reduction at `data/src/cg-omp/cg.c`:392
- **Iterations:** `lastcol - firstcol + 1`
- **Type:** F – computes `sum = Σ (x[j]-r[j])^2`
- **Parent loop:** `conj_grad`
- **Contains:** none
- **Dependencies:** scalar reduction on `sum`
- **Nested bounds:** constant
- **Private vars:** `j`, `d`
- **Arrays:** `x(R)`, `r(R)`
- **Issues:** reduction barrier before `sqrt` and `*rnorm`

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| `main` bench loop (`data/src/cg-omp/cg.c`:222) | A | CRITICAL | none | `NITER` | stage on `conj_grad` | none |
| Norm accumulation (`data/src/cg-omp/cg.c`:227) | F | CRITICAL | main loop | `NA` | reduction | requires reduction tree |
| Vector rescale (`data/src/cg-omp/cg.c`:239) | A | CRITICAL | main loop | `NA` | none | none |
| Init sweep (`data/src/cg-omp/cg.c`:325) | A | CRITICAL | `conj_grad` | `naa` | none | none |
| Initial residual reduction (`data/src/cg-omp/cg.c`:332) | F | CRITICAL | `conj_grad` | `NA` | reduction (`rho`) | reduction sync |
| CG stage loop (`data/src/cg-omp/cg.c`:336) | E | CRITICAL | `conj_grad` | 25 | stage dependency | sequential iteration order |
| Stage mat-vec (`data/src/cg-omp/cg.c`:340) | B | CRITICAL | stage loop | `NA` × avg `NONZER` | none | indirect CSR access |
| Stage dot reduction (`data/src/cg-omp/cg.c`:351) | F | CRITICAL | stage loop | `NA` | reduction | needs tree/atomics |
| Vector update z/r (`data/src/cg-omp/cg.c`:361) | A | CRITICAL | stage loop | `NA` | none | none |
| Residual norm (`data/src/cg-omp/cg.c`:367) | F | CRITICAL | stage loop | `NA` | reduction | needs sync |
| Direction update (`data/src/cg-omp/cg.c`:374) | A | CRITICAL | stage loop | `NA` | none | none |
| Final mat-vec (`data/src/cg-omp/cg.c`:379) | B | CRITICAL | `conj_grad` | `NA` × avg `NONZER` | none | CSR indirect reads |
| Final residual reduction (`data/src/cg-omp/cg.c`:392) | F | CRITICAL | `conj_grad` | `NA` | reduction | reduction sync |

## Data Details
- **Dominant compute loop:** `for (it = 1; it <= NITER; it++)` (`data/src/cg-omp/cg.c`:222) – timed section that repeatedly calls `conj_grad` and performs norm/rescale, so it dominates runtime.
- **Arrays swapped between functions?:** NO – `conj_grad` drives vectors in-place; no ping-pong buffers move between host/device contexts.
- **Scratch arrays?:** YES – `conj_grad` declares `double sum_array[NA+2]` (unused placeholder) and uses temporaries `q`, `z`, `r`, `p` as transient storage between loops.
- **Mid-computation sync?:** NO – there are no explicit barrier/synchronization constructs inside loops beyond sequential order.
- **RNG in timed loop?:** NO – RNG (`randlc`) is only invoked during `makea`/`sprnvc` before timing begins.

### Array Characterization
- `colidx[NZ]` (`data/src/cg-omp/cg.c`:29) – static CSR column indices (flat, contiguous, compile-time size). Read-only in compute loops.
- `rowstr[NA+1]` (`data/src/cg-omp/cg.c`:30) – static row pointers, flat array describing CSR boundaries; read-only in CG stage.
- `a[NZ]` (`data/src/cg-omp/cg.c`:35) – static CSR values; read-only except during setup and mat-vec loops.
- `x`, `z`, `p`, `q`, `r` (`data/src/cg-omp/cg.c`:36-40) – static vectors used for input, residual, direction, and auxiliary state; reused across iterations (RW when updated, R when read).
- `arow`, `acol[][NONZER+1]`, `aelt[][NONZER+1]` – static metadata used during matrix assembly (`makea`/`sparse`). They are pointer-to-array aggregates allocated at compile time via globals.
- `sum_array[NA+2]` – local stack array inside `conj_grad`; currently unused but sized to `NA+2` for intermediate reductions.
- **Global scalars used across loops:** `naa`, `lastrow`, `lastcol`, `firstrow`, `firstcol`, `amult`, `tran`, `nonzer` macros drive bounds and RNG seed progression.

### Flagged Issues
- **Reduction hotspots:** Every dot-product/residual loop relies on scalar reductions (`rho`, `sum`, `norm_temp*`); parallelization must introduce tree reductions or atomics.
- **Stage dependency:** `cgit` iterations depend on previous `rho`, `beta`, and direction `p`, so only intra-iteration parallelism (rows, vector updates) is safe.
- **Indirect sparse access:** CSR loops (lines 340 and 379) scatter via `colidx`/`rowstr`; hardware offload must manage irregular memory access for both reads and writes (writes only to contiguous `q`/`r`).
