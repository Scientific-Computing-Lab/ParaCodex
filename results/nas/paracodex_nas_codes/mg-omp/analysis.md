- **main it loop** (mg.c:195) Type A — driver over `nit`, invokes `mg3P`/`resid` each iteration.
  - contains multigrid V-cycle stages in `mg3P` (mg.c:344, mg.c:355) Type C2
    - `rprj3` restriction loops (mg.c:546, mg.c:561) Type G
    - `zero3` clears (mg.c:352, mg.c:358; body at mg.c:1242) Type A
    - `psinv` smoothing loops (mg.c:389, mg.c:401) Type G
    - `interp` prolongation loops (mg.c:632, mg.c:647, mg.c:660, mg.c:672, mg.c:685 and fallback at mg.c:722+) Type G
    - `resid` residual loops (mg.c:459, mg.c:473; also top-level mg.c:369) Type G
    - `comm3` halo updates (mg.c:913, mg.c:921, mg.c:929) Type A
- Setup/aux loops (mg.c:294, mg.c:299, mg.c:305; mg.c:270 timing print; norm/zran/power/zero) Type A/F/G, run during initialization or diagnostics only.

## Loop Details

### Loop: main at mg.c:195
- **Iterations:** `nit` (class dependent; 4–50)
- **Type:** A — flat driver loop over iterations
- **Parent loop:** none
- **Contains:** calls `mg3P` (V-cycle) then `resid`
- **Dependencies:** none in loop body
- **Nested bounds:** constant
- **Private vars:** `it`
- **Arrays:** indirect through called kernels
- **Issues:** none
- **Priority:** CRITICAL (wraps all work each timestep)

### Loop: mg3P V-cycle down-sweep at mg.c:344
- **Iterations:** levels `lt`→`lb+1` (log2 of problem size; ~5–10)
- **Type:** C2 — multigrid stage invoking coarse-grid restriction
- **Parent loop:** main mg.c:195
- **Contains:** calls `rprj3` (stencil on coarse grids)
- **Dependencies:** stage ordering across levels
- **Nested bounds:** constant level count
- **Private vars:** `k`, `j`
- **Arrays:** `r` (RW) via offsets
- **Issues:** stage dependency; coarse-grid sizes shrink quickly
- **Priority:** CRITICAL (part of each V-cycle)

### Loop: mg3P V-cycle mid-levels at mg.c:355
- **Iterations:** levels `lb+1..lt-1`
- **Type:** C2 — multigrid stage (prolongate/smooth)
- **Parent loop:** main mg.c:195
- **Contains:** `zero3` (Type A), `interp` (Type G), `resid` (Type G), `psinv` (Type G)
- **Dependencies:** level ordering
- **Nested bounds:** constant level count
- **Private vars:** `k`, `j`
- **Arrays:** `u`, `r`
- **Issues:** stage dependency; multiple grid-sized kernels per level
- **Priority:** CRITICAL

### Loop: psinv precompute at mg.c:389
- **Iterations:** `(n3-2)*(n2-2)*n1` (interior planes)
- **Type:** G — 7-point-like stencil producing neighbor sums into `r1/r2`
- **Parent loop:** none (function scope; invoked from mg3P levels)
- **Contains:** inner loops over `i2`, `i1`
- **Dependencies:** none; uses temporaries to break recurrences
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `orr(R)`, `r1(W)`, `r2(W)`
- **Issues:** scratch allocation each call; memory bandwidth bound
- **Priority:** CRITICAL

### Loop: psinv update at mg.c:401
- **Iterations:** `(n3-2)*(n2-2)*(n1-2)`
- **Type:** G — stencil smoothing writing `ou`
- **Parent loop:** none
- **Contains:** inner loops over `i2`, `i1`
- **Dependencies:** none; uses `r1/r2`
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `ou(RW)`, `orr(R)`, `r1(R)`, `r2(R)`
- **Issues:** reuse of `ou` in-place; halo required via `comm3`
- **Priority:** CRITICAL

### Loop: resid neighbor sums at mg.c:459
- **Iterations:** `(n3-2)*(n2-2)*n1`
- **Type:** G — neighbor aggregation into temporaries
- **Parent loop:** none
- **Contains:** inner loops over `i2`, `i1`
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `ou(R)`, `u1(W)`, `u2(W)`
- **Issues:** scratch alloc each call
- **Priority:** CRITICAL

### Loop: resid compute at mg.c:473
- **Iterations:** `(n3-2)*(n2-2)*(n1-2)`
- **Type:** G — stencil residual write to `orr`
- **Parent loop:** none
- **Contains:** inner loops over `i2`, `i1`
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `orr(W)`, `ov(R)`, `ou(R)`, `u1(R)`, `u2(R)`
- **Issues:** halo update needed after (`comm3`)
- **Priority:** CRITICAL

### Loop: rprj3 neighbor prep at mg.c:546
- **Iterations:** `(m3j-2)*(m2j-2)*(m1j-1)` on coarse-grid faces
- **Type:** G — stencil averaging into `x1/y1`
- **Parent loop:** mg3P level loop mg.c:344
- **Contains:** inner loops over `j2`, `j1`
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `j3`, `j2`, `j1`, `i3`, `i2`, `i1`
- **Arrays:** `orr(R)`, `x1(W)`, `y1(W)`
- **Issues:** scratch alloc; irregular offsets (`d1/d2/d3`)
- **Priority:** IMPORTANT

### Loop: rprj3 restrict at mg.c:561
- **Iterations:** `(m3j-2)*(m2j-2)*(m1j-2)`
- **Type:** G — stencil restriction writing coarse grid `os`
- **Parent loop:** mg3P level loop mg.c:344
- **Contains:** inner loops over `j2`, `j1`
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `j3`, `j2`, `j1`, `i3`, `i2`, `i1`
- **Arrays:** `orr(R)`, `x1(R)`, `y1(R)`, `os(W)`
- **Issues:** fractional weights; needs halo after (`comm3`)
- **Priority:** IMPORTANT

### Loop: interp z-prep at mg.c:632
- **Iterations:** `(mm3-1)*(mm2-1)*mm1`
- **Type:** G — computes partial sums (`z1/z2/z3`) for prolongation
- **Parent loop:** mg3P level loop mg.c:355
- **Contains:** inner loops over `i2`, `i1`
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `oz(R)`, `z1/2/3(W)`
- **Issues:** scratch alloc; dominates memory traffic
- **Priority:** CRITICAL

### Loop: interp inject corners/edges/faces (mg.c:647, mg.c:660, mg.c:672, mg.c:685)
- **Iterations:** multiple `(mm3-1)*(mm2-1)*(mm1-1)` style loops
- **Type:** G — stencil-based prolongation writing `ou`
- **Parent loop:** mg3P level loop mg.c:355
- **Contains:** inner loops over `i2`, `i1`
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `ou(RW)`, `oz(R)`, `z1/2/3(R)`
- **Issues:** overlapping writes to `ou` across loops; order-dependent but no cross-iteration dependency
- **Priority:** CRITICAL

### Loop: interp small-grid fallback at mg.c:722–821
- **Iterations:** multiple `(mm3-?)*(mm2-?)*(mm1-?)` depending on `d1/d2/d3`
- **Type:** G — prolongation for 3-point edges
- **Parent loop:** mg3P level loop mg.c:355
- **Contains:** several triple loops
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `ou(RW)`, `oz(R)`
- **Issues:** branchy bounds for tiny grids
- **Priority:** IMPORTANT (smaller grids)

### Loop: comm3 halos at mg.c:913 / 921 / 929
- **Iterations:** surfaces: `(n3-2)*(n2-2)`, `(n3-2)*n1`, `n2*n1`
- **Type:** A — boundary copy/periodic wrap
- **Parent loop:** called after psinv/resid/rprj3
- **Contains:** 2D nested loops within each block
- **Dependencies:** halo must follow stencil updates
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `ou(RW)`
- **Issues:** none; ensure ordering after stencil kernels
- **Priority:** IMPORTANT

### Loop: norm2u3 reduction at mg.c:881
- **Iterations:** `(n3-2)*(n2-2)*(n1-2)`
- **Type:** F — reduction on `s` (sum of squares) and `temp` (max)
- **Parent loop:** none (called pre/post benchmark)
- **Contains:** inner loops over `i2`, `i1`
- **Dependencies:** reduction on scalars
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`, `s`, `a`, `temp`
- **Arrays:** `orr(R)`
- **Issues:** needs reduction/atomic if parallelized; not in main iteration body
- **Priority:** SECONDARY

### Loop: zero3 fill at mg.c:1242
- **Iterations:** `n3*n2*n1`
- **Type:** A — dense memset of 3D field
- **Parent loop:** used inside mg3P level loop mg.c:355 and setup
- **Contains:** inner loops over `i2`, `i1`
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`, `i1`
- **Arrays:** `oz(W)`
- **Issues:** bandwidth bound
- **Priority:** IMPORTANT

### Loop: zran3 RNG fill at mg.c:1026
- **Iterations:** `(e3-1)*(e2-1)` outer with inner `vranlc(d1)` (d1 elements)
- **Type:** A — dense initialization with RNG; inner RNG loop is recurrence (Type E) inside `vranlc`
- **Parent loop:** initialization block (not timed iteration)
- **Contains:** call to `vranlc` (recurrence)
- **Dependencies:** RNG recurrence inside `vranlc`
- **Nested bounds:** constant
- **Private vars:** `i3`, `i2`
- **Arrays:** `oz(W)`
- **Issues:** RNG recurrence; setup only
- **Priority:** SECONDARY

### Loop: power exponentiation at mg.c:1173
- **Iterations:** while over bits of exponent `n` (log2(nx*ny*...))
- **Type:** E — loop-carried state via `randlc`
- **Parent loop:** none
- **Contains:** none
- **Dependencies:** RNG recurrence
- **Nested bounds:** variable
- **Private vars:** `nj`, `aj`
- **Arrays:** none
- **Issues:** sequential; setup-only
- **Priority:** SECONDARY

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| main (it loop) | A | CRITICAL | none | `nit` | none | driver only |
| mg3P level down (mg.c:344) | C2 | CRITICAL | main it | `lt-lb` | stage order | multigrid traversal |
| mg3P mid levels (mg.c:355) | C2 | CRITICAL | main it | `lt-lb-1` | stage order | multiple kernels/level |
| psinv stencil (mg.c:389/401) | G | CRITICAL | mg3P level | `(n3-2)*(n2-2)*n1` etc | none | halo after |
| resid stencil (mg.c:459/473) | G | CRITICAL | mg3P level & main resid | `(n3-2)*(n2-2)*n1` etc | none | halo after |
| rprj3 stencil (mg.c:546/561) | G | IMPORTANT | mg3P down | coarse-grid volume | none | scratch alloc |
| interp stencil (mg.c:632–821) | G | CRITICAL/IMPORTANT (fallback) | mg3P up | fine-grid volume | none | overlapping writes |
| comm3 halos (mg.c:913/921/929) | A | IMPORTANT | stencil kernels | surface sizes | order after compute | boundary copy |
| norm2u3 reduction (mg.c:881) | F | SECONDARY | none | interior volume | reduction on `s/temp` | not per iter |
| zero3 fill (mg.c:1242) | A | IMPORTANT | mg3P/setup | full grid | none | bandwidth bound |
| zran3 RNG (mg.c:1026) | A/E | SECONDARY | init | `(e3-1)*(e2-1)*d1` | RNG recurrence | setup only |
| power (mg.c:1173) | E | SECONDARY | init | O(log n) | RNG recurrence | sequential |

## Data Details
- **Dominant compute loop:** `for (it=1; it<=nit; ++it)` in `main`, executing `mg3P` V-cycle (restriction → smoothing → prolongation) plus `resid` per timestep.
- **Arrays swapped between functions?:** NO; `u`, `v`, `r` are static 1D grids indexed via `I3D`; temporaries (`r1/r2`, `u1/u2`, `x1/y1`, `z1/z2/z3`) are allocated per kernel call, not swapped.
- **Scratch arrays?:** YES; each stencil kernel allocates local work arrays (`r1/r2`, `u1/u2`, `x1/y1`, `z1/z2/z3`) sized to the active grid.
- **Mid-computation sync?:** NO explicit barriers; `comm3` performs halo copies after stencil updates to refresh ghost cells.
- **RNG in timed loop?:** NO; RNG (`zran3`, `power`, `randlc/vranlc`) occurs during setup (T_init) only.
