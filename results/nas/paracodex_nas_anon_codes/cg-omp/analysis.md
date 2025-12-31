# CG Loop Analysis

## Loop Nesting Structure
- `for (v76 = 1; v76 <= NITER; v76++)` (golden_labels/src/cg-serial/cg.c:226) Type A
  ├── `f1(...)` (golden_labels/src/cg-serial/cg.c:530) → contains the CG inner phase
  │   ├── `for (v74 = 0; v74 < v13; v74++)` (golden_labels/src/cg-serial/cg.c:549) Type A
  │   ├── `for (v74 = 0; v74 < v18 - v17 + 1; v74++)` (golden_labels/src/cg-serial/cg.c:556) Type F
  │   ├── `for (v120 = 1; v120 <= v121; v120++)` (golden_labels/src/cg-serial/cg.c:560) Type A
  │   │   ├── `for (v74 = 0; v74 < v77; v74++)` (golden_labels/src/cg-serial/cg.c:565) Type B
  │   │   │   └── `for (v75 = v117; v75 < v118; v75++)` (golden_labels/src/cg-serial/cg.c:569) Type B
  │   │   ├── `for (v74 = 0; v74 < v77; v74++)` (golden_labels/src/cg-serial/cg.c:579) Type F
  │   │   ├── `for (v74 = 0; v74 < v77; v74++)` (golden_labels/src/cg-serial/cg.c:589) Type A
  │   │   ├── `for (v74 = 0; v74 < v77; v74++)` (golden_labels/src/cg-serial/cg.c:595) Type F
  │   │   └── `for (v74 = 0; v74 < v77; v74++)` (golden_labels/src/cg-serial/cg.c:603) Type A
  │   ├── `for (v74 = 0; v74 < v77; v74++)` (golden_labels/src/cg-serial/cg.c:608) Type B
  │   │   └── `for (v75 = v117; v75 < v118; v75++)` (golden_labels/src/cg-serial/cg.c:613) Type B
  │   └── `for (v74 = 0; v74 < v18 - v17 + 1; v74++)` (golden_labels/src/cg-serial/cg.c:621) Type F
  ├── `for (v74 = 0; v74 < v77; v74++)` (golden_labels/src/cg-serial/cg.c:231) Type F
  └── `for (v74 = 0; v74 < v77; v74++)` (golden_labels/src/cg-serial/cg.c:243) Type A

## Loop Details

### Loop: main dot product at golden_labels/src/cg-serial/cg.c:231
- **Iterations:** `v77 = v16 - v15 + 1` (≈ NA rows)
- **Type:** Type F (dense reduction) – computes `v80 = v8·v9` and `v81 = v9·v9` in one pass
- **Priority:** CRITICAL
- **Parent loop:** `for (v76 = 1; v76 <= NITER; v76++)` (line 226)
- **Contains:** none
- **Dependencies:** reduction on `v80` and `v81` before the `sqrt` and `SHIFT` update
- **Nested bounds:** variable (depends on `v16 - v15 + 1` which equals NA)
- **Private vars:** `v74`
- **Arrays:** `v8` (R), `v9` (R), scalars `v80`, `v81` (W)
- **Issues:** needs a reduction clause/explicit accumulation when parallelized; avoids cross-thread interference by accumulating into local temporaries before the final `sqrt`

### Loop: main scaling at golden_labels/src/cg-serial/cg.c:243
- **Iterations:** `v77 = v16 - v15 + 1` (≈ NA)
- **Type:** Type A (dense vector scale)
- **Priority:** IMPORTANT
- **Parent loop:** `for (v76 = 1; v76 <= NITER; v76++)` (line 226)
- **Contains:** none
- **Dependencies:** none (each element updated independently)
- **Nested bounds:** variable (same as above)
- **Private vars:** `v74`
- **Arrays:** `v8` (RW), `v9` (R)
- **Issues:** no cross-iteration dependencies, but the vector is reused as input for the next CG iteration so the update must complete before the next `f1` call

### Loop: f1 vector reset at golden_labels/src/cg-serial/cg.c:549
- **Iterations:** `v13 = NA`
- **Type:** Type A (dense initialization)
- **Priority:** CRITICAL
- **Parent loop:** `f1` (line 530)
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant (`v13` is initialized from NA once per run)
- **Private vars:** `v74`
- **Arrays:** `v65`, `v68` (W), `v69` (W ← `v64`), `v64` (R), `v67` (W ← `v69`)
- **Issues:** resets the CG working vectors; should run before any CG-specific reductions

### Loop: f1 initial norm at golden_labels/src/cg-serial/cg.c:556
- **Iterations:** `v18 - v17 + 1` (≈ NA)
- **Type:** Type F (vector norm reduction)
- **Priority:** CRITICAL
- **Parent loop:** `f1` (line 530)
- **Contains:** none
- **Dependencies:** reduction on `v124` (norm of `v69`)
- **Nested bounds:** variable (depends on static bounds, but resolves to NA)
- **Private vars:** `v74`
- **Arrays:** `v69` (R)
- **Issues:** reduction must be handled carefully when parallelizing inside `f1`

### Loop: f1 CG outer iteration at golden_labels/src/cg-serial/cg.c:560
- **Iterations:** `v121 = 25`
- **Type:** Type A (fixed-count CG iteration)
- **Priority:** CRITICAL
- **Parent loop:** `f1` (line 530)
- **Contains:** mat-vec, inner dot, updates, norms, and `v67` refresh loops
- **Dependencies:** sequential stages inside the loop (each uses results from the preceding loops, especially the dot products before the updates)
- **Nested bounds:** constant (control loop over 25 CG steps)
- **Private vars:** `v120`, `v77`
- **Arrays:** orchestrates `v62`, `v63`, `v65`, `v66`, `v67`, `v68`, `v69`
- **Issues:** loop-carried dataflow (updates to `v124`, `v122`, `v65`, `v69`, `v67`) prevents reordering of the contained loops

### Loop: f1 mat-vec outer at golden_labels/src/cg-serial/cg.c:565
- **Iterations:** `v77 = v16 - v15 + 1` (≈ NA)
- **Type:** Type B (sparse row iteration)
- **Priority:** CRITICAL
- **Parent loop:** `for (v120 = 1; v120 <= v121; v120++)` (line 560)
- **Contains:** the `v75` loop over nonzeros inside each row
- **Dependencies:** each iteration writes to `v68[v74]`; rows are independent so no atomics are needed
- **Nested bounds:** variable (depends on `v63[v74]`/`v63[v74+1]` from the CSR offsets)
- **Private vars:** `v74`, `v117`, `v118`, `v123`
- **Arrays:** `v63` (R), `v62` (R), `v66` (R), `v67` (R), `v68` (W)
- **Issues:** irregular, indirect access through `v62`; the row pointers (`v63`) must be treated as read-only while partitioning rows

### Loop: f1 mat-vec inner at golden_labels/src/cg-serial/cg.c:569
- **Iterations:** `v118 - v117` (≈ NONZER per row)
- **Type:** Type B (sparse nonzero gather)
- **Priority:** CRITICAL
- **Parent loop:** `for (v74 = 0; v74 < v77; v74++)` (line 565)
- **Contains:** none
- **Dependencies:** reduction on `v123` when accumulating contributions to the row result
- **Nested bounds:** variable (depends on each row's nonzero count stored via `v63`)
- **Private vars:** `v75`, `v119`
- **Arrays:** `v62` (R), `v66` (R), `v67` (R)
- **Issues:** gathers from `v67` using indirect indices (`v119`); ensure the gather does not create cross-thread contention when partitioning rows

### Loop: f1 dot product inside CG (golden_labels/src/cg-serial/cg.c:579)
- **Iterations:** `v77 = v18 - v17 + 1` (≈ NA)
- **Type:** Type F (reduction for direction dot product)
- **Priority:** CRITICAL
- **Parent loop:** `for (v120 = 1; v120 <= v121; v120++)` (line 560)
- **Contains:** none
- **Dependencies:** reduction on `v122` (`v67·v68`)
- **Nested bounds:** variable
- **Private vars:** `v74`
- **Arrays:** `v67` (R), `v68` (R)
- **Issues:** reduction again needs explicit handling in a parallel implementation

### Loop: f1 update `v65`/`v69` (golden_labels/src/cg-serial/cg.c:589)
- **Iterations:** `v77 = v18 - v17 + 1`
- **Type:** Type A (dense vector updates)
- **Priority:** CRITICAL
- **Parent loop:** `for (v120 = 1; v120 <= v121; v120++)` (line 560)
- **Contains:** none
- **Dependencies:** sequential updates that reuse `v67` and `v68`
- **Nested bounds:** variable
- **Private vars:** `v74`
- **Arrays:** `v65` (RW), `v67` (RW), `v69` (RW)
- **Issues:** `v65` and `v69` are read-modify-write, so each element must be finished before the next stage of the CG inner loop

### Loop: f1 norm recompute (golden_labels/src/cg-serial/cg.c:595)
- **Iterations:** `v77 = v18 - v17 + 1`
- **Type:** Type F (reduction to recompute `v124`)
- **Priority:** CRITICAL
- **Parent loop:** `for (v120 = 1; v120 <= v121; v120++)` (line 560)
- **Contains:** none
- **Dependencies:** reduction on `v124`
- **Nested bounds:** variable
- **Private vars:** `v74`
- **Arrays:** `v69` (R)
- **Issues:** repeated reduction; the intermediate value `v124` drives the next scalar update so it cannot be deferred

### Loop: f1 update search direction `v67` (golden_labels/src/cg-serial/cg.c:603)
- **Iterations:** `v77 = v18 - v17 + 1`
- **Type:** Type A (dense vector update)
- **Priority:** CRITICAL
- **Parent loop:** `for (v120 = 1; v120 <= v121; v120++)` (line 560)
- **Contains:** none
- **Dependencies:** none beyond reading `v69`
- **Nested bounds:** variable
- **Private vars:** `v74`
- **Arrays:** `v67` (RW), `v69` (R)
- **Issues:** next CG iteration uses updated `v67`, so the update must complete before the next outer `v120` iteration

### Loop: f1 final mat-vec outer at golden_labels/src/cg-serial/cg.c:608
- **Iterations:** `v77 = v16 - v15 + 1`
- **Type:** Type B (sparse row iteration)
- **Priority:** CRITICAL
- **Parent loop:** `f1` (after the CG inner loop completes)
- **Contains:** row's nonzeros loop (line 613)
- **Dependencies:** writes to `v69[v74]`; rows are independent
- **Nested bounds:** variable (CSR bounds from `v63`)
- **Private vars:** `v74`, `v117`, `v118`, `v122`
- **Arrays:** `v63` (R), `v62` (R), `v66` (R), `v65` (R), `v69` (W)
- **Issues:** final residual computation uses this mat-vec result, so it must finish before the residual reduction

### Loop: f1 final mat-vec inner at golden_labels/src/cg-serial/cg.c:613
- **Iterations:** `v118 - v117`
- **Type:** Type B (sparse nonzero gather)
- **Priority:** CRITICAL
- **Parent loop:** `for (v74 = 0; v74 < v77; v74++)` (line 608)
- **Contains:** none
- **Dependencies:** reduction on `v122` when building the row result
- **Nested bounds:** variable
- **Private vars:** `v75`, `v119`
- **Arrays:** `v62` (R), `v66` (R), `v65` (R)
- **Issues:** gathers same as the primary mat-vec, but now the result is used to compute the residual norm

### Loop: f1 residual norm at golden_labels/src/cg-serial/cg.c:621
- **Iterations:** `v18 - v17 + 1`
- **Type:** Type F (reduction computing `||r||`)
- **Priority:** CRITICAL
- **Parent loop:** `f1` (after final mat-vec)
- **Contains:** none
- **Dependencies:** reduction on scalar `v123`
- **Nested bounds:** variable
- **Private vars:** `v74`
- **Arrays:** `v64` (R), `v69` (R)
- **Issues:** reduction against the reference vector to compute the verification residual; requires a parallel reduction on `v123`

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| `main` dot product (231) | F | CRITICAL | `for (v76...)` (226) | `v77 ≈ NA` | reduction: `v80`, `v81` | needs reduction clause |
| `main` vector scale (243) | A | IMPORTANT | `for (v76...)` (226) | `v77 ≈ NA` | none | vector reused next iteration |
| `f1` reset (549) | A | CRITICAL | `f1` (530) | `v13 = NA` | none | must precede CG inner phase |
| `f1` initial norm (556) | F | CRITICAL | `f1` (530) | `v18 - v17 + 1` | reduction: `v124` | requires reduction guard |
| `f1` CG loop (560) | A | CRITICAL | `f1` (530) | 25 | stage dependency on previous updates | sequential stage ordering |
| `f1` mat-vec outer (565) | B | CRITICAL | `f1` CG (560) | `v77 ≈ NA` | none (row independent) | irregular access via `v63`/`v62`|
| `f1` mat-vec inner (569) | B | CRITICAL | mat-vec outer (565) | `v118 - v117 ≈ NONZER` | reduction: `v123` | gathers from `v67` using `v62` |
| `f1` dot (579) | F | CRITICAL | `f1` CG (560) | `v77 ≈ NA` | reduction: `v122` | parallel reduction required |
| `f1` updates `v65`, `v69` (589) | A | CRITICAL | `f1` CG (560) | `v77 ≈ NA` | none | read-modify-write on `v65`, `v69` |
| `f1` norm recompute (595) | F | CRITICAL | `f1` CG (560) | `v77 ≈ NA` | reduction: `v124` | reduction needed |
| `f1` direction update (603) | A | CRITICAL | `f1` CG (560) | `v77 ≈ NA` | none | feeds next CG iteration |
| `f1` final mat-vec outer (608) | B | CRITICAL | `f1` (after CG) | `v77 ≈ NA` | none | same CSR irregular access |
| `f1` final mat-vec inner (613) | B | CRITICAL | final mat-vec outer (608) | `v118 - v117` | reduction: `v122` | gather from `v65` |
| `f1` residual norm (621) | F | CRITICAL | `f1` (after final mat-vec) | `v18 - v17 + 1` | reduction: `v123` | reduction to scalar |

## Data Details
- **Dominant compute loop:** `for (v76 = 1; v76 <= NITER; v76++)` (golden_labels/src/cg-serial/cg.c:226) – each iteration calls `f1` to run 25 internal CG steps, two vector dot/scaling operations, and feeds verification and benchmark timers.
- **Arrays swapped between functions?:** NO – `f1` mutates the passed-in vectors (`v8`–`v12`, `v65`–`v69`) in place; no double-buffered handoff occurs between `main` and `f1`.
- **Scratch arrays?:** YES – `v65`, `v66`, `v67`, `v68`, `v69`, and the timer buffers (`v10`–`v12`) act as temporary working storage for inner CG updates.
- **Mid-computation sync?:** NO explicit synchronization primitives inside the timed loop; dataflow is enforced by sequential loops and scalar reductions.
- **RNG in timed loop?:** NO – all RNG calls (`randlc`) happen inside `f4`/`f6` during setup before `timer_start(T_bench)`.
- **Array definitions:** all working arrays (`v1[NZ]`, `v2[NA+1]`, `v3[NA]`, `v4[NA]`, `v5[NAZ]`, `v7[NZ]`, `v8`–`v12` sized `NA+2`, and `v64`–`v69` sized `NA+2`) are static, flat globals; no pointer-to-pointer or dynamic heap allocation is used in the timed region.
- **Global scalars:** `NA`, `NONZER`, `NITER`, `SHIFT`, `RCOND`, `nz`, `naz`, `na`, and the stateful counters `v13`–`v21`, `v71`, `v72` drive loop bounds and reductions and are read-only inside the timed loops.
- **Issue summary:** the compute kernels rely on repeated vector reductions (`v80`, `v81`, `v122`, `v123`, `v124`) and irregular CSR access (`v63`, `v62`, `v66`); any parallel implementation must expose reductions with a reduction clause and handle `v62` gather patterns carefully to avoid load imbalance.
