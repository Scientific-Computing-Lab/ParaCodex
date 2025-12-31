# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `a` | `NZ * sizeof(double)` | working | host (makea builds) | RO for SpMV, updates during conj_grad, reused | 
| `colidx` | `NZ * sizeof(int)` | index | host (sparse builds) | RO for SpMV loops
| `rowstr` | `(NA+1) * sizeof(int)` | index | host | RO bounds for SpMV
| `x` | `(NA+2) * sizeof(double)` | working | host (vector initialization) | RW (norm, rescale, final check)
| `z` | `(NA+2) * sizeof(double)` | working | zero-initialized on host | RW (conj_grad iterations + norm)
| `p` | `(NA+2) * sizeof(double)` | working | zero-initialized on host | RW within conj_grad
| `q` | `(NA+2) * sizeof(double)` | scratch/working | zero-initialized on host | RW (SpMV output)
| `r` | `(NA+2) * sizeof(double)` | working | zero-initialized on host | RW (conj_grad, final norm)

`sum_array` is declared but unused in timed region, so no dedicated device storage is needed.

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| `main` norm+rescale loops (before timer stop/start and after conj_grad calls) | `x`, `z`, `r` | every iteration of the initial warm-up + NITER (once per norm stage) | device (parallel reductions + scalings) |
| `conj_grad` | `x`, `z`, `p`, `q`, `r`, `rowstr`, `colidx`, `a` | once per `cgitmax` loop inside every benchmark iteration | device (SpMV, reductions, vector SAXPYs) |

## Data Movement Strategy
**Chosen Strategy:** A (target data region)

### Device Allocations (once)
- `#pragma omp target enter data map(to: a[0:NZ], colidx[0:NZ], rowstr[0:NA+1])`
- `#pragma omp target enter data map(alloc: x[0:NA+2], z[0:NA+2], p[0:NA+2], q[0:NA+2], r[0:NA+2])`
- `target data` region keeps those mappings live across warm-up, timed loops, and reductions inside `conj_grad`; the working vectors are initialized in-device, so their host copies never move.

### Host→Device Transfers
- When: immediately after `makea`/`sparse` finish and before GPU work begins
- Arrays: `a` → device (double, `NZ * 8` bytes), `colidx` → device (`NZ * 4` bytes), `rowstr` → device (`(NA+1) * 4` bytes)
- Working vectors (`x,z,p,q,r`) are allocated on-device via `map(alloc: ...)` and populated through device loops, so no host-to-device transfer is required for them. Total H→D ≈ `(8*NZ + 4*NZ + 4*(NA+1))` bytes (~ `N/A` MB depending on class).

### Device→Host Transfers
- When: not required; scalars (`norm_temp*`, `zeta`, `rnorm`) come from device reduction results, not whole arrays
- Arrays: none
- Total D→H: ~0 MB

### Transfers During Iterations: NO
- All vectors remain resident on device via the `target data` region, and reductions feed host scalars directly through OpenMP reduction clauses.

## Critical Checks (for chosen strategy)
**Strategy A:**
- [ ] `target data` region uses `present` for the arrays accessed by device loops (no implicit host copies during computation).
- [ ] Scratch buffers (`q`, temporary reductions) are either allocated via the `target data` map or purely per-thread (`sum` reductions), not reallocated each loop.
- [ ] Every `#pragma omp target teams` inside hot loops lists the `present` clause for the arrays it mutates.

## Expected Transfer Volume
- Total: ~`NZ*(8+4) + (NA+2)*8*5 + (NA+1)*4` bytes for the once-at-start transfer; for Class S that is on the order of a few megabytes (e.g., `~(1400*(7+1)*(7+1)*(12 bytes) + ~1400*8*5) < 200 MB`).
- **Red flag:** An actual transfer volume much higher (>2x) would indicate missing `target data` mapping or unexpected updates.

## Additional Parallelization Notes
- **RNG Replicable?** NO (RNG only used before timed region, `sprnvc` not offloaded).
- **Outer Saturation?** The outer `cgit` iterations are sequential (Type E) and remain host-controlled; inner loops (SpMV, dot, updates) run in parallel on the device.
- **SpMV NONZER?** `NONZER` is small (7 for Class S), so the irregular gathers rely on independent row loops and no intra-row parallelism (Type B).
- **Histogram Strategy?** Not applicable.

**Summary:** 8 arrays (5 working, 3 index), 1 timed function (`conj_grad`) plus two reduction/rescale loops; Strategy A retains everything inside a persistent `target data` region, limiting data movement to the initial upload (~several MB) and relying on in-device reductions for scalar outputs.
