# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| u[NR] | ~19.7M doubles (~158 MB) | working solution | host then device | R/W |
| v[NR] | ~19.7M doubles (~158 MB) | const RHS | host then device | RO |
| r[NR] | ~19.7M doubles (~158 MB) | working residual | host then device | R/W |
| r1/r2 (psinv scratch) | n1\*n2\*n3 doubles per level | scratch | device | R/W |
| u1/u2 (resid scratch) | n1\*n2\*n3 doubles per level | scratch | device | R/W |
| x1/y1 (rprj3 scratch) | m1k\*m2k\*m3k doubles per level | scratch | device | R/W |
| z1/z2/z3 (interp scratch) | mm1\*mm2\*mm3 doubles per level | scratch | device | R/W |

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| resid | u, v, r, scratch u1/u2 | per-iteration (multiple levels) | device |
| mg3P (driver) | u, v, r | per-iteration | device |
| rprj3 | r (fine), r (coarse), scratch x1/y1 | per-level inside mg3P | device |
| psinv | r, u, scratch r1/r2 | per-level inside mg3P | device |
| interp | u (coarse), u (fine), scratch z1/z2/z3 | per-level inside mg3P | device |
| zero3 | u slices | per-level inside mg3P | device |
| comm3 | boundary of u/r | per-call inside mg3P/resid | device |
| norm2u3 | r | once before loop + once after loop | device |

## Data Movement Strategy

**Chosen Strategy:** A

**Device Allocations (once):**
```
target data region holding u[0:NR], v[0:NR], r[0:NR]
Scratch arrays per kernel via map(alloc:) inside each routine
```

**Host→Device Transfers:**
- When: enter benchmark region before first resid in timed loop
- Arrays: u (tofrom, ~158 MB), v (to, ~158 MB), r (tofrom, ~158 MB)
- Total H→D: ~474 MB

**Device→Host Transfers:**
- When: exit benchmark target data after iterations
- Arrays: u (final solution, ~158 MB), r (residual, ~158 MB)
- Total D→H: ~316 MB

**Transfers During Iterations:** NO
- All working and scratch data stay on device for duration of timed loop.

## Critical Checks (for chosen strategy)

**Strategy A:**
- [ ] Functions inside target data use `present` wrappers
- [ ] Scratch arrays use enter/exit data OR device alloc

## Common Mistakes:
-  Some functions on device, others on host (causes copying)
-  Scratch as host arrays in Strategy C
-  Forgetting to offload ALL functions in loop

## Expected Transfer Volume
- Total: ~790 MB for entire execution
- **Red flag:** If actual >2x expected → data management wrong

**Summary:** 3 arrays (2 working, 1 const), 4 scratch groups, 8 functions, Strategy A. Expected: ~474 MB H→D, ~316 MB D→H.
