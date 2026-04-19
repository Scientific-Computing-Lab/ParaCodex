# Loop Classification Guide

## Decision Tree

```
Q0: Nested inside another loop? → Note parent
Q-1 (check FIRST): Outer loop over independent items + inner loop whose body is
    guarded by `if (inner_var == 0)` to restrict writes to one iteration?
    → Type H (CUDA-serial port): vestigial intra-block parallelism serialized for CPU.
      Flag RESTRUCTURE NEEDED.
Q1: Writes A[idx[i]] with varying idx? → Type D (Histogram)
Q2: Reads A[i-1] or accumulates across iterations? → Type E (Recurrence - CPU only)
Q3: Stage loop where L+1 depends on L?
    - Scratch swap (tmp1↔tmp2)? → C1 (FFT/Butterfly)
    - Level traversal with stencil calls? → C2 (Multigrid)
Q4: Inner bound varies with outer index? → Type B (Sparse)
Q5: Accumulates to scalar? → Type F (Reduction)
Q6: Accesses neighbors? → Type G (Stencil)
Default → Type A (Dense)
```

## Structural Check

Before recommending offload, answer these:

- Is the timed region a single GPU-shaped loop nest already?
- Is the work split across many helper functions with tiny outer loops?
- Would preserving the current function boundaries introduce multiple tiny kernels or `target update` traffic?
- If yes, recommend a fused offload unit in `analysis.md` instead of helper-by-helper offload.

## Special Case - Outer A + Inner E
When outer loop iterates over INDEPENDENT samples and inner has RNG:
- Mark outer as Type A (CRITICAL) - parallelizable with per-thread RNG
- Mark inner RNG as Type E - sequential WITHIN each thread
- Note: "RNG replicable: YES - each sample can compute its own seed"

## Type Reference

| Type | Pattern | Parallelizable |
|------|---------|----------------|
| A | Dense, constant bounds | YES |
| B | Sparse (CSR), inner bound varies | Outer only |
| C1 | FFT/Butterfly, scratch swap | Outer only |
| C2 | Multigrid, hierarchical calls | Outer only |
| D | Histogram, indirect write | YES + atomic |
| E | Recurrence, loop-carried dep | NO |
| F | Reduction to scalar | YES + reduction |
| G | Stencil, neighbor access | YES |
| H | CUDA-serial port: outer work-item loop + inner fake-thread loop with `if(var==0)` guard | RESTRUCTURE: eliminate inner loop |

---

## Code Examples

### Finding loops with grep
```bash
# Find all for loops
grep -n "for\s*(" *.c *.cpp 2>/dev/null | head -100

# Find main compute loop (iterations/time-step)
grep -n "for.*iter\|for.*step\|for.*time\|while\s*(" *.c *.cpp 2>/dev/null | head -30

# Find reductions (loop-carried accumulation)
grep -n "+=\|sum\|acc\|total" *.c *.cpp 2>/dev/null | head -30

# Find indirect writes (histogram pattern)
grep -n "\[.*\[" *.c *.cpp 2>/dev/null | head -30
```

### Timing a region for hotspot identification
```c
#include <time.h>
struct timespec t0, t1;
clock_gettime(CLOCK_MONOTONIC, &t0);
// ... region under test ...
clock_gettime(CLOCK_MONOTONIC, &t1);
printf("%.4f s\n", (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9);
```

### Type A — Dense (parallelizable directly)
```c
// Independent iterations: safe to offload
for (int i = 0; i < N; i++)
    C[i] = A[i] * B[i];   // No dependency across iterations
```

### Type B — Sparse/CSR (outer only)
```c
for (int row = 0; row < nrows; row++) {
    double sum = 0.0;
    for (int k = rowptr[row]; k < rowptr[row+1]; k++)  // Inner bound varies!
        sum += val[k] * x[col[k]];
    y[row] = sum;
}
```

### Type E — Recurrence (cannot parallelize)
```c
for (int i = 1; i < N; i++)
    A[i] = A[i-1] * alpha + B[i];  // Loop-carried: each iteration needs previous
```

### Type F — Reduction
```c
double sum = 0.0;
for (int i = 0; i < N; i++)
    sum += A[i] * B[i];   // Reduction to scalar — parallelizable with reduction clause
```

### Small staged hot path — fuse rather than preserve helpers
```c
// Serial structure
forward(...);
backward(...);
update(...);

// Analysis recommendation:
// If each stage has low outer parallelism or would become a tiny device region,
// fuse these stages into one GPU-oriented routine before adding pragmas.
```
