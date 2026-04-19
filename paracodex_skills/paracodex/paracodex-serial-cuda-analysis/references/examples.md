# Analysis Decision Tree

## Q1: Where is the Hotspot?
- Use `gprof`, `perf`, or simple timers.
- Look for loops consuming >50% of runtime.

## Q2: Is it Parallelizable?
- **YES:** Independent iterations (e.g., `C[i] = A[i] + B[i]`).
- **MAYBE:** Reduction (`sum += A[i]`). Requires specific parallel reduction algo.
- **NO:** Loop carried dependency (`A[i] = A[i-1] + 1`).

## Q3: Data Transfer Overhead?
- Estimate transfer time: `Size / Bandwidth`.
- Compare with Compute time benefit.
- If `Transfer Time > Compute Time`, DO NOT OFFLOAD (or fuse kernels).

## Q4: Recursion?
- CUDA supports recursion (Dynamic Parallelism) but effectively limited.
- **Advice:** Convert to iterative if possible.

---

## Code Examples

### Timing a hotspot with POSIX clock
```c
#include <time.h>
struct timespec t0, t1;
clock_gettime(CLOCK_MONOTONIC, &t0);
hotspot_function(...);
clock_gettime(CLOCK_MONOTONIC, &t1);
double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
printf("Hotspot time: %.4f s\n", elapsed);
```

### Parallelizability Checklist
```
[ ] Independent iterations?       → YES: Type A (Dense) — safe to offload
[ ] Reduction only (sum/min/max)?  → YES: Type F — use __reduce__ / atomicAdd
[ ] Loop-carried (A[i] = f(A[i-1])) → NO: recurrence, cannot parallelize
[ ] Indirect write (A[idx[i]])    → YES: Type D (Histogram) — needs atomics
[ ] Inner bound varies?           → Sparse (Type B) — outer loop only
```

### Transfer overhead estimate
```
GPU PCIe bandwidth: ~16 GB/s
For N floats: transfer_time ≈ (N * 4) / 16e9  seconds
GPU compute: ~10 TFLOPS → compute_time ≈ (N * ops_per_elem) / 10e12 seconds
Offload is worthwhile if: compute_time > 2 × transfer_time
```

## Q3b: Is the serial structure already GPU-shaped?
- If one large loop dominates, preserve it and map it to one kernel.
- If the hot path is split across multiple tiny helper calls, recommend a fused kernel plan.
- Do not assume each helper function should become its own kernel.
