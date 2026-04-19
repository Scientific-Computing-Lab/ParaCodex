# Strategy Details

## Structural Decision Rule

Choose the offload unit before choosing pragmas:

- If the timed region is a single large loop nest, preserve that loop nest and offload it directly.
- If the timed region is split across multiple helper functions and one or more stages have tiny outer parallelism, fuse the full timed region into one GPU-oriented routine.
- Prefer a fused routine when the helper-based design would create multiple kernels, `target update` operations, or host-visible scalar staging between steps.
- Do not keep the serial helper structure unless it is already GPU-shaped.

## STRATEGY A: target data Region

**Map Clause Selection:**
| Scenario | Map Clause | Why |
|----------|------------|-----|
| Device-init arrays (zero(), fill()) | `alloc` | Avoid copying garbage |
| Host RNG init then sync | `alloc` + `update to` | Explicit sync after host init |
| Read + modify + write | `tofrom` | Bidirectional |
| Read-only | `to` | One-way |

**Functions Called Inside target data:**
Wrap with `present,alloc`/'to,tofrom', then choose the pragma form that gives the best GPU mapping:
```c
void compute(double *u, double *v, int n) {
  #pragma omp target data map(present,alloc:u[0:n],v[0:n])
  {
    #pragma omp target teams loop
    for (int i = 0; i < n; i++) { ... }
  }
}
```

Prefer `target teams loop` first for OpenMP 5+ offload, but keep or switch to `target teams distribute parallel for` when profiling or compiler behavior shows it maps the kernel better on the active toolchain.

**RNG replicable:**
```c
#pragma omp target teams loop reduction(+:sum1, sum2) firstprivate(seed_base, params)
for (int sample = 0; sample < N; ++sample) {
  double rng_state = compute_seed_for_sample(sample);  // Per-thread seed
  double local_hist[BINS] = {0};  // Per-thread histogram
  
  // Type E (RNG) is sequential WITHIN this thread
  for (int i = 0; i < work_per_sample; ++i) {
    double r = my_rng(&rng_state, A);
    int bin = compute_bin(r);
    local_hist[bin] += 1.0;
    sum1 += ...; sum2 += ...;  // Reduction handles these
  }
  
  // Atomic merge histogram at end
  for (int b = 0; b < BINS; ++b) {
    if (local_hist[b] != 0.0) {
      #pragma omp atomic update
      global_hist[b] += local_hist[b];
    }
  }
}
```

**Scratch Arrays (two options):**

- **Option 1: enter/exit data**
```c
double scratch[N];
#pragma omp target enter data map(alloc:scratch[0:n])
#pragma omp target data map(present,alloc:in[0:n])
{
  #pragma omp target teams loop
  for (...) { /* use scratch */ }
}
#pragma omp target exit data map(delete:scratch[0:n])
```

- **Option 2: omp_target_alloc**
```c
double *scratch = (double*)omp_target_alloc(n*sizeof(double), 0);
#pragma omp target data map(present,alloc:in[0:n])
{
  #pragma omp target teams loop is_device_ptr(scratch)
  for (...) { ... }
}
omp_target_free(scratch, 0);
```

**Mid-computation sync:**
```c
#pragma omp target update from(result)
host_compute(result);
#pragma omp target update to(indices)
```

Avoid this pattern in the timed region unless the algorithm truly requires host participation between stages.

## STRATEGY B: Asynchronous Offload
Use when: Overlapping compute/transfer possible
```c
#pragma omp target teams loop nowait depend(out:x[0])
for (i = 0; i < N; i++) { x[i] = init(i); }

#pragma omp target teams loop nowait depend(in:x[0]) depend(out:y[0])
for (i = 0; i < N; i++) { y[i] = compute(x[i]); }

#pragma omp taskwait
```

## STRATEGY C: Global Device State (Iterative Solvers)
Use omp_target_alloc + is_device_ptr for all device arrays.

**Pattern:**
```c
// Device pointers: static double *d_arr
allocate_device_arrays();  // omp_target_alloc once
copy_to_device();          // omp_target_memcpy once

for (iter ...) {
  #pragma omp target teams is_device_ptr(d_arr1, d_arr2, ...)
  {
  #pragma omp loop            // Outer parallelism
  for (k ...) {
    #pragma omp loop          // Middle parallelism (if needed)
    for (j ...) {
      for (stage ...) { ... }  // NO pragma - stages must be serial!
    }
  }
  }
}

free_device_arrays();
```

**Key Rules:**
- Use `is_device_ptr` everywhere (no map clauses in hot path)
- Reduction helpers (dot, norm) OK - they return scalars
- stage loops: parallelize outer k,j; keep stage loop L serial
- Iterative solvers: inline SpMV, updates in main loop

---

## Map Globals & Functions
```c
#pragma omp declare target
double helper_func() { ... };
#pragma omp end declare target

#pragma omp declare target(global_var)
```

---

## Parallelization patterns

**Type A (Dense):**
```c
#pragma omp target teams loop collapse(2)
for (i = 0; i < N; i++)
  for (j = 0; j < M; j++) ...
```
Use `target teams distribute parallel for collapse(2)` instead when this compiler/runtime maps the dense iteration space better in practice.

**Tiny staged kernel -> fuse first:**
```c
void train_kernel(state_t *s) {
  float *x = s->x;
  float *y = s->y;
  float *w = s->w;

  #pragma omp target data map(to:x[0:N], w[0:N*M]) map(tofrom:y[0:M])
  {
    #pragma omp target teams loop collapse(2)
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {
        // Forward/update logic that used to live in multiple helpers.
      }
    }
  }
}
```
Use this when the serial code has multiple helper calls but each helper exposes only small outer parallelism.

**Type B (Sparse/CSR) - Nested Parallelism:**
```c
int tmp1, tmp2, tmp3;  // Function scope
#pragma omp target teams loop is_device_ptr(...)
for (int row = 0; row < nrows; row++) {
  tmp1 = rowptr[row];
  tmp2 = rowptr[row+1];
  double sum = 0.0;
  ***#pragma omp loop reduction(+:sum)***  // Parallelize inner *based on GPU saturation* 
  for (int k = tmp1; k < tmp2; k++) {
    tmp3 = colidx[k];
    sum += A[k] * x[tmp3];
  }
  y[row] = sum;
}
```

**Type C1 (Iterative Solvers) - Serial Inner:**
```c
#pragma omp target teams is_device_ptr(...)
{
#pragma omp loop collapse(2)
  for (k = 0; k < K; k++) {
    for (j = 0; j < J; j++) {
      for (stage = 0; stage < S; stage++) { ... }  // No pragma - keep inner serial!
    }
  }
}
```
**Rationale:** K×J teams already saturate GPU. Inner serial = better register reuse, no barriers.

**Type C2 (Multigrid):** Wrap with `present,alloc`; each stencil call gets `target teams loop`.

**Type C special rule:** Stage-dependent algorithms (multigrid, iterative stages) 
should NEVER have inner parallelism, regardless of GPU. The barrier overhead between 
stages exceeds any benefit from inner thread parallelism.

**Type D (Histogram):** Add `#pragma omp atomic` on indirect writes.

**Type F (Reduction):** `reduction(+:sum)`

**Dense update with flattening and collapse:**
```c
const int stride = cols;
#pragma omp target teams loop collapse(2)
for (int j = 0; j < cols; ++j) {
  for (int i = 0; i < rows; ++i) {
    int idx = i * stride + j;
    w[idx] += alpha * delta[j] * x[i];
  }
}
```
For pointer-to-pointer inputs, create contiguous storage first if that enables flat indexing and better GPU code generation.
If profiling shows weaker mapping with `teams loop`, switch this pattern to `target teams distribute parallel for collapse(2)`.

**Type G (Stencil):** `collapse(2)` on spatial dimensions.

**Type A+E (Outer parallel, inner RNG):** 
**When analysis says "RNG replicable: YES":**
- Add `declare target` on RNG function - GPU callable.
- Parallelize over samples, each thread has private RNG + histogram
- Atomic merge histogram at the end

## Histogram Optimization 
If histogram bins ≤ 100:
```c
// GOOD: Per-thread local array (80 bytes for 10 bins)
#pragma omp target teams loop reduction(+:sx, sy)
for (int k = 0; k < N; ++k) {
  double q_local[BINS] = {0};  // Thread-private
  // ... accumulate into q_local ...
  for (int b = 0; b < BINS; ++b) {
    if (q_local[b] != 0.0) {
      #pragma omp atomic update
      q[b] += q_local[b];
    }
  }
}
```
**DO NOT** create large scratch arrays for small histograms - the atomic overhead is negligible compared to memory transfer costs.
**Key:** Each thread replicates the RNG state for its sample. Type E becomes parallelizable at the OUTER level.
