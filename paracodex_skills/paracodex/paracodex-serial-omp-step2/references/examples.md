# Optimization Actions & Fusion Rules

## Fusion Rules

**Fuse when:**
- Adjacent independent, same bounds
- Producer-consumer
- Multi-vector ops
- Staged helper offload creates tiny kernels or host/device sync
- Preserved serial helper structure prevents collapse/fusion of the true hot work

**Don't fuse:**
- Different bounds
- Intermediate sync required
- Algorithmic staging is semantically required and cannot stay on device

## 4A. Fix Data Movement

- Hoist target data outside loops
- `omp_target_alloc` + `is_device_ptr` for scratch
- Remove map inside target data
- Wrap functions: `present,alloc`
- Host init: `target update to` after

## 4B. Optimize Hot Kernel

- Use combined `target teams loop`
- Or `target teams distribute parallel for` when it yields better parallel mapping on the active compiler/runtime
- Type B: Add inner `#pragma omp loop reduction(+:sum)`
- `collapse(N)` on nested dense loops
- Add `#pragma omp simd` to innermost
- Cache array accesses (SpMV/CSR)

**simd on innermost loop:**
```c
// Vectorize innermost loop within each GPU thread's work
#pragma omp target teams is_device_ptr(d_A, d_B, d_C)
{
    #pragma omp loop collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++)
                sum += d_A[i*K+k] * d_B[k*N+j];
            d_C[i*N+j] = sum;
        }
    }
}
```

```c
int tmp1, tmp2, tmp3;  // Function scope
#pragma omp target teams loop is_device_ptr(...)
for (int i = 0; i < nrows; i++) {
  tmp1 = d_rowptr[i];
  tmp2 = d_rowptr[i+1];
  double sum = 0.0;
  #pragma omp loop reduction(+:sum)
  for (int k = tmp1; k < tmp2; k++) {
    tmp3 = d_col[k];
    sum += d_val[k] * d_x[tmp3];
  }
  d_y[i] = sum;
}
```

## 4C. Launch Overhead

**Rule:** If kernel instances >> iteration count, inline helper functions in the main loop.
- Keep reduction helpers (dot, norm) - they return scalars
- Inline SpMV, vector updates, scaling operations
- Fuse adjacent loops with same bounds

## 4S. Structural Rewrite

When step1 preserved the serial structure but the profile shows tiny kernels or low outer saturation, rewrite the hot path first:

```c
void fused_train(state_t *s) {
  float *x = s->x;
  float *h = s->h;
  float *y = s->y;
  float *w1 = s->w1;
  float *w2 = s->w2;

  #pragma omp target data map(to:x[0:N], w1[0:N*H], w2[0:H*O]) \
                          map(tofrom:h[0:H], y[0:O])
  {
    #pragma omp target teams distribute parallel for
    for (int j = 0; j < H; ++j) {
      float sum = 0.0f;
      for (int i = 0; i < N; ++i)
        sum += w1[i * H + j] * x[i];
      h[j] = sum;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int j = 0; j < H; ++j)
      for (int i = 0; i < N; ++i)
        w1[i * H + j] += alpha * h[j] * x[i];
  }
}
```

Use this when the winning optimization is to replace helper-level offload with one GPU-oriented routine.

## 4D. Fix Type C1 (Multi-Stage)

- Outer loops: `collapse(2)` on spatial dimensions
- Inner stage loops: **Remove all pragmas (must be serial)**

## 4E. Increase Parallelism

- Increase collapse depth
- Use `tile sizes(32, 32)`
- Remove manual `num_teams`/`thread_limit`
- Flatten pointer-to-pointer arrays into contiguous buffers if that enables more collapse/fusion and lower mapping overhead

## 4F. Performance Portability: metadirective (OpenMP 5.0+)

Use `metadirective` to select different parallelization strategies for GPU vs CPU without `#ifdef`:

```c
// Single source compiles optimally for both GPU and CPU targets
void compute(double *A, double *B, double *C, int N) {
    #pragma omp metadirective \
        when(device={arch("nvptx", "amdgcn")}: target teams loop) \
        default(parallel for simd)
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}
```

## 4G. Device Zero-Init: omp_target_memset (OpenMP 6.0)

```c
// Initialize device buffer without copying from host (OpenMP 6.0)
// Replaces: cudaMemset / hipMemset patterns
void *d_scratch = omp_target_alloc(N * sizeof(double), omp_get_default_device());
omp_target_memset(d_scratch, 0, N * sizeof(double), omp_get_default_device());
```
