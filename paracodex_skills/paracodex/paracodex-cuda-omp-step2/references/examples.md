# Optimization Actions & Context

## Context: CUDA to OMP Migration
Key differences affect optimization:
- CUDA kernels → OMP `target teams loop`
- `cudaMemcpy` → OMP map clauses or `omp_target_memcpy`
- `__syncthreads()` → May have been split into multiple target regions
- Shared memory → Converted to private or eliminated
- `atomicAdd` → OMP atomic

**Common migration bottlenecks:**
1. Excessive data transfers (lost explicit CUDA control)
2. Over-decomposed kernels (from `__syncthreads()` elimination)
3. Missing collapse on nested loops (CUDA had 2D/3D grids)
4. Suboptimal thread mapping (CUDA grid-stride → OMP loop)

## Fusion Rules

**Fuse when:**
- CUDA had single kernel for operations
- Adjacent independent, same bounds
- Producer-consumer in CUDA
- Multi-vector ops in one CUDA kernel

**Don't fuse:**
- Different bounds
- CUDA had separate kernels with `cudaDeviceSynchronize()`
- `__syncthreads()` required synchronization
- Correctness requires keeping phases separate on device

## Optimization Checklist
- [ ] **Wrong offload unit**: if CUDA recovery is structurally wrong, rewrite before micro-tuning.
- [ ] **Transfers dominate**: hoist data; `omp_target_alloc` + `is_device_ptr`; avoid per-iter mapping.
- [ ] **Too many kernels/regions**: fuse adjacent target loops; inline helper kernels when safe.
- [ ] **Missing CUDA grid shape**: add `collapse(N)`.
- [ ] **Hot kernel**: `const`, `restrict`, cache locals, reduce recomputation (and `simd` where safe).

## Code Examples

### 4A. Hoist Data — Avoid per-iteration transfers
```c
// BAD: data transferred every iteration
for (int iter = 0; iter < NITERS; iter++) {
    #pragma omp target teams loop map(to:A[0:N]) map(from:B[0:N])
    for (int i = 0; i < N; i++) B[i] = f(A[i]);
}

// GOOD: allocate once, keep on device
double *d_A = (double*)omp_target_alloc(N*sizeof(double), 0);
double *d_B = (double*)omp_target_alloc(N*sizeof(double), 0);
omp_target_memcpy(d_A, A, N*sizeof(double), 0, 0, 0, omp_get_initial_device());
for (int iter = 0; iter < NITERS; iter++) {
    #pragma omp target teams loop is_device_ptr(d_A, d_B)
    for (int i = 0; i < N; i++) d_B[i] = f(d_A[i]);
}
omp_target_memcpy(B, d_B, N*sizeof(double), 0, 0, omp_get_initial_device(), 0);
omp_target_free(d_A, 0); omp_target_free(d_B, 0);
```

### 4B. Hot Kernel — collapse + simd
```c
// collapse(2): expose more parallelism to the GPU scheduler
// simd: hint for vectorization within each team's work
#pragma omp target teams loop collapse(2) is_device_ptr(A, B, C)
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        C[i*N+j] = A[i*K] * B[j];  // simplified

// simd on innermost loop inside a target teams region
#pragma omp target teams is_device_ptr(d_x, d_y)
{
    #pragma omp loop
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int k = 0; k < K; k++) sum += d_x[i*K+k];
        d_y[i] = sum;
    }
}
```

### 4C. Reduce Launch Overhead — Inline helpers
```c
// BAD: separate target region for each helper (N×3 kernel launches)
for (int i = 0; i < N; i++) {
    device_spmv(...); device_axpy(...); device_dot(...);
}

// GOOD: fuse into one target region per iteration
for (int i = 0; i < N; i++) {
    #pragma omp target teams is_device_ptr(...)
    {
        #pragma omp loop
        for (int j = 0; j < M; j++) {
            // inline spmv, axpy, dot body here
        }
    }
}
```

### 4S. Structural rewrite before micro-tuning
```c
// If literal CUDA->OMP migration created too many target regions,
// replace the staged helper structure with one fused target region
// that matches the original logical hot path.
```

## CRITICAL: Syntax Reminder

OpenMP clauses (`is_device_ptr`, `use_device_addr`, `map`) require bare pointer variables.
**Extract struct members to local variables before the pragma**:
```c
double *d_arr = data.arr;  // Extract first
#pragma omp target teams loop is_device_ptr(d_arr)  // Use local var
```
