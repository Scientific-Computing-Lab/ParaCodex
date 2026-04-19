# CUDA to OpenMP Migration Patterns

## Pattern Recognition for Strategy Selection

```
Pattern 1: cudaMalloc once → kernel loop → cudaFree
  → Strategy C: omp_target_alloc + is_device_ptr

Pattern 2: Single kernel launch with data transfer
  → Strategy A: target data region

Pattern 3: Multiple kernels with dependencies
  → Strategy B: nowait + depend clauses
```

If the CUDA source uses helper wrappers or tiny launches, prefer a fused OpenMP offload unit over a literal one-launch-per-helper translation.

## Data Movement Strategies

**Device Allocations (OMP equivalent):**
```c
// CUDA: cudaMalloc(&d_arr, size)
// OMP Strategy C:
d_arr = omp_target_alloc(size, 0)
// OMP Strategy A:
#pragma omp target data map(alloc:arr[0:n])
```

**Host→Device Transfers (OMP equivalent):**
```c
// CUDA: cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice)
// OMP Strategy C:
omp_target_memcpy(d_arr, h_arr, size, 0, 0, 0, omp_get_initial_device())
// OMP Strategy A:
map(to:arr[0:n])
// OR
#pragma omp target update to(arr[0:n])
```

**Device→Host Transfers (OMP equivalent):**
```c
// CUDA: cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost)
// OMP Strategy C:
omp_target_memcpy(h_arr, d_arr, size, 0, 0, omp_get_initial_device(), 0)
// OMP Strategy A:
map(from:arr[0:n])
// OR
#pragma omp target update from(arr[0:n])
```

## Kernel Conversion

**Example Conversion:**

CUDA:
```c
__global__ void kernel_name(double *arr, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) arr[idx] = ...;
}
```

OMP:
```c
void kernel_name(double *arr, int n) {
  #pragma omp target teams loop is_device_ptr(arr)
  for (int idx = 0; idx < n; idx++) {  
    arr[idx] = ...;
  }
}
```

## Preserve or rewrite CUDA kernel boundaries intentionally
```c
// If CUDA had one dominant kernel, prefer keeping one dominant OpenMP region.
// If CUDA had several tiny kernels or helper wrappers, fuse them when safe.
// Do not preserve launches literally if that creates fragmented OpenMP offload.
```

## Strategy B — Async Dispatch with nowait + depend
Use when multiple independent kernels can overlap (mirrors CUDA multi-stream pattern):
```c
// Kernel 1 and Kernel 2 are independent — dispatch async, overlap execution
#pragma omp target teams loop nowait depend(out: d_A[0:N])
for (int i = 0; i < N; i++) { d_A[i] = compute_A(i); }

#pragma omp target teams loop nowait depend(out: d_B[0:N])
for (int i = 0; i < N; i++) { d_B[i] = compute_B(i); }

// Kernel 3 depends on both
#pragma omp target teams loop nowait depend(in: d_A[0:N], d_B[0:N]) depend(out: d_C[0:N])
for (int i = 0; i < N; i++) { d_C[i] = d_A[i] + d_B[i]; }

#pragma omp taskwait  // Synchronize before reading d_C on host
```

## CRITICAL: OpenMP Clause Syntax Limitation

OpenMP pragma clauses (`is_device_ptr`, `use_device_addr`, `map`) do NOT support struct member access.
You MUST extract struct members to local pointer variables first.

**WRONG (will not compile):**
```c
#pragma omp target teams loop is_device_ptr(data.arr1, data.arr2)
```

**CORRECT:**
```c
double *d_arr1 = data.arr1;
double *d_arr2 = data.arr2;
#pragma omp target teams loop is_device_ptr(d_arr1, d_arr2)
for (int i = 0; i < n; i++) {
    // use d_arr1[i], d_arr2[i] inside the loop
}
```
When converting CUDA code that passes structs to kernels, extract ALL device pointer members to local variables BEFORE the pragma.
