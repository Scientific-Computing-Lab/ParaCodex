# Kernel Classification Guide

## Decision Tree

```
Q0: Is this a __global__ kernel or host loop? → Note context
Q1: Writes A[idx[i]] with varying idx (atomicAdd)? → Type D (Histogram)
Q2: Uses __syncthreads() or shared memory dependencies? → Type E (Block-level recurrence)
Q3: Multi-stage kernel pattern?
    - Separate kernels for stages with global sync? → C1 (FFT/Butterfly)
    - Hierarchical grid calls? → C2 (Multigrid)
Q4: Block/thread indexing varies with outer dimension? → Type B (Sparse)
Q5: Uses atomicAdd to scalar (reduction pattern)? → Type F (Reduction)
Q6: Accesses neighboring threads' data? → Type G (Stencil)
Default → Type A (Dense)
```

## Structural Recommendation Rule
- If CUDA already uses one large kernel for the hot path, preserve that logical unit when possible.
- If CUDA uses several tiny kernels only because of source structure, consider fusing them during OMP migration.
- If `__syncthreads()` is present, decide whether the kernel must be split or can be rewritten without preserving the exact block-level staging.

## Type Reference

|| Type | CUDA Pattern | OMP Equivalent | Notes |
||------|--------------|----------------|-------|
|| A | Dense kernel, regular grid | YES - parallel for | Direct map |
|| B | Sparse (CSR), varying bounds | Outer only | Inner sequential |
|| C1 | Multi-kernel, global sync | Outer only | Barrier between stages |
|| C2 | Hierarchical grid | Outer only | Nested parallelism tricky |
|| D | Histogram, atomicAdd | YES + atomic | Performance loss expected |
|| E | __syncthreads, shared deps | NO | Requires restructuring |
|| F | Reduction, atomicAdd scalar | YES + reduction | OMP reduction clause |
|| G | Stencil, halo exchange | YES | Ghost zone handling |

---

## Code Examples

### grep commands for CUDA kernel analysis
```bash
grep -n "__global__\|__device__" *.cu 2>/dev/null
grep -n "<<<.*>>>" *.cu 2>/dev/null
grep -n "__syncthreads\|cudaDeviceSynchronize" *.cu 2>/dev/null
grep -n "__shared__" *.cu 2>/dev/null
grep -n "atomicAdd\|atomicMax\|atomicMin" *.cu 2>/dev/null
grep -n "cudaMalloc\|cudaMemcpy\|cudaFree" *.cu 2>/dev/null
```

### CUDA → OMP Kernel Conversion (Type A — Dense)
```c
// CUDA kernel
__global__ void scale(double *A, double alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) A[i] *= alpha;
}
// Launch: scale<<<(N+255)/256, 256>>>(d_A, alpha, N);

// OMP equivalent (Strategy C — is_device_ptr)
void scale(double *d_A, double alpha, int N) {
    #pragma omp target teams loop is_device_ptr(d_A)
    for (int i = 0; i < N; i++) d_A[i] *= alpha;
}
```

### __syncthreads() — requires kernel split (Type E)
```c
// CUDA: single kernel with __syncthreads
__global__ void twoPhase(float *A, float *B, int N) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    smem[threadIdx.x] = A[i];   // Phase 1: load
    __syncthreads();
    B[i] = smem[(threadIdx.x + 1) % 256];  // Phase 2: use neighbor
}

// OMP: must split into two target regions
#pragma omp target teams loop is_device_ptr(d_A, d_tmp) map(tofrom: d_A[0:N])
for (int i = 0; i < N; i++) d_tmp[i] = d_A[i];  // Phase 1

#pragma omp target teams loop is_device_ptr(d_tmp, d_B)
for (int i = 0; i < N; i++) d_B[i] = d_tmp[(i+1) % N];  // Phase 2
```

### Shared memory → per-thread private (Strategy C)
```c
// CUDA __shared__ used only for register spilling / local accumulation
// → Replace with scalar private variable in OMP

// CUDA
__shared__ float partial[256];
partial[threadIdx.x] = A[idx] * B[idx];

// OMP: private scalar (each team thread has its own copy)
double local_val = d_A[i] * d_B[i];  // automatically private inside loop body
```

### CUDA launch structure is part of the analysis
```c
// If CUDA had one kernel for the full hot path, analysis should not recommend
// exploding it into several OpenMP regions unless correctness requires it.
```
