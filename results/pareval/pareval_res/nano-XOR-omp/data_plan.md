# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| `input` | Host array (`new int[N*N]`) | `N*N*sizeof(int)` | source for H→D copy once before kernel |
| `output` | Host array (`new int[N*N]`) | `N*N*sizeof(int)` | destination for D→H copy once after kernel |
| `d_input` | `cudaMalloc` | `N*N*sizeof(int)` | `cudaMemcpy` H→D once (before kernel) |
| `d_output` | `cudaMalloc` | `N*N*sizeof(int)` | `cudaMemcpy` D→H once (after kernel) |

**CUDA Operations:**
- cudaMalloc calls: `d_input`, `d_output` (each `N*N*sizeof(int)`).
- cudaMemcpy H→D: `cudaMemcpy(d_input, input, ...)` just before `cellsXOR` launch.
- cudaMemcpy D→H: `cudaMemcpy(output, d_output, ...)` immediately after kernel.
- Kernel launches: `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)` executed exactly once per run.

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| `cellsXOR` | grid `(ceil(N/blockEdge), ceil(N/blockEdge))`, block `(blockEdge, blockEdge)` | once | `d_input` (read), `d_output` (write), constant `N` |

**Kernel Launch Patterns:**
- In outer loop? → No, single launch from `main`.
- Sequential kernels? → No, only one compute kernel.
- Conditional launch? → No.
- Domain: full `N×N` grid with one thread per output cell.

## OMP Data Movement Strategy

**Chosen Strategy:** A

**Rationale:** single kernel with one H→D and one D→H copy surrounding it; everything can be expressed via an OpenMP target data region that maps the two flat grids.

**Device Allocations (OMP equivalent):**
```
CUDA: cudaMalloc(&d_input, N*N*sizeof(int))
OMP Strategy A: #pragma omp target data map(to: input[0:N*N])
CUDA: cudaMalloc(&d_output, N*N*sizeof(int))
OMP Strategy A: #pragma omp target data map(from: output[0:N*N])
```

**Host→Device Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(d_input, input, N*N*sizeof(int), cudaMemcpyHostToDevice)
OMP Strategy A: map(to: input[0:N*N]) via target data region (boundaries before kernel invocation)
```
- When: once before the offloaded region.
- Arrays: `input` (size `N*N*sizeof(int)`).
- Total H→D: ~`N*N*4` bytes (~`N² * 4` bytes).

**Device→Host Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(output, d_output, N*N*sizeof(int), cudaMemcpyDeviceToHost)
OMP Strategy A: map(from: output[0:N*N]) via target data region (after kernel finishes)
```
- When: once immediately after kernel execution.
- Arrays: `output` (size `N*N*sizeof(int)`).
- Total D→H: ~`N*N*4` bytes (~`N² * 4` bytes).

**Transfers During Iterations:** NO
- All transfers happen once outside the repeated compute; the compute kernel is bounded by the single launch.

## Kernel to OMP Mapping (short)
- Replace the `cellsXOR` `__global__` function with a host helper that contains a `#pragma omp target teams loop collapse(2)` region using `is_device_ptr(input, output)` and iterates over the output domain.
- Keep all neighbor checks and boundary guards exactly as in the CUDA kernel but iterate with explicit loop indices `i` and `j` that cover `[0,N)`.
- Launch pattern remains logically `N×N` iterations, so the double loop inside the target region directly mirrors the CUDA grid-stride.

## Critical Migration Issues
- __syncthreads() usage: not present in the CUDA version → no special handling required.
- Shared memory: none used.
- Atomics: none.
- Dynamic indexing: only regular row/column lookups; OpenMP can handle them.

**__syncthreads() Resolution:** N/A (not used).
**Shared memory / barriers:** N/A.

## Expected Performance
- CUDA kernel time: (not measured in repo; baseline run exists but not timed).
- OMP expected: similar per-cell compute cost; offload launch may be slightly different but still single offload.
- Red flag: If OMP target overhead causes ≥3× slower runtime, consider adjusting loop collapse/teams reduction strategy.

**Summary:** 1 kernel, 4 device arrays (2 host, 2 device) tracked, Strategy A. CUDA pattern: host→device copy, single `cellsXOR` launch over `N×N`, device→host copy. OMP approach: map the host grids via `#pragma omp target data`, run the double loop inside `#pragma omp target teams loop collapse(2)` with `is_device_ptr`, and rely on mapped arrays for validation. Expected transfers ~`2*N²*4` bytes (`input` and `output`).
