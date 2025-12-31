# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| d_input | cudaMalloc | N*N*sizeof(int) | H→D once at launch |
| d_output | cudaMalloc | N*N*sizeof(int) | D→H once after kernel |
| input | host array | N*N*sizeof(int) | source for d_input |
| output | host array | N*N*sizeof(int) | destination from d_output |

**CUDA Operations:**
- cudaMalloc calls: `cudaMalloc(&d_input, N * N * sizeof(int))`, `cudaMalloc(&d_output, N * N * sizeof(int))`
- cudaMemcpy H→D: `cudaMemcpy(d_input, input, N * N * sizeof(int), cudaMemcpyHostToDevice)` executed once before kernel
- cudaMemcpy D→H: `cudaMemcpy(output, d_output, N * N * sizeof(int), cudaMemcpyDeviceToHost)` executed once after kernel
- Kernel launches: `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)` once per run

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| cellsXOR | grid={(N+blockEdge-1)/blockEdge, (N+blockEdge-1)/blockEdge}, block={blockEdge, blockEdge} | once | d_input (read), d_output (write), N |

**Kernel Launch Patterns:**
- In outer loop? No, single launch
- Sequential kernels? No
- Conditional launch? No

## OMP Data Movement Strategy
**Chosen Strategy:** A

**Rationale:** Single kernel, single H→D transfer and single D→H transfer; `target data` with `map(to:)`/`map(from:)` mirrors this simple flow while keeping data residence bounded to the launch.

**Device Allocations (OMP equivalent):**
```
#pragma omp target data map(to: input[0:N*N]) map(from: output[0:N*N])
```

**Host→Device Transfers (OMP equivalent):**
- When: before `cellsXOR` offload (managed by `#pragma omp target` mapping)
- Arrays: `input`
- Total H→D: ~N*N*4 bytes

**Device→Host Transfers (OMP equivalent):**
- When: after `cellsXOR` offload (managed by `map(from: output...)`)
- Arrays: `output`
- Total D→H: ~N*N*4 bytes

**Transfers During Iterations:** NO

## Kernel to OMP Mapping (short)
- Replace kernel launch with `#pragma omp target teams loop` inside a helper function that takes `int *input`, `int *output`, `size_t N`.
- `blockIdx/threadIdx` translate into explicit 2D loops over `i` and `j`, or a single loop over linear index.
- Keep the same logical neighbor checks.

## Critical Migration Issues
**From analysis.md "OMP Migration Issues":**
- __syncthreads() usage: not present
- Shared memory: not used
- Atomics: not used
- Dynamic indexing: handled via standard indexing

**__syncthreads() Resolution:**
- Not applicable.

**Shared memory / barriers:**
- Not used; no additional refactoring needed.

## Expected Performance
- CUDA kernel time: not provided (baseline only)
- OMP expected: similar order since kernel is simple
- Red flag: unlikely as only one kernel and simple mapping

**Summary:** 1 kernel, 2 device arrays, Strategy A.
CUDA pattern: single kernel with global memory ops.
OMP approach: `#pragma omp target data` around helper `cellsXOR_device`, `#pragma omp target teams loop` for parallelism.
Expected: ~4*N*N bytes H→D and D→H.
