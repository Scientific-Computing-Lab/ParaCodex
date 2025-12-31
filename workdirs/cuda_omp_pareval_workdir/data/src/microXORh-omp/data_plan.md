# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| d_input       | cudaMalloc      | N*N*sizeof(int) | Host→Device once before kernel |
| d_output      | cudaMalloc      | N*N*sizeof(int) | Device→Host once after kernel |
| input         | new int[] (host)| N*N*sizeof(int) | Source for H→D input |
| output        | new int[] (host)| N*N*sizeof(int) | Destination for D→H output |

**CUDA Operations:**
- cudaMalloc calls: `cudaMalloc(&d_input, N * N * sizeof(int))`, `cudaMalloc(&d_output, N * N * sizeof(int))`
- cudaMemcpy H→D: Copy `input` → `d_input` once before launching `cellsXOR`
- cudaMemcpy D→H: Copy `d_output` → `output` once immediately after kernel
- Kernel launches: `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)` executed once per run

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| cellsXOR <<<numBlocks, threadsPerBlock>>> | grid = ((N+blockEdge-1)/blockEdge)^2, block = (blockEdge, blockEdge) | once | d_input (R), d_output (W) |

**Kernel Launch Patterns:**
- In outer loop? No. Kernel launched once as part of the main compute stage.
- Sequential kernels? Only `cellsXOR` is executed.
- Conditional launch? No.

## OMP Data Movement Strategy
**Chosen Strategy:** Strategy A

**Rationale:** Single dense 2D stencil kernel with a block-global data transfer before/after the compute; a `target data` region with simple maps keeps the structure closest to the CUDA flow.

**Device Allocations (OMP equivalent):**
```
#pragma omp target data map(to: input[0:N*N]) map(from: output[0:N*N])
{
  // kernel invocation
}
```

**Host→Device Transfers (OMP equivalent):**
```
#pragma omp target data map(to: input[0:N*N])
```
- When: prior to the stencil kernel launch
- Arrays: `input`
- Total H→D: ~N*N*sizeof(int) (~4*N^2 bytes)

**Device→Host Transfers (OMP equivalent):**
```
#pragma omp target data map(from: output[0:N*N])
```
- When: immediately after kernel completes
- Arrays: `output`
- Total D→H: ~N*N*sizeof(int) (~4*N^2 bytes)

**Transfers During Iterations:** NO
- All moves happen before/after the single kernel execution; no repeated transfers inside the compute stage.

## Kernel to OMP Mapping (short)
- Replace `cellsXOR` kernel launch with an OMP offload function that uses `#pragma omp target teams loop collapse(2) is_device_ptr(input, output)` over `i` and `j` indices.
- Translate `blockIdx/threadIdx` indexing to nested loops guarded by the `collapse(2)` delegate for the full `N×N` domain.
- Boundary checks can be simplified by looping exactly over valid `i` and `j` ranges.

## Critical Migration Issues
**From analysis.md "OMP Migration Issues":**
- [ ] __syncthreads() usage: not present
- [ ] Shared memory: none
- [ ] Atomics: none
- [ ] Dynamic indexing: straightforward row-major access

**__syncthreads() Resolution:** N/A – no CUDA barriers were used.

**Shared memory / barriers:** No shared state; each thread reads from global arrays independently.

## Expected Performance
- CUDA kernel time: not reported in provided data
- OMP expected: comparable per-element work; `target teams loop` should hit ~N^2 iterations with similar memory behavior
- Red flag: >3x slowdown would suggest the offload mapping is incorrect or data is not kept resident

**Summary:** 1 kernel, 2 device arrays, Strategy A. CUDA pattern: simple 2D stencil with a single kernel and symmetrical transfers. OMP approach: wrap `input`/`output` in a `target data` region and execute a `target teams loop` over the full grid. Expected transfers: ~4*N^2 bytes host→device and ~4*N^2 bytes device→host.
