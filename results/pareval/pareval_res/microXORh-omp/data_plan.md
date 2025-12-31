# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| d_input       | cudaMalloc      | N×N×sizeof(int) | H→D once, D→H once |
| d_output      | cudaMalloc      | N×N×sizeof(int) | D→H once |
| input (host)  | heap (new[])    | N×N×sizeof(int) | source for init and H→D |
| output (host) | heap (new[])    | N×N×sizeof(int) | destination from D→H |

**CUDA Operations:**
- cudaMalloc calls: `cudaMalloc(&d_input, N*N*sizeof(int))`, `cudaMalloc(&d_output, N*N*sizeof(int))`
- cudaMemcpy H→D: copies `input` → `d_input` once before kernel launch
- cudaMemcpy D→H: copies `d_output` → `output` once after kernel launch
- Kernel launches: `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)` executed once per run

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| `cellsXOR`  | grid=(ceil(N/blockEdge), ceil(N/blockEdge)), block=(blockEdge, blockEdge) | once | device `d_input`, `d_output`; parameter `N` |

**Kernel Launch Patterns:**
- Kernel launched one time after initialization → single offload region
- No sequential or conditional kernels

## OMP Data Movement Strategy

**Chosen Strategy:** C

**Rationale:** The CUDA version allocates device buffers once via `cudaMalloc`, transfers data once before the lone kernel launch, and frees them after validation. Strategy C maps cleanly to this pattern by mimicking persistent device pointers with `omp_target_alloc` (or equivalent) and explicit `omp_target_memcpy` calls, keeping the host/device lifetimes similar to CUDA.

**Device Allocations (OMP equivalent):**
```
int *d_input = (int *)omp_target_alloc(N*N*sizeof(int), omp_get_default_device());
int *d_output = (int *)omp_target_alloc(N*N*sizeof(int), omp_get_default_device());
```

**Host→Device Transfers (OMP equivalent):**
```
omp_target_memcpy(d_input, input, N*N*sizeof(int), 0, 0, omp_get_default_device(), omp_get_initial_device());
```
- When: once after host input buffer is populated
- Arrays: `input` (size N×N ints)
- Total H→D: approximately `4 * N²` bytes

**Device→Host Transfers (OMP equivalent):**
```
omp_target_memcpy(output, d_output, N*N*sizeof(int), 0, 0, omp_get_initial_device(), omp_get_default_device());
```
- When: after kernel offload completes
- Arrays: `output` (size N×N ints)
- Total D→H: approximately `4 * N²` bytes

**Transfers During Iterations:** NO

## Kernel to OMP Mapping (short)
- Replace `cellsXOR` kernel with a host function that offloads via `#pragma omp target teams loop is_device_ptr(d_input, d_output)` over the flat 2-D iteration space [0,N)×[0,N).
- Adapt `blockIdx`/`threadIdx` indexing to a pair of loop variables `i` (row) and `j` (col).
- Keep boundary checks for safety (though loops will already be bounded).

## Critical Migration Issues
- `__syncthreads()`: not present.
- Shared memory: not present; direct global reads/writes.
- Atomics: none.
- Dynamic indexing: omitted.

**__syncthreads() Resolution:** Not applicable.

**Shared memory / barriers:** Not applicable.

## Expected Performance
- CUDA kernel time: unknown (baseline not provided) but single stencil over N² grid.
- OMP expected: comparable with careful offload; contiguous row-major should keep bandwidth steady.
- Red flag: more than 3× slowdown would suggest missing parallelism or transfers.

**Summary:** 1 kernel, 2 device arrays, Strategy C. CUDA pattern: single stencil kernel with persistent device buffers. OMP approach: mimic allocations/transfer with `omp_target_alloc` + `omp_target_memcpy`, offload the stencil via `target teams loop` over rows and columns.
