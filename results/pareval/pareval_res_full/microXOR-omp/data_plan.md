# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| d_input | cudaMalloc | `N * N * sizeof(int)` | H→D once before kernel |
| d_output | cudaMalloc | `N * N * sizeof(int)` | D→H once after kernel |
| input | host array | `N * N * sizeof(int)` | Source for H→D copy |
| output | host array | `N * N * sizeof(int)` | Destination for D→H copy |

**CUDA Operations:**
- cudaMalloc calls: `cudaMalloc(&d_input, N * N * sizeof(int))`, `cudaMalloc(&d_output, N * N * sizeof(int))`
- cudaMemcpy H→D: `cudaMemcpy(d_input, input, N * N * sizeof(int), cudaMemcpyHostToDevice)` (once before kernel)
- cudaMemcpy D→H: `cudaMemcpy(output, d_output, N * N * sizeof(int), cudaMemcpyDeviceToHost)` (once after kernel)
- Kernel launches: `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)` (once per run)

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| `cellsXOR<<<numBlocks, threadsPerBlock>>>` | grid = `(ceil(N/blockEdge), ceil(N/blockEdge))`, block = `(blockEdge, blockEdge)` | once per run | `d_input`, `d_output`, `N` |

**Kernel Launch Patterns:**
- launched exactly once from `main()` with fixed grid/block to cover the `N×N` input
- no conditional or outer loops controlling the launch
- purely compute stencil; host validation follows after kernel completes

## OMP Data Movement Strategy
**Chosen Strategy:** A

**Rationale:** single kernel launch with one H→D transfer before the compute phase and one D→H transfer after; data can be mapped into a `target data` region while the kernel body becomes an OpenMP target teams loop over the same `N×N` domain.

**Device Allocations (OMP equivalent):**
```
CUDA: cudaMalloc(&d_input, size)
OMP Strategy A: #pragma omp target data map(alloc: input[0:N*N])
``` 
`output` is similarly mapped with `map(from: output[0:N*N])` and used for writes.

**Host→Device Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice)
OMP Strategy A: the `target data` region opens with `map(to: input[0:N*N])`, so the runtime copies the entire array into the device before the kernel offload.
```
- When: upon entering the `target teams loop` nested in the `cellsXOR` function once per run
- Arrays: `input` (`N*N` ints)
- Total H→D: ~`4 * N * N` bytes (~4 bytes per int)

**Device→Host Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost)
OMP Strategy A: the `target teams loop` keeps `output` mapped with `map(tofrom: output[0:N*N])`, so the final write-back occurs when the target data region closes after the kernel completes.
```
- When: after the `cellsXOR` offload finishes, but before host validation begins
- Arrays: `output` (`N*N` ints)
- Total D→H: ~`4 * N * N` bytes

**Transfers During Iterations:** NO – all transfers occur before/after the single kernel invocation

## Kernel to OMP Mapping (short)
- Replace the CUDA grid/block launch with a `#pragma omp target teams loop` over a flattened `N*N` domain (or a `collapse(2)` over `i` and `j`).
- Drop `blockIdx/threadIdx` arithmetic by computing `(i, j)` from a simple integer loop index.
- Keep the neighbor-count logic intact inside the loop body, with standard C bounds checks as before.

## Critical Migration Issues
- `__syncthreads()` usage: NONE
- Shared memory: NONE
- Atomics: NONE
- Dynamic indexing: handled by the per-cell access pattern (`i*N + j`)

**__syncthreads() Resolution:** Not applicable.

**Shared memory / barriers:** Not applicable.

## Expected Performance
- CUDA kernel time: not provided in this repo (baseline only)
- OMP expected: similar amount of work with a single target launch; in absence of caching, expect comparable throughput for the same `N`.
- Red flag: no more than ~3× slowdown expected for this pattern.

**Summary:** 1 kernel, 2 device arrays (input/output), Strategy A. CUDA pattern: single kernel launching an `N×N` stencil over device buffers. OMP approach: keep `input/output` mapped via `#pragma omp target data`, run the same stencil logic inside a `target teams loop`, and reuse the host driver for validation. Expected: ~`4*N*N` bytes H→D + D→H each.
