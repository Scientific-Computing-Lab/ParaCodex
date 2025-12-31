# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| d_input       | cudaMalloc      | N*N*sizeof(int) | H→D once at start |
| d_output      | cudaMalloc      | N*N*sizeof(int) | D→H once at end |
| input (host)  | new[] (host)    | N*N*sizeof(int) | Source (initial fill) |
| output (host) | new[] (host)    | N*N*sizeof(int) | Destination (final copy) |

**CUDA Operations:**
- cudaMalloc calls: two allocations (`d_input`, `d_output`) sized at `N*N*sizeof(int)`
- cudaMemcpy H→D: one copy of `input` into `d_input` before the kernel launch
- cudaMemcpy D→H: one copy of `d_output` back to `output` after the kernel completes
- Kernel launches: one invocation of `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)`

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| cellsXOR    | grid `ceil(N/blockEdge)` x `ceil(N/blockEdge)`, block `blockEdge x blockEdge` | once | `d_input` (read), `d_output` (write) |

**Kernel Launch Patterns:**
- Launched once from `main`; not nested inside loops.
- No conditional launches.
- Data flow is: fill `input`, copy to device, run kernel, copy `d_output` back to host.

## OMP Data Movement Strategy
**Chosen Strategy:** A

**Rationale:** Single kernel launch with straightforward host-to-device and device-to-host transfers (Pattern 2). Data fits neatly into a `#pragma omp target data` region with a single `cellsXOR` offload per invocation.

**Device Allocations (OMP equivalent):**
```
CUDA: cudaMalloc(&d_input, size)
OMP A: #pragma omp target data map(to: input[0:N*N])
```
- Host array `input` is mapped to the device for the duration of the kernel, and `output` is mapped from the device to host for retrieval after computation.

**Host→Device Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice)
OMP A: map(to: input[0:N*N]) via target data region (no explicit memcpy)
```
- When: immediately before `cellsXOR` offload
- Arrays: `input`
- Total H→D: ~`2 * N * N` bytes (~`0.0038N^2` MB per iteration depending on N)

**Device→Host Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost)
OMP A: map(from: output[0:N*N]) in the same target data region
```
- When: immediately after the kernel offload
- Arrays: `output`
- Total D→H: ~`2 * N * N` bytes

**Transfers During Iterations:** NO
- All transfers happen once before/after the offload; no intra-kernel transfers.

## Kernel to OMP Mapping (short)
- Replace the `cellsXOR` CUDA kernel with a host function that contains `#pragma omp target teams loop collapse(2) is_device_ptr(...)` iterating across the same `i,j` grid.
- Remove `blockIdx/threadIdx` usage and rely on the loop indices for the 2D space.
- Keep boundary checks inside the loop body.

## Critical Migration Issues
- `__syncthreads()`: not present in kernel
- `shared` memory: not used
- atomics/reductions: not used
- Dynamic indexing: direct, in-bounds checks remain valid

## Expected Performance
- CUDA kernel time: not recorded in analysis (single-launch, `N^2` operations)
- OMP expected: Inline with CUDA since work per element is constant
- Red flag: none identified; this kernel should map directly to the device

**Summary:** 1 kernel, 2 device arrays, Strategy A. CUDA pattern: single stencil kernel with `blockIdx/threadIdx` indexing across a 2D grid. OMP approach: `target data` region + `target teams loop collapse(2)` over `i,j`. Expected: ~`8*N^2` bytes transferred total across H→D and D→H.
