# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| `d_input` | `cudaMalloc` | `N*N*sizeof(int)` | Host→Device once before kernel |
| `d_output` | `cudaMalloc` | `N*N*sizeof(int)` | Device→Host once after kernel |
| `input` | host array | `N*N*sizeof(int)` | Source for H→D map and read-only revalidation |
| `output` | host array | `N*N*sizeof(int)` | Destination for D→H map and validation |

**CUDA Operations:**
- cudaMalloc calls: `cudaMalloc(&d_input, size)` and `cudaMalloc(&d_output, size)` (per run)
- cudaMemcpy H→D: `cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice)` before kernel
- cudaMemcpy D→H: `cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost)` after kernel
- Kernel launches: `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)` once per invocation

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| `cellsXOR` | grid = `(N/blockEdge, N/blockEdge)`, block = `(blockEdge, blockEdge)` covering `N×N` cells | Once per run | `input` read-only, `output` write-only |

**Kernel Launch Patterns:**
- In outer loop? → No, single host call in `main`
- Sequential kernels? → No
- Conditional launch? → No

## OMP Data Movement Strategy
**Chosen Strategy:** A

**Rationale:** Single dense 2D kernel with simple H→D/D→H transfers maps cleanly to a single `target data` region where both arrays are mapped once; uses target teams/loop inside `cellsXOR` for the heavy compute.

**Device Allocations (OMP equivalent):**
```
CUDA: cudaMalloc(&d_input, size)
OMP Strategy A: #pragma omp target data map(to: input[0:totalCells])
CUDA: cudaMalloc(&d_output, size)
OMP Strategy A: #pragma omp target data map(from: output[0:totalCells])
```

**Host→Device Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice)
OMP Strategy A: target data map(to: input[0:totalCells]) (implicit transfer at map entry)
```
- When: once before `cellsXOR` offload
- Arrays: `input`
- Total H→D: `N*N*sizeof(int)` (~4×N² bytes)

**Device→Host Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost)
OMP Strategy A: target data map(from: output[0:totalCells]) (implicit transfer at map exit)
```
- When: once after offload completes
- Arrays: `output`
- Total D→H: `N*N*sizeof(int)` (~4×N² bytes)

**Transfers During Iterations:** NO – only at region boundaries, matching original CUDA timings

## Kernel to OMP Mapping (short)
- Replace CUDA `cellsXOR<<<...>>>` kernel with an OpenMP target teams loop over the same NxN logical domain
- Use `collapse(2)` to mirror the 2D grid/block structure; replace `blockIdx/threadIdx` indexing with nested `i`, `j` loops and direct bounds checks
- Preserve neighbor-count logic exactly inside the offloaded loop body

## Critical Migration Issues
**__syncthreads() usage:** none (not present in CUDA kernel)

**Shared memory:** none

**Atomics:** none

**Dynamic indexing:** simple neighbor offsets; OMP handles pointer math without issue

**__syncthreads() Resolution:** not required (no synchronization in CUDA)

**Shared memory / barriers:** not applicable

## Expected Performance
- CUDA kernel time: (baseline) not measured in repo; expect similar runtime as `cellsXOR` is compute-bound with 4 neighbor reads
- OMP expected: close to CUDA if GPU offload works, maybe slightly slower due to additional target teams setup
- Red flag: >3x slower could signal missing offload (e.g., running on host)

**Summary:** 1 kernel, 2 device arrays, Strategy A. CUDA pattern: single dense stencil kernel with straightforward transfers. OMP approach: map input/output for the duration of `cellsXOR` and offload a collapsed teams loop, reusing host validation code. Expected H→D and D→H transfers each move `N²` ints (4×N² bytes).
