# microXOR CUDA Loop Classification

## File Conversion Mapping
```
main.cu → main.cpp
microXOR.cu → microXOR.cpp
```
(When copying, update `#include "microXOR.cuh"` to the C++ header that wraps the OpenMP/host-side kernel declaration so the new `.cpp` files compile with the migrated build.)

## Kernel/Loop Nesting Structure
- `main` ([main.cu:12](main.cu:12)) builds the grid and dispatches a single compute kernel per invocation
  - host loop (initialization, [main.cu:34](main.cu:34)) prepares `input` with N² random bits
  - host loop (validation, [main.cu:64](main.cu:64)) scans each cell after the GPU pass
  └── `cellsXOR<<<numBlocks, threadsPerBlock>>>(...)` ([main.cu:52](main.cu:52)) Type A dense 2D kernel

## Kernel/Loop Details

### Kernel/Loop: cellsXOR at microXOR.cu:21
- **Context:** `__global__` compute kernel that runs over an NxN grid
- **Launch config:** `(N/blockEdge) × (N/blockEdge)` blocks of `blockEdge × blockEdge` threads; blocks cover the whole 2D matrix when `N % blockEdge == 0`
- **Total threads/iterations:** `N × N` threads, each touching a single cell
- **Type:** A – dense stencil-like scan with regular indexing and no sparse/atomic pattern
  - **Parent loop:** single host call from `main` ([main.cu:52](main.cu:52))
- **Contains:** no device-side loops, only a few `if` checks for bounds
- **Dependencies:** none (no atomics, no shared-memory barriers)
- **Shared memory:** NO – only global loads/stores
- **Thread indexing:** `i = blockIdx.y * blockDim.y + threadIdx.y`, `j = blockIdx.x * blockDim.x + threadIdx.x`
- **Private vars:** `count` is thread-private; `i`, `j`, `N` map to registers
- **Arrays:** `input` (R) & `output` (W) – global device buffers passed as pointers; `N` is a value parameter
- **OMP Migration Issues:** minimal – regular 2D access allows mapping to nested `for` loops and OpenMP parallelization without atomics or synchronizations

### Kernel/Loop: random initialization at main.cu:34
- **Context:** host-side loop preparing `input` before kernel launch
- **Launch config:** single-threaded `for (size_t i = 0; i < N * N; ++i)`
- **Total iterations:** `N × N` random assignments
- **Type:** A – dense linear scan, no irregular access
- **Parent loop:** `main` (setup phase)
- **Contains:** `std::uniform_int_distribution` result assignment
- **Dependencies:** reads `std::mt19937` state; no device interaction
- **Shared memory:** not applicable
- **Thread indexing:** sequential host index `i`
- **Private vars:** `i`, RNG objects
- **Arrays:** `input` (W) – host heap; no CUDA constructs
- **OMP Migration Issues:** trivial – can be `#pragma omp parallel for` on host with chunking, RNG seeding must be thread-safe if threaded

### Kernel/Loop: validation nest at main.cu:64
- **Context:** host-side verification loop after copying `output` back
- **Launch config:** nested host loops `i` and `j`, each iterating `N` times
- **Total iterations:** `N × N` checks computing neighbor counts per cell
- **Type:** A – dense per-cell stencil check
- **Parent loop:** `main` (validation phase)
- **Contains:** 4 neighbor checks, conditional prints on failure
- **Dependencies:** reads both `input` and `output` host buffers
- **Shared memory:** not applicable
- **Thread indexing:** sequential host indices `i`, `j`
- **Private vars:** `count` and loop indices
- **Arrays:** `input`/`output` (R) – host arrays built prior to validation
- **OMP Migration Issues:** low – safe-to-parallelize once data dependencies are respected (no writes), but would need to protect the early-exit `return` or collect a failure flag across threads

## Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|-----------------|------|----------|---------|------------|--------------|------------|
| `cellsXOR` (microXOR.cu:21) | A | CRITICAL | dense 2D CUDA kernel | `N²` threads, 5 neighbor reads + 1 write each | none | direct mapping to OpenMP parallel loops; no synchronization or atomics |
| initialization loop (main.cu:34) | A | SECONDARY | host-side RNG fill | `N²` iterations | RNG state | safe to parallelize with thread-safe RNG seeding |
| validation nest (main.cu:64) | A | SECONDARY | host-side verification | `N²` iterations | host read-only access | capture failure flag before exiting when parallelized |

## CUDA-Specific Details
- **Dominant compute kernel:** `cellsXOR` (microXOR.cu:21) – each thread examines four neighbors and writes one output cell.
- **Memory transfers in timed loop?:** YES – `cudaMemcpy` Host→Device before the kernel and Device→Host after it wrap the compute.
- **Shared memory usage:** none (0 bytes).
- **Synchronization points:** none inside the kernel (no `__syncthreads()`); CUDA kernel boundary is the only device synchronization.
- **Atomic operations:** none.
- **Reduction patterns:** none.

## OMP Migration Strategy Notes
- **Direct kernel → parallel for:** `cellsXOR` is a perfect fit for a parallel 2D `for` nest (or flattened `for` over `N²`) with work-sharing; each iteration reads fixed offsets and writes one element, so OpenMP `parallel for collapse(2)` can replace the CUDA grid.
- **Requires restructuring:** none – there are no shared-memory dependencies, GPU-only atomics, or fine-grained barriers.
- **Performance concerns:** moving the `cudaMemcpy` pairs to OpenMP target data regions or unified memory will reduce explicit transfers; threads should avoid divergence in the validation branch when condensing errors.
- **Data management:** replace `cudaMalloc`/`cudaMemcpy` with host allocations and `#pragma omp target data map(to: input[0:totalCells]) map(from: output[0:totalCells])` (as in the current OpenMP baseline) so the kernel sees device copies without manual `cudaMemcpy` calls.
