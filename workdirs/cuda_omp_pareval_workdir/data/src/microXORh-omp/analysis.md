# microXORh Loop Classification for OMP Migration

## File Conversion Mapping
```
golden_labels/src/microXORh-cuda/main.cu → data/src/microXORh-omp/main.cpp
```

## Kernel/Loop Nesting Structure
- `main` (golden_labels/src/microXORh-cuda/main.cu:42) allocates host/device buffers, fills `input`, and launches `cellsXOR`.
  ├── `cellsXOR<<<numBlocks, threadsPerBlock>>>` (golden_labels/src/microXORh-cuda/main.cu:79-83) Type A kernel that maps each thread to one grid cell.
  └── Host validation nested loop (golden_labels/src/microXORh-cuda/main.cu:93-115) compares `output` against sequential compute.
- Host initialization loop (golden_labels/src/microXORh-cuda/main.cu:64-71) populates `input` before kernel launch.

## Kernel/Loop Details
### Kernel/Loop: `cellsXOR` at golden_labels/src/microXORh-cuda/main.cu:22
- **Context:** `__global__` kernel updating each of the N×N grid cells.
- **Launch config:** `grid = ((N + blockEdge-1)/blockEdge, (N + blockEdge-1)/blockEdge)` × `block = (blockEdge, blockEdge)` with `blockEdge ∈ [2,32]`.
- **Total threads/iterations:** `((N/blockEdge)^2) × (blockEdge^2) = N^2`; each thread touches a unique cell.
- **Type:** A – dense, regular 2D stencil that reads four neighbors and writes a single cell.
- **Parent loop:** `main` setup and compute stage (no outer CUDA loop).
- **Contains:** no internal device loops, only neighbor checks and a conditional write.
- **Dependencies:** none beyond per-thread boundary checks; no atomics or synchronizations.
- **Shared memory:** NO – all data comes from global buffers.
- **Thread indexing:** `i = blockIdx.y * blockDim.y + threadIdx.y`, `j = blockIdx.x * blockDim.x + threadIdx.x`; guards ensure `i < N`, `j < N`.
- **Private vars:** `i`, `j`, `count`.
- **Arrays:** `input(R)`, `output(W)` point to `cudaMalloc`-backed global memory buffers; each thread reads neighbors in row-major layout.
- **OMP Migration Issues:** None unique; the kernel is a straightforward per-cell stencil – map to a 2D `parallel for` while preserving boundary checks.

### Kernel/Loop: Host initialization loop at golden_labels/src/microXORh-cuda/main.cu:64-71
- **Context:** Host loop filling `input` with random 0/1 values before compute.
- **Launch config:** single-threaded; iterations = `N^2`.
- **Total threads/iterations:** `N^2` sequential writes.
- **Type:** A – dense initialization of a contiguous buffer.
- **Parent loop:** top-level `main`.
- **Contains:** none.
- **Dependencies:** reads from RNG, writes to host `input`.
- **Shared memory:** N/A.
- **Thread indexing:** single loop index `i` over `input`.
- **Private vars:** `i`.
- **Arrays:** `input(W)` host buffer.
- **OMP Migration Issues:** trivial; could be parallelized with OMP if desired but already sequential one-time setup.

### Kernel/Loop: Host validation nested loops at golden_labels/src/microXORh-cuda/main.cu:93-115
- **Context:** Host loop re-computes the stencil and compares against `output` for correctness.
- **Launch config:** nested loops `i ∈ [0,N)` and `j ∈ [0,N)`; executes once after kernel.
- **Total threads/iterations:** `N^2` iterations (same as kernel work) with simple integer checks.
- **Type:** A – dense sequential check over contiguous grid.
- **Parent loop:** `main` post-kernel validation
- **Contains:** none.
- **Dependencies:** relies on host copies `input` and `output`; no atomic ops.
- **Shared memory:** N/A.
- **Thread indexing:** standard nested indices `i`, `j`.
- **Private vars:** `i`, `j`, `count`.
- **Arrays:** `input(R)`, `output(R)` host buffers.
- **OMP Migration Issues:** Could be parallelized with OMP reduction onto a flag, but sequential check is acceptable for validation.

## Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|-----------------|------|----------|---------|------------|--------------|------------|
| `cellsXOR` | A | CRITICAL | `__global__` dense stencil (main compute) | `N^2` threads/4 neighbor loads + write | none | boundary checks only (no syncthreads/atomics) |
| Host init loop | A | SECONDARY | sequential setup before kernel | `N^2` writes | RNG | none |
| Host validation loop | A | SECONDARY | sequential post-kernel check | `N^2` comparisons | none | might need parallel flag if converted |

## CUDA-Specific Details
- **Dominant compute kernel:** `cellsXOR` is the only kernel and does the bulk of compute for the NxN grid.
- **Memory transfers in timed loop?:** YES – `cudaMemcpyHostToDevice` before the kernel and `cudaMemcpyDeviceToHost` after it; these transfers bracket the main compute and are part of the timed workload.
- **Shared memory usage:** None; the kernel relies entirely on global device pointers.
- **Synchronization points:** Implicit global barrier at kernel launch/return; no `__syncthreads()` calls.
- **Atomic operations:** None.
- **Reduction patterns:** None (per-thread work is independent).
- **Data management:** host buffers allocated with `new`, device buffers via `cudaMalloc`; `cudaMemcpy` invoked twice; cleanup via `cudaFree`/`delete[]`.

## OMP Migration Strategy Notes
1. **Direct kernel → parallel for:** `cellsXOR` can map directly to a nested OMP parallel-for (two-dimensional grid) with private `count` and boundary guards; each iteration processes one cell.
2. **Requires restructuring:** none – the kernel has no shared memory, atomics, or synchronization primitives that obstruct an OMP translation.
3. **Performance concerns:** ensure the parallel-for handles contiguous neighbor accesses to preserve coalesced-like patterns and avoid false sharing when writing to `output`.
4. **Data management:** host allocations remain on the CPU so `cudaMalloc`/`cudaMemcpy` calls disappear; maintain the RNG seeding before compute and reuse the validation logic (which could itself be parallelized if needed for performance).
