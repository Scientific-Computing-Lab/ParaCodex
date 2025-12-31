# microXOR Loop Classification

## File Conversion Mapping
- `golden_labels/src/microXOR-cuda/main.cu` -> `data/src/microXOR-omp/main.cpp`
- `golden_labels/src/microXOR-cuda/microXOR.cu` -> `data/src/microXOR-omp/microXOR.cpp`

## Kernel/Loop Nesting Structure
- main compute flow (`golden_labels/src/microXOR-cuda/main.cu:35-90`)
  - host initialization loop (`golden_labels/src/microXOR-cuda/main.cu:40-42`)
  - kernel launch `cellsXOR<<<numBlocks, threadsPerBlock>>>` (`golden_labels/src/microXOR-cuda/main.cu:50-54`) Type G
  - validation loops (`golden_labels/src/microXOR-cuda/main.cu:65-88`)

## Kernel/Loop Details

### Kernel/Loop: `cellsXOR` at `golden_labels/src/microXOR-cuda/microXOR.cu:21`
- **Context:** `__global__` CUDA kernel
- **Launch config:** grid `((N + blockEdge - 1)/blockEdge)^2` x block `blockEdge x blockEdge`
- **Total threads/iterations:** approx `NxN` threads (each executes constant neighbor checks)
- **Type:** G - stencil-style neighbor access (each thread reads up to 4 neighbors around its cell)
- **Parent loop:** none (launched once from `main`)
- **Contains:** no explicit `for` loops; relies on thread space to cover the grid
- **Dependencies:** independent threads, no atomics or reductions
- **Shared memory:** NO; all accesses use globals
- **Thread indexing:** `i = blockIdx.y * blockDim.y + threadIdx.y`, `j = blockIdx.x * blockDim.x + threadIdx.x`
- **Private vars:** `i`, `j`, `count`
- **Arrays:** `input` (read-only grid in global memory), `output` (write target grid in global memory)
- **OMP Migration Issues:** none beyond the usual boundary checks; data accesses are coalesced and independent (`__syncthreads` not used)

### Kernel/Loop: initialization loop at `golden_labels/src/microXOR-cuda/main.cu:35-42`
- **Context:** host setup loop filling the `input` grid with random bits
- **Launch config:** sequential `for (size_t i = 0; i < N * N; i++)`
- **Total threads/iterations:** `N x N`
- **Type:** A - dense linear work over contiguous array
- **Parent loop:** main (single pass)
- **Contains:** single loop body with random number generator call and store
- **Dependencies:** none; each iteration writes disjoint elements of `input`
- **Shared memory:** not applicable
- **Thread indexing:** sequential on host
- **Private vars:** `i`, RNG state (`rd`, `gen`, `dis`)
- **Arrays:** `input` (write, host)
- **OMP Migration Issues:** straightforward parallelization; iterations are independent so a `parallel for` could replace this loop if needed for speed

### Kernel/Loop: validation nested loops at `golden_labels/src/microXOR-cuda/main.cu:65-88`
- **Context:** host verification loops comparing `input` neighbors to `output`
- **Launch config:** nested loops `for (i = 0; i < N; i++)` and `for (j = 0; j < N; j++)`
- **Total threads/iterations:** `N x N`
- **Type:** A - dense scan over the 2D grid with local checks
- **Parent loop:** main (post-kernel validation phase)
- **Contains:** conditional boundary checks that read `input` neighbors and output values
- **Dependencies:** each iteration reads a disjoint `output[i*N + j]` and neighbors from `input`
- **Shared memory:** not applicable
- **Thread indexing:** sequential host loops, but iteration-space independence allows parallelism
- **Private vars:** `i`, `j`, `count`
- **Arrays:** `input` (read), `output` (read)
- **OMP Migration Issues:** could be parallelized with care to ensure `cleanup`/reporting happens once per failure; currently sequential for deterministic validation

## Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|-----------------|------|----------|---------|------------|--------------|------------|
| `cellsXOR` | G | CRITICAL | `__global__` kernel (`microXOR.cu:21`) | ~`N^2` threads | none | none |
| Input initialization loop | A | SECONDARY | host setup loop (`main.cu:40-42`) | `N^2` | none | none (loop iterations independent) |
| Validation nested loops | A | SECONDARY | host verification (`main.cu:65-88`) | `N^2` | reads input/output neighbors | none (could parallelize with reduction-style failure checks) |

## CUDA-Specific Details
- **Dominant compute kernel:** `cellsXOR` (`microXOR.cu:21`) captures almost all of the runtime; it maps each thread to one grid cell and only reads neighbors.
- **Memory transfers in timed loop?:** YES - host-to-device copy (`main.cu:48`) and device-to-host copy (`main.cu:55`) both happen once per invocation of `main`.
- **Shared memory usage:** none; kernel accesses global arrays directly.
- **Synchronization points:** none; kernel does not use `__syncthreads()` or other explicit synchronization.
- **Atomic operations:** none; each thread writes a unique output slot.
- **Reduction patterns:** none; no scalar accumulations via atomics or reduction clauses.
- **Memory management:** host allocates `input`/`output` via `new[]`, device buffers via `cudaMalloc` (`main.cu:44-46`), and frees them in `cleanup`.
- **Thread/Block geometry:** `threadsPerBlock(blockEdge, blockEdge)` (blockEdge in [2,32]) and grid dimensions ensure a full 2D mapping of the `NxN` space with ceil division.
- **Data lifetime:** `input` seeded on host, copied to device once, kernel computes `output`, and host copy returns results for validation/GATE checksum.

## OMP Migration Strategy Notes
- **Direct kernel -> parallel for:** The stencil kernel only needs neighbor reads and writes to unique output cells, so it can be rewritten as a 2D `#pragma omp parallel for collapse(2)` over the `i, j` indices with the same boundary checks; each thread block essentially becomes an outer chunk of the iteration space.
- **Requires restructuring:** none; there is no shared memory, atomic, or synchronization dependency that would force restructuring beyond expressing the grid bounds in host loops.
- **Performance concerns:** the data is already coalesced (row-major) and the kernel does constant work per cell; the primary CPU concern is the lack of temporal reuse so any OpenMP version should also aim to process contiguous rows for cache efficiency.
- **Data management:** Replace CUDA allocations/copies with host-only buffers; `input`/`output` already reside on host so the OpenMP version can reuse them directly. Remove `cudaMemcpy`/`cudaMalloc`/`cudaFree` calls and keep the `GATE_CHECKSUM_BYTES` validation unchanged.
