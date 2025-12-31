# File Conversion Mapping
- `golden_labels/src/nano-XOR-cuda/nanoXOR.cu → data/src/nano-XOR-omp/nanoXOR.cpp`

# Kernel/Loop Nesting Structure
- `main` (golden_labels/src/nano-XOR-cuda/nanoXOR.cu:41) sets up host arrays, copies them to the device, and launches the compute kernel once
  └── `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)` (golden_labels/src/nano-XOR-cuda/nanoXOR.cu:78‑81) Type A, CRITICAL single-kernel sweep over the NxN grid

# Kernel/Loop Details
## Kernel/Loop: `cellsXOR` at golden_labels/src/nano-XOR-cuda/nanoXOR.cu:21
- **Context:** `__global__` kernel
- **Launch config:** `(ceil(N/blockEdge) × ceil(N/blockEdge))` blocks × `(blockEdge × blockEdge)` threads (lines 78‑80), covering the full NxN domain
- **Total threads/iterations:** ≈ `N × N` threads, one per grid cell (each thread writes exactly one output slot)
- **Type:** Type A – dense 2D neighbor scan on a regular grid, no divergence or irregular indexing
- **Parent loop:** none (single launch from `main` line 81)
- **Contains:** no device-side loops, just per-thread neighbor checks
- **Dependencies:** none (__syncthreads and atomics absent)
- **Shared memory:** NO
- **Thread indexing:** `i = blockIdx.y*blockDim.y + threadIdx.y`, `j = blockIdx.x*blockDim.x + threadIdx.x` (lines 22‑24); each thread checks up to four neighbors
- **Private vars:** `i`, `j`, `count`
- **Arrays:** `input` (R, device global) read-only, `output` (W, device global) written once per thread; host copies via `cudaMemcpy` before/after kernel (lines 76, 83)
- **OMP Migration Issues:** direct parallel-for candidate; must replace CUDA allocations/transfers with host buffers and ensure the same boundary checks on the CPU

# Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|----------------|------|----------|---------|------------|--------------|------------|
| `cellsXOR` | A | CRITICAL | `__global__` kernel | `N²` threads (one per cell) | none | Replace CUDA allocations and cudaMemcpy pairs with host-managed arrays before/after the parallel loop |

# CUDA-Specific Details
- **Dominant compute kernel:** `cellsXOR` (lines 21‒31) performs the only CUDA work and drives the runtime
- **Memory transfers in timed loop?:** YES – host→device copy of `input` before launch (line 76) and device→host copy of `output` after the kernel (line 83)
- **Shared memory usage:** none (no `__shared__` declarations)
- **Synchronization points:** none (no `__syncthreads()`)
- **Atomic operations:** none
- **Reduction patterns:** none, each thread independently computes one cell without cross-thread reductions
- **Data management:** host arrays allocated via `new[]`, device buffers via `cudaMalloc` (lines 63‒75), and cleaned up with `delete[]`/`cudaFree` (lines 34‒39, 110)

# OMP Migration Strategy Notes
- **Direct kernel → parallel for:** `cellsXOR` can be replaced with a nested `#pragma omp parallel for collapse(2)` loop over `i` and `j`, reusing the same neighbor conditionals and writing directly into the shared `output` buffer
- **Requires restructuring:** the explicit `cudaMalloc`/`cudaMemcpy` pair becomes a single host allocation when migrating to OpenMP; the boundary checks stay in the parallel loop, so no extra staging is needed
- **Performance concerns:** memory-bound neighbor loads still dominate, but OpenMP suffers less from kernel launch overhead; ensure the host `output` buffer is zero-initialized or updated safely to avoid false sharing (use `collapse(2)` or tile the loops)
- **Data management:** allocate `input`/`output` in host memory once, run the OpenMP loop in place, and keep the validation/ checksum logic unchanged (no additional copies needed)
