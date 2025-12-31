# microXOR CUDA loop classification for OMP migration

## File Conversion Mapping
- `golden_labels/src/microXOR-cuda/main.cu` → `data/src/microXOR-omp/main.cpp` (C++ driver with CUDA kernels intact)
- `golden_labels/src/microXOR-cuda/microXOR.cu` → `data/src/microXOR-omp/microXOR.cpp` (CUDA kernel definition)
- `golden_labels/src/microXOR-cuda/include/microXOR.cuh` → `data/src/microXOR-omp/include/microXOR.h` (header describing the CUDA kernel)

## Kernel/Loop Nesting Structure
- `main()` driver (main.cpp:12-89)
  ├── host initialization loop (main.cpp:34-41) – fills the `input` grid with random bits before any CUDA work begins
  ├── `cellsXOR<<<numBlocks, threadsPerBlock>>>` (main.cpp:49-53) – launches the sole CUDA kernel (Type A, CRITICAL)
  └── host validation loops (main.cpp:63-85) – scans `output` to confirm exactly-one-neighbor rule
- `cellsXOR` kernel (microXOR.cpp:21-32) – each thread computes the neighbor count for one `(i,j)` cell and writes either `1` or `0`

## Kernel/Loop Details

### Kernel/Loop: cellsXOR at `data/src/microXOR-omp/microXOR.cpp:21`
- **Context:** `__global__` CUDA kernel launched once from `main()` (no additional host loops iterate over the launch)
- **Launch config:** grid `(ceil(N/blockEdge), ceil(N/blockEdge))`, block `(blockEdge, blockEdge)` set by `dim3` objects; launch covers the entire `N×N` grid
- **Total threads/iterations:** ≈ `N^2` threads (`gridDim.x * gridDim.y * blockDim.x * blockDim.y`) with each thread touching one cell
- **Type:** A – dense, regular 2D neighbor stencil with uniform work per thread and no irregular indexing
- **Parent loop:** none (single launch from `main()` before validation)
- **Contains:** no device-internal loops beyond the constant-time neighbor checks
- **Dependencies:** none (no `__syncthreads`, no atomics, no reductions)
- **Shared memory:** NO (kernel only reads/writes global memory)
- **Thread indexing:** maps thread `(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x)` to `(i,j)` in the grid
- **Private vars:** `int i`, `int j`, `int count`
- **Arrays:** `input` (R) and `output` (W) – device global memory allocated with `cudaMalloc`; kernel touches neighboring cells through strided accesses but without indirection
- **OMP Migration Issues:** main challenge is removing the CUDA launch/`,cudaMemcpy`/`cudaMalloc` scaffolding – once data resides on the host, this kernel becomes a straight nested `for` (parallel-for-able) over `i`/`j` with no additional synchronization or atomics

### Kernel/Loop: Input initialization loop at `data/src/microXOR-omp/main.cpp:34-41`
- **Context:** host setup loop in `main()` that seeds the input grid before any device calls
- **Launch config:** not applicable (serial host loop)
- **Total threads/iterations:** `N^2` (fills every slot in the flattened `input` array)
- **Type:** A – dense host-side work writing `input[k]` in row-major order
- **Parent loop:** `main()` entry path
- **Contains:** no nested host loops beyond the single index increment
- **Dependencies:** depends on `std::random_device`, `std::mt19937`, and `std::uniform_int_distribution` for randomness
- **Shared memory:** NO
- **Thread indexing:** uses `size_t i` iterating from `0` to `N*N-1`
- **Private vars:** `i`, RNG objects (`rd`, `gen`, `dis`)
- **Arrays:** `input` (W) – host heap allocation via `new int[N*N]`
- **OMP Migration Issues:** to parallelize, RNG must be made thread-safe (per-thread `std::mt19937` seeding or deterministic streams) and care taken to avoid false sharing when writing sequential slots; otherwise this loop can remain serial as it runs just once

### Kernel/Loop: Validation nested loops at `data/src/microXOR-omp/main.cpp:64-85`
- **Context:** host verification pass that mirrors the stencil logic to check the computed `output`
- **Launch config:** host loops over `i` and `j` covering the full `N×N` grid
- **Total threads/iterations:** `N^2` (outer loop `i` and inner loop `j` multiply to `N×N` comparisons)
- **Type:** A – dense read-only scan with predictable neighbor accesses
- **Parent loop:** `main()` (follows kernel launch)
- **Contains:** inner loop with `j` stepping faster than `i`
- **Dependencies:** sequential dependencies only come from scalar `count`; each iteration inspects contiguous `input` entries
- **Shared memory:** NO
- **Thread indexing:** host indices `i` and `j` map to grid coordinates
- **Private vars:** `int count`, `size_t i`, `size_t j`
- **Arrays:** `input` (R) and `output` (R) – host memory used for validation
- **OMP Migration Issues:** there is no CUDA-specific complexity, but if the validation is parallelized a reduction can verify the entire grid; currently the checks run serially and report the first failure

## Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|-----------------|------|----------|---------|------------|--------------|------------|
| `cellsXOR` | A | CRITICAL | CUDA kernel (microXOR.cpp:21) | ≈ `N^2` threads | none | Need to remove GPU launch / `cudaMemcpy` scaffolding; otherwise a nested `for` is readable |
| Host initialization loop | A | SECONDARY | host loop (main.cpp:34) | `N^2` iterations | RNG state | RNG needs per-thread streams if parallelized |
| Validation nested loops | A | SECONDARY | host loops (main.cpp:64) | `N^2` iterations | scalar count | no CUDA dependencies; can stay serial or become `parallel for` |

## CUDA-Specific Details
- **Dominant compute kernel:** `cellsXOR` (microXOR.cpp:21-32) – the only kernel responsible for the NxN stencil work
- **Memory transfers in timed loop?:** NO – only one host-to-device copy before the kernel and one device-to-host copy after, so the main computation loop is clean
- **Shared memory usage:** NONE
- **Synchronization points:** None inside the kernel (`__syncthreads()` not used); global sync via kernel boundary only
- **Atomic operations:** NONE
- **Reduction patterns:** NONE (per-thread count is local)
- **Thread indexing:** Each thread computes `i = blockIdx.y * blockDim.y + threadIdx.y`, `j = blockIdx.x * blockDim.x + threadIdx.x`
- **Memory types:** `input`/`output` allocated via `cudaMalloc` (device global); host counterparts created with `new` and copied via `cudaMemcpy`
- **CUDA APIs to replace:** `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cellsXOR<<<>>>`, `dim3` + `blockIdx/threadIdx/gridDim` usage; no shared memory, no atomics, no warp intrinsics

## OMP Migration Strategy Notes
- **Direct kernel → parallel `for`:** `cellsXOR` maps to `#pragma omp parallel for collapse(2)` over `i` and `j` once the arrays live on the host; neighbor count logic remains identical
- **Requires restructuring:** GPU memory management needs to be removed (allocate host buffers once, drop `cudaMemcpy`/`cudaFree`) and the driver must treat `input`/`output` as plain host arrays or `std::vector`
- **Performance concerns:** The RNG initialization loop is also dense (`N^2`), so if the OMP version parallelizes it each thread needs its own PRNG stream to avoid contention; false sharing is unlikely with linear indices, but alignment matters if scaling to large `N`
- **Data management:** Keep the validation scan and kernel logic sharing the same host buffers; ensure `N` validation (divisibility by `blockEdge`) is preserved as a correctness guard
