# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| `d_input` | `cudaMalloc` | `N * N * sizeof(int)` | H→D once
| `d_output` | `cudaMalloc` | `N * N * sizeof(int)` | D→H once
| `input` | host array | `N * N * sizeof(int)` | source
| `output` | host array | `N * N * sizeof(int)` | destination

**CUDA Operations:**
- cudaMalloc calls: `d_input` and `d_output`, each allocate `N * N * sizeof(int)` bytes.
- cudaMemcpy H→D: `input` → `d_input` once before the kernel.
- cudaMemcpy D→H: `d_output` → `output` once after the kernel.
- Kernel launches: `cellsXOR<<<grid, block>>>` executed once.

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| `cellsXOR` | `grid = ((N + blockEdge - 1)/blockEdge, (N + blockEdge - 1)/blockEdge)` / `block = (blockEdge, blockEdge)` | once | `d_input`, `d_output` |

**Kernel Launch Patterns:**
- One time launch from `main`; no outer loops or repeated launches.
- No conditional branching around the kernel.

## OMP Data Movement Strategy
**Chosen Strategy:** A (target data region)

**Rationale:** Single kernel with one-shot transfers matches Strategy A. We can map both input/output buffers for the whole execution and offload the `cellsXOR` work through a `target teams loop` without persistent opaque device pointers.

**Device Allocations (OMP equivalent):**
```
#pragma omp target data map(to: input[0:N*N]) map(from: output[0:N*N])
```
The mapped regions allocate buffers on the target device implicitly for the lifetime of the target data region.

**Host→Device Transfers (OMP equivalent):**
- Mechanism: `map(to: input[0:N*N])` inside the `target data` region.
- When: once before invoking the offloaded compute region.
- Arrays: `input`.
- Total H→D: `N * N * sizeof(int)` bytes (~`4 * N * N` bytes).

**Device→Host Transfers (OMP equivalent):**
- Mechanism: `map(from: output[0:N*N])` inside the `target data` region.
- When: automatically performed at region exit after the offload completes.
- Arrays: `output`.
- Total D→H: `N * N * sizeof(int)` bytes (~`4 * N * N` bytes).

**Transfers During Iterations:** NO (all transfers happen outside the compute loop).

## Kernel to OMP Mapping (short)
- Wrap `cellsXOR` body in a host function that emits `#pragma omp target teams loop collapse(2)` so each `(i,j)` pair executes on the device.
- Replace CUDA `i,j` index derivation with the natural two nested loops over `i` and `j`.
- Use the mapped `input`/`output` pointers directly inside the target loop.

## Critical Migration Issues
**From analysis.md "OMP Migration Issues":**
- `__syncthreads()` usage: NONE.
- Shared memory: NONE.
- Atomics: NONE.
- Dynamic indexing: handled naturally by the collapse(2) loops.

**__syncthreads() Resolution:** Not applicable.

**Shared memory / barriers:** Not applicable.

## Expected Performance
- CUDA kernel time: baseline from `baseline_output.txt` (single launch over `N × N`).
- OMP expected: similar memory-bound behavior, targeting the NVIDIA GeForce RTX 4060 (Ada Lovelace) GPU reported in `system_info.txt`; expect comparable memory throughput when `OMP_TARGET_OFFLOAD=MANDATORY` forces GPU execution.
- Red flag: >3× slowdown would indicate missed parallelism or an offload issue.

**Summary:** 1 kernel, 2 device arrays, Strategy A. CUDA pattern: single dense kernel with host-side initialization/validation and one pair of symmetric transfers. OMP approach: `target data` maps the inputs/outputs for the whole run and uses a `target teams loop collapse(2)` to reproduce the per-cell stencil. Expected: ~`4*N*N` bytes H→D and D→H each.
