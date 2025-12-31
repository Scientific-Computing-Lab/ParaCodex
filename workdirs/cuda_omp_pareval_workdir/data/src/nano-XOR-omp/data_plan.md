# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| d_input | `cudaMalloc` | `N * N * sizeof(int)` | Host→Device once before kernel launch |
| d_output | `cudaMalloc` | `N * N * sizeof(int)` | Device→Host once after kernel launch |
| input (host) | host array | `N * N * sizeof(int)` | Source of Host→Device transfer |
| output (host) | host array | `N * N * sizeof(int)` | Destination of Device→Host transfer |

**CUDA Operations:**
- cudaMalloc calls: `d_input` and `d_output`, each sized `N*N*sizeof(int)`
- cudaMemcpy H→D: `cudaMemcpy(d_input, input, ... , cudaMemcpyHostToDevice)` (single batch before kernel)
- cudaMemcpy D→H: `cudaMemcpy(output, d_output, ... , cudaMemcpyDeviceToHost)` (single transfer after kernel)
- Kernel launches: `cellsXOR<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N)` (once per execution)

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| `cellsXOR` | 2D grid with `numBlocks` × `threadsPerBlock` covering `N×N` points | once per run | `input`, `output` |

**Kernel Launch Patterns:**
- Launched once from `main`; no outer loop around the launch.
- Sequential kernel (single invocation) so no need for depend/nowait clauses.
- No conditional launch, just single execution with boundary guards inside the kernel.

## OMP Data Movement Strategy
**Chosen Strategy:** A

**Rationale:** Single CUDA kernel with simple Host→Device copy before launch and D→H copy after. The mapped arrays stay alive for the duration of the kernel and are small enough to map directly.

**Device Allocations (OMP equivalent):**
```
// CUDA: cudaMalloc(&d_input, size) + cudaMalloc(&d_output, size)
// OMP Strategy A: the stencil kernel uses `map(to: input[0:N*N])` and `map(from: output[0:N*N])`
// on the `target teams loop`, so the runtime allocates and tracks device storage for each array.
```

**Host→Device Transfers (OMP equivalent):**
```
// CUDA: cudaMemcpy(d_input, input, size, HostToDevice)
// OMP Strategy A: the `target teams loop` uses `map(to: input[0:N*N])`, so input is copied to the device
// before the loop executes, matching the original cudaMemcpy timing.
```
- When: immediately before the kernel execution (first entry into the `target teams loop`).
- Arrays: `input`.
- Total H→D: `N*N*sizeof(int)` (~`4 * N*N` bytes, i.e., ~`4*10^6` bytes for `N=1000`).

**Device→Host Transfers (OMP equivalent):**
```
// CUDA: cudaMemcpy(output, d_output, size, DeviceToHost)
// OMP Strategy A: the `target teams loop` uses `map(from: output[0:N*N])`, so the result is copied
// to the host when the kernel completes.
```
- When: after the stencil kernel completes and before validation.
- Arrays: `output`.
- Total D→H: `N*N*sizeof(int)` (same as H→D).

**Transfers During Iterations:** NO
- There are no intermediate transfers; all data copies happen before and after the single kernel invocation.

## Kernel to OMP Mapping (short)
- Replace the CUDA kernel launch with a nested `#pragma omp target teams loop collapse(2)` over `i` and `j`, adding `map(to: input[0:N*N]) map(from: output[0:N*N])` clauses.
- The original boundary guards remain; each `i/j` iteration computes `count` exactly as before.
- The body uses the same neighbor accesses and writes to `output[i*N + j]`.

## Critical Migration Issues
- `__syncthreads()` usage: not present.
- Shared memory: none.
- Atomics: none.
- Dynamic indexing: handled via standard C indexing and boundary checks.

**__syncthreads() Resolution:** N/A (kernel already independent).

**Shared memory / barriers:** N/A.

## Expected Performance
- CUDA kernel time: (not provided explicitly in analysis) — assume `cellsXOR` dominates and is highly parallel.
- OMP expected: similar order, but CPU overhead for kernel launch mapping may be slightly higher; please run with `OMP_TARGET_OFFLOAD=MANDATORY`.
- Red flag: If OMP offload runs >3× slower, revisit mapping strategy or data transfers.

**Summary:** 1 kernel, 2 device arrays, Strategy A. CUDA pattern: single stencil kernel with host-managed data copies. OMP approach: a `target teams loop collapse(2)` with map clauses replaces the kernel, keeping the data movement semantics equivalent while host validation stays sequential.
