# Data Management Plan

## OMP Target Memory Overview
This plan documents how the current OpenMP target port mirrors the CUDA data movements described in the original analysis. The port relies on `SimulationData move_simulation_data_to_device` to `omp_target_alloc` the major arrays once and `omp_target_memcpy` their contents onto the default GPU device. Device-only buffers (`verification`, `p_energy_samples`, `mat_samples`) also live on the target for the duration of each simulation run.

| Array/Pointer | OMP Allocation | Size | Data Movement |
|---------------|----------------|------|---------------|
| `GSD.num_nucs` | `allocate_and_copy` → `omp_target_alloc`/`omp_target_memcpy` | `length_num_nucs` ints | host→device once via `move_simulation_data_to_device` |
| `GSD.concs` | same pattern | `length_concs` doubles | host→device once |
| `GSD.mats` | same pattern | `length_mats` ints | host→device once |
| `GSD.unionized_energy_array` | conditional `allocate_and_copy` | `length_unionized_energy_array` doubles | host→device when needed |
| `GSD.index_grid` | conditional `allocate_and_copy` | `length_index_grid` ints | host→device when needed |
| `GSD.nuclide_grid` | `allocate_and_copy` | `length_nuclide_grid` `NuclideGridPoint` | host→device once |
| `GSD.verification` | `omp_target_alloc` | `in.lookups * sizeof(unsigned long)` | device buffer read back after kernels via `omp_target_memcpy` |
| `GSD.p_energy_samples` | `omp_target_alloc` | `in.lookups * sizeof(double)` | written on-device; host shim copies out/in around sorting |
| `GSD.mat_samples` | `omp_target_alloc` | `in.lookups * sizeof(int)` | same as `p_energy_samples` |

**Device operations:**
- `omp_target_alloc` handles persistent allocations for all device arrays.
- `omp_target_memcpy` copies host simulation data into the `GSD` structure before any kernels run.
- Sorting/partition logic intentionally pulls `p_energy_samples`/`mat_samples` back to the host, operates on the host arrays, and pushes the reordered data back to the device when needed.
- `release_device_memory` frees each target allocation after the simulation completes.

## Kernel Inventory

| Kernel Function | OpenMP Mapping | Frequency | Arrays Used |
|-----------------|----------------|-----------|-------------|
| `sampling_kernel` | `#pragma omp target teams loop is_device_ptr(...)` | once per optimized path | `p_energy_samples`, `mat_samples` |
| `xs_lookup_kernel_baseline` | `#pragma omp target teams loop is_device_ptr(...)` | `nwarmups + num_iterations` times | core `GSD` arrays + `verification` |
| `xs_lookup_kernel_optimization_*` | same `target teams loop` | varies per optimization | sample buffers + core arrays |
| Random/search helpers | `#pragma omp declare target` | invoked per lookup inside kernels | host/global constants (no transfers) |

**Kernel launch details:**
- Each kernel uses a teams/loop construct with `is_device_ptr` so that the mapped device pointers are used directly.
- The physical launch grid simplifies to a single `for (int i = 0; i < n; ++i)` in the teams loop, so bounds checks guard the active work.
- No kernels use `distribute parallel for`, matching the directive constraint.

## OMP Data Movement Strategy

**Chosen Strategy:** Strategy B – persistent device allocation with sequential offloads.

**Rationale:** The simulation repeatedly reuses the same large `SimulationData` arrays across multiple kernels (baseline plus the optimization variants) and hosts frequent serial loops (material loops, sorting/partition). Keeping those buffers allocated on the target and issuing `#pragma omp target teams loop` for each compute kernel mirrors the CUDA pattern of `cudaMalloc`/`cudaMemcpy` once with multiple kernel launches.

### Host → Device Transfers
```
SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
```
- `allocate_and_copy` handles the repeated `omp_target_memcpy` for the read-only arrays.
- `verification`, `p_energy_samples`, and `mat_samples` are allocated via `omp_target_alloc` and then written entirely on the target.

### Device → Host Transfers
- After each simulation path finishes, `omp_target_memcpy(SD.verification, GSD.verification, ...)` copies the verification array back to the host before accumulating the checksum.
- Sort/partition helpers move samples (`p_energy_samples`, `mat_samples`) back to the host only when Thrust equivalents (count/sort/partition) are required; they are immediately copied back afterward.

### Transfers During Iterations
- All kernels read/write data that resides on the target buffer provided by `SimulationData GSD`. No additional transfers occur inside the timed loop; the only explicit transfers happen before/after the compute kernels (matching the baseline).

## Critical Migration Issues
- `__syncthreads` / CUDA shared memory: none present. All temporary buffers live inside the teams loop.
- Atomics: none (reduction uses `verification` writes followed by host `std::accumulate`).
- Struct pointer arguments: `is_device_ptr` clauses only reference plain pointers extracted from `SimulationData`, preventing prohibited struct member access.
- `omp_target_memcpy` handles host-device coherence when sample arrays are reordered.

## Expected Performance Observations
- CUDA baseline output is available in `baseline_output.txt`; the OpenMP port should produce the same verification scalar with `OMP_TARGET_OFFLOAD=MANDATORY`.
- Sorting/partition steps now execute on the host, so their cost will be serialized but limited to `O(lookups)` extra work.
- Red flags: if `gpu_output.txt` differs from the baseline or if the run reports `OMP: no devices available`, revisit the data movement strategy to ensure the mapped arrays exist before each kernel.

**Summary:** 6 compute kernels backed by the same `SimulationData` arrays, Strategy B with persistent target allocations, host-side sorting for Thrust replacements, and sequential `#pragma omp target teams loop` kernels for each simulation path.
