# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| `num_nucs` | cudaMalloc | `12 * sizeof(int)` | Host→Device once inside `move_simulation_data_to_device` |
| `concs` | cudaMalloc | `length_mats * sizeof(double)` | Host→Device once |
| `mats` | cudaMalloc | `length_mats * sizeof(int)` | Host→Device once |
| `unionized_energy_array` | cudaMalloc *(conditional)* | `in.n_isotopes * in.n_gridpoints * sizeof(double)` | Host→Device once when `grid_type == UNIONIZED` |
| `index_grid` | cudaMalloc *(conditional)* | `(grid_type == UNIONIZED) ? n_isotopes * (n_isotopes * n_gridpoints) : hash_bins * n_isotopes` | Host→Device once if non-zero |
| `nuclide_grid` | cudaMalloc | `n_isotopes * n_gridpoints * sizeof(NuclideGridPoint)` | Host→Device once |
| `verification` | cudaMalloc | `in.lookups * sizeof(unsigned long)` | Device-only buffer, copied to host once after kernel loop |
| `p_energy_samples` | cudaMalloc (per optimization path) | `in.lookups * sizeof(double)` | Populated by `sampling_kernel`, stays on device for downstream kernels; not transferred back unless needed for host-side sorting/bucketing |
| `mat_samples` | cudaMalloc (per optimization path) | `in.lookups * sizeof(int)` | Same as above |

**CUDA Operations:**
- `cudaMalloc` calls: movable arrays listed above (permanent: num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification; per-optimization: p_energy_samples, mat_samples)
- `cudaMemcpy` H→D: Copy host grids/structures once in `move_simulation_data_to_device`
- `cudaMemcpy` D→H: Copy `verification` back after timed kernel loop; some optimization paths would also read sampled arrays for host-side `thrust` utilities
- Kernel launches: baseline `xs_lookup_kernel_baseline` is launched `(in.num_iterations + in.num_warmups)` times; each optimization path launches the sampling kernel plus its paired lookup kernels (with occasional nested material loops)

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| `xs_lookup_kernel_baseline` | grid=`ceil(in.lookups / 256)`, block=`256` | Per simulation iteration (warmup + timed) | `GSD.verification`, `GSD.num_nucs`, `GSD.concs`, `GSD.unionized_energy_array`, `GSD.index_grid`, `GSD.nuclide_grid`, `GSD.mats`, RNG helpers |
| `sampling_kernel` | grid=`ceil(in.lookups / 32)`, block=`32` | Once per optimized run | `GSD.p_energy_samples`, `GSD.mat_samples` |
| `xs_lookup_kernel_optimization_{1-6}` | grid=`ceil(workset / 32)`, block=`32` | Varies: some run once, some per material/fuel split | Sample buffers + baseline data |

**Kernel Launch Patterns:**
- Baseline is a simple outer loop that launches the same lookup kernel for every iteration. No host-device transfers happen inside the timed region.
- Optimizations 2–6 embed host loops over materials or fuel/no-fuel categories before launching kernels so there are nested dispatches. Material counters (`n_lookups_per_material`) are computed with `thrust::count` and `thrust::sort` to cluster lookups.

## OMP Data Movement Strategy
**Chosen Strategy:** B (persistent device allocations + explicit kernel order control)

**Rationale:** There are many kernels sharing the same data and material-dependent launch schedules. A `target data` region that bounds the entire simulation with data mapped once is insufficient because we need CUDA-style device pointers for repeated kernel launches and additional helper allocations (`p_energy_samples`, `mat_samples`) that behave like temporary GPU scratch space. Strategy B lets us allocate device buffers once via `omp_target_alloc`, copy the static simulation inputs once (matching the CUDA `move_simulation_data_to_device` step), and reuse those device pointers from every offloaded kernel while preserving the sequential dependencies via sequential host calls (we can insert `nowait`/`depend` annotations if needed for overlapping work, but the existing CUDA code already synchronizes between kernel launches).

**Device Allocations (OMP equivalent):**
```
CUDA: cudaMalloc(&d_arr, size)
OMP Strategy B: d_arr = omp_target_alloc(size, omp_get_default_device());
           omp_target_memcpy(d_arr, h_arr, size, omp_get_default_device(), omp_get_initial_device());
```

**Host→Device Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice)
OMP Strategy B: omp_target_memcpy(d_arr, h_arr, size, omp_get_default_device(), omp_get_initial_device());
```
- When: once during `move_simulation_data_to_device` before the timed loop; `p_energy_samples`/`mat_samples` are also populated on-device via offloaded sampling, so we do not need host transfers there unless a host-side sort/count requires their values.
- Arrays: the static grids/pointers plus the verification buffer (~`in.lookups` entries). Total H→D footprint is roughly `(length_mats * (sizeof(int)+sizeof(double))) + grid sizes + unionized structures`, which is on the order of the original CUDA version's GPU allocation (tens to hundreds of MB depending on `in.lookups` and `in.n_gridpoints`).

**Device→Host Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(SD.verification, GSD.verification, size, cudaMemcpyDeviceToHost)
OMP Strategy B: omp_target_memcpy(SD.verification, GSD.verification, size, omp_get_initial_device(), omp_get_default_device());
```
- When: once per simulation (after kernels complete) for the verification buffer. Optimization paths that rely on sorted/partitioned samples will copy `mat_samples`/`p_energy_samples` back, rearrange them on the host, and push the reordered data back to the device before launching the lookup kernels.
- Arrays: `verification` (size `in.lookups * sizeof(unsigned long)`), potentially `mat_samples`/`p_energy_samples` for host-side material/fuel bucket creation. Total D→H is on the order of the lookup buffers when those optimizations run, but baseline runs only move `verification`.

**Transfers During Iterations:** NO – once the data is staged on-device, we reuse it for every iteration. The only transfers inside the timed region are those implicit in `omp_target_memcpy` when we need to refresh `mat_samples`/`p_energy_samples` for host sorting in optimizations, but even these can happen before the timed window if we control timing carefully.

## Kernel to OMP Mapping (short)
- The CUDA kernels become functions that contain `#pragma omp target teams loop is_device_ptr(...)` over the logical work domain (`i` from `0` to `lookups` or material-specific ranges).
- All `blockIdx/threadIdx` indexing turns into an explicit `for (int i = 0; i < n; ++i)` wrapped by the target teams loop.
- RNG helper functions (`fast_forward_LCG`, `LCG_random_double`, `pick_mat`) and grid/search utilities need `#pragma omp declare target` so they can run inside the target region, while still being callable on the host for helper work.
- Kernel-private arrays (e.g., `macro_xs_vector[5]`) become standard stack locals inside the loop body.

## Critical Migration Issues
**From analysis.md "OMP Migration Issues":**
- `__syncthreads()`: not present in these kernels, so no additional barriers are necessary beyond the natural completion of each `target teams loop` region.
- Shared memory: none used; register arrays remain local variables.
- Atomics: none in the CUDA version; OpenMP does not need atomic directives for the per-index writes to `GSD.verification`.
- Dynamic indexing: `calculate_macro_xs` iterates `num_nucs[mat]` per lookup and uses binary search – these loops stay inside the offloaded region.

**__syncthreads() Resolution:** Not applicable.

**Shared memory / barriers:** No conversion required.

## Expected Performance
- CUDA kernel time: the original baseline observer run is captured in `baseline_output.txt`, which can be used as a reference for `xs_lookup_kernel_baseline` throughput.
- OMP expected: target teams loops over the same `lookups` domain should yield similar order-of-magnitude throughput on the RTX 4060 once data sits on device; the extra `omp_target_alloc`/`omp_target_memcpy` overhead is paid only once before timing starts.
- Red flag: If offload throughput ends up >3× slower than the reference, revisit the per-kernel scheduling clauses or consider distributing the material-dependent paths with `nowait`/`depend`.

**Summary:** 5 lookup kernels (baseline + four opt variants) over `in.lookups` samples, several supporting device buffers, Strategy B. CUDA pattern: host staging step + repeated kernel launches + `thrust` utilities. OMP approach: allocate device buffers via `omp_target_alloc`, reuse them inside `target teams loop`s, minimize host<->device transfers, and perform sorting/count operations on the host with explicit memcpy when necessary. Expected H→D ~ tens of MB (dominated by grids and material arrays), D→H ~ `in.lookups * sizeof(unsigned long)` (~8 bytes per lookup).
