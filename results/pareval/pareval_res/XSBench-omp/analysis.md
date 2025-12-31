# File Conversion Mapping
Simulation.cu → Simulation.c
- copied the CUDA implementation verbatim, renaming the file suffix for the OMP workspace and redirecting the include to `XSbench_header.h` so the converted file resolves the headers that already live beside the C files.

# Kernel/Loop Nesting Structure
- run_event_based_simulation_baseline iteration loop (Simulation.cu:14-22) repeatedly launches the dense lookup kernel for every iteration + warmup; the loop runs for `in.num_iterations + in.num_warmups` timesteps.
  └── xs_lookup_kernel_baseline<<<ceil(in.lookups/256),256>>> (Simulation.cu:41-) Type A (dense per-lookup work)
      └── device-side accumulation inside `calculate_macro_xs` (Simulation.cu:156-189) over `num_nucs[mat]` entries and the constant five-entry `macro_xs_vector` reduction.
- run_event_based_simulation_optimization_1 host path (Simulation.cu:304-347)
  ├── sampling_kernel<<<ceil(in.lookups/32),32>>> (Simulation.cu:348-365) Type A (per-lookup RNG)
  └── xs_lookup_kernel_optimization_1<<<ceil(in.lookups/32),32>>> (Simulation.cu:367-402) Type A (reuse sampled batches)
- run_event_based_simulation_optimization_2 host loop (Simulation.cu:407-449)
  ├── sampling_kernel<<<ceil(in.lookups/32),32>>> (Simulation.cu:348-365) Type A
  └── Material dispatch loop `for(m=0; m<12; m++)` (Simulation.cu:438-440) that relaunches xs_lookup_kernel_optimization_2 for each material order
      └── xs_lookup_kernel_optimization_2<<<ceil(in.lookups/32),32>>> (Simulation.cu:452-507) Type A with early `mat != m` exit
- run_event_based_simulation_optimization_4 host workflow (Simulation.cu:586-641)
  ├── sampling_kernel<<<ceil(in.lookups/32),32>>> (Simulation.cu:348-365)
  ├── `thrust::count` loop over 12 materials (Simulation.cu:617-620)
  ├── `thrust::sort_by_key` to cluster lookups (Simulation.cu:621)
  └── Material kernel loop `for(m=0; m<12; m++)` launching xs_lookup_kernel_optimization_4 with `n_lookups_per_material[m]` (Simulation.cu:623-631)
      └── xs_lookup_kernel_optimization_4<<<ceil(n_lookups_per_material[m]/32),32>>> (Simulation.cu:643-691)
- run_event_based_simulation_optimization_5 path (Simulation.cu:697-748)
  ├── sampling_kernel<<<ceil(in.lookups/32),32>>>
  ├── `thrust::count` for fuel lookups + `thrust::partition` to split fuel/non-fuel (Simulation.cu:724-731)
  ├── xs_lookup_kernel_optimization_5<<<ceil(n_fuel_lookups/32),32>>> (Simulation.cu:732-737)
  └── xs_lookup_kernel_optimization_5<<<ceil((in.lookups-n_fuel_lookups)/32),32>>> after offset (Simulation.cu:743-749)
- run_event_based_simulation_optimization_6 host workflow (Simulation.cu:792-854)
  ├── sampling_kernel<<<ceil(in.lookups/32),32>>>
  ├── `thrust::count` over materials and `thrust::sort_by_key` (Simulation.cu:823-833)
  ├── secondary sort per material slice (Simulation.cu:829-834)
  └── Material kernel loop for xs_lookup_kernel_optimization_4 (Simulation.cu:836-843)

# Kernel/Loop Details
## Kernel/Loop: run_event_based_simulation_baseline host loop (Simulation.cu:14)
- **Context:** Host simulation driver iterating `in.num_iterations + in.num_warmups` times and timing only the post-warmup kernels.
- **Launch config:** `nthreads=256`, `nblocks=ceil(in.lookups/256)`; launched once per iteration and warmup.
- **Total threads/iterations:** `(in.num_iterations + in.num_warmups) × ceil(in.lookups/256) × 256` potential threads, though the last block lamps the valid subset.
- **Type:** A – dense per-lookup work executed every simulation iteration.
- **Parent loop:** none; called directly from `main` when `kernel_id == 0`.
- **Contains:** `xs_lookup_kernel_baseline` for every iteration.
- **Dependencies:** only the kernel launches and the timing window (calls `cudaDeviceSynchronize` around warmup boundaries).
- **Shared memory:** NO.
- **Thread indexing:** not applicable (host loop), but the nested kernel uses `i = blockIdx.x * blockDim.x + threadIdx.x`.
- **Private vars:** iteration counter `i`, warmup flag, local timing footprint.
- **Arrays:** none beyond the `Inputs in`, `SimulationData SD`, and the profile pointer; all input data was already staged on the device via `move_simulation_data_to_device` before entering the loop.
- **OMP Migration Issues:** simple outer loop whose iterations are independent except for the final reduction on `SD.verification`; an OpenMP parallel for can map directly, but care must be taken to keep the verification tally in a thread-safe reduction before writing to `profile`.

## Kernel/Loop: xs_lookup_kernel_baseline (Simulation.cu:41)
- **Context:** __global__ kernel invoked from the baseline host loop, executes once per `in.lookups` sample per iteration.
- **Launch config:** `gridDim.x = ceil(in.lookups/256)`, `blockDim.x = 256`; grid-stride loops are not used, so work-per-thread is one lookup.
- **Total threads/iterations:** `ceil(in.lookups/256) × 256 ≈ in.lookups` per host iteration; repeated `(in.num_iterations + in.num_warmups)` times.
- **Type:** A – dense per-lookup work with 1:1 mapping from threads to samplings.
- **Parent loop:** `run_event_based_simulation_baseline` iteration loop (Simulation.cu:14).
- **Contains:** per-thread RNG (`fast_forward_LCG`, `LCG_random_double`, `pick_mat`), `calculate_macro_xs` (Simulation.cu:156-189) whose main loop visits `num_nucs[mat]` entries, and the fixed five-entry reduction that chooses the max.
- **Dependencies:** no atomics, no shared memory, no inter-thread sync (just device-local RNG seeds and global reads).
- **Shared memory:** NO.
- **Thread indexing:** `int i = blockIdx.x * blockDim.x + threadIdx.x`; each thread guards `i < in.lookups`.
- **Private vars:** `seed`, `p_energy`, `mat`, `macro_xs_vector[5]`, `max`, `max_idx`.
- **Arrays:** device-resident `GSD.num_nucs`, `GSD.concs`, `GSD.unionized_energy_array`, `GSD.index_grid`, `GSD.nuclide_grid`, `GSD.mats`, and the verification buffer `GSD.verification`; these are set up on the GPU via `move_simulation_data_to_device`.
- **OMP Migration Issues:** `calculate_macro_xs` iterates `num_nucs[mat]` times (variable per material) and invokes `calculate_micro_xs`/`grid_search` (binary search loops) and per-thread RNG; mapping this to OpenMP will require each `i`-iteration to duplicate the RNG pipeline plus handle the irregular inner loop lengths and global data references without `__syncthreads`, but no atomic hazards exist when writing `verification[i]`.

## Kernel/Loop: sampling_kernel (Simulation.cu:348)
- **Context:** Device kernel that pre-populates `GSD.p_energy_samples` and `GSD.mat_samples`; invoked once before lookup kernels in every optimization path (IDs 1-6).
- **Launch config:** `nthreads=32`, `nblocks=ceil(in.lookups/32)`.
- **Total threads/iterations:** `ceil(in.lookups/32) × 32 ≈ in.lookups` threads, executed once per simulation run in the optimization paths.
- **Type:** A – dense sampling per lookup.
- **Parent loop:** `run_event_based_simulation_optimization_#` functions (Simulation.cu:304, 407, 586, 697, 792).
- **Contains:** RNG that calls `fast_forward_LCG` (while loop log n), `LCG_random_double`, and `pick_mat` (nested loop over 12 materials with cumulative sums).
- **Dependencies:** none, aside from device RNG helper functions.
- **Shared memory:** NO.
- **Thread indexing:** same global index guard as other kernels.
- **Private vars:** per-thread `seed`, `p_energy`, `mat`.
- **Arrays:** device buffers `GSD.p_energy_samples` and `GSD.mat_samples` and the RNG tables embedded in the kernels.
- **OMP Migration Issues:** replicating the RNG logic (fast-forward and pick_mat’s nested loops) on the host must keep deterministic seeding per lookup; a straightforward `#pragma omp parallel for` over lookups can mimic the per-thread RNG pattern, but the random number generator must be re-entrant and per-lookup to avoid races.

## Kernel/Loop: run_event_based_simulation_optimization_2 material loop (Simulation.cu:431-440)
- **Context:** Host loop that dispatches `xs_lookup_kernel_optimization_2` once per material (12 iterations) after sampling.
- **Launch config:** `nthreads=32`, `nblocks=ceil(in.lookups/32)` for every material, even though only the matching lookups survive inside the kernel.
- **Total threads/iterations:** `12 × ceil(in.lookups/32) × 32` thread launches; each iteration filters on `mat == m`.
- **Type:** B – sparse dispatch where the host loop divides the work by material and each kernel quickly aborts on the wrong `mat`.
- **Parent loop:** `run_event_based_simulation_optimization_2` (Simulation.cu:407).
- **Contains:** repeated kernel launch; no nested device loops outside what the kernel already runs.
- **Dependencies:** the sampled `GSD.mat_samples` buffer must remain on device; the loop also relies on `thrust::reduce` for the verification after the kernel sequence.
- **Shared memory:** NO.
- **Thread indexing:** host loop recalculates the same linear grid for each material.
- **Private vars:** loop index `m` and temporary sizing.
- **Arrays:** `GSD.mat_samples` used to gate work, `GSD.p_energy_samples` read-only, `GSD.verification` written per material.
- **OMP Migration Issues:** recreating the material-dispatch dimension on the host would map to a nested `#pragma omp parallel for` (iterate materials and inside, iterate lookups) but the early exit on unmatched `mat` requires either pre-filtering lookups per material or a combined `if (mat != m) continue;` inside the parallel loop to avoid spinning on irrelevant lookups.

## Kernel/Loop: xs_lookup_kernel_optimization_4 (Simulation.cu:643)
- **Context:** Material-specific verification kernel used by optimizations 4 and 6 after sorting lookups by material and storing counts in `n_lookups_per_material`.
- **Launch config:** `nthreads=32`, `nblocks=ceil(n_lookups_per_material[m]/32)`; offset parameter shifts the working window into the sorted arrays.
- **Total threads/iterations:** sum over materials of `ceil(n_lookups_per_material[m]/32) × 32`; the earlier sort ensures each thread sees a contiguous span of lookups for `mat == m`.
- **Type:** A – dense per-lookup compute within each contiguous chunk.
- **Parent loop:** host material loop inside `run_event_based_simulation_optimization_4` and `_6` (Simulation.cu:623-843).
- **Contains:** same `calculate_macro_xs`/reduction logic as the baseline kernel.
- **Dependencies:** depends on `thrust::count` and `thrust::sort_by_key` to provide contiguous material slices; `n_lookups_per_material[m]` must match the offsets used when partitioning.
- **Shared memory:** NO.
- **Thread indexing:** 1D global index with an offset `i += offset` to align with the sorted ranges.
- **Private vars:** per-thread RNG state, `macro_xs_vector`, `max`, `max_idx`.
- **Arrays:** `GSD.p_energy_samples`, `GSD.mat_samples`, `GSD.num_nucs`, `GSD.concs`, `GSD.index_grid`, `GSD.nuclide_grid`, `GSD.mats`, `GSD.verification` (all device arrays pinned before the kernel launches).
- **OMP Migration Issues:** the pre-sorting and per-material offsets must be simulated on the CPU, either by reusing `std::sort` with keyed pairs or by manual bucketing; the kernel itself remains a candidate for a `#pragma omp parallel for` over the contiguous chunk but must be fed the `offset`/`n_lookups` metadata created by the host loop.

# Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|-----------------|------|----------|---------|------------|--------------|------------|
| run_event_based_simulation_baseline iteration (Simulation.cu:14) | A | CRITICAL | Host loop driving default kernel | `(in.num_iterations + in.num_warmups) × lookups` | xs_lookup_kernel_baseline | None beyond kernel and reduction prep |
| xs_lookup_kernel_baseline (Simulation.cu:41) | A | CRITICAL | Default CUDA kernel | `lookups` per iteration (≈gridDim×blockDim) | `calculate_macro_xs` / RNG helpers | Irregular inner loop over `num_nucs[mat]`, RNG needs host equivalent |
| sampling_kernel (Simulation.cu:348) | A | IMPORTANT | Pre-sampling for optimization paths | `lookups` RNG samples | RNG helpers (`fast_forward_LCG`, `pick_mat`) | Host RNG must match GPU RNG seeds |
| run_event_based_simulation_optimization_2 material loop (Simulation.cu:431) | B | SECONDARY | Sequential launches split per material | `12 × lookups` grid launches | sampling_kernel, xs_lookup_kernel_optimization_2 | Need to filter lookups per material in OpenMP and avoid spinning on irrelevant data |
| xs_lookup_kernel_optimization_4 (Simulation.cu:643) | A | IMPORTANT | Material-specific kernels (optimizations 4 & 6) | `sum(ceil(n_lookups_per_material[m]/32)×32)` | sorted `GSD.mat_samples`| Requires sorting/partition metadata computed on CPU |
| run_event_based_simulation_optimization_6 preprocessing loops (Simulation.cu:823-843) | B | SECONDARY | Host sorts/counts per material | `12` material sorts + kernels | `thrust::count`, `sort_by_key` + xs_lookup_kernel_optimization_4 | Must emulate `thrust` operations (counts, sorts, scans) on host |

# CUDA-Specific Details
- **Dominant compute kernel:** `xs_lookup_kernel_baseline` when `kernel_id == 0`; every simulation iteration re-traverses the lookup data with per-thread RNG and matrix evaluations (Simulation.cu:41-136).
- **Memory transfers in timed loop?:** NO – `move_simulation_data_to_device` and the `cudaMemcpy` that copies `GSD.verification` back to `SD.verification` happen outside the timed kernel loop; only device kernels run inside the window.
- **Shared memory usage:** NONE – kernels rely on register arrays (e.g., `double macro_xs_vector[5]`) and device-global reads/writes, so no arrays require manual privatization beyond standard stack temporaries.
- **Synchronization points:** explicit `cudaDeviceSynchronize()` before starting timing, after each kernel sequence, and after `thrust` operations; no intra-kernel sync primitives are used.
- **Atomic operations:** NONE – each thread writes to a distinct `GSD.verification[i]` slot, so atomicAdd is not needed.
- **Reduction patterns:** `thrust::reduce` is called after each optimized kernel path (`Optimization 1` through `6`) to collapse the verification buffer; the host also sums it manually in the baseline path (`Profile`), so OpenMP must replicate these reductions.
- **Sort/count utilities:** `thrust::count`, `thrust::sort_by_key`, and `thrust::partition` are used to cluster lookups by material/fuel status in optimizations 4, 5, and 6; these rely on device-level parallel primitives with global synchronizations.

# OMP Migration Strategy Notes
1. **Direct kernel → parallel for:** The per-lookup kernels (`xs_lookup_kernel_baseline`, `_optimization_1`, `_optimization_2`, `_optimization_4`, `_optimization_5`) all map to a parallel loop over `lookups`, with each iteration performing RNG, material/energy selection, and the same `calculate_macro_xs` logic. Implement a thread-private RNG state so each iteration can mimic `fast_forward_LCG` and `pick_mat` exactly.
2. **Requires restructuring:** Optimization variants that launch per-material kernels depend on Thrust (count/sort/partition) and repeated kernel dispatches; the migration should pre-bucket lookups by material/fuel on the host (e.g., `std::vector` of indices per material) and then spawn OpenMP loops over those buckets rather than invoking macros per material sequentially.
3. **Performance concerns:** The irregular `num_nucs[mat]` loop inside `calculate_macro_xs` and the binary search inside `calculate_micro_xs` mean each lookup has variable work, so dynamic scheduling or chunking might be necessary to avoid load imbalance in OpenMP. The verification buffer is dense but unordered accesses to `nuclide_grid` and `index_grid` could cause cache pressure.
4. **Data management:** `SimulationData` currently stores device pointers (e.g., `GSD.num_nucs`, `GSD.concs`, `GSD.verification`); on the host, those arrays should live in pinned structures, and the `move_simulation_data_to_device`/`release_device_memory` steps reduce to pointer copies or `memcpy` of contiguous data. Ensure any metadata (like `n_lookups_per_material` and `mat_samples`) stays consistent between the sampling stage and the lookup stage.
5. **Global reductions:** The baseline path sums the verification buffer on the host, while optimized variants rely on `thrust::reduce` on the GPU. In OpenMP, perform a single `#pragma omp parallel for reduction(+:verification_scalar)` over `SD.verification` or its bucketed equivalents after the main work.

Baseline run output was captured in `data/src/XSBench-omp/baseline_output.txt` for reference (generated by `golden_labels/src/XSBench-cuda/Makefile.nvc`).
