# XSBench CUDA Loop Analysis

## File Conversion Mapping
- `golden_labels/src/XSBench-cuda/Main.cu` → `data/src/XSBench-omp/Main.cpp`
- `golden_labels/src/XSBench-cuda/io.cu` → `data/src/XSBench-omp/io.cpp`
- `golden_labels/src/XSBench-cuda/GridInit.cu` → `data/src/XSBench-omp/GridInit.cpp`
- `golden_labels/src/XSBench-cuda/XSutils.cu` → `data/src/XSBench-omp/XSutils.cpp`
- `golden_labels/src/XSBench-cuda/Materials.cu` → `data/src/XSBench-omp/Materials.cpp`
- `golden_labels/src/XSBench-cuda/Simulation.cu` → `data/src/XSBench-omp/Simulation.cpp`

## Kernel/Loop Nesting Structure
- `main` (golden_labels/src/XSBench-cuda/Main.cu:4-110) branches on `in.kernel_id` and calls one of the `run_event_based_simulation_*` helpers.
  - Baseline path (`Simulation.cu:3-38`) allocates `GSD`, then performs the warmup/iteration loop (`Simulation.cu:14-24`) that repeatedly launches `xs_lookup_kernel_baseline` (`Simulation.cu:41-85`).
  - Optimization 1 (`Simulation.cu:304-346`) launches `sampling_kernel` (`Simulation.cu:348-365`) followed by `xs_lookup_kernel_optimization_1` (`Simulation.cu:367-405`).
  - Optimization 2 (`Simulation.cu:407-449`) runs `sampling_kernel` and then the host loop `for (m=0; m<12)` that launches `xs_lookup_kernel_optimization_2` (`Simulation.cu:452-494`).
  - Optimization 3 (`Simulation.cu:496-538`) runs `sampling_kernel` then a pair of `xs_lookup_kernel_optimization_3` launches with `is_fuel=0/1` (`Simulation.cu:541-582`).
  - Optimization 4 (`Simulation.cu:586-640`) runs `sampling_kernel`, material counts/sort, and a host loop (`Simulation.cu:623-631`) that dispatches `xs_lookup_kernel_optimization_4` (`Simulation.cu:643-687`) for each sorted chunk.
  - Optimization 5 (`Simulation.cu:697-745`) runs `sampling_kernel`, then `thrust::count`/`partition`, and two `xs_lookup_kernel_optimization_5` launches that process fuel and non-fuel ranges (`Simulation.cu:750-789`).
  - Optimization 6 (`Simulation.cu:792-852`) is like 4 but adds an extra per-material `thrust::sort_by_key` pass (`Simulation.cu:829-834`) before dispatching `xs_lookup_kernel_optimization_4` again (`Simulation.cu:836-843`).
- Device functions (`calculate_macro_xs`, `calculate_micro_xs`, `grid_search`, `grid_search_nuclide`, `pick_mat`, `LCG` helpers) are invoked per lookup inside each kernel.

## Kernel/Loop Details

### Kernel/Loop: run_event_based_simulation_baseline host loop at golden_labels/src/XSBench-cuda/Simulation.cu:3
- **Context:** Host-managed warmup/timed loop that measures kernel performance for the baseline kernel.
- **Launch config:** `nthreads=256`, `nblocks=ceil((double)in.lookups/256.0)` computed once, and every iteration launches `xs_lookup_kernel_baseline<<<nblocks, nthreads>>>`.
- **Total threads/iterations:** ≈ `in.lookups` threads per iteration, repeated `in.num_iterations + in.num_warmups` times (warmup iterations are synchronized separately before timing starts).
- **Type:** A – one-to-one dense mapping from lookups to threads.
- **Parent loop:** `main` event-based branch (Main.cu:59-79).
- **Contains:** kernel launch, `cudaDeviceSynchronize()` for warmup/timed boundary, and the verification copy/reduction that follows.
- **Dependencies:** `move_simulation_data_to_device` (Simulation.cu:5-6) allocates `GSD`; results copied back with `cudaMemcpy` and reduced on the host (Simulation.cu:27-34).
- **Shared memory:** NO.
- **Thread indexing:** N/A (host loop).
- **Private vars:** `nthreads`, `nblocks`, `nwarmups`, `start`, `verification_scalar`.
- **Arrays:** `GSD.verification` (W read back), `GSD.num_nucs`, `concs`, `mats`, `nuclide_grid`, etc accessed by the kernel.
- **OMP Migration Issues:** The loop simply replays the same O(N) particle count. In an OpenMP port the same warmup/timed structure can be preserved by making the kernel body a `#pragma omp parallel for` over `lookups` and running that parallel loop `in.num_iterations` times; warmup iterations can also execute the same parallel loop while timing is skipped.

### Kernel/Loop: xs_lookup_kernel_baseline (golden_labels/src/XSBench-cuda/Simulation.cu:41)
- **Context:** Primary compute kernel for baseline event-based simulation, invoked from the host loop above.
- **Launch config:** `grid=(ceil(lookups/256.0))`, `block=(256)`.
- **Total threads/iterations:** Each thread handles one lookup and writes one verification byte/word.
- **Type:** A – dense cross-section lookup per work-item.
- **Parent loop:** `run_event_based_simulation_baseline` (Simulation.cu:3-38).
- **Contains:** calls `fast_forward_LCG`, `LCG_random_double`, `pick_mat`, `calculate_macro_xs`, and a reduction over the 5 reaction channels to store the dominant channel index.
- **Dependencies:** Device functions `calculate_macro_xs` and `calculate_micro_xs`, `grid_search` helpers, and the global `GSD` arrays.
- **Shared memory:** NO.
- **Thread indexing:** `const int i = blockIdx.x * blockDim.x + threadIdx.x;` (Simulation.cu:43).
- **Private vars:** `seed`, `p_energy`, `mat`, `macro_xs_vector[5]`, `max`, `max_idx`.
- **Arrays:** `GSD.verification[i]` (W), `GSD.num_nucs`/`concs`/`mats` (R), `in.unionized_energy_array`, `GSD.index_grid`, `GSD.nuclide_grid` (R), `GSD.max_num_nucs` (value-driven).
- **OMP Migration Issues:** All accesses are read-only besides the single store to `GSD.verification`, so the kernel maps to a simple parallel for. Irregular data access comes from `calculate_macro_xs`/`calculate_micro_xs` (grid search and material-dependent loops) but no atomics or shared memory dependencies exist.

### Kernel/Loop: calculate_macro_xs device work (golden_labels/src/XSBench-cuda/Simulation.cu:156)
- **Context:** Device helper invoked by every lookup to accumulate contributions from all nuclides in the selected material.
- **Launch config:** Scalar per thread within `xs_lookup_kernel_*`.
- **Total threads/iterations:** The outer `for (int j = 0; j < num_nucs[mat]; j++)` loop iterates over the handful of nuclides in the chosen material; inner `for (int k = 0; k < 5; k++)` sums cross sections for the five reaction channels.
- **Type:** A – dense per-material accumulation with small inner dimension (5) and `num_nucs` varying by material (max controlled by `GSD.max_num_nucs`).
- **Parent loop:** `xs_lookup_kernel_*` (all versions).
- **Contains:** per-nuclide `calculate_micro_xs` calls and the 5-element vector reduction.
- **Dependencies:** `calculate_micro_xs` (Simulation.cu:87), `grid_search`/`grid_search_nuclide`, `GSD` arrays, `in.grid_type`, `hash_bins`.
- **Shared memory:** NO.
- **Thread indexing:** per kernel thread (no `blockIdx` inside the function).
- **Private vars:** `p_nuc`, `idx`, `conc`, `xs_vector[5]`, `macro_xs_vector[5]` reset locally.
- **Arrays:** `num_nucs`/`concs`/`mats` (R), `nuclide_grid` (R), `unionized_energy_array`/`index_grid` (R) whose interpretation depends on `grid_type`.
- **OMP Migration Issues:** The `num_nucs` loop depends on the material but is read-only; in OpenMP it can execute as an inner loop inside a parallel for. The branch on `grid_type` yields different search paths (NUCLIDE, UNIONIZED, HASH) but each path is independent.

### Kernel/Loop: calculate_micro_xs / grid search helpers (golden_labels/src/XSBench-cuda/Simulation.cu:87-233)
- **Context:** Device functions used by `calculate_macro_xs` to interpolate a single nuclide cross section.
- **Launch config:** Invoked from each thread; not a kernel.
- **Total threads/iterations:** `grid_search_nuclide` and `grid_search` perform logarithmic binary search loops (`while (length > 1)`), so each lookup executes a handful of iterations proportional to `log(n_gridpoints)`.
- **Type:** A – memory-bound search/interpolation for a single nuclide and energy value.
- **Parent loop:** `calculate_macro_xs`.
- **Contains:** Binary search, interpolation of 5 floating-point values, and return of interpolated cross sections for the calling kernel.
- **Dependencies:** `NuclideGridPoint` data, `index_data`, `hash_bins`, and the `grid_type` switch.
- **Shared memory:** NO.
- **Thread indexing:** per kernel thread; the work is entirely data-driven.
- **Private vars:** Search bounds (`lowerLimit`, `upperLimit`, `length`) and interpolation weight `f`.
- **Arrays:** `nuclide_grids`, `egrid`, `index_data` (R) – each thread walks these arrays irregularly but only reads.
- **OMP Migration Issues:** The binary search loops are branch-heavy but sequential, so they convert directly to scalar C++ loops inside each OpenMP thread. No additional synchronization is necessary.

### Kernel/Loop: sampling_kernel (golden_labels/src/XSBench-cuda/Simulation.cu:348)
- **Context:** Generates `p_energy_samples` and `mat_samples` for every lookup before the optimized lookup kernels.
- **Launch config:** `nthreads=32`, `nblocks=ceil((double)in.lookups/32.0)`.
- **Total threads/iterations:** `in.lookups` threads, each writing two sample arrays.
- **Type:** A – dense generation, identical pattern to the baseline kernel’s random number generation.
- **Parent loop:** Called from every `run_event_based_simulation_optimization_*` (Simulation.cu:304-852) before their lookup kernels.
- **Contains:** `fast_forward_LCG`, `LCG_random_double`, `pick_mat`, writing to `GSD.p_energy_samples` and `GSD.mat_samples`.
- **Dependencies:** Random number helpers (`fast_forward_LCG` lines 276-301, `LCG_random_double` lines 266-274, `pick_mat` lines 235-264).
- **Shared memory:** NO.
- **Thread indexing:** `i = blockIdx.x * blockDim.x + threadIdx.x` (Simulation.cu:351).
- **Private vars:** `seed`, `p_energy`, `mat`.
- **Arrays:** `GSD.p_energy_samples` (W), `GSD.mat_samples` (W) – both device/global.
- **OMP Migration Issues:** The same random generation strategy can be ported to OpenMP by parallelizing over lookups and using per-thread seeds; only the GPU-specific memory handles (`cudaMalloc`) need to become host arrays.

### Kernel/Loop: xs_lookup_kernel_optimization_1 (golden_labels/src/XSBench-cuda/Simulation.cu:367)
- **Context:** Uses pre-generated samples instead of regenerating randomness per launch, but otherwise mirrors the baseline kernel.
- **Launch config:** `ceil(in.lookups/32)` blocks of 32 threads.
- **Total threads/iterations:** `in.lookups` per launch, single invocation.
- **Type:** A – same per-lookup cross-section accumulation.
- **Parent loop:** `run_event_based_simulation_optimization_1` (Simulation.cu:304-345).
- **Contains:** `calculate_macro_xs` and the five-channel reduction.
- **Dependencies:** Sample arrays, `calculate_macro_xs`/`calculate_micro_xs`.
- **Shared memory:** NO.
- **Thread indexing:** Standard `blockIdx.x * blockDim.x + threadIdx.x` (Simulation.cu:370).
- **Private vars:** same as baseline kernel.
- **Arrays:** Read samples `GSD.p_energy_samples`, `GSD.mat_samples`, plus the same `GSD` data and verification output.
- **OMP Migration Issues:** Directly convertible to OpenMP parallel for over lookups; the only addition is that the sample arrays already contain the random inputs, so the CPU version can skip recomputing them.

### Kernel/Loop: host loop in run_event_based_simulation_optimization_2 (`for (int m = 0; m < 12; m++)`) at golden_labels/src/XSBench-cuda/Simulation.cu:407-441
- **Context:** Serial loop over the 12 materials that dispatches a kernel per material.
- **Launch config:** Inside each iteration, launches `xs_lookup_kernel_optimization_2` with `nthreads=32`, `nblocks=ceil(lookups/32)`.
- **Total threads/iterations:** 12 launches, each covering all `lookups` but filtering by material in the kernel.
- **Type:** B – each kernel activates only the lookups that match the selected `m`, so the work set is sparse but still touched in bulk.
- **Parent loop:** `run_event_based_simulation_optimization_2`.
- **Contains:** `sampling_kernel` before the loop (Simulation.cu:434-437), per-material kernel launches, and a single `thrust::reduce` at the end.
- **Dependencies:** `GSD.mat_samples` must be populated; `thrust::reduce` runs after the loop to aggregate verification.
- **Shared memory:** NO.
- **Thread indexing:** `blockIdx.x * blockDim.x + threadIdx.x` inside the kernels.
- **Private vars:** `m`, `nblocks`, `nthreads`.
- **Arrays:** `GSD.mat_samples` (R), `GSD.p_energy_samples` (R), `GSD.verification` (W) per kernel.
- **OMP Migration Issues:** The serial material loop can become a nested `#pragma omp parallel for` over materials or was even unnecessary on CPU, since filtering by material can happen inside a single parallel loop; the sequential `for (m...)` just enforces material ordering but does not add data dependencies.

### Kernel/Loop: xs_lookup_kernel_optimization_2 (golden_labels/src/XSBench-cuda/Simulation.cu:452)
- **Context:** Same as baseline kernel except every thread first checks `GSD.mat_samples[i] == m` and returns early if not.
- **Launch config:** Same `ceil(lookups/32)` blocks of 32 threads per material.
- **Total threads/iterations:** Each launch still spawns `in.lookups` threads, but most threads exit early when their sample did not match `m`.
- **Type:** B – the active working set is the subset of lookups whose sampled material equals the current `m`.
- **Parent loop:** Host material loop in `run_event_based_simulation_optimization_2`.
- **Contains:** `calculate_macro_xs` and the five-channel reduction, identical to baseline once past the material guard.
- **Dependencies:** `GSD.mat_samples`, `GSD.p_energy_samples` pre-filled by `sampling_kernel`.
- **Shared memory:** NO.
- **Thread indexing:** `blockIdx.x * blockDim.x + threadIdx.x` (Simulation.cu:455).
- **Private vars:** `mat`, `macro_xs_vector[5]`, `max`, `max_idx`.
- **Arrays:** `GSD.mat_samples`, `GSD.p_energy_samples`, `GSD.verification` (per lookup, writes filtered results), `GSD` cross-section data.
- **OMP Migration Issues:** On CPU the early-return branch is cheap; the best mapping is a single parallel loop that checks `mat` and computes only for matching lookups, eliminating the need for repeated kernel launches.

### Kernel/Loop: xs_lookup_kernel_optimization_3 (golden_labels/src/XSBench-cuda/Simulation.cu:541)
- **Context:** Two kernel launches partition lookups into `is_fuel=1` (mat==0) and `is_fuel=0` (mat!=0).
- **Launch config:** Two launches with `nthreads=32`, `nblocks=ceil(lookups/32)`.
- **Total threads/iterations:** `2 × in.lookups` thread launches (one for each partition), though each thread immediately checks `mat` and returns if the condition fails.
- **Type:** A/B hybrid – the kernel structure is dense, but each launch only performs work on the subset matching `is_fuel`.
- **Parent loop:** `run_event_based_simulation_optimization_3`.
- **Contains:** Condition `if ((is_fuel == 1 && mat == 0) || (is_fuel == 0 && mat != 0))` gating the identical `calculate_macro_xs` body.
- **Dependencies:** `sampling_kernel` pre-fills `GSD.mat_samples`, `thrust::reduce` at the end.
- **Shared memory:** NO.
- **Thread indexing:** `blockIdx.x * blockDim.x + threadIdx.x` (Simulation.cu:544).
- **Private vars:** `mat`, `is_fuel` parameter.
- **Arrays:** same as other kernels.
- **OMP Migration Issues:** Partitioned execution can be replaced by a single loop that branches on `mat` and performs `calculate_macro_xs` for each lookup once; the double-launch structure is only needed on CUDA to reduce divergence, so it simplifies on CPU.

### Kernel/Loop: run_event_based_simulation_optimization_4 material sorting & loop (golden_labels/src/XSBench-cuda/Simulation.cu:586-633)
- **Context:** Builds a material-sorted worklist before dispatching `xs_lookup_kernel_optimization_4` sequentially per material.
- **Launch config:** After `sampling_kernel`, counts lookups per material (`thrust::count` on GSD.mat_samples, lines 617-619), sorts key/value pairs (`thrust::sort_by_key`, line 621), and then iterates `for (int m = 0; m < 12; m++)` (lines 623-631) to launch `xs_lookup_kernel_optimization_4` on each chunk sized by `n_lookups_per_material[m]`.
- **Total threads/iterations:** The sorting phase is device-wide (O(lookups) operations) and the subsequent loop launches 12 kernels whose grid size equals the per-material lookup count.
- **Type:** C1/C2 component – multiple kernel launches require global synchronization between `thrust` sort/reduce and each kernel.
- **Parent loop:** `run_event_based_simulation_optimization_4`.
- **Contains:** `thrust::count`, `thrust::sort_by_key`, sequential kernel launches with dynamic `offset` bookkeeping.
- **Dependencies:** `GSD.mat_samples`, `GSD.p_energy_samples`, `n_lookups_per_material`, `xs_lookup_kernel_optimization_4`.
- **Shared memory:** NO.
- **Thread indexing:** inside `xs_lookup_kernel_optimization_4` (see below).
- **Private vars:** `nthreads`, `nblocks`, `offset`, `n_lookups_per_material` array.
- **Arrays:** Input (material keys and energy samples) and indexing arrays used to create contiguous chunks.
- **OMP Migration Issues:** The Thrust count and sort operations need CPU equivalents (e.g., `std::sort` + `std::count_if` or parallel algorithms). The sequential loop over materials can be replaced by an OpenMP `parallel for` over the 12 material chunks, assuming the `offset` bookkeeping is replicated.

### Kernel/Loop: xs_lookup_kernel_optimization_4 (golden_labels/src/XSBench-cuda/Simulation.cu:643)
- **Context:** Processes an ordered chunk of lookups that all share the same material ID.
- **Launch config:** Each launch receives `n_lookups` and `offset`; block size 32 threads, grid `ceil(n_lookups/32)`.
- **Total threads/iterations:** `n_lookups` active threads per material, so overall work equals the total of all `n_lookups_per_material` (≈ `in.lookups`).
- **Type:** B – only the lookups in the current chunk perform arithmetic, while others exit early because of the guard `if (mat != m)`.
- **Parent loop:** `run_event_based_simulation_optimization_4` and `run_event_based_simulation_optimization_6`.
- **Contains:** `calculate_macro_xs` plus `GSD.verification` writes (same as other `xs_lookup` kernels).
- **Dependencies:** Sorted samples and material keys.
- **Shared memory:** NO.
- **Thread indexing:** `int i = blockIdx.x * blockDim.x + threadIdx.x; i += offset;` (Simulation.cu:646-651).
- **Private vars:** `i`, `mat`, `macro_xs_vector`, `max`, `max_idx`.
- **Arrays:** Sorted `GSD.mat_samples`, `GSD.p_energy_samples`, `GSD.verification` chunk range, other device data.
- **OMP Migration Issues:** The chunk-by-chunk processing simply translates to iterating over contiguous subsets when moving to OpenMP; the extra guard `mat != m` becomes redundant if each chunk is already material-homogeneous.

### Kernel/Loop: run_event_based_simulation_optimization_5 partition + kernels (golden_labels/src/XSBench-cuda/Simulation.cu:697-738)
- **Context:** Splits lookups into fuel/non-fuel partitions before launching two specialized kernels.
- **Launch config:** After `sampling_kernel`, uses `thrust::count` (line 728) and `thrust::partition` (line 730) to rearrange `GSD.mat_samples`/`GSD.p_energy_samples`, then launches `xs_lookup_kernel_optimization_5` twice for fuel and non-fuel offsets.
- **Total threads/iterations:** The partition rearrangement is a global operation on `in.lookups` elements; each subsequent kernel handles `n_fuel_lookups` or `in.lookups - n_fuel_lookups` threads.
- **Type:** B – simple partitioning followed by dense processing on the two subranges.
- **Parent loop:** `run_event_based_simulation_optimization_5`.
- **Contains:** `thrust::count`, `thrust::partition`, and two kernel launches with different offsets.
- **Dependencies:** Partitioned sample arrays, plus `xs_lookup_kernel_optimization_5`.
- **Shared memory:** NO.
- **Thread indexing:** `blockIdx.x * blockDim.x + threadIdx.x`, with offsets handled inside the kernel.
- **Private vars:** `n_fuel_lookups`, `offset`.
- **Arrays:** Partitioned `GSD.mat_samples`/`GSD.p_energy_samples`, output `GSD.verification`.
- **OMP Migration Issues:** Replacing `thrust::partition` requires either `std::partition` (serial) or an OpenMP parallel prefix-style implementation. Once partitioned, the two ranges hand off to independent parallel loops.

### Kernel/Loop: xs_lookup_kernel_optimization_5 (golden_labels/src/XSBench-cuda/Simulation.cu:750)
- **Context:** Processes either the fuel range or the non-fuel range after partitioning.
- **Launch config:** Each kernel is launched with the number of lookups for its range and the offset where that range lives.
- **Total threads/iterations:** `n_lookups` per launch (either `n_fuel_lookups` or `in.lookups - n_fuel_lookups`).
- **Type:** A – standard lookup once the partition boundaries are enforced.
- **Parent loop:** `run_event_based_simulation_optimization_5`.
- **Contains:** `calculate_macro_xs` for each lookup, offset addition, and storing the verification index.
- **Dependencies:** Partitioned sample arrays.
- **Shared memory:** NO.
- **Thread indexing:** `int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n_lookups) return; i += offset;` (Simulation.cu:753-759).
- **Private vars:** `i`, `macro_xs_vector`, `max`, `max_idx`.
- **Arrays:** Partitioned `GSD.p_energy_samples`, `GSD.mat_samples`, `GSD.verification`.
- **OMP Migration Issues:** Equivalent to processing two contiguous ranges; easily modeled by two separate parallel loops or by a single loop with `offset` handling.

### Kernel/Loop: run_event_based_simulation_optimization_6 (golden_labels/src/XSBench-cuda/Simulation.cu:792-845)
- **Context:** Adds a second sort pass to optimization 4 before dispatching the same per-material kernels.
- **Launch config:** After sampling, performs `thrust::count`, `thrust::sort_by_key`, another `sort_by_key` inside the `for (m)` loop, then sequentially launches `xs_lookup_kernel_optimization_4` per material (lines 823-843).
- **Total threads/iterations:** Sorting stages touch all lookups twice; the kernel launches total ≈ `in.lookups` work once all chunks are processed.
- **Type:** C1 – multiple global sync points interleaving `thrust` calls and kernel launches.
- **Parent loop:** `run_event_based_simulation_optimization_6`.
- **Contains:** Two `thrust::sort_by_key` passes plus sequential kernel launches similar to optimization 4.
- **Dependencies:** Sorted data must be consistent across two key/value sorts; uses `xs_lookup_kernel_optimization_4` for the actual compute.
- **Shared memory:** NO.
- **Thread indexing:** Same as in optimization 4 kernels.
- **Private vars:** `offset`, `nthreads`, `nblocks`, `n_lookups_per_material` array.
- **Arrays:** `GSD.p_energy_samples`, `GSD.mat_samples`, `GSD.verification`, `GSD.num_nucs`, etc.
- **OMP Migration Issues:** Replacing the Thrust sorts is the main effort; once sorted, the per-material loops reduce to contiguous ranges suitable for OpenMP parallel loops.

### Kernel/Loop: verification accumulation host loop (golden_labels/src/XSBench-cuda/Simulation.cu:32)
- **Context:** Sums the device verification buffer after it has been copied back to host memory.
- **Launch config:** Serial `for (int i = 0; i < in.lookups; i++) verification_scalar += SD.verification[i];`.
- **Total threads/iterations:** `in.lookups` host iterations, executed once after all kernels finished.
- **Type:** SECONDARY – only used for validation.
- **Parent loop:** `run_event_based_simulation_baseline`.
- **Contains:** Simple scalar addition over the verification buffer elements.
- **Dependencies:** Results from the GPU executed kernel writes.
- **Shared memory:** NO.
- **Thread indexing:** N/A.
- **Private vars:** `verification_scalar`, loop counter `i`.
- **Arrays:** `SD.verification` (host copy of `GSD.verification`).
- **OMP Migration Issues:** Easily replaced with an OpenMP reduction over `SD.verification` to preserve the checksum semantics.

## Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|-----------------|------|----------|---------|------------|--------------|------------|
| `run_event_based_simulation_baseline` | A | CRITICAL | Host warmup/timed loop | `(in.num_iterations + in.num_warmups) * lookups` | `GSD` allocations + verification reduction | Preserve warmup/timing but mapping is a repeated `parallel for` over lookups |
| `xs_lookup_kernel_baseline` | A | CRITICAL | __global__ kernel | `ceil(lookups/256) * 256` threads per iteration | `calculate_macro_xs` → `calculate_micro_xs` → `grid_search*`, random helpers | Direct `#pragma omp parallel for`, watch grid search branching |
| `calculate_macro_xs` / micro search | A | CRITICAL | Device loops inside lookup kernels | `num_nucs[mat] * 5` per lookup, with binary search steps | `grid_search`, `grid_search_nuclide` | Keep per-material loops, but all data is read-only |
| `xs_lookup_kernel_optimization_4` | B | IMPORTANT | Per-material kernel for optimizations 4/6 | Sum of `n_lookups_per_material` threads | Requires sorted `GSD.mat_samples`/`p_energy_samples` | CPU version can operate on contiguous ranges without per-material kernel launches |
| Material sorting/partition (`run_event_*_optimization_4/5/6` host loops + `thrust`) | C1/C2 | IMPORTANT | Thrust-based count/sort/partition host loops | O(lookups) operations in Thrust, plus the same lookup kernels | `thrust::count`, `thrust::sort_by_key`, `thrust::partition` | Need CPU equivalents or parallel STL algorithms; sequential host loop ordering can be relaxed |
| `sampling_kernel` | A | IMPORTANT | Generates random inputs before optimized lookups | `ceil(lookups/32) * 32` threads | `fast_forward_LCG`, `LCG_random_double`, `pick_mat` | Map to `parallel for` writing sample arrays; seeds must stay per-thread |

## CUDA-Specific Details
- **Dominant compute kernel:** `xs_lookup_kernel_*` variations (Simulation.cu:41, 367, 452, 541, 643, 750) – each thread performs a full random-energy draw, material selection, and `calculate_macro_xs` accumulation.
- **Memory transfers in timed loop?:** No extra transfers occur inside the timed loop; `move_simulation_data_to_device` (Simulation.cu:5-30) copies the `SimulationData` arrays once at startup, and only the verification buffer is copied back after the loop (`cudaMemcpy` at Simulation.cu:27-30).
- **Shared memory usage:** None of the kernels declare `__shared__` arrays; all state is either thread-private or in global memory.
- **Synchronization points:** Each kernel launch is followed by `cudaPeekAtLastError()` + `cudaDeviceSynchronize()`. Warmup iterations in the baseline path synchronize before timing begins (Simulation.cu:16-21).
- **Atomic operations:** None of the CUDA kernels use `atomicAdd`; reductions are handled via `thrust::reduce` on the verification buffer (Simulation.cu:341, 445, 534, 636, 743, 849).
- **Reduction patterns:** `thrust::reduce` is used repeatedly after kernels to sum `GSD.verification`. Material counts and reorders rely on `thrust::count`, `thrust::sort_by_key`, and `thrust::partition` (Simulation.cu:617-731, 827-834).
- **Thread indexing:** All kernels follow `i = blockIdx.x * blockDim.x + threadIdx.x`. Optimized kernels add `offset` adjustments and material guards to skip non-target lookups.
- **Random and search helpers:** `fast_forward_LCG` (Simulation.cu:276-301), `LCG_random_double` (Simulation.cu:266-274), `pick_mat` (Simulation.cu:235-264), and the binary searches (`grid_search`, `grid_search_nuclide`) run per-thread and introduce irregular memory access patterns.

## OMP Migration Strategy Notes
1. **Direct kernel → parallel for:** All `xs_lookup_kernel_*` variants share the pattern of one independent lookup per thread. In OpenMP we can convert them to `#pragma omp parallel for` over the `lookups` array while keeping `calculate_macro_xs`/`calculate_micro_xs` unchanged. The random helpers (`LCG`) remain per-iteration and can use thread-local seeds.
2. **Requires restructuring:** The `run_event_based_simulation_optimization_2/4/5/6` helpers rely on `thrust::count`, `sort_by_key`, `partition`, and sequential kernel dispatch per material. These need CPU-side equivalents (e.g., `std::sort` plus `std::stable_partition` or parallel STL algorithms) and a strategy for constructing contiguous material ranges before the parallel loops.
3. **Performance concerns:** The grid search branches (unionized vs hash vs nuclide) and `num_nucs[mat]`-length inner loops introduce irregular memory access and varying work per lookup; dynamic scheduling (`schedule(dynamic)`) might help. Material-specific kernels (optimizations 2/4/6) reduce warp divergence on CUDA but become redundant on CPU; a single parallel loop with conditional logic is sufficient.
4. **Data management:** `move_simulation_data_to_device` currently duplicates every pointer in `SimulationData`; the OpenMP version should operate directly on host arrays (no `cudaMalloc`/`cudaMemcpy`). Temporary arrays such as `p_energy_samples`, `mat_samples`, and `verification` can stay as host vectors, and the final verification reduction can be replaced with an OpenMP reduction or sequential loop.
