# Data Management Plan

## Arrays Inventory

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `temp` | `grid_rows * grid_cols * sizeof(FLOAT)` | working | host, then device copy | `R/W` |
| `result` | `grid_rows * grid_cols * sizeof(FLOAT)` | working | host, then device copy | `W` |
| `power` | `grid_rows * grid_cols * sizeof(FLOAT)` | const | host, then device copy | `R` |
| `d_temp` | same as `temp` | working | device alloc | `R/W` |
| `d_result` | same as `result` | working | device alloc | `W` |
| `d_power` | same as `power` | const | device alloc | `R` |

## Functions in Timed Region

| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| `compute_tran_temp` | `temp`, `result`, `power`, `d_temp`, `d_result`, `d_power` | once | host orchestration |
| `single_iteration` | `result`, `temp`, `power` device pointers | per timestep | device |

## Offload Unit Decision

- **Chosen Offload Unit:** flattened per-timestep stencil kernel
- **Why this unit:** the CPU chunking structure is cache-oriented and creates branch-heavy boundary/interior staging; GPU execution is better served by one flat `row x col` kernel per timestep with the time recurrence kept on the host
- **Timed-region stage count if left unfused:** multiple chunk/boundary branches per timestep
- **Structural Risk:** helper launch overhead and branchy tiling
- **Required rewrite before pragmas?** YES
- **Combined GPU+mem budget:** one kernel launch per timestep, one H→D setup copy, one D→H final copy

## Data Movement Strategy

**Chosen Strategy:** C

**Device Allocations (once):**
```
d_temp, d_result, d_power via omp_target_alloc
```

**Host→Device Transfers:**
- When: once before the timestep loop
- Arrays: `temp` -> `d_temp`, `power` -> `d_power`
- Total H→D: ~12 MB for 1024x1024, ~48 MB for 2048x2048

**Device→Host Transfers:**
- When: once after the timestep loop
- Arrays: `d_t` -> final host output array (`temp` or `result` depending on parity)
- Total D→H: ~4 MB for 1024x1024, ~16 MB for 2048x2048

**Transfers During Iterations:** NO
- All timestep work stays on the device after the initial copy

**Mid-computation sync in timed region:** NO
- No `target update` or host participation between timesteps

## Critical Checks

**Strategy C:**
- [x] ALL functions in iteration loop use `is_device_ptr`
- [x] Scratch arrays allocated on device
- [x] No map clauses in the hot path

**Common Mistakes:**
- Keep `temp/result/power` as host arrays only for I/O and verification
- Do not preserve the old CPU tile/chunk decomposition on the GPU
- Avoid `target update` inside the timestep loop

## Expected Transfer Volume

- Total: ~16 MB for 1024x1024, ~64 MB for 2048x2048
- **Red flag:** repeated per-timestep transfers would dominate and indicate the data plan is wrong

## Additional Parallelization Notes

- **RNG Replicable?** NO
- **Outer Saturation?** `row x col`
- **Sparse Matrix NONZER?** N/A
- **Histogram Strategy?** N/A
- **Kernel Granularity Check:** good
- **Preferred GPU pragma form:** `target teams distribute parallel for collapse(2)` - best fit for the flat dense stencil on this compiler/runtime

## Scalability Check

- **Default correctness size:** `1024 x 1024`, `sim_time=2`
- **Larger practical profiling size:** `2048 x 2048`, `sim_time=12`
- **Why this size materially exercises the GPU:** millions of cells per timestep keep the kernel busy long enough to reflect real offload behavior instead of launch overhead
- **Likely small-input sensitive?** NO
- **Scaling risk if this design is chosen:** none
- **Chosen structure still plausible at larger size?** YES

**Summary:** 3 arrays (1 scratch/working output, 2 working/const), 2 functions, Strategy C, offload unit flattened per-timestep stencil. Expected: ~12 MB H→D and ~4 MB D→H for correctness, ~48 MB H→D and ~16 MB D→H for profiling, 1 hot kernel per timestep.

## Build / Run Readiness

- **Plain build command works?** YES
- **Plain `make -f Makefile.nvc run` works?** YES
- **Unresolved placeholders remaining?** NO
- **`{nsys_profile_cmd} > {profile_log_path} 2>&1` shows GPU kernels?** YES
