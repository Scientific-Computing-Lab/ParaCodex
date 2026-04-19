# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `input_itemsets` | `max_rows * max_cols * sizeof(int)` | working | host, copied to device once | RW |
| `referrence` | `max_rows * max_cols * sizeof(int)` | const | host, copied to device once | RO |
| `input_itemsets_l` | `289 * sizeof(int)` | scratch | device stack inside each tile iteration | RW |
| `reference_l` | `256 * sizeof(int)` | scratch | device stack inside each tile iteration | RO |

**Types:** working (main data), scratch (temp), const (read-only), index (maps)

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| `nw_optimized` | `input_itemsets`, `referrence`, `input_itemsets_l`, `reference_l` | once | device for tile loops, host for stage control |
| `maximum` | scalar values only | per-cell update | device |

## Offload Unit Decision
- **Chosen Offload Unit:** fused routine
- **Why this unit:** the timed path is a staged wavefront with repeated tile-local copy/compute/copy-back work; preserving helper boundaries would force repeated tiny offloads and extra host/device staging around each stage
- **Timed-region stage count if left unfused:** O(`max_cols / 16`) stage waves, each containing an independent tile loop
- **Structural Risk:** tiny outer parallelism
- **Required rewrite before pragmas?** YES
- **Combined GPU+mem budget:** one target-data copy in, a sequence of device tile kernels, one copy out

## Data Movement Strategy

**Chosen Strategy:** A

**Device Allocations (once):**
```
input_itemsets and referrence live in a target data region
```

**Host→Device Transfers:**
- When: once at the start of the timed region
- Arrays: `input_itemsets` and `referrence`
- Total H→D: about 2 * `max_rows * max_cols * sizeof(int)` bytes

**Device→Host Transfers:**
- When: once at the end of the timed region
- Arrays: `input_itemsets`
- Total D→H: about `max_rows * max_cols * sizeof(int)` bytes

**Transfers During Iterations:** NO
- If YES: none
- If NO: all working data stays on device for the hot path

**Mid-computation sync in timed region:** NO
- If YES: none
- If NO: no `target update` or host participation inside the hot path

## Critical Checks (for chosen strategy)

**Strategy A:**
- [ ] Functions inside target data use `present,alloc` wrapper?
- [ ] Scratch arrays use enter/exit data OR omp_target_alloc?
- [x] Chosen offload unit avoids avoidable tiny-kernel staging?

**Common Mistakes:**
- Some functions on device, others on host (causes copying)
- Scratch as host arrays in Strategy C
- Forgetting to offload ALL functions in loop
- Leaving generated placeholders such as `<RUN_ARGS>` in `Makefile.nvc` or scripts

## Expected Transfer Volume
- Total: about 3 * `max_rows * max_cols * sizeof(int)` bytes
- **Red flag:** if actual is more than 2x expected, data management is wrong

## Additional Parallelization Notes
- **RNG Replicable?** NO
- **Outer Saturation?** one tile loop per stage, with enough tiles to occupy the GPU for practical sizes
- **Sparse Matrix NONZER?** N/A
- **Histogram Strategy?** N/A
- **Kernel Granularity Check:** good for step1, but still stage-fragmented by design
- **Preferred GPU pragma form:** `target teams loop` - the current nvc++ toolchain compiled this form cleanly and generated GPU kernels for both stage loops

## Scalability Check
- **Default correctness size:** `2048 10 2`
- **Larger practical profiling size:** `4096 10 2`
- **Likely small-input sensitive?** YES
- **Scaling risk if this design is chosen:** kernel count and stage sync
- **Chosen structure still plausible at larger size?** YES

**Summary:** 4 arrays (2 scratch, 2 working), 2 functions, Strategy A, offload unit fused routine. Expected: about 33.6 MB H→D and 16.8 MB D→H for the default size, with O(`max_cols / 16`) hot stages.

## Build / Run Readiness
- **Plain build command works?** YES
- **Plain `make -f Makefile.nvc run` works?** YES
- **Unresolved placeholders remaining?** NO
- **`{nsys_profile_cmd} > {profile_log_path} 2>&1` shows GPU kernels?** YES
