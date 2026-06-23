# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `knodesD` | tree-sized contiguous node buffer | working/const | host-built, copied once per command | R/RO in kernels |
| `recordsD` | record buffer | working/const | host-built, copied once per `k` command | R/RO in `findK` |
| `keysD` | `count` ints | index | host RNG init | R in `findK` |
| `ansD` | `count` records | working/output | host init to `-1` | W in `findK` |
| `x_2` | `count` longs | scratch | host init | unused after rewrite |
| `v_3` | `count` longs | scratch | host init | unused after rewrite |
| `data_4` | `count` longs | scratch | host init | unused after rewrite |
| `offset_2D` | `count` longs | scratch | host init | unused after rewrite |
| `x_0` | `count` ints | index | host RNG init | R in `findRangeK` |
| `var_7` | `count` ints | index | host RNG init | R in `findRangeK` |
| `RecstartD` | `count` ints | working/output | host init to `0` | W in `findRangeK` |
| `elem_1` | `count` ints | working/output | host init to `0` | W in `findRangeK` |

**Types:** working (main data), scratch (temp), const (read-only), index (maps)

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| `findK` | `knodesD`, `recordsD`, `keysD`, `ansD` | once per `k` command | device |
| `findRangeK` | `knodesD`, `x_0`, `var_7`, `RecstartD`, `elem_1` | once per `j` command | device |

## Offload Unit Decision
- **Chosen Offload Unit:** fused routine per command, with the outer query loop in each kernel offloaded directly
- **Why this unit:** each query is independent, but the CUDA-style fake thread lane loop is serial noise; keeping the whole batch in one target region avoids tiny-kernel fragmentation and avoids host/device staging between helper steps
- **Timed-region stage count if left unfused:** 1 per command, but the original code still contains serialized fake-thread structure
- **Structural Risk:** helper launch overhead and serialized lane emulation
- **Required rewrite before pragmas?** YES
- **Combined GPU+mem budget:** one kernel launch plus one transfer-in/out set per command

## Data Movement Strategy

**Chosen Strategy:** A

**Device Allocations (once):**
```
target data region around each command invocation; no manual omp_target_alloc required
```

**Host->Device Transfers:**
- When: once at the start of the `k` command and once at the start of the `j` command
- Arrays: `knodesD`, `recordsD`, `keysD`/`x_0`/`var_7`, outputs copied as needed for the command
- Total H->D: tree-sized input plus `O(count)` query buffers

**Device->Host Transfers:**
- When: once at the end of the `k` command and once at the end of the `j` command
- Arrays: `ansD`, `RecstartD`, `elem_1`
- Total D->H: `O(count)` outputs

**Transfers During Iterations:** NO
- If YES: none
- If NO: All data stays on device for the duration of each kernel call

**Mid-computation sync in timed region:** NO
- If YES: none
- If NO: No host/device synchronization inside the hot path

## Critical Checks (for chosen strategy)

**Strategy A:**
- [x] Functions inside target data use `present,alloc` wrapper?
- [x] Scratch arrays use enter/exit data OR omp_target_alloc?
- [x] Chosen offload unit avoids avoidable tiny-kernel staging?

**Common Mistakes:**
- Some functions on device, others on host (causes copying)
- Scratch as host arrays in Strategy C
- Forgetting to offload ALL functions in loop
- Leaving generated placeholders such as `<RUN_ARGS>` in `Makefile.nvc` or scripts

## Expected Transfer Volume
- Total: roughly the tree size plus a few `count`-sized buffers per command
- **Red flag:** If actual >2x expected -> data management wrong

## Additional Parallelization Notes
- **RNG Replicable?** NO
- **Outer Saturation?** `count` independent queries per command
- **Sparse Matrix NONZER?** N/A
- **Histogram Strategy?** N/A
- **Kernel Granularity Check:** good
- **Preferred GPU pragma form:** `target teams distribute parallel for` - one independent query per team-thread iteration maps cleanly to the GPU and keeps the inner traversal serial

## Scalability Check
- **Default correctness size:** `count = 4` for the bundled run path
- **Larger practical profiling size:** `count = 65535`
- **Likely small-input sensitive?** YES
- **Scaling risk if this design is chosen:** transfer volume and host-side tree build, not kernel fragmentation
- **Chosen structure still plausible at larger size?** YES

**Summary:** 6 arrays in the `k` path and 6 arrays in the `j` path, 2 device kernels, Strategy A, offload unit per-command query loop. Expected: one tree transfer per command, one output transfer per command, 2 hot kernels.

## Build / Run Readiness
- **Plain build command works?** YES
- **Plain `make -f Makefile.nvc run` works?** YES
- **Unresolved placeholders remaining?** NO
- **`{nsys_profile_cmd} > {profile_log_path} 2>&1` shows GPU kernels?** EXPECTED YES
