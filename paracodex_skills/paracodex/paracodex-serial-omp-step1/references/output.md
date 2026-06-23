# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| [name] | [bytes] | working/scratch/const/index | host/device | R/W/RO |

**Types:** working (main data), scratch (temp), const (read-only), index (maps)

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| [name] | [list] | per-iteration/once | device/host |

## Offload Unit Decision
- **Chosen Offload Unit:** [single loop nest / fused routine / helper-by-helper]
- **Why this unit:** [state why this is better than preserving the serial helper structure]
- **Timed-region stage count if left unfused:** [count]
- **Structural Risk:** [none / tiny outer parallelism / helper launch overhead / host-device sync]
- **Required rewrite before pragmas?** [YES/NO]
- **Combined GPU+mem budget:** [kernel + memcpy + sync]

## Data Movement Strategy

**Chosen Strategy:** [A/B/C]

**Device Allocations (once):**
```
Strategy C: d_[array]: [size] via omp_target_alloc
Strategy A: [arrays] in target data region
```

**Host→Device Transfers:**
- When: [before iterations/once at start]
- Arrays: [array1]→d_[array1] ([size] MB)
- Total H→D: ~[X] MB

**Device→Host Transfers:**
- When: [after iterations/once at end]
- Arrays: d_[array1]→[array1] ([size] MB)
- Total D→H: ~[Y] MB

**Transfers During Iterations:** [YES/NO]
- If YES: [which arrays and why]
- If NO: All data stays on device

**Mid-computation sync in timed region:** [YES/NO]
- If YES: [which `target update`, scalar staging, or host participation remains and why]
- If NO: No host/device synchronization inside the hot path

## Critical Checks (for chosen strategy)

**Strategy A:**
- [ ] Functions inside target data use `present,alloc` wrapper?
- [ ] Scratch arrays use enter/exit data OR omp_target_alloc?
- [ ] Chosen offload unit avoids avoidable tiny-kernel staging?

**Strategy C:**
- [ ] ALL functions in iteration loop use is_device_ptr?
- [ ] Scratch arrays allocated on device (not host)?
- [ ] No map() clauses (only is_device_ptr)?

**Common Mistakes:**
-  Some functions on device, others on host (causes copying)
-  Scratch as host arrays in Strategy C
-  Forgetting to offload ALL functions in loop
-  Leaving generated placeholders such as `<RUN_ARGS>` in `Makefile.nvc` or scripts

## Expected Transfer Volume
- Total: ~[X+Y] MB for entire execution
- **Red flag:** If actual >2x expected → data management wrong

## Additional Parallelization Notes
- **RNG Replicable?** [YES/NO] - If YES, use `#pragma omp declare target` on RNG function
- **Outer Saturation?** [outer iters]
- **Sparse Matrix NONZER?** [value]
- **Histogram Strategy?** For small bin (≤ 100) counts: use per-thread local array + atomic merge (NO scratch arrays needed!)
- **Kernel Granularity Check:** [good / too fragmented / must fuse]
- **Preferred GPU pragma form:** [`target teams loop` / `target teams distribute parallel for` / other] - [why this compiler/runtime prefers it]

## Scalability Check
- **Default correctness size:** [value]
- **Larger practical profiling size:** [value or rule]
- **Why this size materially exercises the GPU:** [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- **Hardware basis from `system_info_summary.txt`:** [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]
- **Likely small-input sensitive?** [YES/NO]
- **Scaling risk if this design is chosen:** [kernel count / transfer count / transfer volume / host sync / memory footprint / none]
- **Chosen structure still plausible at larger size?** [YES/NO]

**Summary:** [num] arrays ([num] scratch, [num] working), [num] functions, Strategy [A/B/C], offload unit [unit]. Expected: ~[X] MB H→D, ~[Y] MB D→H, [N] hot kernels/stages.

## Build / Run Readiness
- **Plain build command works?** [YES/NO]
- **Plain `make -f Makefile.nvc run` works?** [YES/NO]
- **Unresolved placeholders remaining?** [YES/NO]
- **`{nsys_profile_cmd} > {profile_log_path} 2>&1` shows GPU kernels?** [YES/NO]
