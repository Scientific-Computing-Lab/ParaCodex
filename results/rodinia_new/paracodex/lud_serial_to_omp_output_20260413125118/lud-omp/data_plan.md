# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `a` | `size * size * sizeof(float)` | working | host, then device-mapped | R/W |
| `temp` | `BS * BS * sizeof(float)` | scratch | device-local automatic storage | R |
| `temp_top` | `BS * BS * sizeof(float)` | scratch | device-local automatic storage | R |
| `temp_left` | `BS * BS * sizeof(float)` | scratch | device-local automatic storage | R |
| `sum` | `BS * sizeof(float)` | scratch | device-local automatic storage | R/W |

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| `lud_omp` | `a` | once | host for orchestration, device for compute phases |
| `lud_diagonal_omp` or fused diagonal block code | `a` | once per block offset | device |
| perimeter chunk update code | `a`, `temp` | once per chunk | device |
| interior chunk update code | `a`, `temp_top`, `temp_left`, `sum` | once per chunk | device |

## Offload Unit Decision
- **Chosen Offload Unit:** fused routine
- **Why this unit:** the LU factorization is staged by block offset, but the diagonal factorization and trailing block updates must stay on the same device-resident matrix to avoid repeated host/device copies. Keeping the whole routine fused avoids tiny helper launches and mid-step synchronization.
- **Timed-region stage count if left unfused:** 3 per block offset, plus a scalar diagonal helper if preserved separately
- **Structural Risk:** helper launch overhead
- **Required rewrite before pragmas?** YES
- **Combined GPU+mem budget:** one matrix transfer in, one matrix transfer out, no staged transfers during the hot path

## Data Movement Strategy

**Chosen Strategy:** A

**Device Allocations (once):**
```
target data map(tofrom:a[0:size*size])
```

**Host->Device Transfers:**
- When: once before the block loop
- Arrays: `a` -> device copy of the matrix
- Total H->D: ~16 MB for the 2048 profiling size

**Device->Host Transfers:**
- When: once after the block loop
- Arrays: device copy of `a` -> host `a`
- Total D->H: ~16 MB for the 2048 profiling size

**Transfers During Iterations:** NO
- All matrix updates remain on the device until the final exit from the target data region.

**Mid-computation sync in timed region:** NO
- No `target update` is needed inside the block loop; each block phase executes on the device and the next phase sees the updated device-resident matrix.

## Critical Checks (for chosen strategy)

**Strategy A:**
- [x] Functions inside target data use `present,alloc` wrapper?
- [x] Scratch arrays use enter/exit data OR omp_target_alloc?
- [x] Chosen offload unit avoids avoidable tiny-kernel staging?

**Common Mistakes:**
- Some functions on device, others on host, causing hidden copies
- Leaving the diagonal helper on host
- Keeping the serial chunk loop unchanged instead of parallelizing the independent chunks
- Forgetting to replace placeholder values in `Makefile.nvc`

## Expected Transfer Volume
- Total: ~32 MB for the full 2048 run
- **Red flag:** if actual traffic grows beyond this, the matrix is being recopied instead of staying resident

## Additional Parallelization Notes
- **RNG Replicable?** NO
- **Outer Saturation?** yes, by block-chunk loops at each offset
- **Sparse Matrix NONZER?** N/A
- **Histogram Strategy?** N/A
- **Kernel Granularity Check:** good after fusion
- **Preferred GPU pragma form:** `target teams distribute parallel for` for the chunk updates, with a scalar `target` region for the diagonal block

## Scalability Check
- **Default correctness size:** 32
- **Larger practical profiling size:** 2048
- **Why this size materially exercises the GPU:** it produces enough block offsets and trailing-block updates to amortize launch overhead and keep the device busy beyond a toy run
- **Likely small-input sensitive?** YES
- **Scaling risk if this design is chosen:** kernel count at small sizes
- **Chosen structure still plausible at larger size?** YES

**Summary:** 5 logical array/scratch items (`a`, `temp`, `temp_top`, `temp_left`, `sum`), 4 timed functions/regions, Strategy A, offload unit fused routine. Expected: ~16 MB H->D, ~16 MB D->H, 3 device phases per block offset.

## Build / Run Readiness
- **Plain build command works?** YES
- **Plain `make -f Makefile.nvc run` works?** YES
- **Unresolved placeholders remaining?** NO
- **`{nsys_profile_cmd} > {profile_log_path} 2>&1` shows GPU kernels?** YES
