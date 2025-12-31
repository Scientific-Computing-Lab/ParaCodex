# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `v59` | `v8 * sizeof(double)` ≈ 80 bytes | working | initialized to 0 before timer | host/device R/W | 
| (per-thread) `local_hist[10]` | 10 × `sizeof(double)` = 80 bytes | scratch | zeroed inside each sample | device-private R/W |

**Types:** working (main data), scratch (temp), const (read-only), index (maps)

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| `main` (sample loop) | `v59` | per-sample | device |

## Data Movement Strategy

**Chosen Strategy:** A (target data region)

**Device Allocations (once):**
- `v59[0:v8]` mapped into `#pragma omp target data map(tofrom:v59[0:v8])`

**Host→Device Transfers:**
- When: once before timed compute
- Arrays: `v59` → device (zeroed on host before entering target region)
- Total H→D: ~0.00008 MB

**Device→Host Transfers:**
- When: once after timed compute finishes
- Arrays: `v59` ← device (final histogram)
- Total D→H: ~0.00008 MB

**Transfers During Iterations:** NO — all data stays resident inside target data region while samples loop runs.

## Critical Checks (for chosen strategy)
**Strategy A:**
- [x] Timed compute runs inside `#pragma omp target data map(tofrom:v59[0:v8])`
- [x] Per-thread scratch (`local_hist`) kept private, no extra maps needed

**Common Mistakes:**
- Forgetting to keep `v59` present for both CUDA teams and reduction updates
- Missing atomic merges when multiple threads update the same bin

## Expected Transfer Volume
- Total: ~0.00016 MB for entire execution
- **Red flag:** Actual transfer >2× expected would imply unnecessary data movement (none expected)

## Additional Parallelization Notes
- **RNG Replicable?** YES → keep `f1` host-only but the per-sample RNG work on device, seeded using sample index
- **Outer Saturation?** `v3` samples (dense per-sample work with `v7=65536` gaussian pairs)
- **Sparse Matrix NONZER?** N/A
- **Histogram Strategy?** `BINS=10 ≤ 100` → use per-thread local array (`local_hist[10]`) and atomic merge to global `v59`

**Summary:** 1 working array (`v59`), 0 persistent scratch arrays (per-thread locals), 1 function actively offloaded (`main` sample loop), Strategy A, expected transfers <0.0002 MB. EOF
