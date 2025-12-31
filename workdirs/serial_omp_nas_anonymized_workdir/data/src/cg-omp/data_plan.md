# Data Management Plan

## Arrays Inventory

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `colidx` | 89600 entries (~0.35 MB) | index | Host (built by `makea` + shift) | Device read-only during CG |
| `rowstr` | 1401 entries (~5.5 KB) | index | Host (built by `sparse`) | Device read-only |
| `a` | 89600 entries (~0.68 MB) | const | Host (constructed by `makea`) | Device read-only SpMV data |
| `x` | 1402 entries (~11 KB) | working | Host (initialized to 1.0) | Device R/W during solver; host read/written between iterations |
| `z` | 1402 entries (~11 KB) | working | Host (zeroed before warm-up) | Device R/W during solver; host read for norm/update after each CG |
| `p` | 1402 entries (~11 KB) | working | Host (copied from `r`) | Device R/W inside CG |
| `q` | 1402 entries (~11 KB) | working | Host (initialized to 0) | Device R/W inside CG |
| `r` | 1402 entries (~11 KB) | working | Host (`r[j]=x[j]`) | Device R/W inside CG |

## Functions in Timed Region

| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| `conj_grad` (main CG kernel) | `a`, `colidx`, `rowstr`, `x`, `z`, `p`, `q`, `r` | per CG iteration | device (offloaded loops via `target teams loop`) |
| `compute norms & x update` | `x`, `z` | per CG iteration | host (sequential norm + x scaling) |

## Data Movement Strategy

**Chosen Strategy:** C (Global device state)

**Device Allocations (once):**
- `d_colidx` (`89600` ints) via `omp_target_alloc`
- `d_rowstr` (`1401` ints) via `omp_target_alloc`
- `d_a` (`89600` doubles) via `omp_target_alloc`
- `d_x`, `d_z`, `d_p`, `d_q`, `d_r` (`1402` doubles each) via `omp_target_alloc`

**Host→Device Transfers:**
- When: once after initialization, plus once per iteration for `x` after the host update
- Arrays: `colidx`, `rowstr`, `a` (~1.04 MB once); `x` (~11 KB) at start and after each host `x` update
- Total H→D per iteration: ~0.011 MB (after iteration); start-up cost ~1.05 MB

**Device→Host Transfers:**
- When: after each `conj_grad` call (before norm computations)
- Arrays: `z` (~11 KB) transferred D→H per iteration so host can do norms and update `x`
- Total D→H per iteration: ~0.011 MB

**Transfers During Iterations:** YES
- `z` is copied back to host after each CG iteration for the norm and scaling steps because the host still manages the outer loop and validation prints. `x` is copied back to the device right after the host computes the scaled values so the next CG iteration has the latest vector.

## Critical Checks (for chosen strategy)

- [x] ALL device-critical loops use `is_device_ptr` to reference the `d_*` arrays when launching `target teams loop`
- [x] Scratch arrays exist only on device via `omp_target_alloc` (no extra host scratch buffers needed)
- [x] No `map()` clauses appear inside the repeated CG loop—only explicit `omp_target_memcpy` and kernel launches
- [ ] `conj_grad` reductions and stage logic are fully offloaded with matching reduction clauses (will verify after implementation)
- [ ] Host norms and updates only operate on host copies resulting from the latest D→H transfer

**Common Mistakes to Avoid:**
- Mixing host and device pointers in the inner loops (ensured by passing `d_*` pointers instead of host data)
- Forgetting to copy the updated `x` back before the next CG iteration (handled via explicit H→D memcpy after the host loop)
- Leaving any zero-initialized arrays only on host and then dereferencing on device (all `d_*` arrays are populated before offload)

## Expected Transfer Volume
- Total (one-time): ~1.05 MB (matrix + structure + initial vectors)
- Per iteration: ~0.022 MB (11 KB D→H for `z`, 11 KB H→D for updated `x`)
- **Red flag:** If transfers exceed ~0.05 MB/iteration, data movement is not optimal

## Additional Parallelization Notes
- **Hardware note:** NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace) with 8 GB VRAM is the offload target; use `target teams loop` to keep enough work per team and avoid `distribute parallel for` since it is forbidden.
- **RNG replicable?** Not applicable (no RNG on hot path)
- **Outer Saturation?** CG runs for `NITER=15`; each iteration launches the entire SpMV/reduction sequence on device
- **Sparse Matrix NONZER:** 7 (SpMV depth is small; inner loop uses per-row sequential scan, consistent with Type C1)
- **Histogram Strategy?** Not applicable (no histogram usage)

**Summary:** 7 arrays (5 working, 2 index/const) and 1 main device-offloaded function (`conj_grad`). Strategy C keeps arrays resident with `omp_target_alloc`/`is_device_ptr`. Expect ~1.05 MB one-time transfers and ~0.022 MB per iteration. This keeps data movement minimal and matches the RTX 4060 device's capacity.
