# SYCL Migration Plan

## Phase 1: Source Conversion
### Tool Selection
- [ ] Manual conversion (Fallback if tools not installed)
- [ ] `dpct` (Intel DPC++ Compatibility Tool) / `c2s` (SYCLomatic) (If available)
- [ ] `intercept-build` (for complex Makefiles)

### File Mapping
| Original CUDA | Target SYCL | Status |
| :--- | :--- | :--- |
| `kernel.cu` | `kernel.dp.cpp` | [ ] |
| `main.cu` | `main.dp.cpp` | [ ] |

## Phase 2: Manual Fixes (Post-Migration)
### Warnings Review
- [ ] Review all `DPCTxxxx` warnings.
- [ ] Fix `DPCT1003` (Error code to Exception).
- [ ] Fix `DPCT1065` (Atomic sync).

### Optimization & Cleanup
- [ ] Verify `nd_item` usage vs `item`.
- [ ] Check for `sycl::malloc_device` USM usage (preferred) vs Buffers.
- [ ] Choose final decomposition: preserve kernels / fuse kernels / rewrite hot path.
- [ ] Remove avoidable `.wait()` calls from the hot path.
- [ ] Ensure the submission structure is coherent for end-to-end runtime.
- [ ] Budget combined GPU kernel + memcpy + sync cost, not just kernel time.
- [ ] Record the default correctness size and one larger practical profiling size.
- [ ] Reject a submission structure that only wins on tiny inputs while scaling poorly.

## Phase 3: Build System
- [ ] Use the provided Makefile (do not force `icpx`).
- [ ] Link against `sycl` and `dpct` helper headers if used.

## Phase 4: Verification
- [ ] Compile using the provided Makefile.
- [ ] Run on destination device (GPU) using `ONEAPI_DEVICE_SELECTOR=cuda:gpu`.
- [ ] Compare output with baseline.
- [ ] Confirm the step1 structure is a good optimization base.
- [ ] All generated placeholders resolved.
- [ ] Plain run path works without ad hoc overrides.
- [ ] `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- [ ] The chosen structure is still plausible at the larger practical profiling size.
