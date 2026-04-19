# Correctness Verification

**Kernel:** `/root/codex_baseline/custom_serial_to_omp_workdir_20260412123745/data/src/backprop-omp`
**Candidate files:** `backprop.c, backprop_kernel.c, facetrain.c, imagenet.c`
**Golden reference:** `/root/codex_baseline/custom_serial_to_omp_workdir_20260412123745/golden_labels/src/backprop-serial/imagenet.c`
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| `loaded_input_units` | matched | matched | ✓ |
| `gpu_input_units` | matched | matched | ✓ |
| `gpu_hidden_units` | matched | matched | ✓ |
| `gpu_output_units` | matched | matched | ✓ |
| `gpu_hidden_delta` | matched | matched | ✓ |
| `gpu_output_delta` | matched | matched | ✓ |
| `gpu_input_weights` | matched | matched | ✓ |
| `gpu_hidden_weights` | matched | matched | ✓ |
| `gpu_input_prev_weights` | matched | matched | ✓ |
| `gpu_hidden_prev_weights` | matched | matched | ✓ |
| `gpu_out_err` | matched | matched | ✓ |
| `gpu_hid_err` | matched | matched | ✓ |

## Fixes Applied

- Added `gate.h` instrumentation to the serial reference loader and the OpenMP loader so the harness can compare the loaded input vector.
- Fixed the OpenMP training kernel to keep GPU offloading while correcting data movement:
  - copied updated weight matrices back to the host with `map(tofrom: ...)`
  - synchronized the hidden bias unit back to the device with `target update`
  - computed the final scalar error values on the host from the offloaded deltas for deterministic comparison
- Added gate checks for the final network state after training, using row-major serialization for 2D weight buffers so the serial and OpenMP allocation layouts compare identically.
- Kept the computation on the GPU; no CPU-only fallback was introduced.

## Makefile Changes

- Updated [`Makefile.nvc`](./Makefile.nvc) to resolve `GATE_ROOT` from the repository root instead of the `data/` subdirectory.
- Updated [`Makefile.nvc`](./Makefile.nvc) to use the golden tree's `Makefile` for `ref_build`.
- Updated the golden [`Makefile`](../../golden_labels/src/backprop-serial/Makefile) so the injected `CFLAGS` include path reaches `gate.h`.
