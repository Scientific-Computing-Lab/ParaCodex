# Correctness Verification

**Kernel:** `/root/codex_baseline/hotspot_serial_to_omp_baseline_workdir_20260413145546/data/src/hotspot-omp`
**Candidate files:** `hotspot_openmp.cpp`
**Golden reference:** `/root/codex_baseline/hotspot_serial_to_omp_baseline_workdir_20260413145546/golden_labels/src/hotspot-serial/hotspot_openmp.cpp`
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| `final_output` | emitted by harness | emitted by harness | ✓ |

## Fixes Applied
- Restored the hotspot stencil in `hotspot_openmp.cpp` so the GPU offload path matches the golden edge/corner handling.
- Added `gate.h` and a guarded `GATE_STATS_F32("final_output", ...)` on the final output buffer in both candidate and golden files.

## Makefile Changes
- Corrected `GATE_ROOT` to the workspace root so `gate.h` resolves.
- Pointed `REF_BIN` at `golden_labels/src/hotspot-serial/hotspot` and `REF_MAKE` at `Makefile`.
- Updated the golden build recipes to pass `CC_FLAGS` instead of the candidate-only `CFLAGS`/GPU flags.
