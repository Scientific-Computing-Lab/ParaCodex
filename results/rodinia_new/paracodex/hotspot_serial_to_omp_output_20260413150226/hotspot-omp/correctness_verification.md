# Correctness Verification

**Kernel:** `/root/codex_baseline/hotspot_serial_to_omp_workdir_20260413150226/data/src/hotspot-omp`
**Candidate files:** `hotspot_openmp.cpp`
**Golden reference:** `/root/codex_baseline/hotspot_serial_to_omp_workdir_20260413150226/golden_labels/src/hotspot-serial/hotspot_openmp.cpp`
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| `final_output` | `n=1048576 min=79.9996 max=80.9605 mean=80.4801504019 L1=84389554.1878 L2=82412.1720` | `n=1048576 min=80.0002 max=80.9599 mean=80.4801508451 L1=84389554.6525 L2=82412.1724` | ✓ |

## Fixes Applied
- Added `gate.h` and a guarded `GATE_STATS_F32` over the final output buffer in both candidate and golden files.
- Kept GPU offload in the candidate kernel and aligned the stencil arithmetic with the reference implementation.

## Makefile Changes
- Updated `REF_MAKE` to `Makefile`.
- Updated `REF_BIN` to the golden binary path and passed `CC_FLAGS` so the reference build picks up `gate.h` during `check-correctness`.
