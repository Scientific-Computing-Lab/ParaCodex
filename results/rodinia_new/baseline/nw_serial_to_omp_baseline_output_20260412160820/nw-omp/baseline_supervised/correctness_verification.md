# Correctness Verification

**Kernel:** /root/codex_baseline/custom_serial_to_omp_workdir_20260412160820/data/src/nw-omp
**Candidate files:** needle.cpp
**Golden reference:** /root/codex_baseline/custom_serial_to_omp_workdir_20260412160820/golden_labels/src/nw-serial/needle.cpp
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| input_itemsets checksum | `65c4552e37550bcd` | `65c4552e37550bcd` | ✓ |

## Fixes Applied
- Added guarded `gate.h` inclusion and a final `GATE_CHECKSUM_BYTES` on the contiguous `input_itemsets` buffer in `needle.cpp` so verification compares the final DP matrix only.

## Makefile Changes
- Updated `/root/codex_baseline/custom_serial_to_omp_workdir_20260412160820/golden_labels/src/nw-serial/Makefile` to pass `$(CFLAGS)` through to the reference build, which enables `-DGATE_VERIFY` during `check-correctness`.
