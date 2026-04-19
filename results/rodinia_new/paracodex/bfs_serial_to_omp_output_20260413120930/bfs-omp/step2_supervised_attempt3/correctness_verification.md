# Correctness Verification

**Kernel:** `/root/codex_baseline/bfs_serial_to_omp_workdir_20260413120930/data/src/bfs-omp`
**Candidate files:** `bfs.cpp`
**Golden reference:** `/root/codex_baseline/bfs_serial_to_omp_workdir_20260413120930/golden_labels/src/bfs-serial/bfs.cpp`
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| `h_cost` | matched | matched | ✓ |

## Fixes Applied
- Added `gate.h` plus a guarded `GATE_CHECKSUM_BYTES("h_cost", ...)` at the end of both golden and candidate BFS programs.
- Kept the OpenMP GPU offload path intact in the candidate.
- Fixed `Makefile.nvc` so `GATE_ROOT` resolves to the workdir root, enabling the gate SDK include path.
- Fixed the reference build wiring so the golden binary uses `Makefile`, targets `bfs`, and receives `gate.h` plus `GATE_VERIFY` through the injected compiler command.

## Makefile Changes
- Updated `GATE_ROOT` to `$(abspath ../../..)`.
- Updated `REF_MAKE` to `Makefile` and `REF_BIN` to `$(REF_DIR)/bfs`.
- Injected `-I$(GATE_ROOT)/gate_sdk -DGATE_VERIFY` into the reference build command via `CC`.
