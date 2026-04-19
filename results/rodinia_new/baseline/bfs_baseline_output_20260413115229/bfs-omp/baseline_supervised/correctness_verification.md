# Correctness Verification

**Kernel:** /root/codex_baseline/bfs_baseline_workdir_20260413115229/data/src/bfs-omp
**Candidate files:** bfs.cpp
**Golden reference:** /root/codex_baseline/bfs_baseline_workdir_20260413115229/golden_labels/src/bfs-serial/bfs.cpp
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| `h_cost_final` | harness matched | harness matched | ✓ |

## Fixes Applied
- Added `gate.h` includes and wrapped final `h_cost` checksums in `#ifdef GATE_VERIFY` on both the candidate and golden reference.
- Kept the OpenMP GPU offload path intact; no CPU fallback was introduced.

## Makefile Changes
- Updated `data/src/bfs-omp/Makefile.nvc` to point `GATE_ROOT` at the repository root, use the existing golden `Makefile`, and reference the correct golden binary name `bfs`.
- Updated `golden_labels/src/bfs-serial/Makefile` to append injected `CFLAGS` so verification builds can pick up `-DGATE_VERIFY` and the `gate_sdk` include path.
