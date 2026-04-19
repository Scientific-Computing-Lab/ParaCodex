# Correctness Verification

**Kernel:** `/root/codex_baseline/lud_serial_to_omp_workdir_20260413125118/data/src/lud-omp`
**Candidate files:** `base/lud.c, base/lud_base.c, common/common.c, omp/lud.c, omp/lud_omp.c, tools/gen_input.c`
**Golden reference:** `/root/codex_baseline/lud_serial_to_omp_workdir_20260413125118/golden_labels/src/lud-serial/common/common.c`
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| `lu_final` stats | matched | matched | ✓ |

## Fixes Applied
- Added a shared `gate_lu_stats()` helper in `common/common.c` and the golden reference, then called it after the LU kernel returns in both `base/lud.c` and `omp/lud.c`.
- Kept the OpenMP offload path intact; no CPU fallback was introduced.

## Makefile Changes
- Updated `Makefile.nvc` so verification builds can include `gate.h`, use the actual reference makefile, pass offload link flags through to the reference build, and point `REF_BIN` at `golden_labels/src/lud-serial/omp/lud_omp`.
