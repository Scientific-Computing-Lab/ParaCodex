# Correctness Verification

**Kernel:** /root/codex_baseline/lud_serial_to_omp_baseline_workdir_20260413124213/data/src/lud-omp
**Candidate files:** base/lud.c, base/lud_base.c, common/common.c, omp/lud.c, omp/lud_omp.c, tools/gen_input.c
**Golden reference:** /root/codex_baseline/lud_serial_to_omp_baseline_workdir_20260413124213/golden_labels/src/lud-serial/common/common.c
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| matrix:f32 stats | min=0.0154986382 max=10 mean=0.501613292 L1=32873.7287 L2=218.544832 | min=0.0154986382 max=10 mean=0.501611657 L1=32873.6215 L2=218.544373 | ✓ |

## Fixes Applied
- Restored the blocked LU offload kernel in `omp/lud_omp.c` so the candidate follows the reference decomposition structure.
- Added `GATE_STATS_F32` verification in `omp/lud.c` and the matching golden file to compare the final matrix numerically.

## Makefile Changes
- Fixed `GATE_ROOT` to point at the repository root.
- Corrected the reference binary path to the golden offload executable.
- Pointed the reference build at the golden `omp/Makefile.offload` and added offload link flags for verification builds.
