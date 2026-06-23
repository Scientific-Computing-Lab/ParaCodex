# Correctness Verification

**Kernel:** /root/codex_baseline/custom_serial_to_omp_workdir_20260413084046/data/src/b+tree-omp
**Candidate files:** kernel/kernel_cpu.c, kernel/kernel_cpu_2.c, main.c, util/num/num.c, util/timer/timer.c
**Golden reference:** /root/codex_baseline/custom_serial_to_omp_workdir_20260413084046/golden_labels/src/b+tree-serial/main.c
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| k_ans:bytes | b44507848f422328 | b44507848f422328 | ✓ |
| j_recstart:u32 | e8e71bf8fcd89711 | e8e71bf8fcd89711 | ✓ |
| j_reclength:u32 | 2fc01f2a7a13f233 | 2fc01f2a7a13f233 | ✓ |

## Fixes Applied

- Added verification-only GATE checksums for the `k` and `j` command result buffers in the candidate `main.c`.
- No candidate kernel logic changes were required; the OpenMP GPU-offloaded kernels already matched the reference outputs.

## Makefile Changes

- Fixed `data/src/b+tree-omp/Makefile.nvc` so `GATE_ROOT` resolves to the workspace root and the reference verification build uses a compatible `Makefile.nvc` plus `EXTRA_CFLAGS`.
- Added a minimal `golden_labels/src/b+tree-serial/Makefile.nvc` so the instrumented reference can be built for the correctness harness.
