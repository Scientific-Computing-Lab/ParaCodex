# Correctness Verification

**Kernel:** `/root/codex_baseline/custom_serial_to_omp_workdir_20260412112457/data/src/b+tree-omp`
**Candidate files:** `kernel/kernel_cpu.c`, `kernel/kernel_cpu_2.c`, `main.c`, `util/num/num.c`, `util/timer/timer.c`
**Golden reference:** `/root/codex_baseline/custom_serial_to_omp_workdir_20260412112457/golden_labels/src/b+tree-serial/main.c`
**Status:** PASS

## GATE Checks

| Metric | Golden | Candidate | Match |
|--------|--------|-----------|-------|
| Gate harness run | PASS | PASS | ✓ |

## Fixes Applied
- Added `gate.h` instrumentation to the golden and candidate source files so the correctness harness could compare final query-result state.
- Added `gate.h` includes to the requested candidate source files.
- Adjusted `Makefile.nvc` so the correctness workflow can build a separate reference binary in the candidate tree and run the gate harness successfully.

## Makefile Changes
- Updated `GATE_ROOT` to the workspace root.
- Pointed the reference build at a separate `b+tree.ref.out` binary.
- Made the reference `clean` step non-fatal when files are absent.
- Ensured the reference build uses the GPU offload flags required by the candidate build.
