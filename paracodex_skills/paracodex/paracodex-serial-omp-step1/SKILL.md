---
name: paracodex-serial-omp-step1
description: "ParaCodex prompt for serial omp step1 step."
---

# GPU Offload with OpenMP
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`
**Required:** 
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for all runs
- Prefer `target teams loop` first for OpenMP 5+ GPU offload, but keep or switch to `target teams distribute parallel for` when it produces better device code or more reliable compiler/runtime behavior on this target.
- Treat combined GPU kernel + memcpy + sync cost as the real performance target; do not accept a design that only improves kernel time.
- Prefer an OpenMP structure that remains efficient as the problem grows; reject designs that only win on tiny runs by adding scaling-hostile launches, transfers, or host waits.

**IMPORTANT:** YOU MAY MODIFY THE MAKEFILE TO ADD ANYTHING YOU NEED TO RUN THE CODE.

**Structural Goal:** Choose the right offload unit before adding pragmas. Do not preserve a weak serial helper decomposition if a fused or flattened OpenMP offload unit will perform better.

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > baseline_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > baseline_output.txt 2>&1
grep -E "Verification|SUCCESSFUL|FAILED" baseline_output.txt
DO NOT SKIP THIS STEP.
```

### 2. Choose Data Strategy
Walk through IN ORDER, stop at first match:
```
RULE 1: Type B (Sparse/CSR)?              → STRATEGY A/C
RULE 2: Type C1 (Iterative Solvers).      → STRATEGY C
RULE 3: Type C2 (Multigrid)?              → STRATEGY A
RULE 4: Outer A + inner E (per-thread RNG) → STRATEGY A
RULE 5: Multiple independent kernels?     → STRATEGY B
RULE 6: Otherwise                         → STRATEGY A
```

### 3. Create Data Management Plan
**MANDATORY:** Create `data_plan.md` in `{kernel_dir}` using the template in `references/output.md`.
**Reference:** `references/output.md` (Use entire file as template).

**FIRST: Check if original algorithm can be simplified for GPU:**
- Large scratch arrays for intermediate results → Can per-thread locals replace them?
- Block-based iteration (for cache) → REMOVE blocking, use single parallel loop over ALL work items
- Multi-stage with host sync → Can everything run in one kernel?

**Rule:** If scratch arrays exist ONLY to avoid atomics on small data (<1KB), 
DELETE them and use per-thread locals + atomic merge instead.

**Block elimination:** If code has `for (blk = 0; blk < numblks; blk++)` with scratch arrays,
this is a CPU cache optimization. For GPU: remove blocking, parallelize over all N items directly.

**CUDA-serial port (Type H):** Eliminate the inner fake-thread loop entirely.
Rewrite as scalar logic per outer work-item. Remove from map clauses any arrays
only accessed inside the eliminated loop.

Analyze ALL arrays and functions in the timed region to fill the plan.
- The plan must decide whether to preserve the original loop/helper boundaries, fuse them, or rewrite them for GPU mapping.
- The plan must record the default correctness size and one larger practical profiling size.
- The larger profiling size must be chosen to expose sustained GPU behavior on the available hardware, not just millisecond-scale launch or transfer overhead.
- Increase the profiling size until the GPU work is materially exercised or until memory/time limits make further growth impractical; do not stop at a token size increase.
- Choose that size using `system_info_summary.txt`; push to the largest short-run, memory-safe input that should load the GPU meaningfully on this device.
- If the benchmark framework offers multiple classes or problem sizes, prefer the largest size that fits comfortably and completes in a practical time budget.
- Do not choose a structure only because it wins on a toy input if it creates per-stage, per-row, or per-launch overhead that will scale badly.

**CRITICAL: After creating data_plan.md, you MUST proceed to Implement Data Plan in this same step1 phase. Do not stop after planning.**

### 4. Implement Data Plan (REQUIRED - part of step1)
**Use `data_plan.md` as implementation guide.**
**Reference:** `references/examples.md` (Contains Code Examples, Map Clauses, and Parallelization Patterns).

#### Step 4.1: Setup Data Structures
From "Arrays Inventory" and "Data Movement Strategy" in your plan:
- Declare device arrays/pointers as needed for chosen strategy.
- Create allocation/initialization functions based on strategy (See `references/examples.md` for STRATEGY A/B/C details).

#### Step 4.2: Implement Transfers
From "H→D Transfers" and "D→H Transfers" sections:
- Implement each transfer listed with timing specified in plan.
- Use method appropriate for strategy (map clauses, omp_target_memcpy, update, etc.).

#### Step 4.3: Offload Functions
Use "Functions in Timed Region" table:
- For each function where "Must Run On" = device:
  - Add appropriate pragma for strategy (See `references/examples.md`).
  - Include arrays from "Arrays Accessed" column.
- If preserving helper boundaries creates fragmented tiny kernels, helper launch overhead, or repeated host/device sync, inline/fuse the affected stages instead of offloading each helper literally.

#### Step 4.4: Main Program Flow
Follow "Data Movement Strategy" timing:
```
[setup from plan]
[H→D transfers at specified time]
[timed computation - call functions]
[D→H transfers at specified time]
[cleanup]
```

#### Step 4.5: Profile Size Selection
- Pick a default correctness size for fast validation and a separate larger profiling size for performance decisions.
- The larger profiling size must be large enough to materially exercise the GPU on this hardware; avoid sizes that still produce only tiny, launch-dominated work unless the benchmark is inherently small.
- Base that choice on `system_info_summary.txt`. The target is the largest short-run, memory-safe size that should use the device meaningfully, not merely a modest increase over the default.
- If the first larger size is still weakly utilized or dominated by fixed launch/setup cost, increase it again within the practical memory/time budget and record why the previous size was insufficient.
- Treat the larger profiling size as the performance reference for step1 structure choices.

### 5. Update Makefile (REQUIRED)
**CRITICAL: Update Makefile to ensure it compiles successfully.**
- Check if code uses files from subdirectories (e.g., `common/`, `utilities/`).
- Update `source` variable to include all required source files.
- Update `obj` variable.
- Add include paths (`CFLAGS += -Icommon`).
- Ensure `make -f Makefile.nvc` works.
- Replace all placeholders in generated files (`<RUN_ARGS>`, `<PROGRAM_NAME>`, `<SOURCE_FILES>`, etc.) with real values.
- Ensure the produced `Makefile.nvc` supports a plain `make -f Makefile.nvc run` with no ad hoc variable overrides.
- If a compiler switch is required, document the exact failure and
    why the switch was necessary

### 6. Compile and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {correctness_run_cmd} > gpu_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > gpu_output.txt 2>&1
```
If timeout/segfault: Remove `#pragma omp loop` from Type C inner loops.

### 7. Verify Correctness
```bash
diff baseline_output.txt gpu_output.txt
```

### 8. Profile
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found"
```

### 9. Step1 Exit Criteria
Before finishing step1, verify all of the following:
- The hot compute path is on the GPU.
- The offload unit is appropriate for the workload size; do not leave a many-stage tiny-kernel design in place when fusion is possible.
- There are no avoidable `target update` operations in the timed region.
- The implementation is a plausible optimization base for step2, not just a minimally offloaded version.
- The produced `Makefile.nvc` and related files contain no unresolved placeholders.
- A plain `make -f Makefile.nvc run` succeeds as the final run check.
- `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- The chosen structure is still plausible for larger practical inputs, not only the smallest tested run, and the larger profiling size must materially exercise the GPU rather than merely increasing runtime by a small constant factor.

# Rules
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- ALWAYS CLEAN BEFORE BUILD.
- You may create backup/output files.
- Never run git commands.
