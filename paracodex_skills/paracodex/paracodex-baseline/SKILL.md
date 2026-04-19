---
name: paracodex-baseline
description: "ParaCodex baseline skill: translate serial / parallel code to GPU-offloaded target API, optimize, compile, and run — all in a single session."
---

# GPU Offload — Baseline (Single Session)

**Source:** `{source_dir}/`
**Target API:** `{target_api}`
**Deliverable directory:** `{kernel_dir}/`
**Files to create:** {file_listing}
**Executable:** `./{executable}`

---

## Your Task

1. **Translate** the serial source code in `{source_dir}/` to a GPU-offloaded `{target_api}` version.
2. **Apply** GPU offloading directives/pragmas/kernels as appropriate for `{target_api}`.
3. **Optimize** the code for GPU performance while preserving its original functionality.
4. **Compile** with `{clean_cmd_str}` then `{build_cmd_str}`, and fix any errors until it succeeds.
5. **Run** with `{run_cmd}` and confirm the output looks correct.
6. **Deliver** the complete, modified source code to `{kernel_dir}/`.

---

## Workflow

### 1. Read the Source Code

Read all relevant files in `{source_dir}/` to understand the data structures, compute kernels, and build requirements.

### 2. Translate to GPU Offload

Write the translated file(s) to `{kernel_dir}/`:
{file_listing}

Offload compute-intensive work to the GPU using `{target_api}` constructs. Keep I/O and initialization on the CPU. Manage data movement explicitly.

### 3. Update Makefile if Needed

Check `{kernel_dir}/Makefile.nvc` and update if required — missing source files, include paths, or linker flags.

### 4. Compile and Fix

```bash
{clean_cmd_str}
{build_cmd_str}
```

Fix any compiler errors and repeat until the build succeeds.

### 5. Run and Verify

```bash
{run_cmd}
```

Confirm the output is correct.

---

## Deliverable

The complete, modified source code for:
{file_listing}

in `{kernel_dir}/`, that compiles with `{build_cmd_str}` and produces correct output when run with `{run_cmd}`.

---

## Rules

- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- ALWAYS CLEAN BEFORE BUILD (`{clean_cmd_str}` before every `{build_cmd_str}`).
- ALWAYS run with exactly `{run_cmd}` — do not substitute or simplify this command.
- The **core compute loops** MUST run on the GPU. Offloading only peripheral or setup code while leaving the main computation on the CPU is a failure — fix it.
- If `{run_cmd}` aborts with an offload error, the GPU pragmas are wrong or missing. Fix the source code. Do NOT remove or relax the run command.
