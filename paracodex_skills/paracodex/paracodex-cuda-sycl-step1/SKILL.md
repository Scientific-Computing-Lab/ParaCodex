---
name: paracodex-cuda-sycl-step1
description: "ParaCodex prompt for cuda to sycl step1 (implementation) step."
---

# CUDA to SYCL Migration - Implementation
**Directory:** `{kernel_dir}/`
**Files:** {file_listing}
**Reference:** `{kernel_dir}/analysis.md`

**Required:**
- Verify correctness against baseline.
- Treat combined GPU kernel + memcpy + sync cost as the real performance target; do not accept a design that only improves kernel time.
- Prefer a SYCL structure that remains efficient as the problem grows; reject designs that only win on tiny runs by adding scaling-hostile submissions, waits, or transfers.

## Workflow

### 1. Get Baseline
```bash
# Baseline CUDA output is in baseline_output.txt in {kernel_dir}/
```

### 2. Create SYCL Migration Plan
**MANDATORY:** Create `sycl_migration_plan.md` in `{kernel_dir}` using template in `references/output.md`.
**Reference:** `references/examples.md` (dpct usage).
- The plan must choose the final SYCL decomposition, preferred memory model, and whether any CUDA kernel structure should be fused or rewritten.
- The plan must record the default correctness size and one larger practical profiling size.
- The larger profiling size must be chosen to expose sustained GPU behavior on the available hardware, not just millisecond-scale launch or transfer overhead.
- Increase the profiling size until the GPU work is materially exercised or until memory/time limits make further growth impractical; do not stop at a token size increase.
- Choose that size using `system_info_summary.txt`; push to the largest short-run, memory-safe input that should load the GPU meaningfully on this device.
- If the benchmark framework offers multiple classes or problem sizes, prefer the largest size that fits comfortably and completes in a practical time budget.

### 3. Implement Migration

#### Phase 1: Source Conversion
If `dpct` (Compatibility Tool) or `syclomatic` is available, you may use it to convert source files.
**CRITICAL:** If `dpct` or `syclomatic` is NOT installed, **DO NOT** attempt to install it via `apt` or by downloading packages. Proceed immediately to perform the migration **manually**.
```bash
# Example if available:
dpct --in-root=. --out-root=dpct_output {source_file}.cu
```

#### Phase 2: Manual Fixes / Best Practices
- Review **Warnings** (DPCTxxxx) and resolve them.
- Fix **Error Handling**: Remove CUDA-style status codes, use SYCL exceptions (`try/catch sycl::exception`).
- **Prefer USM over buffer/accessor model** for performance and code clarity:
  ```cpp
  sycl::queue q{sycl::gpu_selector_v};
  T* d_ptr = sycl::malloc_device<T>(N, q);
  q.memcpy(d_ptr, h_ptr, N * sizeof(T)).wait();
  q.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> it) {
      int i = it.get_global_id(0);
      // kernel body
  }).wait();
  q.memcpy(h_ptr, d_ptr, N * sizeof(T)).wait();
  sycl::free(d_ptr, q);
  ```
- Use `q.submit([&](sycl::handler& h){ ... })` with chained events (`.depends_on()`) to overlap transfers with compute. For strictly linear pipelines, `sycl::property::queue::in_order{}` is a cleaner alternative — no explicit `depends_on` needed:
  ```cpp
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  // All operations execute in submission order automatically
  ```
- Verify **Work Group Sizes**: query at runtime if unsure (see step2).
- If the migrated structure creates many tiny submissions or unnecessary waits, rewrite the hot path before proceeding.

#### Phase 3: Build System
- Use the provided Makefile to build. **Do not** modify the Makefile to use `icpx` (the environment manages the correct SYCL compiler).
- You may still replace placeholders in the provided Makefile or helper scripts (`<RUN_ARGS>`, `<PROGRAM_NAME>`, `<SOURCE_FILES>`, etc.) with real values.

### 4. Build and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > sycl_output.txt 2>&1
```

The final deliverable must also support its normal run path without manual placeholder overrides.

### 5. Verify Correctness
```bash
diff baseline_output.txt sycl_output.txt
```

### 6. Step1 Exit Criteria
- The chosen SYCL memory model is intentional and documented.
- The queue/submission structure is plausible for performance.
- There are no avoidable `.wait()` calls in the hot path.
- Build/run files contain no unresolved placeholders.
- The normal run path works without ad hoc placeholder substitution.
- `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- The chosen queue/submission structure is still plausible for larger practical inputs, not only the smallest tested run, and the larger profiling size must materially exercise the GPU rather than merely increasing runtime by a small constant factor.

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- **DO NOT INSTALL PACKAGES:** If `dpct` is missing, do it manually.
- **VERIFY CORRECTNESS.**
