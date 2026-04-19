---
name: paracodex-cuda-omp-step1
description: "ParaCodex prompt for cuda omp step1 step."
---

# CUDA to OpenMP Migration
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for all runs
- Prefer `target teams loop` first for OpenMP 5+ GPU offload, but keep or switch to `target teams distribute parallel for` when it better matches the original CUDA structure or produces better device code on this compiler/runtime.
- **REMOVE ALL CUDA API CALLS** (`cudaMalloc`, `cudaMemcpy`, `kernel<<<>>>`)
- **CONVERT ALL __global__ FUNCTIONS TO REGULAR FUNCTIONS**
- Treat combined GPU kernel + memcpy + sync cost as the real performance target; do not accept a design that only improves kernel time.
- Prefer an OpenMP structure that remains efficient as the problem grows; reject designs that only win on tiny runs by adding scaling-hostile launches, transfers, or host waits.

**IMPORTANT:** YOU MAY MODIFY THE MAKEFILE TO ADD ANYTHING YOU NEED TO RUN THE CODE.

**Structural Goal:** Recover or improve on the original CUDA hot-path structure. Do not preserve a weak literal translation if a fused or flattened OpenMP offload unit will perform better.

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline
```bash
# Baseline cuda output is in baseline_output.txt in {kernel_dir}/
```

### 2. Choose Data Strategy
Walk through IN ORDER, stop at first match:
```
RULE 1: Type B (Sparse/CSR)?              → STRATEGY A/C
RULE 2: Type C1 (Iterative Solvers)?      → STRATEGY C
RULE 3: Type C2 (Multigrid)?              → STRATEGY A
RULE 4: Multiple independent kernels?     → STRATEGY B
RULE 5: Otherwise                         → STRATEGY A
```

### 3. Create Data Management Plan
**MANDATORY:** Create `data_plan.md` in `{kernel_dir}` using the template in `references/output.md`.

**FIRST: Understand CUDA memory model and map to OMP:**
- `cudaMalloc` → `omp_target_alloc` OR `target data map(alloc)`
- `cudaMemcpy H→D` → `map(to)` OR `omp_target_memcpy`
- `cudaMemcpy D→H` → `map(from)` OR `omp_target_memcpy`
- Kernel launches → `target teams loop` with `is_device_ptr`

**Pattern Recognition:** See `references/examples.md`.

Analyze ALL arrays and kernels in timed region.
- The plan must decide whether to preserve CUDA kernel boundaries, fuse them, or split them for correctness.
- The plan must record the default correctness size and one larger practical profiling size.
- The larger profiling size must be chosen to expose sustained GPU behavior on the available hardware, not just millisecond-scale launch or transfer overhead.
- Increase the profiling size until the GPU work is materially exercised or until memory/time limits make further growth impractical; do not stop at a token size increase.
- Choose that size using `system_info_summary.txt`; push to the largest short-run, memory-safe input that should load the GPU meaningfully on this device.
- If the benchmark framework offers multiple classes or problem sizes, prefer the largest size that fits comfortably and completes in a practical time budget.

### 4. Implement Data Plan
**Use `data_plan.md` as implementation guide.**
**Reference:** `references/examples.md` (Migration Pattern & Syntax Limitations).

#### Step 4.1: Remove CUDA API Calls
- Remove `cudaMalloc`/`cudaFree`.
- Remove `cudaMemcpy`.
- Remove `<<<grid, block>>>`. Use OpenMP pragmas.

#### Step 4.2: Convert Kernels to Functions
- Convert `__global__` void functions to standard C void functions.
- Add `#pragma omp target teams loop is_device_ptr(...)` inside the function.
- **Reference:** See Conversion Example in `references/examples.md`.
- If preserving the CUDA kernel boundary creates fragmented tiny regions or poor OpenMP mapping, inline/fuse the affected kernels instead.

#### Step 4.3: Setup Data Structures & Transfers
- Create OMP allocations/transfers based on logic in `data_plan.md`.
- Match timing from original CUDA code.
- Use `nowait` on independent `target` regions to allow async dispatch; synchronize with `#pragma omp taskwait` only where results are needed:
  ```c
  #pragma omp target teams loop nowait depend(out: d_a[0:N])
  for (int i = 0; i < N; i++) { ... }
  #pragma omp taskwait  // wait before D→H transfer
  ```

#### Step 4.4: Convert Thread Indexing
- Replace `blockIdx/threadIdx` with loop iterators.
- Remove `if (idx < N)` guards (handled by loop bounds).
- Convert grid-stride loops to simple loops.

#### Step 4.5: Handle Special CUDA Constructs
- `atomicAdd` → `#pragma omp atomic update`
- `__syncthreads()` → Split kernel OR remove if not critical.
- Shared memory → Per-thread private OR elimination.
- **Limitation:** OpenMP clauses cannot access struct members directly (e.g. `data.arr` BAD). Extract to `double *d_arr = data.arr` FIRST (See `references/examples.md`).

#### Step 4.6: Build / Run Readiness
- Replace all placeholders in generated files (`<RUN_ARGS>`, `<PROGRAM_NAME>`, `<SOURCE_FILES>`, etc.) with real values.
- Ensure the produced `Makefile.nvc` supports a plain `make -f Makefile.nvc run` with no ad hoc variable overrides.

### 5. Compile and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > gpu_output.txt 2>&1
```

### 6. Verify Correctness
```bash
diff baseline_output.txt gpu_output.txt
```

### 7. Profile
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10
```

### 8. Step1 Exit Criteria
- The CUDA hot path has been migrated to working OpenMP GPU offload.
- The generated build/run files contain no unresolved placeholders.
- A plain `make -f Makefile.nvc run` succeeds as the final run check.
- `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- The chosen OpenMP structure is still plausible for larger practical inputs, not only the smallest tested run, and the larger profiling size must materially exercise the GPU rather than merely increasing runtime by a small constant factor.

# Rules
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- ALWAYS CLEAN BEFORE BUILD.
- Remove ALL CUDA-specific syntax.
