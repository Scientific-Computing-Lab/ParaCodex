---
name: paracodex-ocl-cuda-step1
description: "ParaCodex prompt for ocl cuda step1 step."
---

# OpenCL to CUDA Migration - Implementation
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- CUDA 11.0 or later
- Verify correctness against baseline
- Treat combined GPU kernel + memcpy + sync cost as the real performance target; do not accept a design that only improves kernel time.
- Prefer a CUDA structure that remains efficient as the problem grows; reject designs that only win on tiny runs by adding scaling-hostile launch or transfer overhead.

## Workflow

### 1. Get Baseline
```bash
# Baseline OpenCL output is in baseline_output.txt in {kernel_dir}/
```

### 2. Create CUDA Migration Plan
**MANDATORY:** Create `cuda_migration_plan.md` in `{kernel_dir}` using template in `references/output.md`.
**Reference:** `references/examples.md` (Indexing, Dynamic Shared, etc.).

Fill plan from `analysis.md`:
- Map OpenCL primitives to CUDA.
- Plan Atomic and Math function conversions.
- Plan Memory management replacements.
- Choose whether to preserve OpenCL kernel boundaries or fuse/simplify them for CUDA.
- Record the default correctness size and one larger practical profiling size.
- Choose that larger profiling size using `system_info_summary.txt`; prefer the largest short-run, memory-safe input that should load the GPU meaningfully on this device, not merely a token size increase.

### 3. Implement Migration Plan

#### Phase 1: Kernel Code Translation (.cu)
- Convert `__kernel` → `__global__`, `__local` → `__shared__`.
- Convert indexing (`get_global_id` → `blockIdx*blockDim + threadIdx`).
- Handle `barrier()` and `atomic_add`.
- **Reference:** `references/examples.md` (Indexing & Dynamic Shared).
- If the OpenCL structure is awkward or too fragmented for CUDA, rewrite the hot path instead of preserving every enqueue boundary literally.

#### Phase 2: Host Code Translation (.cu)
- Replace OpenCL boilerplate with CUDA setup.
- Create a CUDA stream for async execution:
  ```cpp
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  ```
- Replace `clCreateBuffer`/`clEnqueueWriteBuffer` with `cudaMalloc` + `cudaMemcpyAsync(..., stream)`.
- Convert Launch: `clEnqueueNDRangeKernel` → `kernel<<<grid, block, 0, stream>>>`.
- Transfer results back asynchronously: `cudaMemcpyAsync(h_out, d_out, ..., cudaMemcpyDeviceToHost, stream)`.
- Synchronize with `cudaStreamSynchronize(stream)` and destroy with `cudaStreamDestroy(stream)`.
- Block size: 256 threads (multiple of 32); grid = `(N + 255) / 256`.
- **Reference:** `references/examples.md` (Launch Mapping & Error Handling).

#### Phase 3: Verify Checklist
- Check all conversions in `references/output.md` checklist.
- Add `CUDA_CHECK` macros.
- Replace all placeholders in generated files (`<RUN_ARGS>`, `<PROGRAM_NAME>`, `<SOURCE_FILES>`, etc.) with real values.
- Ensure the produced build/run path works without manual placeholder overrides.

### 4. Build and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > cuda_output.txt 2>&1
```

**Debug Issues:**
- Wrong results? Check Grid/Block dimensions logic.
- Runtime errors? Check Shared Memory limits.

### 5. Verify Correctness
```bash
diff baseline_output.txt cuda_output.txt
```

### 6. Profile
```bash
{clean_cmd_str}
{profile_cmd_str} > {profile_log_path} 2>&1
```

### 7. Step1 Exit Criteria
- CUDA kernel structure is plausible for performance.
- Avoidable OpenCL-era fragmentation or host overhead has been removed.
- Generated build/run files contain no unresolved placeholders.
- The normal run path works without ad hoc variable substitution.
- `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- The chosen CUDA structure is still plausible for larger practical inputs, not only the smallest tested run, and the larger profiling size must materially exercise the GPU rather than merely increasing runtime by a small constant factor.

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- ONLY EDIT SOURCE CODE IN: {file_listing}
- **REMOVE ALL OpenCL API CALLS.**
- **CONVERT ALL SYNTAX TO CUDA.**
- **VERIFY CORRECTNESS.**
