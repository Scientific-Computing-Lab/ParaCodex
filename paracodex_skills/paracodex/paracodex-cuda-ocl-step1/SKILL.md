---
name: paracodex-cuda-ocl-step1
description: "ParaCodex prompt for cuda ocl step1 step."
---

# CUDA to OpenCL Migration - Implementation
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- OpenCL 1.2 or later compatibility
- Test on target device (GPU)
- Verify correctness against baseline
- Treat combined GPU kernel + memcpy + sync cost as the real performance target; do not accept a design that only improves kernel time.
- Prefer a kernel/program split that remains efficient as the problem grows; reject designs that only win on tiny runs by adding scaling-hostile enqueue, build, or transfer overhead.

**Structural Goal:** Choose the kernel/program split that gives the best end-to-end OpenCL implementation, not necessarily the most literal CUDA translation.

## Workflow

### 1. Get Baseline
```bash
# Baseline CUDA output is in baseline_output.txt in {kernel_dir}/
```

### 2. Create OpenCL Migration Plan
**MANDATORY:** Create `opencl_migration_plan.md` in `{kernel_dir}` using template in `references/output.md`.
**Reference:** `references/examples.md` (Syntax Conversion, Math Functions, etc.).

Analyze `analysis.md` and mapped files to fill the plan.
- Map global/device/shared syntax.
- Plan host code setup.
- Decide whether to preserve CUDA kernel boundaries or fuse/split them for a better OpenCL implementation.
- Record the default correctness size and one larger practical profiling size.
- Choose that larger profiling size using `system_info_summary.txt`; prefer the largest short-run, memory-safe input that should load the GPU meaningfully on this device, not merely a token size increase.

### 3. Implement Migration Plan

#### Phase 1: Kernel Code Translation (.cl)
- Convert `__global__`, `__device__`, `__shared__` to OpenCL keywords.
- Convert indexing (`threadIdx` → `get_local_id`).
- Handle `__syncthreads()` and atomics.
- **Reference:** `references/examples.md` (Common Syntax).
- If the CUDA split is performance-poor or awkward in OpenCL, rewrite the hot path rather than preserving every kernel boundary literally.

#### Phase 2: Host Code Translation (.c/.cpp)
- Add OpenCL Setup Boilerplate (Platform, Context, Queue, Program).
- Create a command queue. Out-of-order execution is **optional in OpenCL 3.0** — query device support before enabling:
  ```c
  // Check support first
  cl_bool ooo_support;
  clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, sizeof(cl_bool), &ooo_support, NULL);
  // Note: CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE is a property bit, check via:
  cl_command_queue_properties queue_props;
  clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                  sizeof(cl_command_queue_properties), &queue_props, NULL);
  cl_bool supports_ooo = (queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;

  cl_queue_properties props[] = {CL_QUEUE_PROPERTIES,
      supports_ooo ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0, 0};
  queue = clCreateCommandQueueWithProperties(ctx, device, props, &err);
  ```
- Replace `cudaMalloc`/`cudaMemcpy` with `clCreateBuffer`/`clEnqueueWriteBuffer`.
- Use `CL_MEM_ALLOC_HOST_PTR` on host-side buffers for pinned memory and faster H↔D transfers.
- Replace kernel launches with `clSetKernelArg` + `clEnqueueNDRangeKernel`.
- Use `cl_event` objects to express dependencies between transfers and kernels instead of `clFinish`.
- **Reference:** `references/examples.md` (Setup Boilerplate & Memory mapping).

#### Phase 3: Verify Checklist
- Check all conversions in `references/output.md` checklist.
- Add error checking (`CL_CHECK`).
- Replace all placeholders in generated files (`<RUN_ARGS>`, `<PROGRAM_NAME>`, `<SOURCE_FILES>`, etc.) with real values.
- Ensure the produced build/run path works without manual placeholder overrides.

### 4. Build and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > opencl_output.txt 2>&1
```

**Debug Issues:** (See `references/examples.md` for Common Issues)
- Compilation failure? Check build log (`clGetProgramBuildInfo`).
- Wrong results? Check work-group sizes.

### 5. Verify Correctness
```bash
diff baseline_output.txt opencl_output.txt
```

### 6. Profile
```bash
{clean_cmd_str}
{profile_cmd_str} > {profile_log_path} 2>&1
```

### 7. Step1 Exit Criteria
- Kernel/program split is plausible for performance.
- Build/program creation happens outside the timed region.
- No obvious avoidable host-device traffic remains in the hot path.
- Generated build/run files contain no unresolved placeholders.
- The normal run path works without ad hoc variable substitution.
- `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- The chosen structure is still plausible for larger practical inputs, not only the smallest tested run, and the larger profiling size must materially exercise the GPU rather than merely increasing runtime by a small constant factor.

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- ONLY EDIT SOURCE CODE IN: {file_listing}
- **REMOVE ALL CUDA API CALLS & SYNTAX.**
- **ADD PROPER OPENCL SETUP.**
