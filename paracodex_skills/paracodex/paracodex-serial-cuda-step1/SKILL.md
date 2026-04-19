---
name: paracodex-serial-cuda-step1
description: "ParaCodex prompt for serial to cuda step1 (implementation) step."
---

# Serial to CUDA Migration - Implementation
**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- CUDA 11.0 or later.
- Verify correctness against serial baseline.
- Treat combined GPU kernel + memcpy + sync cost as the real performance target; do not accept a design that only improves kernel time.
- Prefer a CUDA structure that remains efficient as the problem grows; reject designs that only win on tiny runs by adding scaling-hostile launch or transfer overhead.

## Workflow

### 1. Get Baseline
```bash
# Baseline Serial output is in baseline_output.txt in {kernel_dir}/
```

### 2. Create Migration Plan
**MANDATORY:** Create `cuda_migration_plan.md` in `{kernel_dir}` using template in `references/output.md`.
**Reference:** `references/examples.md`.
- The plan must identify the natural CUDA offload unit and say whether a fused kernel is required.
- The plan must record the default correctness size and one larger practical profiling size.
- The larger profiling size must be chosen to expose sustained GPU behavior on the available hardware, not just millisecond-scale launch or transfer overhead.
- Increase the profiling size until the GPU work is materially exercised or until memory/time limits make further growth impractical; do not stop at a token size increase.
- Choose that size using `system_info_summary.txt`; push to the largest short-run, memory-safe input that should load the GPU meaningfully on this device.
- If the benchmark framework offers multiple classes or problem sizes, prefer the largest size that fits comfortably and completes in a practical time budget.

### 3. Implement Migration

#### Phase 1: Data Management
- Create a CUDA stream: `cudaStreamCreate(&stream)`.
- Allocate device memory: `cudaMallocAsync(&d_ptr, N * sizeof(T), stream)` (CUDA 11.2+, stream-ordered, preferred) or `cudaMalloc` for static allocations.
- Transfer data asynchronously: `cudaMemcpyAsync(d_ptr, h_ptr, N * sizeof(T), cudaMemcpyHostToDevice, stream)`.
- Initialize data on host before transfers.

#### Phase 2: Kernel Implementation
- Write `__global__` kernel(s) for the identified hotspot.
- Prefer a fused kernel when the serial hot path is split across helper calls that would otherwise launch tiny kernels or force extra host/device traffic.
- Use `__launch_bounds__(BLOCK_SIZE)` to cap register usage.
- Block dimension: default to 256 threads (multiple of 32, the warp size).
- Grid dimension: `dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE)`.
- Launch on stream: `kernel<<<grid, block, 0, stream>>>(...args)`.
- Transfer results back: `cudaMemcpyAsync(h_out, d_out, ..., cudaMemcpyDeviceToHost, stream)`.
- Synchronize: `cudaStreamSynchronize(stream)`.
- **Reference:** `references/examples.md` (Vector Add example).

#### Phase 2b: Structural Check
- If the current design would create many small kernels, repeated scalar staging, or avoidable host-device round trips, rewrite the hot path before tuning details.
- Flatten pointer-heavy structures when that simplifies kernel arguments and memory access.

#### Phase 3: Build System
- Update Makefile to use `nvcc` with `-O3 --use_fast_math`.
- Replace all placeholders in generated files (`<RUN_ARGS>`, `<PROGRAM_NAME>`, `<SOURCE_FILES>`, etc.) with real values.
- Destroy stream after use: `cudaStreamDestroy(stream)`.

### 4. Build and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > cuda_output.txt 2>&1
```

The deliverable must also support its plain run target or command with no ad hoc placeholder overrides. Do not rely on manual `RUN_ARGS=...` retries to pass step1.

### 5. Verify Correctness
```bash
diff baseline_output.txt cuda_output.txt
```

### 6. Step1 Exit Criteria
- The main compute path is on the GPU.
- The kernel decomposition is plausible for performance; avoid a helper-by-helper tiny-kernel port when fusion is possible.
- There are no avoidable host-device transfers inside the hot path.
- Build/run files contain no unresolved placeholders.
- The normal run path for the generated deliverable works without manual placeholder substitution.
- `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- The chosen kernel/transfer structure is a plausible base for larger practical inputs, not only the smallest run, and the larger profiling size must materially exercise the GPU rather than merely increasing runtime by a small constant factor.

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- **Start with HOTSPOTS.**
- **VERIFY CORRECTNESS.**
