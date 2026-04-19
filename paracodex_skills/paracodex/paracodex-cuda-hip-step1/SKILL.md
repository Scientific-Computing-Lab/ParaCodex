---
name: paracodex-cuda-hip-step1
description: "ParaCodex prompt for cuda to hip step1 (implementation) step."
---

# CUDA to HIP Migration - Implementation
**Directory:** `{kernel_dir}/`
**Files:** {file_listing}
**Reference:** `{kernel_dir}/analysis.md`

**Required:**
- ROCm / HIP installed.
- Verify correctness against baseline.

## Workflow

### 1. Get Baseline
```bash
# Baseline CUDA output is in baseline_output.txt in {kernel_dir}/
```

### 2. Create HIP Migration Plan
**MANDATORY:** Create `hip_migration_plan.md` in `{kernel_dir}` using template in `references/output.md`.
**Reference:** `references/examples.md` (Hipify tools & API mapping).

### 3. Implement Migration (Hipify)

#### Phase 1: Run Hipify
Use `hipify-perl` or `hipify-clang` to convert source files.
```bash
hipify-perl {source_file}.cu > {source_file}.hip.cpp
```
**Reference:** `references/examples.md` (Tool usage).

#### Phase 2: Manual Fixes
- Fix **Warp Size** assumptions: AMD wavefront is **architecture-dependent** — CDNA2/CDNA3 = 64, RDNA3/RDNA4 = 32. Always use the `warpSize` built-in at runtime (never hardcode 32 or 64). Note: `__AMDGCN_WAVEFRONT_SIZE` macro was **removed in ROCm 7.x** — do not use it. Default block sizes of 256 are safe (multiple of both 32 and 64).
- Fix **Inline PTX** (rewrite in GCN ASM or use HIP intrinsics).
- Add `hipStream_t` for async operations; use `hipMallocAsync`/`hipFreeAsync` (ROCm 5.2+) for dynamic per-kernel allocations:
  ```cpp
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipMemcpyAsync(d_ptr, h_ptr, sz, hipMemcpyHostToDevice, stream);
  kernel<<<grid, block, 0, stream>>>(...);
  hipStreamSynchronize(stream);
  hipStreamDestroy(stream);
  ```
- Verify `<<<>>>` syntax (keep or convert if needed).

#### Phase 3: Build System
- Update Makefile to use `hipcc` with `-O3`.

### 4. Build and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > hip_output.txt 2>&1
```

### 5. Verify Correctness
```bash
diff baseline_output.txt hip_output.txt
```

# Rules
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- ALWAYS CLEAN BEFORE BUILD
- **USE HIPIFY TOOLS** first, then manual fix.
- **VERIFY CORRECTNESS.**
