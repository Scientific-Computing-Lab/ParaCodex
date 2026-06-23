# Performance Tuning - CUDA to OpenCL Migration

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Reference:** `{kernel_dir}/opencl_migration_plan.md`

**Required:**
- **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

## Context: CUDA to OpenCL Migration
The code was migrated from CUDA to OpenCL. Performance differences may arise from:
- OpenCL runtime overhead (context, queue, program compilation)
- Different compiler optimizations
- Local memory size differences
- Work-group scheduling differences
- Memory coalescing patterns

**Common migration bottlenecks:**
1. Suboptimal work-group sizes
2. Unnecessary global memory barriers
3. Non-native math functions (precision vs speed)
4. Missing memory access optimizations
5. Kernel compilation overhead in timing

## Workflow

**MANDATORY:** Create opencl_optimization_plan.md in {kernel_dir} before implementation.

### 1. Verify Baseline Correctness
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt
```

If results differ, fix correctness issues first before optimization.

### 2. Analyze Performance Profile

Read profile data:
```bash
cat {profile_log_path} | grep -A20 "kernel\|GPU\|OpenCL"
```

Compare with CUDA baseline (if available):
- CUDA kernel time: [X] ms
- OpenCL kernel time: [Y] ms
- Ratio: [Y/X]x
- If >1.5x: Significant optimization needed

### 3. Create Optimization Plan

Create optimization_plan.md in {kernel_dir}:

```markdown
# Performance Analysis - CUDA to OpenCL

## Current Metrics
- Total runtime: [X]s
- Main kernel: [name], [Y]ms total, [Z] calls
- Memory transfer: [W]ms, [V]MB total
- Kernel compilation time: [U]ms (if significant)

## CUDA vs OpenCL Comparison
- CUDA runtime: [X]s (baseline)
- OpenCL runtime: [Y]s (current)
- Slowdown: [Y/X]x
- Target: <1.3x slowdown (acceptable for portability)

## Bottleneck Analysis

### [ ] 1. Work-Group Size Suboptimal
- Current: [X] work-items per group
- Device max: [check CL_DEVICE_MAX_WORK_GROUP_SIZE]
- Device preferred multiple: [check CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE]
- **Fix:** Tune local_work_size to device characteristics
- Expected gain: [X]%

### [ ] 2. Memory Transfer Overhead
- Transfer time: [X]% of total
- If >50%: Consider:
  - CL_MEM_ALLOC_HOST_PTR for pinned memory
  - Async transfers with events
  - Reduce transfer frequency
- Expected gain: [X]%

### [ ] 3. Math Function Precision
- Using sin/cos (precise) vs native_sin/native_cos (fast)
- Using pow vs native_powr
- **Fix:** Replace with native_* variants if precision allows
- Expected gain: [X]%

### [ ] 4. Local Memory Bank Conflicts
- Check __local memory access patterns
- Stride should avoid 32-word (AMD) or 64-word (NVIDIA) conflicts
- **Fix:** Pad __local arrays or change access pattern
- Expected gain: [X]%

### [ ] 5. Kernel Launch Overhead
- Kernel calls: [N]
- If many small kernels: Consider fusion
- If compilation in timing: Move clBuildProgram outside timer
- Expected gain: [X]%

### [ ] 6. Global Memory Coalescing
- Check access patterns (same as CUDA requirements)
- Sequential threads should access sequential memory
- **Fix:** Restructure access patterns
- Expected gain: [X]%

### [ ] 7. Barrier Overhead
- barrier() calls: [count and locations]
- Check if CLK_LOCAL_MEM_FENCE sufficient (vs CLK_GLOBAL_MEM_FENCE)
- Consider algorithm restructuring to reduce barriers
- Expected gain: [X]%

## Optimization Strategy (priority order)
1. [ACTION]: [description] - expected [X]% gain
2. [ACTION]: [description] - expected [X]% gain
3. [ACTION]: [description] - expected [X]% gain

## Target Performance
- Target runtime: [X]s (<1.3x CUDA baseline)
- Target slowdown: [X]x
```

### 4. Execute Optimizations

#### 4A. Optimize Work-Group Size
```c
// Query device properties
size_t max_work_group_size;
clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

// Query kernel preferred size
size_t preferred_multiple;
clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                         sizeof(size_t), &preferred_multiple, NULL);

// Tune local_work_size to be multiple of preferred_multiple
// Try powers of 2: 32, 64, 128, 256, 512 (within device max)
```

#### 4B. Use Native Math Functions
Replace in kernel code (.cl):
```c
// Before optimization
float result = sin(x) * cos(y);
float power = pow(base, exp);

// After optimization (if precision allows)
float result = native_sin(x) * native_cos(y);
float power = native_powr(base, exp);  // base must be positive
```

Precision trade-offs:
- `native_*`: Fast, lower precision (acceptable for graphics, some simulations)
- Standard: Slower, IEEE 754 precision
- Test correctness after change!

#### 4C. Optimize Memory Transfers
```c
// Use pinned memory for faster transfers
cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
                                size, NULL, &err);

// Async transfers with events
cl_event write_event;
clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, size, host_ptr, 0, NULL, &write_event);
clEnqueueNDRangeKernel(queue, kernel, ..., 1, &write_event, &kernel_event);
clEnqueueReadBuffer(queue, buffer, CL_FALSE, 0, size, host_ptr, 1, &kernel_event, NULL);
```

#### 4D. Reduce Barriers
```c
// Use minimal barrier scope
barrier(CLK_LOCAL_MEM_FENCE);  // Only sync __local memory
// vs
barrier(CLK_GLOBAL_MEM_FENCE); // Sync global memory (heavier)

// Consider algorithm changes to eliminate barriers
// Example: Change two-phase reduction to single-phase if possible
```

#### 4E. Optimize Local Memory
```c
// Avoid bank conflicts by padding
// Before:
__local float shared[256];  // 256 threads access shared[tid]

// After (if bank conflicts detected):
__local float shared[256 + 16];  // Pad to avoid conflicts
```

#### 4F. Kernel Fusion
If multiple small kernels in sequence:
```c
// Before: Two kernel launches
clEnqueueNDRangeKernel(queue, kernel1, ...);
clEnqueueNDRangeKernel(queue, kernel2, ...);

// After: Fused into single kernel (if safe)
clEnqueueNDRangeKernel(queue, fused_kernel, ...);
```

Only fuse if:
- Same work dimensions
- No global synchronization needed between them
- No D→H→D transfer between them

#### 4G. Compiler Optimizations
Add build options:
```c
// Fast math (if precision allows)
clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);

// Additional options
// -cl-mad-enable: Allow a*b+c fusion
// -cl-no-signed-zeros: Optimize sign handling
// -cl-unsafe-math-optimizations: Aggressive (check correctness!)
// -cl-finite-math-only: No inf/nan handling
```

### 5. Micro-Optimizations

- [ ] Use `restrict` keyword for non-aliasing pointers (OpenCL C 2.0)
- [ ] Cache frequently accessed values in private variables
- [ ] Unroll small loops manually if compiler doesn't
- [ ] Use vector types (float4, int4) for coalesced access
- [ ] Reduce register pressure (check CL_KERNEL_PRIVATE_MEM_SIZE)

Example:
```c
__kernel void optimized(__global const float *restrict input,
                        __global float *restrict output) {
    int gid = get_global_id(0);
    
    // Cache in private variable
    float val = input[gid];
    
    // Vectorized load (if alignment allows)
    float4 vec = vload4(gid/4, input);
    
    // Computation
    val = native_sin(val) * 2.0f;
    
    output[gid] = val;
}
```

### 6. Verify After Each Change
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > optimized_output.txt 2>&1
diff baseline_output.txt optimized_output.txt
```

If correctness breaks, revert last change!

### 7. Profile Optimized Version
```bash
{clean_cmd_str}
{profile_cmd_str} > {profile_log_path}_optimized 2>&1
```

### 8. Final Summary

Update optimization_plan.md:
```markdown
# Final Performance Summary

## Baseline (CUDA)
- Runtime: [X]s
- Main kernel: [Y]ms

## Initial OpenCL
- Runtime: [A]s
- Slowdown vs CUDA: [A/X]x
- Main kernel: [B]ms

## Optimized OpenCL
- Runtime: [C]s
- Slowdown vs CUDA: [C/X]x (target: <1.3x)
- Speedup vs initial OpenCL: [A/C]x
- Main kernel: [D]ms

## Optimizations Applied
1. [X] [ACTION]: [description] → [±X%]
2. [X] [ACTION]: [description] → [±X%]
3. [ ] [ACTION]: REVERTED (broke correctness or slower)

## Micro-Optimizations
1. [X] Native math functions → [±X%]
2. [X] Work-group size tuned → [±X%]
3. [X] Memory access optimized → [±X%]

## Key Insights
- [Most impactful optimization]
- [Remaining performance gap vs CUDA: X%]
- [OpenCL-specific bottlenecks]
- [Limitations compared to CUDA]

## Device-Specific Notes
- Device: [name]
- Max work-group size: [X]
- Local memory: [Y] KB
- Optimal work-group size found: [Z]
```

## OpenCL-Specific Optimization Checklist
- [ ] Work-group size tuned to device (query CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)
- [ ] Native math functions used (native_sin, native_sqrt, etc.)
- [ ] Memory transfers minimized and/or async
- [ ] Pinned memory used (CL_MEM_ALLOC_HOST_PTR)
- [ ] Kernel compilation outside timing
- [ ] Barrier scope minimized (CLK_LOCAL_MEM_FENCE vs GLOBAL)
- [ ] Local memory bank conflicts addressed
- [ ] Compiler flags optimized (-cl-fast-relaxed-math, etc.)
- [ ] Vector types used where appropriate
- [ ] Memory coalescing verified

## RULES - BREAKING A RULE = FAILURE
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- DO NOT EDIT MAKEFILES
- ALWAYS CLEAN BEFORE BUILD
- You may create documentation/backup/output files (opencl_optimization_plan.md, *.bak, *.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}
- PRESERVE CORRECTNESS - diff against baseline after each change
- VERIFY PERFORMANCE GAIN after each optimization
- REVERT changes that break correctness or hurt performance
```

---

## Notes

- All prompts contain template variables (e.g., `{kernel_dir}`, `{file_listing}`, `{clean_cmd_str}`, etc.) that are filled in at runtime by the Python scripts.
- CUDA to OpenCL migration typically achieves 0.7x-1.5x performance compared to CUDA, depending on optimization effort.
- OpenCL provides broader device portability (AMD, Intel, NVIDIA) at the cost of more verbose host code.
- Template variables are replaced with actual values when the prompts are used in the codebase.

