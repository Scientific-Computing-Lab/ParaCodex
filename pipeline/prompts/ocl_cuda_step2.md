# Performance Tuning - OpenCL to CUDA Migration

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Reference:** `{kernel_dir}/cuda_migration_plan.md`

**Required:**
- **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

## Context: OpenCL to CUDA Migration
The code was migrated from OpenCL to CUDA. Performance differences may arise from:
- Simpler CUDA API (less overhead)
- Different compiler optimizations (nvcc vs OpenCL JIT)
- CUDA-specific features (warp primitives, faster texture cache)
- Different memory hierarchy tuning
- Shared memory vs local memory optimizations

**Common migration opportunities:**
1. Leverage CUDA warp-level primitives
2. Use faster CUDA intrinsics (__sinf, __cosf)
3. Simplify memory management (unified memory)
4. Better coalescing with CUDA profiler guidance
5. Cooperative groups for complex synchronization

**Target: Match or exceed OpenCL performance (often 1.0x-1.3x faster)**

## Workflow

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
cat {profile_log_path} | grep -A20 "cuda_gpu_kern_sum"
cat {profile_log_path} | grep -A10 "cuda_api_sum"
cat {profile_log_path} | grep -A10 "cuda_gpu_mem_time_sum"
```

Compare with OpenCL baseline:
- OpenCL kernel time: [X] ms
- CUDA kernel time: [Y] ms
- Ratio: [Y/X]x
- Target: <1.0x (CUDA should match or beat OpenCL)

### 3. Create Optimization Plan

Create optimization_plan.md in {kernel_dir}:

```markdown
# Performance Analysis - OpenCL to CUDA

## Current Metrics
- Total runtime: [X]s
- Main kernel: [name], [Y]ms total, [Z] calls
- Memory transfer: [W]ms, [V]MB total

## OpenCL vs CUDA Comparison
- OpenCL runtime: [X]s (baseline)
- CUDA runtime: [Y]s (current)
- Speedup: [X/Y]x
- Target: >1.0x (CUDA should be faster or equal)
- If slower: Major optimization needed

## Bottleneck Analysis

### [ ] 1. Block Size Suboptimal
- Current: [X] threads per block
- Recommended: Multiple of 32 (warp size), typically 128-512
- Check occupancy: Use CUDA Occupancy Calculator
- **Fix:** Tune block dimensions to maximize occupancy
- Expected gain: [X]%

### [ ] 2. Memory Transfer Overhead
- Transfer time: [X]% of total
- If >50%: Consider:
  - Pinned memory (cudaHostAlloc)
  - Async transfers (cudaMemcpyAsync + streams)
  - Unified memory (cudaMallocManaged)
- Expected gain: [X]%

### [ ] 3. Slow Math Functions
- Using sinf/cosf (precise) vs __sinf/__cosf (fast)
- Using powf vs __powf
- **Fix:** Use intrinsics (__sinf, __cosf, __expf, etc.) if precision allows
- Expected gain: [X]%

### [ ] 4. Shared Memory Bank Conflicts
- Check __shared__ memory access patterns
- Use nvprof/nsys to detect conflicts
- Stride should avoid 32-bank conflicts
- **Fix:** Pad __shared__ arrays or restructure access
- Expected gain: [X]%

### [ ] 5. Warp Divergence
- Check if/else branches vary across warp
- Use --metrics branch_efficiency in nvprof
- **Fix:** Restructure to minimize divergence
- Expected gain: [X]%

### [ ] 6. Memory Coalescing
- Sequential threads should access sequential memory
- Use nvprof --metrics gld_efficiency, gst_efficiency
- **Fix:** Restructure access patterns
- Expected gain: [X]%

### [ ] 7. Occupancy Too Low
- Check with nvprof --metrics achieved_occupancy
- Limited by: registers, shared memory, or block size
- **Fix:** Reduce register usage, smaller __shared__, tune block size
- Expected gain: [X]%

### [ ] 8. Missing CUDA-Specific Optimizations
- Warp shuffles for reduction (vs atomic)
- Texture cache for read-only data
- Constant memory for small read-only arrays
- Expected gain: [X]%

## Optimization Strategy (priority order)
1. [ACTION]: [description] - expected [X]% gain
2. [ACTION]: [description] - expected [X]% gain
3. [ACTION]: [description] - expected [X]% gain

## Target Performance
- Target runtime: [X]s (≤ OpenCL baseline)
- Target speedup: [X]x vs OpenCL
```

### 4. Execute Optimizations

#### 4A. Optimize Block Size and Occupancy
```bash
# Use CUDA Occupancy Calculator or nvprof
nvprof --metrics achieved_occupancy ./program

# Typical optimal block sizes: 128, 256, 512
# Must be multiple of 32 (warp size)
```

In code:
```c
// Before
dim3 block(64);  // Suboptimal

// After (test different sizes)
dim3 block(256);  // Better occupancy
```

#### 4B. Use CUDA Fast Math Intrinsics
Replace in kernel code:
```c
// Before
float result = sinf(x) * cosf(y);
float power = powf(base, exp);
float root = sqrtf(val);

// After (if precision allows)
float result = __sinf(x) * __cosf(y);
float power = __powf(base, exp);
float root = __fsqrt_rn(val);  // Or rsqrtf for 1/sqrt
```

Compile with fast math:
```bash
nvcc -use_fast_math ...
```

#### 4C. Leverage Warp-Level Primitives
For reductions within a warp (instead of atomics):
```c
// Before: Using atomicAdd for reduction
__shared__ float shared[32];
// ... complex reduction with atomics ...

// After: Using warp shuffle
__inline__ __device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kernel(...) {
    float val = ...;
    val = warp_reduce(val);  // Reduce within warp
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&result, val);  // One atomic per warp
    }
}
```

#### 4D. Optimize Memory Transfers
```c
// Use pinned memory for faster transfers
float *h_pinned;
cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);
cudaMemcpy(d_ptr, h_pinned, size, cudaMemcpyHostToDevice);  // Faster
cudaFreeHost(h_pinned);

// Or use async transfers with streams
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);

// Or simplify with unified memory
float *data;
cudaMallocManaged(&data, size);
// No explicit transfers needed
kernel<<<grid, block>>>(data, ...);
cudaDeviceSynchronize();
```

#### 4E. Optimize Shared Memory
```c
// Avoid bank conflicts by padding
// Before: 32 banks, conflicts on stride 32
__shared__ float shared[32][32];

// After: Pad to avoid conflicts
__shared__ float shared[32][33];  // Extra column eliminates conflicts
```

Check for conflicts:
```bash
nvprof --metrics shared_load_transactions_per_request ./program
# Target: 1.0 (no conflicts)
```

#### 4F. Reduce Warp Divergence
```c
// Before: Divergent branches
if (threadIdx.x % 2 == 0) {
    // Even threads
} else {
    // Odd threads
}

// After: Restructure to minimize divergence
// Process even/odd separately or use warp-uniform conditions
```

Check divergence:
```bash
nvprof --metrics branch_efficiency ./program
# Target: >90%
```

#### 4G. Use Texture Memory for Read-Only Data
For read-only, spatially cached access:
```c
// Declare texture
texture<float, 1, cudaReadModeElementType> texRef;

// Bind before kernel
cudaBindTexture(0, texRef, d_data, size);

// Read in kernel
__global__ void kernel() {
    float val = tex1Dfetch(texRef, idx);
}

// Unbind after
cudaUnbindTexture(texRef);
```

Or use texture objects (CUDA 5.0+):
```c
cudaTextureObject_t texObj;
// ... create texture object ...
__global__ void kernel(cudaTextureObject_t tex) {
    float val = tex1Dfetch<float>(tex, idx);
}
```

#### 4H. Use Constant Memory
For small (<64KB) read-only data accessed uniformly:
```c
__constant__ float const_data[1024];

// Copy once
cudaMemcpyToSymbol(const_data, h_data, size);

// Access in kernel (cached per multiprocessor)
__global__ void kernel() {
    float val = const_data[idx];
}
```

### 5. Micro-Optimizations

- [ ] Use `restrict` keyword (__restrict__)
- [ ] Cache frequently accessed values in registers
- [ ] Unroll loops with #pragma unroll
- [ ] Use vector types (float4, int4) for coalesced access
- [ ] Minimize register pressure (nvprof --metrics reg_per_thread)
- [ ] Use shared memory for frequently accessed data

Example:
```c
__global__ void optimized(const float * __restrict__ input,
                          float * __restrict__ output) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cache in register
    float val = input[gid];
    
    // Vector load (if aligned)
    float4 vec = reinterpret_cast<const float4*>(input)[gid/4];
    
    // Computation with fast math
    val = __sinf(val) * 2.0f;
    
    // Unrolled loop
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        val += vec.x;  // Simplified example
    }
    
    output[gid] = val;
}
```

### 6. Compile-Time Optimizations

```bash
# Enable fast math
nvcc -use_fast_math ...

# Optimize for specific GPU
nvcc -arch=sm_75 ...  # For compute capability 7.5

# Maximum optimization
nvcc -O3 -use_fast_math -lineinfo ...

# Check PTX/SASS
nvcc -ptx ...  # Generate PTX
nvcc -cubin ...  # Generate SASS
```

### 7. Verify After Each Change
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > optimized_output.txt 2>&1
diff baseline_output.txt optimized_output.txt
```

If correctness breaks, revert last change!

### 8. Profile Optimized Version
```bash
{clean_cmd_str}
nvprof --metrics achieved_occupancy,branch_efficiency,gld_efficiency,gst_efficiency ./program
{profile_cmd_str} > {profile_log_path}_optimized 2>&1
```

### 9. Final Summary

Update optimization_plan.md:
```markdown
# Final Performance Summary

## Baseline (OpenCL)
- Runtime: [X]s
- Main kernel: [Y]ms

## Initial CUDA
- Runtime: [A]s
- Speedup vs OpenCL: [X/A]x
- Main kernel: [B]ms

## Optimized CUDA
- Runtime: [C]s
- Speedup vs OpenCL: [X/C]x (target: >1.0x)
- Speedup vs initial CUDA: [A/C]x
- Main kernel: [D]ms

## Optimizations Applied
1. [X] [ACTION]: [description] → [±X%]
2. [X] [ACTION]: [description] → [±X%]
3. [ ] [ACTION]: REVERTED (broke correctness or slower)

## CUDA-Specific Optimizations
1. [X] Fast math intrinsics (__sinf, __cosf) → [±X%]
2. [X] Warp-level primitives (shuffle) → [±X%]
3. [X] Block size tuned (occupancy: [X]%) → [±X%]
4. [X] Unified memory (simplified code) → [±X%]

## Micro-Optimizations
1. [X] __restrict__ pointers → [±X%]
2. [X] Register caching → [±X%]
3. [X] Loop unrolling → [±X%]

## Key Insights
- [Most impactful optimization]
- [Performance vs OpenCL: +X%]
- [CUDA advantages leveraged]
- [Remaining bottlenecks]

## Device-Specific Metrics
- Device: [name]
- Compute capability: [X.Y]
- Optimal block size: [Z]
- Achieved occupancy: [X]%
- Branch efficiency: [Y]%
- Memory coalescing: [Z]%
```

## CUDA-Specific Optimization Checklist
- [ ] Block size optimized (multiple of 32, check occupancy)
- [ ] Fast math intrinsics used (__sinf, __cosf, __expf, etc.)
- [ ] Warp-level primitives for reductions (__shfl_down_sync)
- [ ] Memory transfers optimized (pinned/unified/async)
- [ ] Shared memory bank conflicts eliminated
- [ ] Warp divergence minimized (branch efficiency >90%)
- [ ] Memory coalescing verified (nvprof metrics)
- [ ] Texture/constant memory for read-only data
- [ ] Register pressure managed (nvprof --metrics reg_per_thread)
- [ ] Compiled with -use_fast_math -O3

## Profiling Commands
```bash
# Basic profiling
nvprof ./program

# Detailed metrics
nvprof --metrics achieved_occupancy,branch_efficiency,gld_efficiency,gst_efficiency ./program

# Nsys for newer GPUs
nsys profile --stats=true ./program

# Check specific kernel
nvprof --kernels "kernel_name" --metrics all ./program
```

## RULES - BREAKING A RULE = FAILURE
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- DO NOT EDIT MAKEFILES
- ALWAYS CLEAN BEFORE BUILD
- You may create documentation/backup/output files (cuda_optimization_plan.md, *.bak, *.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}
- PRESERVE CORRECTNESS - diff against baseline after each change
- VERIFY PERFORMANCE GAIN after each optimization
- REVERT changes that break correctness or hurt performance
- TARGET: Match or exceed OpenCL baseline performance
```

---

## Notes

- All prompts contain template variables (e.g., `{kernel_dir}`, `{file_listing}`, `{clean_cmd_str}`, etc.) that are filled in at runtime by the Python scripts.
- OpenCL to CUDA migration typically achieves 1.0x-1.3x speedup compared to OpenCL, due to CUDA's optimized compiler and warp-level features.
- CUDA provides cleaner API (no context/queue/program boilerplate) and better development tools (nvprof, nsys, cuda-gdb).
- CUDA-specific features like warp shuffles, unified memory, and cooperative groups can provide significant performance gains.
- Template variables are replaced with actual values when the prompts are used in the codebase.

