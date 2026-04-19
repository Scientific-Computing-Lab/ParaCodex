# OpenCL to CUDA Migration - Implementation

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- CUDA 11.0 or later
- nvcc compiler
- Verify correctness against baseline
- **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline
```bash
Baseline OpenCL output is in baseline_output.txt in {kernel_dir}/
```

### 2. Create CUDA Migration Plan
**MANDATORY:** Create cuda_migration_plan.md in {kernel_dir} before implementation.

**Use the analysis from `{kernel_dir}/analysis.md` to inform this migration plan.**

**Understand OpenCL to CUDA mapping:**

```markdown
# CUDA Migration Plan

## Phase 1: Kernel Code Translation

### Kernel Files
From analysis.md "File Conversion Mapping":
- Original: [kernels.cl]
- Target: [combined.cu] (kernels + host code)

### Kernel Syntax Conversion
For each kernel in analysis.md:

|| OpenCL | CUDA | Status |
||--------|------|--------|
|| __kernel void kernel(...) | __global__ void kernel(...) | [ ] |
|| __local float arr[256] | __shared__ float arr[256] | [ ] |
|| __constant float arr[N] | __constant__ float arr[N] | [ ] |
|| get_local_id(0/1/2) | threadIdx.x/y/z | [ ] |
|| get_group_id(0/1/2) | blockIdx.x/y/z | [ ] |
|| get_local_size(0/1/2) | blockDim.x/y/z | [ ] |
|| get_num_groups(0/1/2) | gridDim.x/y/z | [ ] |
|| get_global_id(0) | blockIdx.x*blockDim.x + threadIdx.x | [ ] |
|| barrier(CLK_LOCAL_MEM_FENCE) | __syncthreads() | [ ] |
|| barrier(CLK_GLOBAL_MEM_FENCE) | __threadfence() | [ ] |
|| atomic_add(&x, v) | atomicAdd(&x, v) | [ ] |

### Work-Item Indexing Conversion
Standard pattern:
```c
// OpenCL
int gid = get_global_id(0);
int lid = get_local_id(0);
int group = get_group_id(0);

// CUDA
int gid = blockIdx.x * blockDim.x + threadIdx.x;
int lid = threadIdx.x;
int group = blockIdx.x;
```

Multi-dimensional:
```c
// OpenCL 2D
int gid_x = get_global_id(0);
int gid_y = get_global_id(1);

// CUDA 2D
int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
```

### Dynamic Local Memory
```c
// OpenCL: Pass as kernel parameter
__kernel void kernel(..., __local float *smem) { }
// Size set at: clSetKernelArg(kernel, argIndex, localMemSize, NULL);

// CUDA: Declare as extern shared, size at launch
extern __shared__ float smem[];
__global__ void kernel(...) { }
// Launch: kernel<<<grid, block, sharedMemSize>>>(...);
```

### Atomic Operations
|| OpenCL | CUDA |
||--------|------|
|| atomic_add | atomicAdd |
|| atomic_sub | atomicSub |
|| atomic_xchg | atomicExch |
|| atomic_cmpxchg | atomicCAS |
|| atomic_inc | atomicInc (behavior differs!) |
|| atomic_dec | atomicDec (behavior differs!) |
|| atomic_min | atomicMin |
|| atomic_max | atomicMax |

**Note:** CUDA atomicInc/Dec have different semantics than OpenCL atomic_inc/dec!

### Math Functions
|| OpenCL | CUDA |
||--------|------|
|| native_sin, native_cos | __sinf, __cosf |
|| sin, cos | sinf, cosf |
|| native_sqrt | __fsqrt_rn or sqrtf |
|| native_rsqrt | rsqrtf |
|| native_powr | __powf |
|| convert_int_rtn | __float2int_rn |

### Barrier Types
```c
// OpenCL: barrier(CLK_LOCAL_MEM_FENCE)  - sync work-group, local memory
// CUDA:   __syncthreads()                - sync block, shared memory

// OpenCL: barrier(CLK_GLOBAL_MEM_FENCE) - sync work-group, global memory
// CUDA:   __threadfence()                - ensure global writes visible

// OpenCL: barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
// CUDA:   __syncthreads(); __threadfence();
```

### Image Objects to Texture Memory
From analysis.md "Image/sampler usage":

OpenCL images require conversion to CUDA textures:
```c
// OpenCL
__kernel void kernel(read_only image2d_t img) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef(img, sampler, coord);
}

// CUDA (requires texture setup)
texture<float4, 2, cudaReadModeElementType> tex;

__global__ void kernel() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float4 pixel = tex2D(tex, x, y);
}
```

**Note:** Texture migration is complex - document separately if needed.

## Phase 2: Host Code Translation

### Remove OpenCL Boilerplate
DELETE all OpenCL setup code:
```c
// Remove these:
clGetPlatformIDs()
clGetDeviceIDs()
clCreateContext()
clCreateCommandQueue()
clCreateProgramWithSource()
clBuildProgram()
clCreateKernel()
```

CUDA requires minimal setup (automatic).

### Memory Management Mapping
From analysis.md "OpenCL-Specific Data Analysis":

|| OpenCL Operation | CUDA Equivalent |
||------------------|-----------------|
|| clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, ...) | cudaMalloc(&d_ptr, size) |
|| clCreateBuffer(ctx, CL_MEM_READ_ONLY, size, ...) | cudaMalloc(&d_ptr, size) |
|| clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, size, h_ptr, ...) | cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice) |
|| clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, size, h_ptr, ...) | cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost) |
|| clEnqueueWriteBuffer(..., CL_FALSE, ..., &event) | cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream) |
|| clEnqueueReadBuffer(..., CL_FALSE, ..., &event) | cudaMemcpyAsync(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost, stream) |
|| clReleaseMemObject(buf) | cudaFree(d_ptr) |
|| clFinish(queue) | cudaDeviceSynchronize() |

### Kernel Launch Mapping
For each clEnqueueNDRangeKernel in analysis.md:

```c
// OpenCL
size_t global_work_size[3] = {N*M, 1, 1};
size_t local_work_size[3] = {M, 1, 1};
clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_buf);
clSetKernelArg(kernel, 1, sizeof(int), &n);
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

// CUDA
dim3 block(M);
dim3 grid(N);
kernel<<<grid, block>>>(d_buf, n);
```

Multi-dimensional:
```c
// OpenCL 2D
size_t global_work_size[2] = {W, H};
size_t local_work_size[2] = {16, 16};
// work_dim = 2

// CUDA 2D
dim3 block(16, 16);
dim3 grid((W + 15)/16, (H + 15)/16);
kernel<<<grid, block>>>(...);
```

### Dynamic Shared Memory at Launch
```c
// OpenCL: clSetKernelArg(kernel, argIndex, sharedMemSize, NULL);

// CUDA: Third launch parameter
kernel<<<grid, block, sharedMemSize>>>(...);
```

### Error Handling
```c
// OpenCL: Check return codes
cl_int err;
cl_mem buf = clCreateBuffer(..., &err);
if (err != CL_SUCCESS) { /* error */ }

// CUDA: Check after calls
cudaMalloc(&d_ptr, size);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
}

// Or use macro
#define CUDA_CHECK(call) \
    { cudaError_t err = call; \
      if (err != cudaSuccess) { \
          fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
          exit(1); \
      } }

CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

## Phase 3: Implementation Checklist

### Kernel Code (.cu file - kernel section)
- [ ] All __kernel → __global__
- [ ] All __local → __shared__
- [ ] All __constant → __constant__
- [ ] All get_global_id() → blockIdx * blockDim + threadIdx
- [ ] All get_local_id() → threadIdx
- [ ] All get_group_id() → blockIdx
- [ ] All get_local_size() → blockDim
- [ ] All get_num_groups() → gridDim
- [ ] All barrier(CLK_LOCAL_MEM_FENCE) → __syncthreads()
- [ ] All atomic_add → atomicAdd (and other atomics)
- [ ] Math functions converted
- [ ] Dynamic __shared__ as extern (if needed)
- [ ] No OpenCL-specific syntax remains

### Host Code (.cu file - host section)
- [ ] CUDA headers included (#include <cuda_runtime.h>)
- [ ] All clCreateBuffer → cudaMalloc
- [ ] All clEnqueueWriteBuffer → cudaMemcpy H→D
- [ ] All clEnqueueReadBuffer → cudaMemcpy D→H
- [ ] All clEnqueueNDRangeKernel → kernel<<<grid,block>>>()
- [ ] All clFinish → cudaDeviceSynchronize()
- [ ] All clReleaseMemObject → cudaFree
- [ ] Removed all OpenCL boilerplate (context, queue, program, etc.)
- [ ] Error checking on CUDA calls
- [ ] No OpenCL API calls remain

### Build System
- [ ] Compile with nvcc
- [ ] Link CUDA runtime (-lcudart)
- [ ] Appropriate compute capability (-arch=sm_XX)
- [ ] Test on target GPU

## Phase 4: Common Issues

### Issue: Compilation errors
- Check all OpenCL syntax converted
- Verify thread indexing math (blockIdx * blockDim + threadIdx)
- Check atomic function names (atomic_add → atomicAdd)
- Verify __shared__ vs __local

### Issue: Wrong results
- Verify grid/block dimensions match original global/local work sizes
- Check barrier conversion (CLK_LOCAL vs CLK_GLOBAL)
- Verify atomicInc/atomicDec semantics (differ from OpenCL!)
- Check boundary conditions in thread indexing

### Issue: Runtime errors
- Check grid/block dimensions are valid
- Verify shared memory size doesn't exceed limit (48KB typical)
- Check for out-of-bounds memory access
- Use cuda-memcheck for debugging

### Issue: Performance worse than OpenCL
- Use fast math functions (__sinf, __cosf)
- Verify coalesced memory access (same as OpenCL)
- Check shared memory bank conflicts
- Use nvprof/nsys for profiling
- Consider CUDA-specific optimizations (see Phase 5)

### Issue: Dynamic shared memory
```c
// OpenCL: Size set per kernel arg
clSetKernelArg(kernel, argIdx, sharedMemSize, NULL);

// CUDA: Size set at launch (sum of all dynamic shared)
extern __shared__ float smem[];  // In kernel
kernel<<<grid, block, sharedMemSize>>>(...);  // Launch

// Multiple dynamic arrays: use offsets
extern __shared__ char sharedMem[];
float *smem1 = (float*)sharedMem;
float *smem2 = (float*)&sharedMem[size1];
```

## Phase 5: CUDA-Specific Enhancements (Optional)

Consider these CUDA-specific features for better performance:

### Warp-Level Primitives
```c
// OpenCL sub-groups are limited
// CUDA offers warp shuffle for efficient communication
float val = __shfl_down_sync(0xffffffff, val, offset);
```

### Cooperative Groups
```c
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void kernel() {
    auto block = this_thread_block();
    block.sync();  // Like __syncthreads() but more flexible
}
```

### Unified Memory (Simplify transfers)
```c
// Instead of cudaMalloc + cudaMemcpy:
float *data;
cudaMallocManaged(&data, size);
// Automatically migrated between host/device
```

### Streams for Concurrency
```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid1, block1, 0, stream1>>>(...);
kernel2<<<grid2, block2, 0, stream2>>>(...);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

## Target Deliverables
- [ ] combined.cu - CUDA kernel and host code
- [ ] cuda_migration_plan.md - Complete mapping documentation
- [ ] Compiles with nvcc
- [ ] Runs to completion
- [ ] Ready for correctness verification
```

### 3. Implement Migration Plan

Follow cuda_migration_plan.md phases:

**Phase 1:** Convert kernel syntax (.cl → __global__ functions)  
**Phase 2:** Replace OpenCL host API with CUDA API  
**Phase 3:** Verify checklist items  
**Phase 4:** Debug common issues  
**Phase 5:** (Optional) Add CUDA-specific optimizations

### 4. Build and Test
```bash
cd {kernel_dir}
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > cuda_output.txt 2>&1
```

If compilation fails:
- Check all OpenCL syntax converted
- Verify nvcc compiler flags
- Check compute capability compatibility

If runtime fails:
- Use cuda-memcheck: `cuda-memcheck ./program`
- Verify grid/block dimensions
- Check shared memory limits

### 5. Verify Correctness
```bash
diff baseline_output.txt cuda_output.txt
```

Check for numerical differences (acceptable tolerance for floating-point).

### 6. Profile
```bash
cd {kernel_dir}
{clean_cmd_str}
{profile_cmd_str} > {profile_log_path} 2>&1
```

Document initial performance for optimization phase.

## RULES - BREAKING A RULE = FAILURE
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY
- DO NOT EDIT MAKEFILES
- ALWAYS CLEAN BEFORE BUILD
- You may create documentation/backup/output files (cuda_migration_plan.md, *.bak, *.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}
- REMOVE ALL OpenCL API CALLS (clCreateBuffer, clEnqueueNDRangeKernel, etc.)
- CONVERT ALL __kernel TO __global__
- REMOVE ALL OpenCL-SPECIFIC SYNTAX (get_global_id → blockIdx*blockDim+threadIdx, etc.)
- SIMPLIFY HOST CODE (remove context/queue/program boilerplate)
- VERIFY CORRECTNESS AGAINST BASELINE
```

---

## Optimization Step 2: Performance Tuning

This prompt is used for optimizing the CUDA implementation.

```markdown
# Performance Tuning - OpenCL to CUDA Migration

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Reference:** `{kernel_dir}/cuda_migration_plan.md`

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
