# OpenCL Migration Plan

## Phase 1: Kernel Code Translation

## Structural Plan
- [ ] Choose natural OpenCL kernel/file split.
- [ ] Preserve CUDA kernel boundaries only when they are still performance-justified.
- [ ] Move program build / setup out of the timed region.
- [ ] Budget combined GPU kernel + memcpy + sync cost, not just kernel time.
- [ ] Record the default correctness size and one larger practical profiling size.
- [ ] Reject a split that only wins on tiny inputs while scaling poorly.

### Kernel Files
From analysis.md "File Conversion Mapping":
- Original: [file.cu]
- Kernel code → [kernels.cl]
- Host code → [host.c/cpp]

### Kernel Syntax Conversion
For each kernel in analysis.md:
|| CUDA | OpenCL | Status |
||------|--------|--------|
|| __global__ void kernel(...) | __kernel void kernel(...) | [ ] |
|| __device__ void helper(...) | Regular function (or kernel) | [ ] |
|| __shared__ float arr[256] | __local float arr[256] | [ ] |
|| __constant__ float arr[N] | __constant float arr[N] | [ ] |
|| threadIdx.x/y/z | get_local_id(0/1/2) | [ ] |
|| blockIdx.x/y/z | get_group_id(0/1/2) | [ ] |
|| blockDim.x/y/z | get_local_size(0/1/2) | [ ] |
|| gridDim.x/y/z | get_num_groups(0/1/2) | [ ] |
|| __syncthreads() | barrier(CLK_LOCAL_MEM_FENCE) | [ ] |
|| atomicAdd(&x, v) | atomic_add(&x, v) | [ ] |

### Warp-Level Primitives
From analysis.md "Warp-level operations":
- [ ] __shfl → Requires manual implementation with __local memory
- [ ] __ballot → No equivalent, restructure algorithm
- [ ] __any/__all → Use atomic flags or restructure

## Phase 2: Host Code Translation
See `references/examples.md` for Boilerplate.

## Phase 3: Implementation Checklist

### Kernel Code (.cl file)
- [ ] All __global__ → __kernel
- [ ] All __device__ functions inlined or moved to .cl
- [ ] All __shared__ → __local
- [ ] All thread indexing converted
- [ ] All __syncthreads() → barrier()
- [ ] All atomics converted
- [ ] Math functions updated
- [ ] Dynamic __local as kernel parameter (if needed)
- [ ] No CUDA-specific syntax remains

### Host Code (.c/.cpp file)
- [ ] OpenCL headers included (#include <CL/cl.h>)
- [ ] Platform/device/context/queue setup
- [ ] Kernel source loading and compilation
- [ ] Kernel object creation
- [ ] All cudaMalloc → clCreateBuffer
- [ ] All cudaMemcpy → clEnqueue{Write|Read}Buffer
- [ ] All kernel launches → clSetKernelArg + clEnqueueNDRangeKernel
- [ ] All cudaFree → clReleaseMemObject
- [ ] Cleanup: clRelease* for all objects
- [ ] Error checking on all OpenCL calls

### Build System
- [ ] Link OpenCL library (-lOpenCL)
- [ ] Include OpenCL headers path
- [ ] Kernel .cl file accessible at runtime
- [ ] Test on target device

## Phase 4: Common Issues
See `references/examples.md` for specific issue resolution.

## Target Deliverables
- [ ] kernels.cl - Pure OpenCL kernel code
- [ ] host.cpp - OpenCL host code
- [ ] opencl_migration_plan.md - Complete mapping documentation
- [ ] Compiles with OpenCL flags
- [ ] Runs to completion
- [ ] Ready for correctness verification
- [ ] Uses a performance-plausible kernel decomposition
- [ ] All generated placeholders resolved
- [ ] Plain run path works without ad hoc overrides
- [ ] `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- [ ] The chosen structure is still plausible at the larger practical profiling size.
