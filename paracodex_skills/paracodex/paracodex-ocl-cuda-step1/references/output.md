# CUDA Migration Plan

## Phase 1: Kernel Code Translation

## Structural Plan
- [ ] Choose CUDA kernel/file split intentionally.
- [ ] Preserve OpenCL kernel boundaries only when they still make performance sense.
- [ ] Remove avoidable host-side enqueue overhead from the hot path.
- [ ] Budget combined GPU kernel + memcpy + sync cost, not just kernel time.
- [ ] Record the default correctness size and one larger practical profiling size.
- [ ] Reject a structure that only wins on tiny inputs while scaling poorly.

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

### Math Functions
|| OpenCL | CUDA |
||--------|------|
|| native_sin, native_cos | __sinf, __cosf |
|| sin, cos | sinf, cosf |
|| native_sqrt | __fsqrt_rn or sqrtf |
|| native_rsqrt | rsqrtf |
|| native_powr | __powf |
|| convert_int_rtn | __float2int_rn |

## Phase 2: Host Code Translation

### Memory Management Mapping
|| OpenCL Operation | CUDA Equivalent |
||------------------|-----------------|
|| clCreateBuffer | cudaMalloc(&d_ptr, size) |
|| clEnqueueWriteBuffer | cudaMemcpy(d_ptr, h_ptr, size, HtoD) |
|| clEnqueueReadBuffer | cudaMemcpy(h_ptr, d_ptr, size, DtoH) |
|| clEnqueueWriteBuffer(async) | cudaMemcpyAsync(..., stream) |
|| clReleaseMemObject | cudaFree(d_ptr) |
|| clFinish | cudaDeviceSynchronize() |

## Phase 3: Implementation Checklist

### Kernel Code (.cu file)
- [ ] All __kernel → __global__
- [ ] All __local → __shared__
- [ ] All __constant → __constant__
- [ ] All indexing functions converted
- [ ] All barriers converted
- [ ] All atomics converted
- [ ] Math functions converted
- [ ] Dynamic __shared__ as extern (not kernel arg)

### Host Code (.cu file)
- [ ] Included <cuda_runtime.h>
- [ ] Replaced all Memory/Kernel/Event API calls
- [ ] Removed OpenCL setup (context/queue/program)
- [ ] Added CUDA checkpoints/error handling
- [ ] Uses a performance-plausible kernel decomposition
- [ ] All generated placeholders resolved
- [ ] Plain run path works without ad hoc overrides
- [ ] `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- [ ] The chosen structure is still plausible at the larger practical profiling size.

## Phase 4: Common Issues
- **AtomicInc/Dec:** Different semantics (Modulo vs Saturation).
- **Dynamic Shared:** Passed as kernel launch param vs OpenCL arg.
- **Barriers:** CLK_LOCAL vs CLK_GLOBAL distinctions.
