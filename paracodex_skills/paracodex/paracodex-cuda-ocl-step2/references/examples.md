# OpenCL Optimization Examples

## 4A. Optimize Work-Group Size
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

## 4B. Use Native Math Functions
Replace in kernel code (.cl):
```c
// After optimization (if precision allows)
float result = native_sin(x) * native_cos(y);
float power = native_powr(base, exp);  // base must be positive
```

## 4C. Optimize Memory Transfers
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

## 4D. Reduce Barriers
```c
// Use minimal barrier scope
barrier(CLK_LOCAL_MEM_FENCE);  // Only sync __local memory
```

## 4E. Optimize Local Memory
```c
// Avoid bank conflicts by padding
// After (if bank conflicts detected):
__local float shared[256 + 16];  // Pad to avoid conflicts
```

## 4F. Kernel Fusion
Only fuse if same work dimensions, no global sync, no host transfers.

## 4S. Structural rewrite
```c
// If total runtime is dominated by setup, build, or tiny kernel overhead,
// replace a weak step1 decomposition before applying micro-optimizations.
```

## 4G. Compiler Optimizations
```c
clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
// -cl-mad-enable, -cl-no-signed-zeros
```

## 4H. Vector Loads (float4 / int4)
```c
// Coalesced read of 4 floats at once — improves memory throughput
__kernel void addVec4(__global const float4 *restrict A,
                      __global const float4 *restrict B,
                      __global float4 *restrict C, int N4) {
    int i = get_global_id(0);
    if (i < N4) C[i] = A[i] + B[i];
}
// Launch with global = N/4 (N must be divisible by 4)
```

## Micro-Optimizations
- Use `restrict` keyword on pointer arguments
- Cache global values in private (register) variables
- Manual `#pragma unroll` or `__attribute__((opencl_unroll_hint(N)))`

## Optimization Checklist
- [ ] Kernel/program structure is sane for end-to-end runtime
- [ ] Work-group size tuned to device
- [ ] Native math functions used
- [ ] Memory transfers minimized and/or async
- [ ] Pinned memory used
- [ ] Kernel compilation outside timing
- [ ] Barrier scope minimized
- [ ] Local memory bank conflicts addressed
- [ ] Compiler flags optimized
- [ ] Vector types used
- [ ] Memory coalescing verified
