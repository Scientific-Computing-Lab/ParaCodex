# OpenCL Migration Examples

## Dynamic Shared Memory
```c
// CUDA: extern __shared__ float smem[];
// OpenCL: Pass as kernel parameter
__kernel void kernel(..., __local float *smem) { }
// Set at enqueue: clSetKernelArg(kernel, argIndex, localMemSize, NULL);
```

## Math Functions
|| CUDA | OpenCL |
||------|--------|
|| __float2int_rn | convert_int_rtn |
|| __sinf, __cosf | native_sin, native_cos (or sin, cos for precision) |
|| __powf | native_powr (or pow) |
|| rsqrtf | native_rsqrt (or rsqrt) |

## OpenCL Setup Boilerplate (Best Practice)
```c
cl_int err;
cl_platform_id platform;
cl_device_id device;
clGetPlatformIDs(1, &platform, NULL);
clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

// Context
cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

// Out-of-order queue — OPTIONAL in OpenCL 3.0; query device support first
cl_command_queue_properties dev_props;
clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                sizeof(dev_props), &dev_props, NULL);
cl_bool supports_ooo = (dev_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;

cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES,
    supports_ooo ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0,
    0
};
cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);

// Program compilation with fast math
const char *source = load_kernel_source("kernels.cl");
cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);

cl_kernel kernel1 = clCreateKernel(program, "kernel1", &err);
```

OpenCL setup/build belongs outside the hot path. If preserving the CUDA decomposition would force expensive queue/build behavior, simplify the structure first.

## Memory Management Mapping
|| CUDA Operation | OpenCL Equivalent |
||----------------|-------------------|
|| cudaMalloc(&d_arr, size) | clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err) |
|| (pinned alloc) | clCreateBuffer(ctx, CL_MEM_READ_WRITE\|CL_MEM_ALLOC_HOST_PTR, size, NULL, &err) |
|| cudaMemcpy H→D (blocking) | clEnqueueWriteBuffer(queue, d_arr, CL_TRUE, 0, size, h_arr, 0, NULL, NULL) |
|| cudaMemcpyAsync H→D | clEnqueueWriteBuffer(queue, d_arr, CL_FALSE, 0, size, h_arr, 0, NULL, &ev) |
|| cudaMemcpy D→H (blocking) | clEnqueueReadBuffer(queue, d_arr, CL_TRUE, 0, size, h_arr, 0, NULL, NULL) |
|| cudaMemcpyAsync D→H | clEnqueueReadBuffer(queue, d_arr, CL_FALSE, 0, size, h_arr, 1, &ev, NULL) |
|| cudaFree(d_arr) | clReleaseMemObject(d_arr) |
|| cudaDeviceSynchronize() | clFinish(queue) |

## Async Transfer + Kernel with Events
```c
// Overlapping H→D transfer, kernel, D→H using events (out-of-order queue)
cl_event ev_write, ev_kernel, ev_read;

// Non-blocking H→D
clEnqueueWriteBuffer(queue, d_buf, CL_FALSE, 0, size, h_buf,
                     0, NULL, &ev_write);

// Kernel depends on transfer completing
clSetKernelArg(kernel1, 0, sizeof(cl_mem), &d_buf);
clEnqueueNDRangeKernel(queue, kernel1, 1, NULL, &global, &local,
                        1, &ev_write, &ev_kernel);

// Non-blocking D→H depends on kernel
clEnqueueReadBuffer(queue, d_buf, CL_FALSE, 0, size, h_buf,
                    1, &ev_kernel, &ev_read);

// Wait only for the final event
clWaitForEvents(1, &ev_read);
clReleaseEvent(ev_write); clReleaseEvent(ev_kernel); clReleaseEvent(ev_read);
```

## Kernel Launch Mapping
```c
// CUDA: kernel<<<gridDim, blockDim>>>(arg1, arg2, ...);

// OpenCL:
size_t global_work_size[3] = {gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z};
size_t local_work_size[3] = {blockDim.x, blockDim.y, blockDim.z};
int work_dim = 1;  // or 2, 3 depending on kernel

clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_arg1);
// ... for each argument

clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
```

## Structural rewrite instead of preserving weak CUDA structure blindly
```c
// If a CUDA path used multiple tiny kernels or helper wrappers,
// OpenCL migration may be better as fewer kernels with clearer buffer residency.
// Favor the decomposition that minimizes command queue overhead and transfers.
```

## Error Handling
```c
#define CL_CHECK(err) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    }
// Usage: CL_CHECK(err);
```

## Common Issues
- **Kernel won't compile?** Check build log: `clGetProgramBuildInfo`.
- **Wrong results?** Check work-group sizes and local memory limits.
- **Worse performance?** Use native math functions, check coalescing.
- **Dynamic __local?** Must pass size at `clSetKernelArg`.
- **Good kernel time but bad total runtime?** Check command queue structure, build timing, and kernel granularity.
