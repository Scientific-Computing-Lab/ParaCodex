# CUDA to OpenCL Migration - Implementation

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- OpenCL 1.2 or later compatibility
- Test on target device (GPU)
- Verify correctness against baseline
- **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline
```bash
Baseline CUDA output is in baseline_output.txt in {kernel_dir}/
```

### 2. Create OpenCL Migration Plan
**MANDATORY:** Create opencl_migration_plan.md in {kernel_dir} before implementation.

**Use the analysis from `{kernel_dir}/analysis.md` to inform this migration plan.**

**Understand CUDA to OpenCL mapping:**

```markdown
# OpenCL Migration Plan

## Phase 1: Kernel Code Translation

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

### Dynamic Shared Memory
If analysis.md shows dynamic __shared__:
```c
// CUDA: extern __shared__ float smem[];
// OpenCL: Pass as kernel parameter
__kernel void kernel(..., __local float *smem) { }
// Set at enqueue: clSetKernelArg(kernel, argIndex, localMemSize, NULL);
```

### Warp-Level Primitives
From analysis.md "Warp-level operations":
- [ ] __shfl → Requires manual implementation with __local memory
- [ ] __ballot → No equivalent, restructure algorithm
- [ ] __any/__all → Use atomic flags or restructure

### Math Functions
|| CUDA | OpenCL |
||------|--------|
|| __float2int_rn | convert_int_rtn |
|| __sinf, __cosf | native_sin, native_cos (or sin, cos for precision) |
|| __powf | native_powr (or pow) |
|| rsqrtf | native_rsqrt (or rsqrt) |

## Phase 2: Host Code Translation

### OpenCL Setup Boilerplate
Required OpenCL initialization:

```c
// 1. Platform and device
cl_platform_id platform;
cl_device_id device;
clGetPlatformIDs(1, &platform, NULL);
clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

// 2. Context and queue (out-of-order for async overlap)
cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
cl_queue_properties props[] = {CL_QUEUE_PROPERTIES,
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);

// 3. Program compilation
const char *source = load_kernel_source("kernels.cl");
cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);

// 4. Kernel objects
cl_kernel kernel1 = clCreateKernel(program, "kernel1", &err);
```

### Memory Management Mapping
From analysis.md "CUDA-Specific Data Analysis":

|| CUDA Operation | OpenCL Equivalent |
||----------------|-------------------|
|| cudaMalloc(&d_arr, size) | clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err) |
|| cudaMemcpy H→D | clEnqueueWriteBuffer(queue, d_arr, CL_TRUE, 0, size, h_arr, 0, NULL, NULL) |
|| cudaMemcpy D→H | clEnqueueReadBuffer(queue, d_arr, CL_TRUE, 0, size, h_arr, 0, NULL, NULL) |
|| cudaMemcpyAsync H→D | clEnqueueWriteBuffer with CL_FALSE + event |
|| cudaMemcpyAsync D→H | clEnqueueReadBuffer with CL_FALSE + event |
|| cudaFree(d_arr) | clReleaseMemObject(d_arr) |
|| cudaDeviceSynchronize() | clFinish(queue) |

### Memory Flags Guide
- **CL_MEM_READ_WRITE:** Kernel reads and writes (default)
- **CL_MEM_READ_ONLY:** Kernel only reads (optimize)
- **CL_MEM_WRITE_ONLY:** Kernel only writes (optimize)
- **CL_MEM_COPY_HOST_PTR:** Initialize from host pointer

### Kernel Launch Mapping
For each kernel launch in analysis.md:

```c
// CUDA: kernel<<<gridDim, blockDim>>>(arg1, arg2, ...);

// OpenCL:
size_t global_work_size[3] = {gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z};
size_t local_work_size[3] = {blockDim.x, blockDim.y, blockDim.z};
int work_dim = 1;  // or 2, 3 depending on kernel

clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_arg1);
clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_arg2);
// ... for each argument

clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
```

### Dynamic Shared Memory at Launch
If kernel uses dynamic __local:
```c
// After regular arguments:
clSetKernelArg(kernel, argIndex, sharedMemSize, NULL);  // NULL means device local allocation
```

### Error Handling
Add error checking macro:
```c
#define CL_CHECK(err) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    }

// Use after every OpenCL call
cl_int err;
cl_mem buffer = clCreateBuffer(context, flags, size, NULL, &err);
CL_CHECK(err);
```

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

### Issue: Kernel won't compile
- Check clBuildProgram status
- Get build log: clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ...)
- Common causes: CUDA syntax not converted, unsupported features

### Issue: Wrong results
- Verify work-group sizes (blockDim) are legal for device
- Check local memory size doesn't exceed device limit
- Verify global work size = num_groups × local_work_size
- Check barrier() flags (CLK_LOCAL_MEM_FENCE vs CLK_GLOBAL_MEM_FENCE)

### Issue: Performance worse than CUDA
- Use native math functions (native_sin vs sin)
- Verify memory coalescing (same as CUDA)
- Check local memory bank conflicts
- Profile with vendor tools

### Issue: Dynamic __local memory
- Must pass size at clSetKernelArg(kernel, argIndex, localMemSize, NULL)
- Size in bytes (not elements)
- Counts toward device local memory limit

## Target Deliverables
- [ ] kernels.cl - Pure OpenCL kernel code
- [ ] host.cpp - OpenCL host code
- [ ] opencl_migration_plan.md - Complete mapping documentation
- [ ] Compiles with OpenCL flags
- [ ] Runs to completion
- [ ] Ready for correctness verification
```

### 3. Implement Migration Plan

Follow opencl_migration_plan.md phases:

**Phase 1:** Convert kernel syntax to .cl file  
**Phase 2:** Replace host CUDA API with OpenCL API  
**Phase 3:** Verify checklist items  
**Phase 4:** Debug common issues

### 4. Build and Test
```bash
cd {kernel_dir}
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > opencl_output.txt 2>&1
```

If compilation fails:
- Check clBuildProgram and get build log
- Verify all CUDA syntax converted
- Check for unsupported OpenCL features

If runtime fails:
- Verify device selection
- Check local memory limits
- Verify work-group sizes are valid

### 5. Verify Correctness
```bash
diff baseline_output.txt opencl_output.txt
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
- You may create documentation/backup/output files (opencl_migration_plan.md, *.bak, *.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}
- REMOVE ALL CUDA API CALLS (cudaMalloc, cudaMemcpy, cudaFree, kernel<<<>>>)
- CONVERT ALL __global__ TO __kernel
- REMOVE ALL CUDA-SPECIFIC SYNTAX (blockIdx → get_group_id, threadIdx → get_local_id, etc.)
- ADD PROPER OpenCL SETUP (context, queue, program compilation)
- VERIFY CORRECTNESS AGAINST BASELINE
```

---

## Optimization Step 2: Performance Tuning
