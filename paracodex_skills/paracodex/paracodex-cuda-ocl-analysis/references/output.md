# Analysis Output Template

## File Conversion Mapping
```
original.cu → kernels.cl (kernel code)
original.cu → host.cpp (host code)
utils.cuh → utils.h
```

## Kernel Structure
```
- host_loop (line:X) launches kernel1
  └── kernel1<<<grid,block>>> (line:Y) Type A
      └── device_loop (line:Z) Type A
      └── __syncthreads() (line:W) - maps to barrier()
- kernel2<<<grid,block>>> (line:V) Type D
    └── atomicAdd operations - maps to atomic_add()
```

## Kernel Details
For each CRITICAL/IMPORTANT/SECONDARY kernel:
```
## Kernel: [name] at [file:line]
- **Context:** [__global__ kernel / __device__ function]
- **Launch config:** [grid_size × block_size]
- **Total work-items:** [count]
- **Type:** [A-H] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [device loops or none]
- **Dependencies:** [none / atomicAdd / __syncthreads / reduction]
- **Shared memory:** [YES/NO - size (dynamic/static)]
- **Thread indexing pattern:** [1D/2D/3D]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW) - memory type]
- **OpenCL Migration Issues:** [flags from section 6]
```

## OpenCL Mapping Table
|| CUDA Construct | OpenCL Equivalent | Complexity |
||----------------|-------------------|------------|
|| __global__ void kernel() | __kernel void kernel() | Trivial |
|| threadIdx.x | get_local_id(0) | Trivial |
|| blockIdx.x | get_group_id(0) | Trivial |
|| blockDim.x | get_local_size(0) | Trivial |
|| gridDim.x | get_num_groups(0) | Trivial |
|| __syncthreads() | barrier(CLK_LOCAL_MEM_FENCE) | Trivial |
|| __shared__ float arr[N] | __local float arr[N] | Trivial |
|| __constant__ | __constant | Trivial |
|| atomicAdd(&var, val) | atomic_add(&var, val) | Simple |
|| cudaMalloc | clCreateBuffer | Moderate |
|| cudaMemcpy | clEnqueueWriteBuffer/Read | Moderate |
|| kernel<<<G,B>>>() | clEnqueueNDRangeKernel | Moderate |
|| __shfl_down | NO EQUIVALENT | Complex |
|| Dynamic __shared__ | Local mem + clSetKernelArg | Moderate |

## Summary Table
|| Kernel/Function | Type | Priority | Launch Config | Total Work | Dependencies | OpenCL Issues |
||-----------------|------|----------|---------------|------------|--------------|---------------|

## CUDA-Specific Details
- **Dominant compute kernel:** [main timed kernel]
- **Memory transfers in timed loop?:** YES/NO
- **Shared memory usage:** [total bytes (static/dynamic), patterns]
- **Synchronization points:** [__syncthreads locations, kernel boundaries]
- **Atomic operations:** [locations and variables]
- **Texture/surface usage:** [bindings and access patterns]
- **Warp-level operations:** [locations - CRITICAL migration issue]
- **Dynamic parallelism:** [YES/NO - requires restructuring]

## OpenCL Migration Strategy Notes
- **Direct kernel conversion:** [list]
- **Requires restructuring:** [list with reasons]
- **Performance concerns:** [atomics overhead, local memory size, barriers]
- **Memory management changes:** [cudaMalloc → clCreateBuffer, etc.]
- **API setup overhead:** [context, command queue, program compilation]
- **Expected complexity:** [LOW/MEDIUM/HIGH based on issues]

## Structural Recommendations
- **Natural OpenCL kernel/file split:** [preserve CUDA kernels / fuse / split]
- **Program build strategy:** [single `.cl` / few kernels / multiple source units]
- **Fragmentation risk:** [none / low / medium / high]

## Scalability Check
- **Default input likely too small to guide migration?** [YES/NO]
- **Small-input sensitive?** [YES/NO]
- **Expected scaling risks:** [kernel count / transfer count / transfer volume / enqueue overhead / build/setup cost / none]
- **Larger practical profile size recommendation:** [value or rule]
- **Why this size materially exercises the GPU:** [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- **Hardware basis from `system_info_summary.txt`:** [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]

## Constraints
- Identify warp-level primitives (CRITICAL)
- Note texture memory usage
- Flag dynamic __shared__ memory (must be set at enqueue)
