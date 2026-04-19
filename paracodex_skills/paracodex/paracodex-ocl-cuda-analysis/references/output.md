# Analysis Output Template

## File Conversion Mapping
```
kernels.cl → combined.cu (kernel code as __global__)
host.cpp → combined.cu (host code in same file)
utils.h → utils.cuh (or keep as .h)
```

## Kernel Structure
```
- host_loop (line:X) enqueues kernel1
  └── kernel1 NDRange (line:Y) Type A
      └── work-item loop (line:Z) Type A
      └── barrier() (line:W) - maps to __syncthreads()
- kernel2 NDRange (line:V) Type D
    └── atomic_add operations - maps to atomicAdd()
```

## Kernel Details
For each CRITICAL/IMPORTANT/SECONDARY kernel:
```
## Kernel: [name] at [file:line]
- **Context:** [__kernel function]
- **NDRange config:** [global_work_size / local_work_size]
- **Work dimensions:** [1D/2D/3D]
- **Total work-items:** [count]
- **Type:** [A-H] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [work-item loops or none]
- **Dependencies:** [none / atomic_add / barrier / reduction]
- **Local memory:** [YES/NO - size bytes]
- **Work-item indexing pattern:** [1D/2D/3D]
- **Private vars:** [list]
- **Buffers:** [name(R/W/RW) - memory type]
- **CUDA Migration Issues:** [flags from section 6]
```

## CUDA Mapping Table
|| OpenCL Construct | CUDA Equivalent | Complexity |
||------------------|-----------------|------------|
|| __kernel void kernel() | __global__ void kernel() | Trivial |
|| get_local_id(0) | threadIdx.x | Trivial |
|| get_group_id(0) | blockIdx.x | Trivial |
|| get_local_size(0) | blockDim.x | Trivial |
|| get_num_groups(0) | gridDim.x | Trivial |
|| get_global_id(0) | blockIdx.x*blockDim.x + threadIdx.x | Trivial |
|| barrier(CLK_LOCAL_MEM_FENCE) | __syncthreads() | Trivial |
|| __local float arr[N] | __shared__ float arr[N] | Trivial |
|| __constant | __constant__ | Trivial |
|| atomic_add(&var, val) | atomicAdd(&var, val) | Simple |
|| clCreateBuffer | cudaMalloc | Moderate |
|| clEnqueueWriteBuffer | cudaMemcpy H→D | Moderate |
|| clEnqueueNDRangeKernel | kernel<<<grid,block>>>() | Moderate |
|| Image objects | Texture memory | Moderate |
|| Sub-groups (explicit) | Warp primitives | Moderate |
|| Device enqueue | NO EQUIVALENT | Complex |
|| Pipes | NO EQUIVALENT | Complex |

## Summary Table
|| Kernel/Function | Type | Priority | NDRange Config | Total Work | Dependencies | CUDA Issues |
||-----------------|------|----------|----------------|------------|--------------|-------------|

## OpenCL-Specific Details
- **Dominant compute kernel:** [main timed kernel]
- **Memory transfers in timed loop?:** YES/NO
- **Local memory usage:** [total bytes, patterns]
- **Synchronization points:** [barrier locations, clFinish calls]
- **Atomic operations:** [locations and variables]
- **Image/sampler usage:** [formats and access patterns]
- **Sub-group operations:** [locations - consider warp-level optimization]
- **Device enqueue:** [YES/NO - requires restructuring for CUDA]

## CUDA Migration Strategy Notes
- **Direct kernel conversion:** [list]
- **Requires restructuring:** [list with reasons]
- **Performance opportunities:** [CUDA warp primitives, faster texture cache]
- **Memory management simplification:** [CUDA API simpler]
- **API setup simplification:** [No context/queue boilerplate]
- **Expected complexity:** [LOW/MEDIUM/HIGH]

## Structural Recommendations
- **Natural CUDA kernel/file split:** [preserve OpenCL kernels / fuse / simplify]
- **Fragmentation risk:** [none / low / medium / high]
- **Notes:** [why]

## Scalability Check
- **Default input likely too small to guide migration?** [YES/NO]
- **Small-input sensitive?** [YES/NO]
- **Expected scaling risks:** [kernel count / transfer count / transfer volume / launch overhead / occupancy / none]
- **Larger practical profile size recommendation:** [value or rule]
- **Why this size materially exercises the GPU:** [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- **Hardware basis from `system_info_summary.txt`:** [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]

## Constraints
- Identify device-side enqueue (CRITICAL)
- Note image object usage
- Flag sub-group operations
