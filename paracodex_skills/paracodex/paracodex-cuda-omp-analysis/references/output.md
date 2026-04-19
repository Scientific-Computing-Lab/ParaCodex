# Analysis Output Template

## File Conversion Mapping
```
original.cu → converted.c
kernel_utils.cu → kernel_utils.cpp
```

## Kernel/Loop Nesting Structure
```
- host_loop (line:X) calls kernel1 
  └── kernel1<<<grid,block>>> (line:Y) Type A
      └── device_loop (line:Z) Type A
- kernel2<<<grid,block>>> (line:W) Type D
```

## Kernel/Loop Details
For each CRITICAL/IMPORTANT/SECONDARY kernel or loop:
```
## Kernel/Loop: [name] at [file:line]
- **Context:** [__global__ kernel / host loop / __device__ function]
- **Launch config:** [grid_size × block_size] or [iterations]
- **Total threads/iterations:** [count]
- **Type:** [A-G] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [device loops or none]
- **Dependencies:** [none / atomicAdd / __syncthreads / reduction]
- **Shared memory:** [YES/NO - size and usage]
- **Thread indexing:** [pattern used]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW) - memory type]
- **OMP Migration Issues:** [flags]
```

## Summary Table
|| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
||-----------------|------|----------|---------|------------|--------------|------------|

## CUDA-Specific Details
- **Dominant compute kernel:** [main timed kernel]
- **Memory transfers in timed loop?:** YES/NO
- **Shared memory usage:** [total bytes, patterns]
- **Synchronization points:** [__syncthreads locations]
- **Atomic operations:** [locations and variables]
- **Reduction patterns:** [manual vs atomicAdd]

## Structural Recommendations
- **Natural OpenMP offload unit:** [single region / fused routine / helper-by-helper]
- **Should CUDA kernel boundaries be preserved?** [YES/NO/PARTIALLY]
- **Fragmentation risk if migrated literally:** [none / low / medium / high]
- **Notes:** [kernel fusion / split / flattening guidance]

## Scalability Check
- **Default input likely too small to guide migration?** [YES/NO]
- **Small-input sensitive?** [YES/NO]
- **Expected scaling risks:** [kernel count / transfer count / transfer volume / host sync / occupancy / none]
- **Larger practical profile size recommendation:** [value or rule]
- **Why this size materially exercises the GPU:** [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- **Hardware basis from `system_info_summary.txt`:** [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]

## OMP Migration Strategy Notes
- **Direct kernel → parallel for:** [list]
- **Requires restructuring:** [list with reasons]
- **Performance concerns:** [atomics, false sharing, etc.]
- **Data management:** [allocation changes needed]
