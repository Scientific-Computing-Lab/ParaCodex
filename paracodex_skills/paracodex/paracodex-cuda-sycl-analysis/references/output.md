# Analysis Output Template

## File Conversion Mapping
```
original.cu → original.dp.cpp
main.cu → main.dp.cpp
```

## Migration Complexity
- **Overall Difficulty:** [Low/Medium/High]
- **CUDA Features:** [Texture/Surface/Dynamic Parallelism] - Hard to migrate.
- **Libraries used:** [cuBLAS, cuFFT, etc.] - Mapped to oneMKL?

## API Inventory
| API Group | Count | Notes |
|-----------|-------|-------|
| Memory (Malloc/Memcpy) | [N] | Auto-convertible to USM |
| Kernel Launch | [N] | Auto-convertible |
| Atomics | [N] | Supported |
| Warp Shuffle | [N] | Mapped to Sub-groups |

## Kernel Details
For each critical kernel:
```
## Kernel: [name]
- **Launch config:** [grid/block]
- **Sub-group Dependencies:** [YES/NO]
- **Shared Memory:** [YES/NO]
```

## Structural Recommendations
- **Natural SYCL decomposition:** [preserve kernels / fuse kernels / flatten helpers]
- **Preferred memory model:** [USM / buffers-accessors / mixed]
- **Fragmentation risk:** [none / low / medium / high]

## Scalability Check
- **Default input likely too small to guide migration?** [YES/NO]
- **Small-input sensitive?** [YES/NO]
- **Expected scaling risks:** [submission count / transfer count / host waits / occupancy / memory footprint / none]
- **Larger practical profile size recommendation:** [value or rule]
- **Why this size materially exercises the GPU:** [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- **Hardware basis from `system_info_summary.txt`:** [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]

## Action Items
1. [ ] Run `dpct` / `intercept-build`
2. [ ] Manually fix warnings
3. [ ] Replace Makefile compiler
