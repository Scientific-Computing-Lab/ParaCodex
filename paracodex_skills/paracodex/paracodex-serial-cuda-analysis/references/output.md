# Analysis Output Template

## Serial Hotspot Analysis
**Profile used:** [profile tool/output log]

| Function | Time (%) | Time (s) | Calls | Parallelizable? |
| :--- | :--- | :--- | :--- | :--- |
| `func_A` | 80% | 12.5s | 1 | YES (Data Independent) |
| `func_B` | 15% | 2.3s | 100 | NO (Recursive) |

## Data Analysis
- **Main Data Structures:** Arrays `A`, `B` (size N).
- **Data Movement:** Host → Device (Start) → Device → Host (End).
- **Size:** [X] MB total.

## Migration Strategy
- **Hotspot:** `func_A`
- **Approach:** Write CUDA Kernel `vector_add_kernel` for loop at line X.
- **Complexity:** Low (Embarrassingly Parallel).

## Structural Risk
- **Natural CUDA offload unit:** [single kernel / fused pipeline / multiple kernels]
- **Fragmentation risk:** [none / low / medium / high]
- **Data movement risk:** [none / low / medium / high]
- **Notes:** [why preserving helper structure is fine or harmful]

## Scalability Check
- **Default input likely too small to guide migration?** [YES/NO]
- **Small-input sensitive?** [YES/NO]
- **Expected scaling risks:** [kernel count / transfer count / transfer volume / host sync / occupancy / none]
- **Larger practical profile size recommendation:** [value or rule]
- **Why this size materially exercises the GPU:** [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- **Hardware basis from `system_info_summary.txt`:** [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]

## GPU Resource Estimation
- **Threads:** N = [Count].
- **Blocks:** N / 256.
- **Memory:** [X] MB < 16GB (Device Limit).
